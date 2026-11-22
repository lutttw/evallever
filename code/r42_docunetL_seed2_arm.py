# -*- coding: utf-8 -*-
"""r42_docunetL_seed2_arm.py — 修订77:docunet_large 第二种子(20260918,本地5090)臂计分。

与 r32 同机器;差异仅 run 目录与产出文件;另含与 s20260917 的双种子池化。

复用 r31 机器;logits 已由 docunet_large_infer_score 产出(本管线列序,含
relation_to_index 置换)。端点(修订61 冻结):
  1) 金标臂 |Γ(gD)−Γ(gR)| 与解码(分组)臂 |Γ(ESS)−Γ(ASS)|(gR, atlop_native),
     doc bootstrap 10k / seed 20260926——解码臂即 B-W-b 混淆检验对照
     (docunet bert-base 快速重训的 0.088);
  2) 与既有 7 生成器的金标方向翻转(描述性);
  3) 密度点;4) dev F1(云端训练日志 0.6219,998-doc 全量口径,仅参照)。
产出 out/revision_v5_experiments/statistics/r32_docunet_large.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "jcip_audit"))
import eval_spec as ES

PREP = ROOT / "out" / "lre_rebuttal_v2" / "prepared"
RUN_DIR = ROOT / "out" / "revision_v5_experiments" / "docunet_large" / "runs" / "docunetL_pe8_s20260918"
OUTD = ROOT / "out" / "revision_v5_experiments" / "statistics" / "r42_docunet_large_seed2.json"
R1 = ROOT / "out" / "revision_v5_experiments" / "statistics" / "r1_rank_flip.json"
BOOT, SEED = 10000, 20260926


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def canon(o):
    return json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def doc_id_of(d):
    import hashlib
    return hashlib.sha256(canon({"title": str(d.get("title", "")).strip(),
                                 "sents": d["sents"]}).encode("utf-8")).hexdigest()


def decode_native(logits):
    th = logits[:, :1]
    mask = logits > th
    topv = np.sort(logits, axis=1)[:, -4][:, None]
    mask &= logits >= topv
    return mask


def metrics(cnt):
    cnt = np.asarray(cnt, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = cnt[..., 0] / (cnt[..., 0] + cnt[..., 1])
        r = cnt[..., 0] / (cnt[..., 0] + cnt[..., 2])
        f = 2 * p * r / (p + r)
    return np.stack([np.nan_to_num(p), np.nan_to_num(r), np.nan_to_num(f)], axis=-1)


def main() -> int:
    logits = np.load(RUN_DIR / "logits_evaluation.npy").astype(np.float64)
    assert np.isfinite(logits).all()
    cands = load_jsonl(PREP / "candidates/evaluation.jsonl")
    assert len(logits) == len(cands)

    manifest = {}
    for line in (ROOT / "out/jcip_audit_v1/manifests/canonical_manifest.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        manifest.setdefault(row["split"], {})[row["doc_id"]] = [
            tuple((int(a), int(b), int(c)) for a, b, c in x) for x in row["common_signatures"]]
    sig = manifest.get("evaluation") or next(iter(manifest.values()))

    gold_gD, gold_gR = {}, {}
    for row in load_jsonl(PREP / "golds_gD.jsonl"):
        gold_gD[(row["doc_id"], int(row["h"]), int(row["t"]))] = set(row["rels"])
    for row in load_jsonl(PREP / "golds_gR.jsonl"):
        gold_gR[(row["doc_id"], int(row["h"]), int(row["t"]))] = set(row["rels"])
    rel2id = json.load(open(PREP / "relation_to_index.json", encoding="utf-8"))

    n = len(cands)
    g_ess = np.zeros(n, dtype=np.int8)
    g_ass = np.zeros(n, dtype=np.int8)
    gset_D, gset_R = [], []
    for i, c in enumerate(cands):
        sh, st = sig[c["doc_id"]][int(c["h"])], sig[c["doc_id"]][int(c["t"])]
        g_ess[i] = ES.GROUP_FN["earliest_mention"](sh, st)
        g_ass[i] = ES.GROUP_FN["any_shared_sentence"](sh, st)
        assert g_ess[i] == int(c["group"])
        gset_D.append({rel2id[r] for r in gold_gD.get((c["doc_id"], int(c["h"]), int(c["t"])), ()) if r in rel2id})
        gset_R.append({rel2id[r] for r in gold_gR.get((c["doc_id"], int(c["h"]), int(c["t"])), ()) if r in rel2id})
    doc_ids = sorted({c["doc_id"] for c in cands})
    didx = np.array([{d: i for i, d in enumerate(doc_ids)}[c["doc_id"]] for c in cands])
    n_docs = len(doc_ids)
    doc_slices = [np.where(didx == j)[0] for j in range(n_docs)]

    pred = decode_native(logits)[:, 1:]
    n_pred = pred.sum(1)
    density = float(n_pred.mean())

    # sigmoid-0.5 校准解码(阈值维/校准臂,B-W-b 混淆检验的对照口径:
    # docunet_bert 的 native vs sigmoid 臂 = 0.088)
    probs = 1.0 / (1.0 + np.exp(-logits[:, 1:]))
    pred_sig = probs > 0.5
    n_pred_sig = pred_sig.sum(1)

    def pair_counts_against(gsets, pr, npr):
        tp = np.zeros(n, dtype=np.int64)
        for i in np.where(npr > 0)[0]:
            if gsets[i]:
                tp[i] = len(set(np.where(pr[i])[0].tolist()) & gsets[i])
        fp = npr - tp
        fn = np.array([len(s) for s in gsets], dtype=np.int64) - tp
        return tp, fp, fn

    def gamma(tp, fp, fn, grp):
        cnt = np.zeros((n_docs, 2, 3), dtype=np.int64)
        for j, sl in enumerate(doc_slices):
            for side, gv in ((0, 0), (1, 1)):
                m = sl[grp[sl] == gv]
                cnt[j, side] = (tp[m].sum(), fp[m].sum(), fn[m].sum())
        m = metrics(cnt.sum(0))
        return float(m[0, 2] - m[1, 2]), cnt

    tpD, fpD, fnD = pair_counts_against(gset_D, pred, n_pred)
    tpR, fpR, fnR = pair_counts_against(gset_R, pred, n_pred)
    g_gD, cnt_gD_ess = gamma(tpD, fpD, fnD, g_ess)
    g_gR, cnt_gR_ess = gamma(tpR, fpR, fnR, g_ess)
    g_ass_R, cnt_ass_R = gamma(tpR, fpR, fnR, g_ass)

    tpS, fpS, fnS = pair_counts_against(gset_R, pred_sig, n_pred_sig)
    g_sig_R, cnt_sig_R = gamma(tpS, fpS, fnS, g_ess)
    arm_threshold = abs(g_gR - g_sig_R)

    rng = np.random.default_rng(SEED)
    arm_gold = abs(g_gD - g_gR)
    arm_grp = abs(g_ass_R - g_gR)
    boot_gold, boot_grp = np.zeros(BOOT), np.zeros(BOOT)
    for b in range(BOOT):
        idx = rng.integers(0, n_docs, n_docs)
        mD = metrics(cnt_gD_ess[idx].sum(0))
        mR = metrics(cnt_gR_ess[idx].sum(0))
        mA = metrics(cnt_ass_R[idx].sum(0))
        boot_gold[b] = abs((mD[0, 2] - mD[1, 2]) - (mR[0, 2] - mR[1, 2]))
        boot_grp[b] = abs((mA[0, 2] - mA[1, 2]) - (mR[0, 2] - mR[1, 2]))
    ci_gold = [float(np.percentile(boot_gold, 2.5)), float(np.percentile(boot_gold, 97.5))]
    ci_grp = [float(np.percentile(boot_grp, 2.5)), float(np.percentile(boot_grp, 97.5))]

    r1 = json.load(open(R1, encoding="utf-8"))

    def mean_point(blk):
        return float(np.mean([p["point"] for p in blk["per_run_points"]]))

    flips = {}
    for gen, blk in r1["gaps"].items():
        if "gold_gD" not in blk or "gold_gR" not in blk:
            continue
        d_gen = mean_point(blk["gold_gD"]) - mean_point(blk["gold_gR"])
        d_t = g_gD - g_gR
        flips[gen] = {"delta_gen": round(d_gen, 4), "delta_L": round(d_t, 4),
                      "flip": bool(d_gen * d_t < 0)}

    results = {
        "meta": {
            "arm": "docunet_large (official published config: RoBERTa-large, trained on cloud "
                   "RTX 24GB GPU 24GB, seed 20260917, dev F1 0.6219 on 998-doc PE8 protocol; "
                   "amendment 61 Exp-B)",
            "decode": "atlop_native; grouping per registered rules",
            "boot": BOOT, "seed": SEED, "amendment": 61,
        },
        "docunet_large": {
            "gamma_gD": round(g_gD, 4), "gamma_gR": round(g_gR, 4),
            "arm_gold": round(arm_gold, 4), "arm_gold_ci95": [round(x, 4) for x in ci_gold],
            "gamma_ess_gR": round(g_gR, 4), "gamma_ass_gR": round(g_ass_R, 4),
            "arm_grouping": round(arm_grp, 4), "arm_grouping_ci95": [round(x, 4) for x in ci_grp],
            "gamma_sig05_gR": round(g_sig_R, 4),
            "arm_threshold_native_vs_sig": round(arm_threshold, 4),
            "density_mean_rels_per_pair": round(density, 4),
        },
        "confound_check": {
            "docunet_bert_threshold_arm": 0.088,
            "docunetL_threshold_arm": round(arm_threshold, 4),
            "reading": "published-config threshold (calibration) arm vs docunet_bert 0.088 "
                       "decides B-W-b weak-training confound",
        },
        "flips_vs_7": flips,
        "summary": {
            "n_flip_goldswap": sum(v["flip"] for v in flips.values()),
            "arm_gold_excl0": bool(ci_gold[0] > 0),
            "arm_grouping_excl0": bool(ci_grp[0] > 0),
        },
    }
    OUTD.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUTD, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(results["docunet_large"], ensure_ascii=False, indent=1))
    print("confound:", results["confound_check"])
    print("flips:", {k: v["flip"] for k, v in flips.items()})
    print("saved ->", OUTD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
