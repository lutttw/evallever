# -*- coding: utf-8 -*-
"""r31_ttmre_arm.py — 修订62 执行:TTM-RE v0.2 第 8 生成器臂接入契约管线。

输入(冻结):data/ttmre_work/ttmre_dev_logits.npz(r31_ttmre_infer 产出,
998 docs × 395,572 对 × 97 logits,TTM-RE rel2id 列序)。
关键对齐:
  (1) 列置换:TTM-RE rel2id ≠ 本文 rel2id(同为 97 类不同排列)——按关系名
      重排列序(Na=0 不动,96 名 ours[n]=ttm[n] 一一映射,断言双射);
  (2) 实体空间:候选 (h,t) 为签名索引 → 经 canonical manifest common_signatures
      与 docred_aligned/dev_revised.json 的 vertexSet 映射到原始 vertex 索引,
      再查 TTM-RE 该 doc 的 (vh,vt) 行(双向全对枚举已核实)。

产出:
  out/revision_v5_experiments/ttmre/runs/ttmre_s74_release02/logits_evaluation.npy
  out/revision_v5_experiments/ttmre/runs/ttmre_s74_release02/keys_evaluation.jsonl
  out/revision_v5_experiments/statistics/r31_ttmre.json(端点见修订62)
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
RUN_DIR = ROOT / "out" / "revision_v5_experiments" / "ttmre" / "runs" / "ttmre_s74_release02"
OUTD = ROOT / "out" / "revision_v5_experiments" / "statistics" / "r31_ttmre.json"
BOOT, SEED = 10000, 20260926
R1 = ROOT / "out" / "revision_v5_experiments" / "statistics" / "r1_rank_flip.json"


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def canon(o):
    return json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def doc_id_of(d):
    import hashlib
    return hashlib.sha256(canon({"title": str(d.get("title", "")).strip(),
                                 "sents": d["sents"]}).encode("utf-8")).hexdigest()


def sig_of(v):
    return tuple(sorted({(m["sent_id"], m["pos"][0], m["pos"][1]) for m in v}))


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
    # ---------- 1. 列置换(完整性门 1) ----------
    ours = json.load(open(ROOT / "data/docred_local/rel2id.json", encoding="utf-8"))
    ttm = json.load(open(ROOT / "data/ttmre_work/repo/meta/rel2id.json", encoding="utf-8"))
    assert ours["Na"] == 0 and ttm["Na"] == 0
    names = [n for n in ours if n != "Na"]
    assert len(names) == 96 and set(names) == {n for n in ttm if n != "Na"}
    assert len({ours[n] for n in names}) == 96 and len({ttm[n] for n in names}) == 96
    perm = np.zeros(97, dtype=np.int64)          # perm[target]=source
    perm[0] = 0
    for n in names:
        perm[ours[n]] = ttm[n]

    # ---------- 2. 冻结 logits ----------
    z = np.load(ROOT / "data/ttmre_work/ttmre_dev_logits.npz", allow_pickle=True)
    titles_arr = [str(t) for t in z["titles"]]
    pair_counts = z["pair_counts"].astype(np.int64)
    hts = z["hts"].astype(np.int64)
    logits_raw = z["logits"].astype(np.float64)
    offsets = np.concatenate([[0], np.cumsum(pair_counts)])
    doc_row0 = {t: i for i, t in enumerate(titles_arr)}
    lookup = {}
    for di, t in enumerate(titles_arr):
        rows = hts[offsets[di]:offsets[di + 1]]
        lookup[t] = {(int(a), int(b)): offsets[di] + k for k, (a, b) in enumerate(rows)}
    print(f"[frozen] docs={len(titles_arr)} pairs={logits_raw.shape[0]} dim={logits_raw.shape[1]}")

    # ---------- 3. 实体空间映射(完整性门 2) ----------
    manifest = {}
    for line in (ROOT / "out/jcip_audit_v1/manifests/canonical_manifest.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        manifest.setdefault(row["split"], {})[row["doc_id"]] = [
            tuple((int(a), int(b), int(c)) for a, b, c in x) for x in row["common_signatures"]]
    sig = manifest.get("evaluation") or next(iter(manifest.values()))
    D = json.load(open(ROOT / "data/docred_aligned/dev_revised.json", encoding="utf-8"))
    id2doc = {doc_id_of(d): d for d in D}
    cands = load_jsonl(PREP / "candidates/evaluation.jsonl")
    print(f"[cands] {len(cands)} pairs, {len({c['doc_id'] for c in cands})} docs")

    # 每个 doc:sig 索引 → 原始 vertex 索引(D 的 vertexSet 内唯一签名)
    sig2ver = {}
    for did, sigs in sig.items():
        d = id2doc.get(did)
        assert d is not None, f"candidate doc missing in aligned dev: {did[:12]}"
        m = defaultdict(list)
        for i, v in enumerate(d["vertexSet"]):
            m[sig_of(v)].append(i)
        smap = []
        for s in sigs:
            hits = m.get(s, [])
            assert len(hits) == 1, f"sig not unique in doc {did[:12]}: {s} -> {hits}"
            smap.append(hits[0])
        sig2ver[did] = smap

    # ---------- 4. 组装 canonical logits(完整性门 3:行覆盖) ----------
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    N = len(cands)
    out_logits = np.zeros((N, 97), dtype=np.float64)
    missing = 0
    for i, c in enumerate(cands):
        did = c["doc_id"]
        d = id2doc[did]
        title = str(d.get("title", "")).strip()
        lk = lookup.get(title)
        if lk is None:
            missing += 1
            continue
        vh = sig2ver[did][int(c["h"])]
        vt = sig2ver[did][int(c["t"])]
        row = lk.get((vh, vt))
        assert row is not None, f"pair ({vh},{vt}) not enumerated for {title}"
        out_logits[i] = logits_raw[row]
    assert missing == 0, f"{missing} candidate docs absent from TTM-RE inference"
    # 门:置换后逐行有限、置换为双射
    out_logits = out_logits[:, perm]
    assert np.isfinite(out_logits).all()
    keys = [{"doc_id": c["doc_id"], "h": c["h"], "t": c["t"]} for c in cands]
    np.save(RUN_DIR / "logits_evaluation.npy", out_logits)
    with open(RUN_DIR / "keys_evaluation.jsonl", "w", encoding="utf-8") as f:
        for k in keys:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"[assembled] logits_evaluation.npy {out_logits.shape} -> {RUN_DIR.name}")

    # ---------- 5. 金标/分组/翻转端点 ----------
    gold_gD, gold_gR = {}, {}
    for row in load_jsonl(PREP / "golds_gD.jsonl"):
        gold_gD[(row["doc_id"], int(row["h"]), int(row["t"]))] = set(row["rels"])
    for row in load_jsonl(PREP / "golds_gR.jsonl"):
        gold_gR[(row["doc_id"], int(row["h"]), int(row["t"]))] = set(row["rels"])
    rel2id = json.load(open(PREP / "relation_to_index.json", encoding="utf-8"))

    n = N
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

    pred = decode_native(out_logits)[:, 1:]
    n_pred = pred.sum(1)
    density = float(n_pred.mean())

    def pair_counts_against(gsets):
        tp = np.zeros(n, dtype=np.int64)
        for i in np.where(n_pred > 0)[0]:
            if gsets[i]:
                tp[i] = len(set(np.where(pred[i])[0].tolist()) & gsets[i])
        fp = n_pred - tp
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

    tpD, fpD, fnD = pair_counts_against(gset_D)
    tpR, fpR, fnR = pair_counts_against(gset_R)
    g_gD, cnt_gD_ess = gamma(tpD, fpD, fnD, g_ess)      # r1 口径:ESS 分组
    g_gR, cnt_gR_ess = gamma(tpR, fpR, fnR, g_ess)
    g_ess_R = g_gR
    g_ass_R, cnt_ass_R = gamma(tpR, fpR, fnR, g_ass)

    def prf(gsets):
        tp = np.zeros(n, dtype=np.int64)
        for i in np.where(n_pred > 0)[0]:
            if gsets[i]:
                tp[i] = len(set(np.where(pred[i])[0].tolist()) & gsets[i])
        p_ = tp.sum() / max(1, n_pred.sum())
        r_ = tp.sum() / max(1, sum(len(s) for s in gsets))
        f_ = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        return round(float(p_), 4), round(float(r_), 4), round(float(f_), 4)

    f1_D, f1_R = prf(gset_D), prf(gset_R)

    rng = np.random.default_rng(SEED)
    arm_gold = abs(g_gD - g_gR)
    arm_grp = abs(g_ass_R - g_ess_R)
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

    # 翻转检查(描述性):vs r1 gaps 的 7 生成器(带符号 Γ_gD−Γ_gR)
    r1 = json.load(open(R1, encoding="utf-8"))

    def mean_point(gold_block):
        vals = [p["point"] for p in gold_block["per_run_points"]]
        return float(np.mean(vals))

    flips = {}
    for gen, blk in r1["gaps"].items():
        if "gold_gD" not in blk or "gold_gR" not in blk:
            continue
        d_gen = mean_point(blk["gold_gD"]) - mean_point(blk["gold_gR"])
        d_t = g_gD - g_gR
        flips[gen] = {"delta_gen": round(d_gen, 4), "delta_ttmre": round(d_t, 4),
                      "flip": bool(d_gen * d_t < 0)}

    results = {
        "meta": {
            "arm": "ttmre (TTM-RE v0.2 release finetune_state_dict, roberta-large, "
                   "Re-DocRED train S-PU seed74 — train-data confound disclosed)",
            "rel2id_permutation": "TTM-RE columns permuted by relation name to pipeline order (bijective, Na fixed)",
            "entity_space": "signature index -> aligned DocRED vertex via canonical manifest; "
                            "TTM-RE pair rows looked up by (vertex_h, vertex_t) over all-pairs enumeration",
            "decode": "atlop_native (same as r29); thresholds contract f as registered",
            "boot": BOOT, "seed": SEED,
            "amendment": 62,
        },
        "ttmre": {
            "gamma_gD": round(g_gD, 4), "gamma_gR": round(g_gR, 4),
            "arm_gold": round(arm_gold, 4), "arm_gold_ci95": [round(x, 4) for x in ci_gold],
            "gamma_ess_gR": round(g_ess_R, 4), "gamma_ass_gR": round(g_ass_R, 4),
            "arm_grouping": round(arm_grp, 4), "arm_grouping_ci95": [round(x, 4) for x in ci_grp],
            "density_mean_rels_per_pair": round(density, 4),
            "f1_vs_gD": list(f1_D), "f1_vs_gR": list(f1_R),
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
    print(json.dumps(results["ttmre"], ensure_ascii=False, indent=1))
    print("flips:", {k: v["flip"] for k, v in flips.items()})
    print("saved ->", OUTD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
