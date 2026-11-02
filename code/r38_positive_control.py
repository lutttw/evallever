# -*- coding: utf-8 -*-
"""r38_positive_control.py — 修订73(冻结设计):契约侧合成阳性对照(ORACLE)。

载体:atlop_pe8 D 臂 s20260917(主)+ R 臂 s20260917(翻转检测)。
注入(嵌套,seed 20260926):eligible = ESS intra 且 native 解码 n_pred>0 的候选对;
  随机排列取前 k(δ) 对,将其已判阳性的关系列 logits 置 -1e9(不动 NA 列)。
  δ ∈ {0(sham), 0.01, 0.02, 0.05, 0.10}。
门 G1:sham 经未修改 e3 run_score 重算的 per_doc 必须与冻结件逐位一致(tp/fp/fn,gR+gD)。
端点:ΔΓ 方向/灵敏度+MDE、ΔL 杠杆移动、Γ_D(δ)−Γ_R(0) 翻转检出;10k doc bootstrap。
输出:out/revision_v5_experiments/statistics/r38_positive_control.json
"""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "jcip_audit"))

RETRAIN = ROOT / "out" / "revision_v5_experiments" / "independent_model_retrain"
PC = ROOT / "out" / "revision_v5_experiments" / "positive_control"
STAT = ROOT / "out" / "revision_v5_experiments" / "statistics"
CANON_KEYS = ROOT / "data" / "dreeam_repo" / "dreeam" / "dreeam_pe8_s20260917_l01" / "keys_evaluation.jsonl"

LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10]
SEED_INJ, BOOT, SEED_BOOT = 20260926, 10000, 20260926
RUN_D, RUN_R = "atlop_D_s20260917", "atlop_R_s20260917"


def log(*a):
    print(f"[PC {datetime.now().strftime('%H:%M:%S')}]", *a, flush=True)


def decode_native(logits, topk=4):
    th = logits[:, :1]
    mask = logits > th
    topv = np.sort(logits, axis=1)[:, -topk][:, None]
    return mask & (logits >= topv)


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def frozen_counts(path, tables):
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    for tb in tables:
        t = d["per_doc"][tb]
        low = {k.lower(): v for k, v in t["intra"].items()}
        out[tb] = {"docs": t["docs"],
                   "intra": np.array([low[c] for c in ("tp", "fp", "fn")], dtype=np.int64).T,
                   "inter": np.array([{k.lower(): v for k, v in t["inter"].items()}[c]
                                      for c in ("tp", "fp", "fn")], dtype=np.int64).T}
    return out


def pc_counts(run_id):
    d = json.load(open(PC / f"per_doc_{run_id}.json", encoding="utf-8"))
    out = {}
    for tb in ("evaluation|gR|atlop_native|earliest_mention",
               "evaluation|gD|atlop_native|earliest_mention"):
        t = d["per_doc"][tb]
        low = {k.lower(): v for k, v in t["intra"].items()}
        out[tb.split("|")[1]] = {
            "docs": t["docs"],
            "intra": np.array([low[c] for c in ("tp", "fp", "fn")], dtype=np.int64).T,
            "inter": np.array([{k.lower(): v for k, v in t["inter"].items()}[c]
                               for c in ("tp", "fp", "fn")], dtype=np.int64).T}
    return out


def pooled_gamma(cd, sel=None):
    arr = cd["intra"], cd["inter"]
    c = np.stack([(a if sel is None else a[sel]).sum(0) for a in arr]).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = c[:, 0] / (c[:, 0] + c[:, 1])
        r = c[:, 0] / (c[:, 0] + c[:, 2])
        f = 2 * p * r / (p + r)
    f = np.nan_to_num(f)
    return float(f[0] - f[1])


def ci(dist):
    return [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))]


def main() -> int:
    import e3_atlop_infer_score as E3
    import argparse

    PC.mkdir(parents=True, exist_ok=True)
    logits0 = np.load(RETRAIN / "runs" / RUN_D / "logits_evaluation.npy").astype(np.float64)
    keys = load_jsonl(CANON_KEYS)
    assert len(keys) == len(logits0)
    grp = np.array([k["group"] for k in keys])

    dec0 = decode_native(logits0)
    n_pred = dec0[:, 1:].sum(1)
    eligible = np.where((grp == 0) & (n_pred > 0))[0]
    log(f"eligible intra pairs with predictions: {len(eligible)}")

    rng = np.random.default_rng(SEED_INJ)
    perm = rng.permutation(eligible)

    results = {"meta": {
        "amendment": 73, "carrier": RUN_D, "flip_partner": RUN_R,
        "levels": LEVELS, "seed_injection": SEED_INJ, "boot": BOOT, "seed_boot": SEED_BOOT,
        "n_docs": 498, "n_eligible": int(len(eligible)),
        "note": "revision73 frozen design; nested injection via one permutation; "
                "unmodified e3 run_score; gate G1 = sham == frozen per_doc (tp/fp/fn, gR+gD)",
        "ts": datetime.now().isoformat(timespec="seconds")},
        "gate": {}, "levels": {}, "flip": {}}

    tables = ("evaluation|gR|atlop_native|earliest_mention",
              "evaluation|gD|atlop_native|earliest_mention")
    fz = frozen_counts(RETRAIN / f"per_doc_{RUN_D}.json", tables)

    for lev in LEVELS:
        rid = f"pc_d{int(round(lev * 1000)):03d}"
        rdir = PC / "runs" / rid
        rdir.mkdir(parents=True, exist_ok=True)
        logits = logits0.copy()
        k = int(round(lev * len(eligible)))
        sel = perm[:k]
        if k:
            sub = dec0[sel].copy()
            sub[:, 0] = False                       # NA 列不动
            logits[np.asarray(sel)[:, None].repeat(sub.shape[1], 1)[sub]] = -1e9
        np.save(rdir / "logits_evaluation.npy", logits)
        shutil.copy(CANON_KEYS, rdir / "keys_evaluation.jsonl")
        E3.run_score(argparse.Namespace(dir=str(PC), run_id=rid))
        cc = pc_counts(rid)
        row = {"k_cleared": k}
        if lev == 0.0:
            ok = all(np.array_equal(cc[g][side], fz[tb][side])
                     for g, tb in (("gR", tables[0]), ("gD", tables[1]))
                     for side in ("intra", "inter"))
            ok_doc = all(cc[g]["docs"] == fz[tb]["docs"]
                         for g, tb in (("gR", tables[0]), ("gD", tables[1])))
            results["gate"]["G1_sham_equals_frozen"] = bool(ok and ok_doc)
            log(f"G1 sham == frozen: {ok and ok_doc}")
            if not (ok and ok_doc):
                json.dump(results, open(STAT / "r38_positive_control.json", "w",
                                        encoding="utf-8"), ensure_ascii=False, indent=1)
                print("[GATE FAIL] sham != frozen — abort per amendment 73")
                return 1
        for g in ("gR", "gD"):
            row[f"gamma_{g}"] = round(pooled_gamma(cc[g]), 6)
        results["levels"][rid] = row

    # 端点:ΔΓ / ΔL / 翻转,doc-level 配对 bootstrap
    gam = {rid: pc_counts(rid) for rid in results["levels"]}
    sham = [r for r in results["levels"] if results["levels"][r]["k_cleared"] == 0][0]
    docs0 = gam[sham]["gR"]["docs"]
    n = len(docs0)
    rngb = np.random.default_rng(SEED_BOOT)
    idxs = rngb.integers(0, n, (BOOT, n))

    for rid, row in results["levels"].items():
        d0 = pooled_gamma(gam[rid]["gR"]) - pooled_gamma(gam[sham]["gR"])
        dg = np.array([pooled_gamma(gam[rid]["gR"], i) - pooled_gamma(gam[sham]["gR"], i)
                       for i in idxs])
        l0 = abs(pooled_gamma(gam[rid]["gR"]) - pooled_gamma(gam[rid]["gD"])) - \
             abs(pooled_gamma(gam[sham]["gR"]) - pooled_gamma(gam[sham]["gD"]))
        dl = np.array([abs(pooled_gamma(gam[rid]["gR"], i) - pooled_gamma(gam[rid]["gD"], i))
                       - abs(pooled_gamma(gam[sham]["gR"], i) - pooled_gamma(gam[sham]["gD"], i))
                       for i in idxs])
        row["dgamma"] = {"point": round(float(d0), 6), "ci95": [round(x, 6) for x in ci(dg)]}
        row["dleverage"] = {"point": round(float(l0), 6), "ci95": [round(x, 6) for x in ci(dl)]}
        row["dgamma_ci_excl0"] = bool(ci(dg)[0] > 0 or ci(dg)[1] < 0)
        row["dleverage_ci_excl0"] = bool(ci(dl)[0] > 0 or ci(dl)[1] < 0)
        log(f"{rid}: ΔΓ={d0:+.4f} CI[{ci(dg)[0]:+.4f},{ci(dg)[1]:+.4f}] "
            f"| ΔL={l0:+.4f} CI[{ci(dl)[0]:+.4f},{ci(dl)[1]:+.4f}]")

    # H-PC4 翻转检出:Γ_D(δ) − Γ_R(0) under gR
    fzr = frozen_counts(RETRAIN / f"per_doc_{RUN_R}.json", (tables[0],))
    assert fzr[tables[0]]["docs"] == docs0, "doc order mismatch D vs R"
    g_r0 = pooled_gamma(fzr[tables[0]])
    g_r0_b = np.array([pooled_gamma(fzr[tables[0]], i) for i in idxs])
    s0 = np.sign(pooled_gamma(gam[sham]["gR"]) - g_r0)
    for rid, row in results["levels"].items():
        pt = pooled_gamma(gam[rid]["gR"]) - g_r0
        bs = np.array([pooled_gamma(gam[rid]["gR"], i) - g_r0_b[j]
                       for j, i in enumerate(idxs)])
        c = ci(bs)
        row_flip = {"point": round(float(pt), 6), "ci95": [round(x, 6) for x in c],
                    "sign_flipped_vs_sham": bool(np.sign(pt) != s0 and pt != 0),
                    "ci_excl0": bool(c[0] > 0 or c[1] < 0)}
        results["flip"][rid] = row_flip
        log(f"flip {rid}: Γ_D−Γ_R={pt:+.4f} CI[{c[0]:+.4f},{c[1]:+.4f}] "
            f"flip={row_flip['sign_flipped_vs_sham']} excl0={row_flip['ci_excl0']}")

    # 汇总判定
    lvls = [results["levels"][r] for r in sorted(results["levels"],
            key=lambda r: results["levels"][r]["k_cleared"])]
    pc2 = all(v["dgamma"]["point"] < 0 for v in lvls if v["k_cleared"] > 0)
    mde = next((v["k_cleared"] / len(eligible) for v in lvls
                if v["k_cleared"] > 0 and v["dgamma_ci_excl0"]), None)
    pc3 = lvls[-1]["dgamma_ci_excl0"]
    dl_mono = all(abs(lvls[i + 1]["dleverage"]["point"]) >= abs(lvls[i]["dleverage"]["point"])
                  for i in range(len(lvls) - 1))
    flip_min = next(({"delta": k / len(eligible), **results["flip"][r]}
                     for r in sorted(results["levels"],
                                     key=lambda r: results["levels"][r]["k_cleared"])
                     for k in [results["levels"][r]["k_cleared"]]
                     if results["flip"][r]["sign_flipped_vs_sham"]
                     and results["flip"][r]["ci_excl0"]), None)
    results["summary"] = {
        "G1": results["gate"].get("G1_sham_equals_frozen"),
        "H_PC2_direction_all_negative": bool(pc2),
        "H_PC3_max_level_excl0": bool(pc3),
        "MDE_delta_smallest_ci_excl0": mde,
        "H_PC5_leverage_monotone_descriptive": bool(dl_mono),
        "H_PC5_max_level_excl0": lvls[-1]["dleverage_ci_excl0"],
        "H_PC4_flip_detected": flip_min,
        "gamma_sham_gR": round(pooled_gamma(gam[sham]["gR"]), 6),
        "gamma_sham_gD": round(pooled_gamma(gam[sham]["gD"]), 6),
        "gamma_R0_gR": round(g_r0, 6),
        "verdict": "PASS" if (results["gate"].get("G1_sham_equals_frozen")
                              and pc2 and pc3) else "FAIL",
    }
    log(f"VERDICT: {results['summary']['verdict']} | MDE δ={mde} | flip={flip_min}")
    json.dump(results, open(STAT / "r38_positive_control.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("saved ->", STAT / "r38_positive_control.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
