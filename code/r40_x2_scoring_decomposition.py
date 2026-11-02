# -*- coding: utf-8 -*-
"""r40_x2_scoring_decomposition.py — 修订76(D 类):X2 金标交换的计分函数分解。

动机:B3 行 "X2, n≤10, sign undetermined at every n"(F1 口径)——本分解检验
  "符号不稳定"本身是否是计分函数契约的效果:同一批冻结预测、同一次金标交换,
  在不同计分函数 s 下的 Δ_s = gap_s(gD) − gap_s(gR)。
口径:20 个 X2 run(5 种子 × {r025,r100} × {D,R})冻结 per_doc 计数,
  表键 evaluation|{g}|fixed_0.5|sigmoid_multilabel|earliest_mention(E5 同口径);
  池化 R/F1/hit/exact;无新推断;配对 doc bootstrap(10k,seed 20260926)。
输出:statistics/r40_x2_scoring_decomposition.json
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "out/jcip_audit_v1/runs"
STAT = ROOT / "out/revision_v5_experiments/statistics"
SEED, BOOT = 20260926, 10000
KEY = "evaluation|{g}|fixed_0.5|sigmoid_multilabel|earliest_mention"
FIELDS = ("tp", "fp", "fn", "n_gold_pairs", "hit", "exact")
SCORES = ("recall", "f1", "hit", "exact")


def side_metrics(c):
    """c: {tp,fp,fn,n_gold_pairs,hit,exact} 逐文档数组 → 池化指标。"""
    tp, fp, fn = c["tp"].sum(), c["fp"].sum(), c["fn"].sum()
    ngp, hit, ex = c["n_gold_pairs"].sum(), c["hit"].sum(), c["exact"].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {"recall": float(np.nan_to_num(rec)),
            "f1": float(np.nan_to_num(f1)),
            "hit": float(hit / ngp) if ngp else 0.0,
            "exact": float(ex / ngp) if ngp else 0.0}


def load_run(rdir):
    out = {}
    for g in ("gD", "gR"):
        src = (rdir / "per_doc" / f"evaluation__{g}.json")
        if not src.exists():
            src = rdir / f"evaluation__{g}.json"
        tb = json.loads(src.read_text(encoding="utf-8"))[KEY.format(g=g)]
        out[g] = {side: {k: np.array(tb[side][k], dtype=np.int64) for k in FIELDS}
                  for side in ("intra", "inter")}
    return out


def gap(cnt):
    per_g = {}
    for g in ("gD", "gR"):
        i, e = side_metrics(cnt[g]["intra"]), side_metrics(cnt[g]["inter"])
        per_g[g] = {s: i[s] - e[s] for s in SCORES}
    return {s: per_g["gD"][s] - per_g["gR"][s] for s in SCORES}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    run_rows, docs = [], {}
    for rdir in sorted(RUNS.glob("ml_*")):
        has = ((rdir / "per_doc" / "evaluation__gD.json").exists()
               or (rdir / "evaluation__gD.json").exists())
        if not has:
            continue
        d = load_run(rdir)
        docs[rdir.name] = d
        g0 = gap(d)
        run_rows.append({"run": rdir.name, "n_docs": int(len(d["gD"]["intra"]["tp"])),
                         **{f"delta_{s}": round(g0[s], 6) for s in SCORES}})
    assert len(run_rows) == 20, f"expect 20 X2 runs, got {len(run_rows)}"
    print(f"runs: {len(run_rows)}")

    rng = np.random.default_rng(SEED)
    boot = {s: np.empty(BOOT) for s in SCORES}
    for b in range(BOOT):
        means = {s: [] for s in SCORES}
        for name, d in docs.items():
            n = len(d["gD"]["intra"]["tp"])
            idx = rng.integers(0, n, n)
            dd = {g: {side: {k: v[idx] for k, v in d[g][side].items()}
                      for side in ("intra", "inter")} for g in ("gD", "gR")}
            gb = gap(dd)
            for s in SCORES:
                means[s].append(gb[s])
        for s in SCORES:
            boot[s][b] = float(np.mean(means[s]))

    res = {"meta": {"amendment": 76, "n_runs": len(run_rows),
                    "table_key": KEY.format(g="{g}"),
                    "note": "X2 gold-swap delta decomposed by scoring function; frozen per_doc "
                            "counts; mean over 20 runs with paired doc bootstrap CI (10k, "
                            "seed 20260926); no new inference",
                    "ts": datetime.now().isoformat(timespec="seconds")},
           "per_run": run_rows, "summary": {}}
    for s in SCORES:
        pts = [r[f"delta_{s}"] for r in run_rows]
        res["summary"][s] = {
            "mean_delta": round(float(np.mean(pts)), 4),
            "sign_positive_runs": int(sum(p > 0 for p in pts)),
            "sign_negative_runs": int(sum(p < 0 for p in pts)),
            "ci95": [round(float(np.percentile(boot[s], 2.5)), 4),
                     round(float(np.percentile(boot[s], 97.5)), 4)],
            "ci_excludes_zero": bool(np.percentile(boot[s], 2.5) > 0
                                     or np.percentile(boot[s], 97.5) < 0)}
        print(f"{s:7s} delta={res['summary'][s]['mean_delta']:+.4f}  "
              f"+{res['summary'][s]['sign_positive_runs']}/-"
              f"{res['summary'][s]['sign_negative_runs']}  "
              f"CI[{res['summary'][s]['ci95'][0]:+.4f},{res['summary'][s]['ci95'][1]:+.4f}]")
    # ---- 分析2:训练源对比(D 训练 vs R 训练)的计分函数分解(legacy crossarm 的口径) ----
    pairs = {}
    for r in run_rows:
        parts = r["run"].split("_")          # ml_{D|R}_{r025|r100}_s{seed}
        arm, spar, seed = parts[1], parts[2], parts[3]
        pairs.setdefault((spar, seed), {})[arm] = r["run"]
    src_rows = []
    boot2 = {s: np.empty(BOOT) for s in SCORES}
    pair_maps = {}
    for (spar, seed), arms in sorted(pairs.items()):
        if set(arms) != {"D", "R"}:
            continue
        cntD, cntR = docs[arms["D"]], docs[arms["R"]]
        gD_run = gap(cntD)                    # D 训练 run 的 gap(gD)−gap(gR)
        gR_run = gap(cntR)
        row = {"spar": spar, "seed": seed}
        for s in SCORES:
            row[f"delta_{s}"] = round(gD_run[s] - gR_run[s], 6)
        src_rows.append(row)
        pair_maps[(spar, seed)] = (cntD, cntR)
    rng2 = np.random.default_rng(SEED)
    for b in range(BOOT):
        means = {s: [] for s in SCORES}
        for (spar, seed), (cntD, cntR) in pair_maps.items():
            n = len(cntD["gD"]["intra"]["tp"])
            idx = rng2.integers(0, n, n)
            dd = {tag: {g: {side: {k: v[idx] for k, v in c[g][side].items()}
                          for side in ("intra", "inter")} for g in ("gD", "gR")}
                  for tag, c in (("D", cntD), ("R", cntR))}
            gD_s, gR_s = gap(dd["D"]), gap(dd["R"])
            for sc in SCORES:
                means[sc].append(gD_s[sc] - gR_s[sc])
        for sc in SCORES:
            boot2[sc][b] = float(np.mean(means[sc]))
    res["source_contrast"] = {"note": "delta_s = [gap_s(D-trained) - gap_s(R-trained)] per "
                                      "(sparsity, seed) pair; legacy crossarm 口径",
                              "per_pair": src_rows, "summary": {}}
    for sc in SCORES:
        pts = [r[f"delta_{sc}"] for r in src_rows]
        res["source_contrast"]["summary"][sc] = {
            "mean_delta": round(float(np.mean(pts)), 4),
            "sign_positive_pairs": int(sum(p > 0 for p in pts)),
            "sign_negative_pairs": int(sum(p < 0 for p in pts)),
            "ci95": [round(float(np.percentile(boot2[sc], 2.5)), 4),
                     round(float(np.percentile(boot2[sc], 97.5)), 4)],
            "ci_excludes_zero": bool(np.percentile(boot2[sc], 2.5) > 0
                                     or np.percentile(boot2[sc], 97.5) < 0)}
        print(f"src {sc:7s} delta={res['source_contrast']['summary'][sc]['mean_delta']:+.4f}  "
              f"+{res['source_contrast']['summary'][sc]['sign_positive_pairs']}/-"
              f"{res['source_contrast']['summary'][sc]['sign_negative_pairs']}  "
              f"CI[{res['source_contrast']['summary'][sc]['ci95'][0]:+.4f},"
              f"{res['source_contrast']['summary'][sc]['ci95'][1]:+.4f}]")

    out = STAT / "r40_x2_scoring_decomposition.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
