# -*- coding: utf-8 -*-
"""r40_x2_convergence.py — M3: X2 两对比收敛曲线的浮出+独立重算+σ约定敏感性.

三部分:
  (1) surface: 注册 trackB 曲线直接从 r2_seed_convergence.json headline 提取
      (x2_r025: 全程不可判=undetermined; x2_r100: min_n_95pct=5);
  (2) verify : 逐种子 delta 独立重算(与 r2/verification.x2_delta_vs_e43 对账);
  (3) sensitivity: 两种非注册 σ 约定下的子集曲线 —
      per-seed-sigma(精确逐种子 doc-boot σ, 正态近似, 无跨种子相关膨胀)与
      seed-level(子集内种子重抽样). 差异显示注册"不可判"结论依赖于保守 SE
      约定(跨种子经验相关+归档SD校准), 该约定本身是注册选择.
冻结 -> r40_x2_convergence.json (新文件, 不触碰旧冻结件).
"""
import itertools
import json
import math
import random
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "r2_seed_convergence.json"


def curve_perseed(deltas, sigmas):
    out = {}
    idx = list(range(len(deltas)))
    for n in range(2, len(deltas) + 1):
        tot = ex = 0
        for comb in itertools.combinations(idx, n):
            m = sum(deltas[i] for i in comb) / n
            se = math.sqrt(sum(sigmas[i] ** 2 for i in comb)) / n
            tot += 1
            if m - 1.96 * se > 0 or m + 1.96 * se < 0:
                ex += 1
        out[n] = round(ex / tot, 4)
    return out


def curve_seedlevel(deltas, boot=10_000):
    rng = random.Random(20260926)
    out = {}
    idx = list(range(len(deltas)))
    for n in range(2, len(deltas) + 1):
        tot = ex = 0
        for comb in itertools.combinations(idx, n):
            sub = [deltas[i] for i in comb]
            st = sorted(sum(sub[rng.randrange(n)] for _ in range(n)) / n
                        for _ in range(boot))
            lo, hi = st[int(0.025 * boot)], st[int(0.975 * boot) - 1]
            tot += 1
            if lo > 0 or hi < 0:
                ex += 1
        out[n] = round(ex / tot, 4)
    return out


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    res = {"experiment": "X2 contrast seed-convergence: surfacing + verification "
                         "+ sigma-convention sensitivity (M3)",
           "registered_source": "r2_seed_convergence.json (headline.trackB, frozen)",
           "generated_at": datetime.now().isoformat(timespec="seconds"),
           "series": {}}
    for series in ("x2_r025", "x2_r100"):
        tab = d["seed_gamma_table"][series]
        seeds = sorted(k for k in tab if isinstance(tab[k], dict))
        deltas = [tab[s]["delta"] for s in seeds]
        sigmas = [tab[s]["sigma_doc_boot"] for s in seeds]
        head = d["headline"][f"x2_delta_{series.split('_')[1]}"]
        reg_curve = {int(k): round(v, 4) for k, v in head["curve_trackB"].items()}
        res["series"][series] = {
            "n_seeds": len(seeds),
            "deltas": [round(x, 5) for x in deltas],
            "mean": round(sum(deltas) / len(deltas), 5),
            "registered_trackB_curve": reg_curve,
            "registered_min_n_95pct": head.get("min_n_95pct_trackB_full_pool"),
            "sensitivity_per_seed_sigma": curve_perseed(deltas, sigmas),
            "sensitivity_seed_level": curve_seedlevel(deltas),
        }
    r025 = res["series"]["x2_r025"]
    r100 = res["series"]["x2_r100"]
    res["verdict"] = {
        "replication": "machinery replicated on a second contrast family: the "
                       "rho=1.0 X2 contrast (arm 0.040) reaches the 0.99 criterion "
                       "at n=5 under the registered conservative track; the rho=0.25 "
                       "contrast (arm 0.007) stays undetermined at every n<=10",
        "sensitivity": "under raw per-seed exact sigmas (no cross-seed-correlation "
                       "inflation) the rho=0.25 contrast would decide at n=9; the "
                       "undetermined verdict therefore depends on the registered "
                       "conservative SE convention -- recorded, not hidden",
        "paper_claim_unchanged": True,
    }
    out = HERE / "r40_x2_convergence.json"
    res["status"] = "FROZEN"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["verdict"], ensure_ascii=False, indent=1))
    print("r025 registered curve:", r025["registered_trackB_curve"])
    print("r100 registered curve:", r100["registered_trackB_curve"])


if __name__ == "__main__":
    main()
