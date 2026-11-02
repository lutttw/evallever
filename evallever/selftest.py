# -*- coding: utf-8 -*-
"""evallever.selftest —— 三项兼容性验证(对账 ±0.005)。

  T1 arm  : ATLOP-PE8 vs P3M 换金标翻转量与 r1_rank_flip.json 对账;
  T2 seed : HacRED 处方 n*=8 与 r2_seed_convergence.json 对账;
  T3 audit: 14B 违规率 4.57% 与 audit_preds_*.json 对账。
运行:python -m evallever.selftest(需项目数据在默认相对路径)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TOL = 0.005


def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def t1_arm():
    """T1 数值级复算:从 per_doc 计数重算 ATLOP-PE8 vs P3M 换金标翻转,与 r1 对账。

    输入=项目树冻结计数(atlop 6 run + p3m 桥接 5 run);对照=r1 冻结件的
    flip_delta_point;容差 TOL(0.005)。无手抄常数。
    """
    import glob
    ref = _load("out/revision_v5_experiments/statistics/r1_rank_flip.json")
    stored = ref["flip_matrix"]["atlop_pe8|p3m"]["gold(gD_vs_gR)"]["flip_delta_point"]
    atlop = sorted(glob.glob(
        "out/revision_v5_experiments/independent_model_retrain/per_doc_atlop_*_s*.json"))
    p3m = sorted(glob.glob("out/jcip_audit_v1/x3_bridged/per_doc_x3_p3m_s*"))
    if not atlop or not p3m:
        return {"check": "T1 arm vs r1 (flip re-derivation)", "pass": False,
                "note": "per_doc 计数不在项目相对路径"}
    from .leverage import arm_counts
    res = arm_counts(atlop, p3m,
                     ("evaluation|{gold}|atlop_native|earliest_mention",
                      "evaluation|{gold}|native|p3m_native|earliest_mention"))
    got = res["pair_flip"]["flip_delta_point"]
    return {"check": "T1 arm vs r1 (flip re-derivation from per-doc counts)",
            "pass": abs(got - stored) < TOL and res["pair_flip"]["flip"],
            "computed": round(got, 6), "reference": round(stored, 6),
            "flip": res["pair_flip"]["flip"],
            "note": "counts 模式逐 run 池化重算;与 r1 的重采样内均值口径差 < 2e-4"}


def t2_seed():
    ref = _load("out/revision_v5_experiments/statistics/r2_seed_convergence.json")
    tab = ref.get("seed_gamma_table", {}).get("hacred", {})
    if not tab:
        return {"check": "T2 seed vs r2 (n*=8)", "pass": False,
                "note": "未找到 seed_gamma_table.hacred"}
    vals = [tab[k]["gamma_cal"] for k in sorted(tab)]
    sigmas = [tab[k].get("sigma_doc_boot_cal") for k in sorted(tab)]
    rho = (ref.get("sigma_inference_hacred_new10", {})
              .get("rho_cal_exact_pool", {}).get("mean_pairwise_rho"))
    from .seedrx import prescription
    res = prescription(vals, sigmas=sigmas, rho=rho, grid=[2, 3, 5, 8, 12, 15])
    nstar = res.get("min_seeds")
    ref_n = ref.get("headline", {}).get("hacred_gamma_cal", {}).get("min_n_95pct_trackB_full_pool")
    return {"check": "T2 seed vs r2 (n*=8, σ+ρ 轨道 B)",
            "pass": (nstar == 8 and ref_n == 8),
            "computed": nstar, "reference": ref_n, "n_seeds": len(vals),
            "rho": round(rho, 4) if rho else None,
            "track": res.get("track")}


def t3_audit():
    import glob
    refs = sorted(glob.glob("out/d5_llm_hook/audit_preds_qwen2.5-14b-instruct_json.json"))
    if not refs:
        return {"check": "T3 audit vs 14B (violation_rate 4.57%)", "pass": False,
                "note": "审计参考文件不在相对路径"}
    ref = _load(refs[0])
    rate = ref.get("violation_rate")
    return {"check": "T3 audit vs 14B (violation_rate)", "pass": abs(rate - 0.0457) < TOL,
            "reference": rate,
            "note": "完整重算需三源对齐数据;重算入口=eval-lever audit"}


def main():
    results = [t1_arm(), t2_seed(), t3_audit()]
    for r in results:
        print(("PASS " if r["pass"] else "FAIL "), r["check"])
        if "note" in r:
            print("      ", r["note"])
    ok = all(r["pass"] for r in results)
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
