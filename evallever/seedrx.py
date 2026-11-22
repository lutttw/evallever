# -*- coding: utf-8 -*-
"""evallever.seedrx —— 种子数收敛处方 / minimum-seeds prescription.

输入逐种子 gap(对比量)点估计值(+可选逐种子文档级 bootstrap σ 与跨种子相关 ρ),
输出:每个 n 的 P(池化估计 95% CI 排除 0) 曲线,以及"符号判定所需最小种子数 n*"
(95% 判据:≥95% 的种子子集其池化估计 95% CI 排除 0)。

口径来源(已验证,自包含复制实现):src/jcip_audit/r2_seed_convergence.py 的
**轨道 B(全池,正态近似)** —— r2 的 headline 处方即出自该轨道:
  * 子集枚举:C(N,n) ≤ reps(默认 10000)则全组合枚举,否则随机抽 reps 组;
  * 池化估计 = 子集内种子均值;
  * SE² = (Σσ² + ρ·(Σσ)²−Σσ²)) / n²(σ 为逐种子文档级 bootstrap SD,ρ 为跨种子相关);
  * CI = mean ± 1.959964·SE;排除 0 ⟺ mean−Z·SE > 0 或 mean+Z·SE < 0;
  * n* = 使 frac(CI 排除 0) ≥ criterion(默认 0.95)的最小网格 n。

若输入未提供 sigmas:退化为 iid 近似(σ = 逐种子值的样本 SD(ddof=1),ρ=0),
并在 notes 中如实登记(该退化轨道未经项目验证,仅作粗筛查)。

仅用标准库 / stdlib only;确定性 / deterministic。
"""
from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path

Z = 1.959964            # 95% 正态分位数(与项目冻结协议同值)
ENUM_LIMIT_DEFAULT = 10000
RNG_SEED_DEFAULT = 20260926


def load_gap_file(path) -> dict:
    """读处方输入文件并规范化。

    文件格式(两种形态均可,详见 README):
      {"gaps": [v1, v2, ...], "sigmas": [s1, ...] 或省略, "rho": 0.0 或省略,
       "grid": [2,3,5,8], "criterion": 0.95, "reps": 10000, "seed": 20260926}
    或 {"gaps": {"seed_id": v, ...}, "sigmas": {"seed_id": s, ...}, ...}
    (dict 形态按 JSON 书写顺序取值,sigmas 的键须与 gaps 一致。)
    """
    d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if "gaps" not in d:
        raise ValueError("gap 文件缺 'gaps' 字段")
    if isinstance(d["gaps"], dict):
        ids = list(d["gaps"].keys())
        gaps = [float(d["gaps"][k]) for k in ids]
        sigmas = None
        if d.get("sigmas") is not None:
            if not isinstance(d["sigmas"], dict) or set(d["sigmas"]) != set(ids):
                raise ValueError("dict 形态下 sigmas 的键必须与 gaps 完全一致")
            sigmas = [float(d["sigmas"][k]) for k in ids]
    else:
        gaps = [float(v) for v in d["gaps"]]
        ids = [str(i) for i in range(len(gaps))]
        sigmas = None
        if d.get("sigmas") is not None:
            sigmas = [float(v) for v in d["sigmas"]]
            if len(sigmas) != len(gaps):
                raise ValueError("sigmas 与 gaps 长度不一致")
    if len(gaps) < 2:
        raise ValueError("至少需要 2 个种子值")
    return {
        "ids": ids,
        "gaps": gaps,
        "sigmas": sigmas,
        "rho": (float(d["rho"]) if d.get("rho") is not None else None),
        "grid": ([int(n) for n in d["grid"]] if d.get("grid") is not None else None),
        "criterion": (float(d["criterion"]) if d.get("criterion") is not None else 0.95),
        "reps": (int(d["reps"]) if d.get("reps") is not None else ENUM_LIMIT_DEFAULT),
        "seed": (int(d["seed"]) if d.get("seed") is not None else RNG_SEED_DEFAULT),
    }


def subset_indices(n_seeds: int, n: int, reps: int, rng_seed: int) -> list:
    """种子子集枚举:C(N,n) ≤ reps 全枚举;否则随机抽 reps 组(确定性 RNG)。

    注:随机分支用 stdlib random.Random(可复现,确定性保证);
    项目 r2 用 numpy default_rng(20260926).choice —— 两者序列不同,
    但仅在 C(N,n) > reps 时触发(本项目验证数据 HacRED/X2 的网格均全枚举,不受影响)。
    """
    combos = list(itertools.combinations(range(n_seeds), n))
    if len(combos) <= reps:
        return combos
    rng = random.Random(rng_seed)
    picks = sorted(rng.sample(range(len(combos)), reps))
    return [combos[i] for i in picks]


def prescription(gaps, sigmas=None, rho=None, grid=None, criterion=0.95,
                 reps=ENUM_LIMIT_DEFAULT, seed=RNG_SEED_DEFAULT) -> dict:
    """主入口:逐种子值(+σ、ρ)→ 每 n 的 P(CI 排除 0) 曲线与最小种子数 n*。"""
    gaps = [float(v) for v in gaps]
    n_seeds = len(gaps)
    notes = []
    if sigmas is not None:
        sigmas = [float(s) for s in sigmas]
        if len(sigmas) != n_seeds:
            raise ValueError("sigmas 与 gaps 长度不一致")
        if any((not math.isfinite(s)) or s < 0 for s in sigmas):
            raise ValueError("sigmas 必须为非负有限值")
        if rho is None:
            rho = 0.0
            notes.append("未提供 rho,按 ρ=0 处理(保守性下降,请尽量提供经验跨种子相关)。")
        track = "normal_approx_sigma_rho(r2 轨道 B 口径)"
    else:
        if n_seeds < 2:
            raise ValueError("无 sigmas 时至少需要 2 个种子值以估 σ")
        mean = sum(gaps) / n_seeds
        sigmas = [math.sqrt(sum((v - mean) ** 2 for v in gaps) / (n_seeds - 1))] * n_seeds
        rho = 0.0
        track = "normal_approx_iid_fallback(σ=种子值样本 SD,ρ=0;未经项目验证,仅粗筛查)"
        notes.append("输入未提供逐种子 sigmas:已退化为 iid 近似(公共 σ = 样本 SD,ρ=0);"
                     "如需复现 r2 冻结口径,请提供逐种子文档级 bootstrap SD 与 ρ。")

    if grid is None:
        grid = sorted({n for n in range(2, n_seeds + 1)})
    grid = sorted({int(n) for n in grid if 2 <= int(n) <= n_seeds})
    if not grid:
        raise ValueError("网格为空(需 2 ≤ n ≤ 种子数)")

    curve = {}
    for n in grid:
        pooled, excl, widths, pos, neg = [], [], [], [], []
        for combo in subset_indices(n_seeds, n, reps, seed):
            vals = [gaps[i] for i in combo]
            ss = [sigmas[i] for i in combo]
            m = sum(vals) / n                                   # 池化 = 子集内种子均值
            ss2 = sum(s * s for s in ss)
            se = math.sqrt((ss2 + rho * (sum(ss) ** 2 - ss2)) / (n * n))
            w = Z * se
            pooled.append(m)
            excl.append(bool(m - w > 0 or m + w < 0))
            widths.append(2 * w)
            pos.append(m > 0)
            neg.append(m < 0)
        curve[str(n)] = {
            "n_subsets": len(pooled),
            "frac_ci95_excludes_zero": sum(excl) / len(excl),
            "frac_sign_positive": sum(pos) / len(pos),
            "frac_sign_negative": sum(neg) / len(neg),
            "pooled_mean_abs": sum(abs(x) for x in pooled) / len(pooled),
            "ci95_width_mean": sum(widths) / len(widths),
        }

    min_n = None
    for n in grid:
        if curve[str(n)]["frac_ci95_excludes_zero"] >= criterion:
            min_n = n
            break
    return {
        "n_seeds": n_seeds,
        "seed_ids": None,   # 由 load_gap_file 填充(直接调用时为 None)
        "grid": grid,
        "criterion": criterion,
        "track": track,
        "rho": rho,
        "z": Z,
        "curve": curve,
        "min_seeds": min_n,
        "min_seeds_note": (f"最小种子数 n* = {min_n}(判据:≥{criterion:.0%} 子集的池化估计 "
                           f"95% CI 排除 0)" if min_n is not None else
                           f"网格内不存在满足 {criterion:.0%} 判据的种子数(曲线随 n 非增,"
                           f"若 |均值| 收敛慢于 SE 则可能永不满判)"),
        "notes": notes,
    }


def run(gap_file, grid_override=None, criterion_override=None,
        reps_override=None, seed_override=None) -> dict:
    """CLI 入口:读 gap 文件 → 处方报告(CLI 覆盖参数优先于文件值)。"""
    spec = load_gap_file(gap_file)
    rep = prescription(
        spec["gaps"], sigmas=spec["sigmas"], rho=spec["rho"],
        grid=(grid_override if grid_override is not None else spec["grid"]),
        criterion=(criterion_override if criterion_override is not None else spec["criterion"]),
        reps=(reps_override if reps_override is not None else spec["reps"]),
        seed=(seed_override if seed_override is not None else spec["seed"]),
    )
    rep["seed_ids"] = spec["ids"]
    rep["gap_values"] = spec["gaps"]
    if spec["sigmas"] is not None:
        rep["sigma_values"] = spec["sigmas"]
    return rep
