# -*- coding: utf-8 -*-
"""r64_ec2_judge_degenerate_ci.py — N7 (R14 R2 P1-2): degenerate judge-grid CI
[0.70, 0.70] (tab:judge / H-J4) — mechanism diagnosis and repair.

E1 mechanism: reproduce the published layer exactly. In score_judge.py the AER
   ensemble reading z-scores each judge's SYSTEM MEAN and broadcasts that value
   to every (segment, system) row, so segment resampling cannot move any system
   mean -> the bootstrap distribution is a point mass by construction.
E2 repair: per-segment ensemble — z per (judge, segment, system) against the
   judge's global segment distribution, ensemble = mean z across judges;
   segment bootstrap (B=2000, seed 7, same scheme) -> percentile CI.
E3 BCa on the E2 bootstrap distribution (jackknife-over-segments influence).
E4 jackknife-over-systems (R2's suggested comparator) on the published
   system-level ensemble reading: LOO over systems -> jackknife SE -> CI.
E5 judge-level bootstrap: all 27 compositions of the 3 judges with
   replacement -> tau range across judge compositions.
E6 discreteness calibration: tau_b lattice resolution at n systems.

Validation gates: reproduced published aer_tau (0.7000), single-judge taus
(0.7167/0.7167/0.6833), and the degenerate CI [0.70, 0.70] itself.
Output: r64_ec2_judge_degenerate_ci.json (alongside this script).
"""
import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
JUDGE = r'E:\WorkBuddy\参考文献解读\04_文档级IE\out\multi_task\judge'
MT = r'E:\WorkBuddy\参考文献解读\04_文档级IE\out\multi_task\mt'
OUT = os.path.join(BASE, 'r64_ec2_judge_degenerate_ci.json')
B = 2000
SEED = 7
TAGS = ['qwen7b', 'llama', 'glm', 'mistral']
JUDGES = {'qwen7b': 'Qwen2.5-7B', 'llama': 'Llama-3.1-8B',
          'glm': 'GLM-4-9B', 'mistral': 'Mistral-7B-v0.3'}
EXPECTED = {'base': 32000}


def load_cell(tag, cell):
    """Replicates score_judge.load_cell: {(seg,sys): mean score} or None."""
    p = os.path.join(JUDGE, 'runs', tag, f'{cell}.jsonl')
    if not os.path.exists(p):
        return None
    n_lines = sum(1 for _ in open(p, encoding='utf-8'))
    if n_lines < EXPECTED.get(cell, 1) * 0.98:
        return None
    scores = {}
    for line in open(p, encoding='utf-8'):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        key = rec['key'].split('#')[0]
        seg, _, sysname = key.partition('|')
        if rec.get('violation') or not rec.get('parsed') \
                or 'score' not in rec.get('parsed', {}):
            continue
        try:
            sc = float(rec['parsed']['score'])
        except Exception:
            continue
        scores.setdefault((seg, sysname), []).append(sc)
    return {k: float(np.mean(v)) for k, v in scores.items()}


def anchor_raw():
    df = pd.read_csv(os.path.join(MT, 'data', 'metrics_task_de-en.csv'), sep=';')
    df = df[df['dataset'] == 'wmt19']
    return df.groupby('mt_system')['wmt-raw'].mean().to_dict()


def kendall_tau_b(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    conc = disc = tx = ty = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            sx = np.sign(x[i] - x[j]); sy = np.sign(y[i] - y[j])
            if sx == 0 and sy == 0: tx += 1; ty += 1
            elif sx == 0: tx += 1
            elif sy == 0: ty += 1
            elif sx * sy > 0: conc += 1
            else: disc += 1
    den = np.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / den if den else 1.0


def bca_interval(theta_hat, boots, loo_vals):
    """BCa percentile interval from bootstrap draws + jackknife influence."""
    boots = np.asarray(boots, float)
    loo_vals = np.asarray(loo_vals, float)
    n_b, n_j = len(boots), len(loo_vals)
    z0 = _norm_ppf(np.mean(boots < theta_hat))
    theta_dot = loo_vals.mean()
    d = theta_dot - loo_vals
    denom = np.sum(d ** 2) ** 1.5
    a = np.sum(d ** 3) / (6 * denom) if denom > 0 else 0.0
    out = {}
    for alpha in (0.025, 0.975):
        az = _norm_ppf(alpha)
        adj = z0 + (z0 + az) / (1 - a * (z0 + az))
        p = _norm_cdf(adj)
        out[f'{alpha:.3f}'] = float(np.percentile(boots, 100 * p))
    out.update({'z0': round(float(z0), 4), 'acceleration': round(float(a), 4)})
    return out


def _norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _norm_ppf(p):
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    # ---------- load ----------
    cells = {t: load_cell(t, 'base') for t in TAGS}
    used = [t for t in TAGS if cells[t]]
    assert set(used) == {'qwen7b', 'llama', 'mistral'}, f'judge set mismatch: {used}'
    systems = sorted({k[1] for t in used for k in cells[t]})
    anch_d = anchor_raw()
    assert all(s in anch_d for s in systems), 'anchor missing systems'
    anch = np.array([anch_d[s] for s in systems])

    # per-judge per-segment matrices (n_seg x n_sys), NaN = missing
    seg_sets = [sorted({k[0] for k in cells[t]}) for t in used]
    segs = sorted(set().union(*[set(s) for s in seg_sets]))
    n_seg, n_sys, n_j = len(segs), len(systems), len(used)
    M = np.full((n_j, n_seg, n_sys), np.nan)
    for ji, t in enumerate(used):
        idx_s = {s: i for i, s in enumerate(systems)}
        idx_g = {g: i for i, g in enumerate(segs)}
        for (g, s), v in cells[t].items():
            M[ji, idx_g[g], idx_s[s]] = v
    # global per-judge standardization over all segment scores (published family)
    allv = [M[ji][~np.isnan(M[ji])] for ji in range(n_j)]
    mu = np.array([v.mean() for v in allv])
    sd = np.array([v.std() for v in allv])

    # ---------- E1: reproduce published layer ----------
    z_sys = np.empty((n_j, n_sys))                      # judge x system z of system mean
    for ji in range(n_j):
        m_sys = np.nanmean(M[ji], axis=0)               # per-system mean
        z_sys[ji] = (m_sys - mu[ji]) / sd[ji]
    ens_sys = z_sys.mean(0)                             # published aer reading per system
    tau_pub = float(kendall_tau_b(ens_sys, anch))
    taus_single = {JUDGES[t]: round(float(kendall_tau_b(z_sys[ji], anch)), 4)
                   for ji, t in enumerate(used)}
    # published bootstrap faithfully reproduced: aer_seg rows = system z broadcast
    aer_rows = np.tile(ens_sys, (n_seg, 1))
    rng = np.random.default_rng(SEED)
    point_mass = set()
    for _ in range(500):
        idx = rng.choice(n_seg, size=n_seg, replace=True)
        means = aer_rows[idx].mean(axis=0)
        point_mass.add(round(float(kendall_tau_b(means, anch)), 6))
    e1 = {'aer_seg_row_constant_within_system': True,
          'bootstrap_distinct_taus_500_draws': len(point_mass),
          'reproduced_aer_tau': round(float(tau_pub), 4),
          'reproduced_single_taus': taus_single,
          'note': 'published aer_seg[(i,s)] holds one system-level z per system; '
                  'segment bootstrap cannot move system means -> point mass; '
                  'degeneracy is structural (ensemble layer), not tau_b discreteness alone'}

    # ---------- E2: per-segment ensemble repair ----------
    with np.errstate(invalid='ignore'):
        Z = (M - mu[:, None, None]) / sd[:, None, None]  # judge x seg x sys
    ens_seg = np.nanmean(Z, axis=0)                      # seg x sys ensemble
    sys_means_pt = np.nanmean(ens_seg, axis=0)
    tau2_pt = float(kendall_tau_b(sys_means_pt, anch))
    col_ok = ~np.isnan(ens_seg).all(axis=0)
    boots2_tau = []
    for _ in range(B):
        idx = rng.choice(n_seg, size=n_seg, replace=True)
        means = np.nanmean(ens_seg[idx], axis=0)
        boots2_tau.append(kendall_tau_b(means, anch))
    boots2_tau = np.array(boots2_tau)
    lo90, hi90 = np.percentile(boots2_tau, [2.5, 97.5])
    e2 = {'point_tau': round(float(tau2_pt), 4),
          'ci95_percentile': [round(float(lo90), 4), round(float(hi90), 4)],
          'B': B, 'seed': SEED,
          'distinct_taus': int(len(np.unique(np.round(boots2_tau, 8)))),
          'systems_without_ensemble_coverage': int((~col_ok).sum())}

    # ---------- E3: BCa on E2 ----------
    # jackknife-over-segments influence on tau2 (vectorized LOO means)
    sums = np.nansum(ens_seg, axis=0)
    cnts = np.sum(~np.isnan(ens_seg), axis=0)
    loo_taus = np.empty(n_seg)
    for i in range(n_seg):
        m = (sums - np.nan_to_num(ens_seg[i])) / np.maximum(cnts - (~np.isnan(ens_seg[i])).astype(int), 1)
        loo_taus[i] = kendall_tau_b(m, anch)
    e3 = bca_interval(tau2_pt, boots2_tau, loo_taus)

    # ---------- E4: jackknife-over-systems on the published reading ----------
    loo_sys = np.empty(n_sys)
    for i in range(n_sys):
        keep = [j for j in range(n_sys) if j != i]
        loo_sys[i] = kendall_tau_b(ens_sys[keep], anch[keep])
    se = float(np.std(loo_sys, ddof=1) * np.sqrt(n_sys - 1))
    e4 = {'point_tau': round(float(tau_pub), 4),
          'loo_taus': [round(float(v), 4) for v in loo_sys],
          'jackknife_se': round(se, 4),
          'ci95_jackknife': [round(float(tau_pub - 1.96 * se), 4),
                             round(float(tau_pub + 1.96 * se), 4)],
          'range_loo': [round(float(loo_sys.min()), 4), round(float(loo_sys.max()), 4)]}

    # ---------- E5: judge-level bootstrap (all 27 compositions) ----------
    from itertools import combinations_with_replacement
    e5 = {}
    tvals = []
    for comp in combinations_with_replacement(range(n_j), 3):
        ez = np.mean([z_sys[j] for j in comp], axis=0)
        tv = kendall_tau_b(ez, anch)
        tvals.append(float(tv))
    e5 = {'n_compositions': len(tvals),
          'tau_min': round(min(tvals), 4), 'tau_max': round(max(tvals), 4),
          'tau_sd': round(float(np.std(tvals, ddof=1)), 4),
          'note': 'judge-composition variability of the system-level ensemble reading'}

    # ---------- E6: discreteness calibration ----------
    n_pairs = n_sys * (n_sys - 1) // 2
    e6 = {'n_systems': n_sys, 'n_pairs': n_pairs,
          'lattice_step_no_ties': round(2.0 / n_pairs, 5),
          'realized_distinct_boots2_taus': e2['distinct_taus'],
          'note': 'tau_b is lattice-valued at n=16; any CI on it is grid-constrained'}

    res = {
        'meta': {'experiment': 'E-C2/N7: judge-grid degenerate CI [0.70,0.70] diagnosis+repair',
                 'tier': 'exploratory, post-review R14',
                 'inputs': {'judge_runs': JUDGE + '/runs/{qwen7b,llama,mistral}/base.jsonl',
                            'anchor': 'metrics_task_de-en.csv wmt-raw (wmt19)'},
                 'bootstrap': {'B': B, 'seed': SEED}},
        'E1_mechanism': e1, 'E2_per_segment_ensemble': e2, 'E3_bca': e3,
        'E4_jackknife_over_systems': e4, 'E5_judge_composition': e5,
        'E6_discreteness': e6,
        'validation_gates': {
            'published_aer_tau_07000': abs(tau_pub - 0.7000) < 1e-6,
            'published_ci_point_mass': len(point_mass) == 1,
            'single_taus_match_r54': taus_single == {'Qwen2.5-7B': 0.7167,
                                                     'Llama-3.1-8B': 0.7167,
                                                     'Mistral-7B-v0.3': 0.6833}},
    }
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('saved', OUT)
    print(json.dumps(res['validation_gates'], indent=1))
    print('E2:', json.dumps(e2))
    print('E3:', json.dumps(e3))
    print('E4:', json.dumps(e4))


if __name__ == '__main__':
    main()
