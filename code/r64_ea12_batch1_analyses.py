# -*- coding: utf-8 -*-
"""_p12_ea12_batch1.py — r64/E-A12: batch-1 frozen-artifact analyses for the
r6 revision (five review items, one artifact).

E1  TOST at anchor-floor margins + analytic TOST power + tau_b discreteness
    floors (R-02; DA C2). Reads stored CI90 from r60; no rerun of draws.
E2  Holm(2) family sensitivity for C1/C2 (R-03).
E3  unstable-pair recount under the repaired per-segment ensemble
    (R-05; DA M4): replicates score_judge.load_cell + unstable_pairs
    (B=200, seed=5) on (a) the three single judges (validation: must
    reproduce 15/14/13) and (b) the E-C2 per-segment ensemble.
E4  AER ensemble false-tie rate on the held-out grids at the same
    definition/scope as the stored single-contract rates (R-06; DA M3).
E5  CI-form gate calibration: max violation count/rate that passes a
    95% Wilson upper bound below the 5% threshold, by n (R-14).

Output: r64_ea12_batch1_analyses.json (this directory).
"""
import json
import math
import os
from itertools import combinations

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, '..')                       # 04_文档级IE/out
STATS = os.path.join(OUTDIR, 'evallever_release_v1_submission', 'statistics')
JUDGE = os.path.join(OUTDIR, 'multi_task', 'judge')
RUNS = os.path.join(JUDGE, 'runs')
OUT = os.path.join(BASE, 'r64_ea12_batch1_analyses.json')

TAGS = {'qwen7b': 'Qwen2.5-7B|R1json-en',
        'llama': 'Llama-3.1-8B|R1json-en',
        'mistral': 'Mistral-7B-v0.3|R1json-en'}
EXPECTED = 32000
FLOORS = {'de-en': 0.033, 'zh-en': 0.026}
N_GRID = {'de-en': 16, 'zh-en': 15}


def load_cell(tag, cell='base'):
    """Replica of score_judge.load_cell -> {(seg, sys): mean score}."""
    p = os.path.join(RUNS, tag, f'{cell}.jsonl')
    n_lines = sum(1 for _ in open(p, encoding='utf-8'))
    if n_lines < EXPECTED * 0.98:
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


def unstable_pairs(seg, systems, B=200, seed=5):
    """Replica of score_judge.unstable_pairs."""
    rng = np.random.default_rng(seed)
    segs = sorted({k[0] for k in seg})
    signs = {}
    for _ in range(B):
        idx = rng.choice(segs, len(segs), replace=True)
        m = {s: np.mean([seg[(i, s)] for i in idx if (i, s) in seg])
             for s in systems}
        for a, b in combinations(systems, 2):
            signs.setdefault((a, b), set()).add(m[a] > m[b])
    return sum(1 for v in signs.values() if len(v) > 1)


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def tost_power(mu, sigma, eps, z=1.645):
    """P( 90% CI of D ~ N(mu, sigma) lies inside (-eps, eps) )."""
    if sigma <= 0:
        return 1.0 if abs(mu) < eps else 0.0
    return (norm_cdf((eps - z * sigma - mu) / sigma)
            - norm_cdf((-eps + z * sigma - mu) / sigma))


def wilson_upper(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    ctr = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (ctr + half) / den)


# ---------------------------------------------------------------- E1
def e1_floor_power():
    r60 = json.load(open(os.path.join(STATS, 'r60_canonical_aer.json'),
                         encoding='utf-8', errors='replace'))
    out = {}
    for g in ('de-en', 'zh-en'):
        d = r60[g]
        lo, hi = d['delta_aer_vs_best_single']['ci90']
        mu = d['tau_point']['aer'] - d['tau_point']['singles'][d['tau_point']['best_single']]
        hw = (hi - lo) / 2.0
        sigma = hw / 1.645
        eps_list = sorted({FLOORS[g], 0.05, 0.10})
        rows = {}
        for eps in eps_list:
            rows[f'eps={eps:.3f}'] = {
                'tost_pass_ci_in_band': bool(lo >= -eps and hi <= eps),
                'analytic_tost_power_at_point': round(max(0.0, tost_power(mu, sigma, eps)), 4)}
        n = N_GRID[g]
        out[g] = {
            'delta_point': round(mu, 4),
            'ci90_stored': [lo, hi],
            'sigma_hat_from_ci90': round(sigma, 4),
            'anchor_split_half_floor': FLOORS[g],
            'verdicts': rows,
            'tau_b_lattice_step': round(2.0 / (n * (n - 1) / 2), 5),
            'lattice_note': f'n={n} systems, no-tie tau_b grid step 2/C(n,2); '
                            f'differences below it are not representable'}
    return {'meta': {'source': 'r60_canonical_aer.json (CI90 stored; draws not '
                                 'needed: TOST at eps <=> CI90 within +/-eps)',
                     'power': 'analytic normal approximation, sigma from CI90 '
                              'half-width / 1.645; mu at stored point delta'},
           'grids': out}


# ---------------------------------------------------------------- E2
def e2_holm2():
    ps = {'C1': 0.0006, 'C2': 0.0016}
    order = sorted(ps, key=ps.get)
    m = len(ps)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, ps[k] * (m - i))
        adj[k] = min(1.0, running)
    return {'family': 'C1, C2 (C3 removed; Holm over the remaining two)',
            'raw_p': ps, 'holm_adjusted_p': {k: round(v, 5) for k, v in adj.items()},
            'reject_both_at_alpha_05': all(v < 0.05 for v in adj.values()),
            'note': 'mechanical recomputation; C3 removel does not change '
                    'C1/C2 conclusions'}


# ---------------------------------------------------------------- E3
def e3_unstable_recount():
    cells = {t: load_cell(t) for t in TAGS}
    assert all(cells.values()), 'a judge run missing/incomplete'
    systems = sorted({k[1] for c in cells.values() for k in c})
    r54 = json.load(open(os.path.join(STATS, 'r54_multitask_judge.json'),
                         encoding='utf-8', errors='replace'))
    assert systems == r54['systems'], 'system roster mismatch vs r54'

    # per-judge raw seg dicts -> validation recount (must match 15/14/13)
    singles = {}
    for t, name in TAGS.items():
        singles[name] = unstable_pairs(cells[t], systems)

    # E-C2 per-segment ensemble: global z per judge, mean z across judges
    mats = {}
    segs = sorted({k[0] for c in cells.values() for k in c})
    for t in TAGS:
        vals = list(cells[t].values())
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        mats[t] = {k: (v - mu) / sd for k, v in cells[t].items()}
    ens = {}
    for seg in segs:
        for s in systems:
            zs = [mats[t][(seg, s)] for t in TAGS if (seg, s) in mats[t]]
            if zs:
                ens[(seg, s)] = float(np.mean(zs))
    ens_unstable = unstable_pairs(ens, systems)

    # which pairs are unstable under the repaired ensemble (one draw set)
    rng = np.random.default_rng(5)
    segs2 = sorted({k[0] for k in ens})
    signs = {}
    for _ in range(200):
        idx = rng.choice(segs2, len(segs2), replace=True)
        m = {s: np.mean([ens[(i, s)] for i in idx if (i, s) in ens])
             for s in systems}
        for a, b in combinations(systems, 2):
            signs.setdefault((a, b), set()).add(m[a] > m[b])
    detail = sorted('|'.join(p) for p, v in signs.items() if len(v) > 1)

    stored = r54['hj3_flip_suppression']['unstable_pairs_bootstrap']
    return {
        'meta': {'mechanism_diagnosis': 'published aer_seg rows are the system z '
                 'broadcast to every segment (E-C2 E1), so every bootstrap draw '
                 'returns identical system means -> sign set size 1 -> 0 unstable '
                 'BY CONSTRUCTION; the recount replaces the broadcast layer with '
                 'the repaired per-segment ensemble (E-C2 E2), B=200 seed=5 as in '
                 'score_judge.unstable_pairs',
                 'inputs': 'judge/runs/{qwen7b,llama,mistral}/base.jsonl (frozen)'},
        'validation_single_recount': singles,
        'stored_single_counts': stored['single'],
        'validation_match': singles == stored['single'],
        'repaired_ensemble_unstable': ens_unstable,
        'repaired_unstable_pairs': detail,
        'stored_aer_count_degenerate_layer': stored['aer'],
        'hj3_verdict_after_repair': {
            'still_suppresses_min_single': bool(ens_unstable < min(singles.values())),
            'min_single': min(singles.values())},
    }


# ---------------------------------------------------------------- E4
def e4_aer_false_tie():
    e = json.load(open(os.path.join(STATS, 'r64_ea3_holdout_coverage.json'),
                       encoding='utf-8', errors='replace'))
    SINGLES = ('bleu13a', 'chrf', 'bertscore')
    out = {}
    for g, gd in e['grids'].items():
        det = gd['flips_detail']

        def rate(rule):
            den = sum(1 for r in det if r[rule] is True)
            others = [c for c in SINGLES if c != rule]
            num = sum(1 for r in det if r[rule] is True
                      and any(r[c] is False for c in others))
            return f'{num}/{den}'

        den_aer = sum(1 for r in det if r['aer'] is True)
        num_aer = sum(1 for r in det if r['aer'] is True
                      and any(r[c] is False for c in SINGLES))
        check = {c: rate(c) for c in SINGLES}
        stored = gd['false_tie_rate']
        out[g] = {
            'scope': f'{len(det)} contract-induced flip pairs (of {gd["n_pairs"]})',
            'aer_ensemble_false_tie': f'{num_aer}/{den_aer}',
            'aer_rate_pct': round(100 * num_aer / den_aer, 1) if den_aer else None,
            'single_contract_recount': check,
            'stored_single_rates': stored,
            'recount_matches_stored': all(
                check[c] == stored[c] for c in SINGLES if c in stored),
        }
    return {'meta': {'definition': 'false tie = pair the rule declares tied while '
                     'some single contract CI-excludes; identical convention and '
                     'scope as the stored single-contract rates (flip pairs only)'},
           'grids': out}


# ---------------------------------------------------------------- E5
def e5_gate_calibration():
    ns = [500, 998, 1000, 1500, 1701, 2507]
    rows = {}
    for n in ns:
        k = 0
        while wilson_upper(k + 1, n) < 0.05:
            k += 1
        rows[str(n)] = {'max_violations_passing': k,
                        'max_rate_passing_pct': round(100 * k / n, 2)}
    return {'meta': {'rule': 'CI-form gate passes iff the 95% Wilson upper bound '
                             'of the violation rate is below the 5% threshold',
                     'note': 'cells with n below ~1000 cannot pass at any rate '
                             '>= 0 unless the upper bound clears 5%; zero-event '
                             'cells pass for n >= ' + str(min(n for n in ns
                             if wilson_upper(0, n) < 0.05))},
           'by_n': rows}


def main():
    out = {
        'meta': {'experiment': 'E-A12 batch-1 frozen-artifact analyses (r6 '
                 'revision items R-02/R-03/R-05/R-06/R-14; DA C2/M3/M4)',
                 'tier': 'exploratory, post-review r6',
                 'inputs': ['r60_canonical_aer.json', 'r54_multitask_judge.json',
                            'r64_ea3_holdout_coverage.json',
                            'judge/runs/{qwen7b,llama,mistral}/base.jsonl']},
        'E1_floor_power': e1_floor_power(),
        'E2_holm2': e2_holm2(),
        'E3_unstable_recount': e3_unstable_recount(),
        'E4_aer_false_tie': e4_aer_false_tie(),
        'E5_gate_calibration': e5_gate_calibration(),
    }
    json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1)
    print('saved ->', OUT)
    # console digest
    print(json.dumps({
        'E1_de-en': out['E1_floor_power']['grids']['de-en']['verdicts'],
        'E1_zh-en': out['E1_floor_power']['grids']['zh-en']['verdicts'],
        'E1_lattice': {g: out['E1_floor_power']['grids'][g]['tau_b_lattice_step']
                       for g in ('de-en', 'zh-en')},
        'E2': out['E2_holm2']['holm_adjusted_p'],
        'E3_singles_recount': out['E3_unstable_recount']['validation_single_recount'],
        'E3_validation_match': out['E3_unstable_recount']['validation_match'],
        'E3_repaired_ensemble_unstable':
            out['E3_unstable_recount']['repaired_ensemble_unstable'],
        'E3_pairs': out['E3_unstable_recount']['repaired_unstable_pairs'],
        'E4': {g: out['E4_aer_false_tie']['grids'][g]['aer_ensemble_false_tie']
               for g in out['E4_aer_false_tie']['grids']},
        'E5': out['E5_gate_calibration']['by_n'],
    }, indent=1))


if __name__ == '__main__':
    main()
