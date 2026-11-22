# -*- coding: utf-8 -*-
"""E-C1: DocRE registry — AER-analog tie coverage + false-tie rate.

The DocRE flip registry (r1_rank_flip) has per-check readings D1/D2 under two
contract settings with bootstrap CIs. The registry's 13 rank flips (sign change
D1->D2) decompose by axis. This script computes, over ALL 57 checks:

  - tie-under-reader rules (CI-based, percentile 95% of the paired delta,
    "touching counts as tied"), for each reader:
      * AER-analog ensemble: cannot be computed as a z-marginalization here
        because the registry stores per-check deltas, not per-contract
        readings per system. Instead, the honest AER-analog is the
        two-contract average reading per model, i.e. mean(D1, D2) per model
        per axis family, then paired difference of the two models' averaged
        readings with CI -> ensemble tie.
      * union-of-ties: tied if tied under EITHER contract setting's CI.
      * single-contract ties: tied under setting1 CI only / setting2 only.
  - coverage of the 13 flips under each reader.
  - false-tie rate: checks tied under a reader but flip_delta CI-excluding
    (i.e., the registry flip actually happened — the reader missed it).

Axis families: gold (gD_vs_gR), threshold (fixed_vs_cal), scoring (A_vs_R).
The ensemble within a family averages the family's two settings.
Cross-family ensemble is not defined for 2-contract-per-family data; we
report per-family ensembles and the union over all families' ties.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
HERE = Path(__file__).resolve().parent
SRC = HERE / 'r1_rank_flip.json'
OUT = HERE / 'r64_ec1_docre_aer_coverage.json'
REPS = 10000
BOOTSEED = 20260926  # registered convention
FLOOR = 0.0039       # preregistered resolution floor (R38, sign-symmetrized)


def main():
    d = json.load(open(SRC, encoding='utf-8'))
    fm = d['flip_matrix']

    checks = []
    for pair, dims in fm.items():
        for dim, e in dims.items():
            if not e.get('available'):
                continue
            d1p, d2p = e['D1_point'], e['D2_point']
            d1ci, d2ci = e['D1_ci95'], e['D2_ci95']
            fdp = e['flip_delta_point']
            fdci = e['flip_delta_ci95']
            checks.append({
                'pair': pair, 'dim': dim,
                'd1': d1p, 'd2': d2p,
                'd1_ci': d1ci, 'd2_ci': d2ci,
                'delta': fdp, 'delta_ci': fdci,
                'flip': d1p * d2p < 0,
                'flip_ci_excl': (fdci[0] > 0) or (fdci[1] < 0),
            })
    n = len(checks)
    n_flips = sum(c['flip'] for c in checks)
    n_flip_excl = sum(c['flip'] and c['flip_ci_excl'] for c in checks)

    def tied_setting(ci):
        return ci[0] <= 0 <= ci[1]

    def tied_floor(v):
        return abs(v) <= FLOOR

    res = {'meta': {
        'experiment': 'E-C1: DocRE registry tie coverage (AER-analog per-family '
                      'ensemble vs union-of-ties vs single-contract ties)',
        'n_checks': n, 'n_flips': n_flips, 'n_flip_ci_excl': n_flip_excl,
        'tie_rule': 'CI-based percentile 95%, touching counts as tied; '
                    'floor cross-check at preregistered +/-0.0039',
        'ensemble_note': 'per-axis-family mean of the two contract readings '
                         '(2-contract marginalization); cross-family union reported separately',
        'bootstrap': {'reps': REPS, 'seed': BOOTSEED},
        'tier': 'exploratory, post-review (E-C1)',
    }, 'coverage': {}, 'false_tie': {}, 'detail': {}}

    flips = [c for c in checks if c['flip']]

    # reader: per-axis-family ensemble reading per model
    # For pair (A,B) under axis family F with settings (s1,s2):
    #   A's ensemble gap = mean(D1, D2) where D1 = gap(A,s1) - gap(B,s1), D2 = gap(A,s2) - gap(B,s2)
    #   -> ens(A,B,F) = (D1 + D2)/2; ensemble tie = CI of (D1+D2)/2 contains 0.
    # We have D1/D2 CIs; the sum's CI is approximated by [min-sum, max-sum] of
    # correlated extremes (conservative) — we instead reuse the stored
    # flip_delta (= D1 - D2) CI and the individual D1/D2 CIs to bound the sum CI:
    #   sum = (D1+D2); sum CI approx = [lo(D1)+lo(D2), hi(D1)+hi(D2)] assuming
    #   independence (optimistic) — but D1 and D2 share the same documents, so we
    #   ALSO report the floor-based tie as a conservative cross-check.
    for reader in ['ens_per_family', 'union_of_ties', 'setting1_only', 'setting2_only', 'floor']:
        covered = 0
        ft = 0
        n_tied = 0
        for c in checks:
            if reader == 'ens_per_family':
                lo_sum = c['d1_ci'][0] + c['d2_ci'][0]
                hi_sum = c['d1_ci'][1] + c['d2_ci'][1]
                tied = lo_sum <= 0 <= hi_sum
            elif reader == 'union_of_ties':
                tied = tied_setting(c['d1_ci']) or tied_setting(c['d2_ci'])
            elif reader == 'setting1_only':
                tied = tied_setting(c['d1_ci'])
            elif reader == 'setting2_only':
                tied = tied_setting(c['d2_ci'])
            else:
                tied = tied_floor(c['d1']) or tied_floor(c['d2'])
            n_tied += tied
            if c['flip']:
                covered += tied
            else:
                ft += tied
        res['coverage'][reader] = f'{covered}/{n_flips}'
        res['false_tie'][reader] = f'{ft}/{n_tied}'

    # Holm-survivor flips only (the 10 threshold + PE8|P3M gold that passed Holm)
    holm = set()
    holmjson = json.load(open(HERE / 'r26_holm_flips.json', encoding='utf-8'))
    for s in holmjson['survivor_list']:
        holm.add((s['pair'], s['dim']))
    for reader in ['ens_per_family', 'union_of_ties']:
        covered = 0
        n_h = 0
        for c in checks:
            if c['flip'] and (c['pair'], c['dim']) in holm:
                n_h += 1
                if reader == 'ens_per_family':
                    lo = c['d1_ci'][0] + c['d2_ci'][0]
                    hi = c['d1_ci'][1] + c['d2_ci'][1]
                    tied = lo <= 0 <= hi
                else:
                    tied = tied_setting(c['d1_ci']) or tied_setting(c['d2_ci'])
                covered += tied
        res['coverage'][reader + '_holm_survivors'] = f'{covered}/{n_h}'

    res['summary'] = dict(res['coverage'])
    res['summary'].update(res['false_tie'])
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(res['summary'], ensure_ascii=False, indent=1))
    print('saved', OUT)


if __name__ == '__main__':
    main()
