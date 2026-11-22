# -*- coding: utf-8 -*-
"""r64_ec3_seed_variance.py — N8 (R14 R2 P1-3): training-seed variance vs
document-bootstrap variance. All published CIs condition on a single
checkpoint; this script quantifies the omitted seed component on every arm
where >=2 training seeds exist, and reports the naive combined width.

Sources (all frozen, post-review tier):
  e43_5seed_extension.json      X2 r=.25 / r=1.0 Delta per-seed (5 seeds);
                                HaCRED Gamma_fixed / Gamma_gold_bearing / J per-seed (5 seeds)
  hacred_5seed_tau_absolute.json HaCRED per-seed absolute Gamma for
                                F1_multi_fixed / F1_multi_cal (5 seeds)
  r31_ttmre.json                TTM-RE seed-1 arm_gold (+CI)
  r35_ttmre_seed2.json          TTM-RE seed-2 arm_gold (+CI)

For each quantity: per-seed values, n, mean, SD (ddof=1), t-CI on the seed
mean where df>=2, the published document-bootstrap CI, and the naive combined
95% half-width sqrt( (t*SD/sqrt(n))^2 + boot_half^2 ) (seed and document
resampling treated as independent; approximation, disclosed).
Output: r64_ec3_seed_variance.json
"""
import io
import json
import math
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'r64_ec3_seed_variance.json')
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def seed_stats(vals, boot_ci=None, label=''):
    v = np.asarray(vals, float)
    n = len(v)
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else float('nan')
    out = {'label': label, 'per_seed': [round(x, 5) for x in vals], 'n_seeds': n,
           'seed_mean': round(mean, 5), 'seed_sd': round(sd, 5) if n > 1 else None,
           'seed_range': [round(float(v.min()), 5), round(float(v.max()), 5)] if n > 1 else None}
    if n >= 2 and sd == sd:
        t = T975.get(n - 1)
        half_t = t * sd / math.sqrt(n)
        out['t_ci95_seedmean'] = [round(mean - half_t, 5), round(mean + half_t, 5)]
        out['seed_cv'] = round(sd / abs(mean), 3) if mean != 0 else None
        if boot_ci:
            bh = (boot_ci[1] - boot_ci[0]) / 2
            comb = math.sqrt(half_t ** 2 + bh ** 2)
            out['published_boot_ci95'] = [round(boot_ci[0], 5), round(boot_ci[1], 5)]
            out['boot_halfwidth'] = round(bh, 5)
            out['combined_halfwidth_naive'] = round(comb, 5)
            out['width_inflation_factor'] = round(comb / bh, 3) if bh > 0 else None
            out['seed_sd_over_boot_halfwidth'] = round(sd / bh, 3) if bh > 0 else None
    # sign stability across seeds
    if n > 1:
        signs = set(np.sign(v[v != 0]).tolist())
        out['sign_stable_across_seeds'] = len(signs) <= 1
    return out


def main():
    e43 = json.load(open(os.path.join(HERE, 'e43_5seed_extension.json'), encoding='utf-8'))
    hacred_abs = json.load(open(os.path.join(HERE, 'hacred_5seed_tau_absolute.json'), encoding='utf-8'))
    r31 = json.load(open(os.path.join(HERE, 'r31_ttmre.json'), encoding='utf-8'))['ttmre']
    r35 = json.load(open(os.path.join(HERE, 'r35_ttmre_seed2.json'), encoding='utf-8'))['ttmre']

    arms = []
    # X2 dose arms (5 seeds each), published = e43 5-seed CI on the seed-mean
    for key, lab in [('X2_r025_Delta_5seed', 'X2 dose r=.25 Delta (gR-gD re-annotation arm)'),
                     ('X2_r100_Delta_5seed', 'X2 dose r=1.0 Delta (gR-gD re-annotation arm)')]:
        e = e43[key]
        arms.append(seed_stats(e['per_seed'], boot_ci=e.get('ci95'), label=lab))
    # HaCRED seed-variance quantities
    for key, lab in [('Gamma_fixed', 'HaCRED Gamma under fixed contract'),
                     ('Gamma_gold_bearing', 'HaCRED Gamma gold-bearing'),
                     ('J(gb-fixed)', 'HaCRED J = gold-bearing minus fixed')]:
        e = e43['HaCRED_5seed'][key]
        arms.append(seed_stats(e['per_seed'], boot_ci=e.get('ci95'), label=lab))
    # HaCRED absolute Gamma per contract (5 seeds, from archived tau table)
    seeds = sorted(hacred_abs)
    for contract in ['F1_multi_fixed', 'F1_multi_cal']:
        vals = [hacred_abs[s][contract]['intra'] - hacred_abs[s][contract]['inter'] for s in seeds]
        arms.append(seed_stats(vals, boot_ci=None,
                               label=f'HaCRED absolute Gamma, {contract} (n=5)'))
    # TTM-RE 2 seeds
    arms.append(seed_stats([r31['arm_gold'], r35['arm_gold']], boot_ci=None,
                           label='TTM-RE arm_gold, seed1(20260910) vs seed2(20261010)'))

    res = {
        'meta': {'experiment': 'E-C3/N8: training-seed variance vs document-bootstrap variance',
                 'tier': 'exploratory, post-review R14',
                 'note': 'published CIs condition on a single checkpoint (document '
                         'bootstrap only); this file quantifies the seed component '
                         'on every arm with >=2 seeds; combined width is the naive '
                         'independent-variance approximation, disclosed as such'},
        'arms': arms,
        'summary_counts': {
            'arms_with_5_seeds': sum(1 for a in arms if a['n_seeds'] == 5),
            'arms_sign_stable': sum(1 for a in arms if a.get('sign_stable_across_seeds')),
            'arms_total': len(arms),
        },
    }
    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('saved', OUT)
    for a in arms:
        print(f"- {a['label']}: n={a['n_seeds']} mean={a['seed_mean']} sd={a['seed_sd']} "
              f"sign_stable={a.get('sign_stable_across_seeds')} "
              f"infl={a.get('width_inflation_factor')}")


if __name__ == '__main__':
    main()
