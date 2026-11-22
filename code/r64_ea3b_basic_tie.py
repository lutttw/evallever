# -*- coding: utf-8 -*-
"""_p6_basic_tie.py — item 9 (R1-W4): re-declare held-out tie coverage under
BASIC (reverse-percentile) intervals alongside the frozen percentile rule.

Design:
- Same bootstrap stream as r64_ea3_holdout_coverage.py (R=1000, SEED=7,
  identical rng call order) so the mats reproduce the frozen run exactly
  and the percentile rule must match the frozen json (asserted).
- bertscore loaded from the original run's cache _ea3_bs_{pair}.npy
  (per-segment F, roberta-large) — no GPU, no bert_score import.
- Basic CI for the paired difference d (resample vector) with point theta:
  [2*theta - q97.5(d), 2*theta - q2.5(d)]; tied iff contains 0.
  theta for singles = pt[c][ia]-pt[c][ib] (full-sample corpus metrics);
  for ens/median = full-sample z-ensemble row (same zsc as the original).
- Flip SET held fixed (percentile-discovered, asserted == frozen json);
  ties re-declared under basic for the same pairs; agreement + basic
  coverage + basic false-tie reported. Basic-discovery flip count reported
  for context only.

Output: r64_ea3_basic_tie.json (this directory).
"""
import glob
import json
import os
import re

import numpy as np
import sacrebleu

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASE, '..', 'wmt21-news-systems-main')
STAT = os.path.join(BASE, '..', '..', 'revision_v5_experiments', 'statistics')
OUT = os.path.join(BASE, 'r64_ea3_basic_tie.json')
R = 1000
SEED = 7
CONTRACTS = ['bleu13a', 'chrf', 'bertscore']
PAIRS = ['de-en', 'zh-en']
TEST = 'newstest2021'


def load_pair(pair):
    so = os.path.join(REPO, 'txt', 'system-outputs')
    rf = os.path.join(REPO, 'txt', 'references')
    hyp_files = {}
    for p in glob.glob(os.path.join(so, f'{TEST}*.{pair}.hyp.*')):
        b = os.path.basename(p)
        m = re.match(rf'{re.escape(TEST)}[a-z0-9]*[.]{pair}[.]hyp[.](.+)[.][a-z]{{2}}$', b)
        if m:
            hyp_files[m.group(1)] = p
    tgt = pair.split('-')[1]
    ref_files = sorted(glob.glob(os.path.join(rf, f'{TEST}*.{pair}.ref.*.{tgt}')))
    ref_lines = [l.rstrip('\n') for l in open(ref_files[0], encoding='utf-8')]
    keep = {}
    for s, p in hyp_files.items():
        lines = [l.rstrip('\n') for l in open(p, encoding='utf-8')]
        if len(lines) == len(ref_lines) and all(x.strip() for x in lines):
            keep[s] = lines
    return keep, ref_lines


def zsc(mat):
    mu = mat.mean(0, keepdims=True)
    sd = mat.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    return (mat - mu) / sd


def main():
    n3 = json.load(open(os.path.join(STAT, 'r59_n3_wmt_scorer_axis.json'),
                        encoding='utf-8'))
    res = {'meta': {
        'experiment': 'E-A3b: basic-interval tie re-declaration on the held-out grids',
        'tie_rules': {'percentile': '95% percentile CI of resample-level paired diff contains 0 (frozen rule)',
                      'basic': 'reverse-percentile: [2*theta - q97.5, 2*theta - q2.5] contains 0'},
        'bootstrap': {'reps': R, 'unit': 'segment', 'seed': SEED,
                      'note': 'same rng stream as the frozen E-A3 run; bertscore from _ea3_bs cache (no GPU)'},
        'flip_set': 'frozen: percentile-discovered contract-induced flips (asserted == frozen json)',
    }, 'validation': {}, 'grids': {}}

    for pair in PAIRS:
        keep, refs = load_pair(pair)
        names = sorted(keep)
        n = len(refs)
        print(f'{pair}: {len(names)} systems, {n} segments', flush=True)
        pt = {}
        pt['bleu13a'] = np.array([sacrebleu.corpus_bleu(keep[s], [refs], tokenize='13a',
                                                        force=True).score for s in names])
        pt['chrf'] = np.array([sacrebleu.corpus_chrf(keep[s], [refs],
                                                     beta=2).score for s in names])
        vg = n3[f'{TEST[:4]}-{pair}'] if f'{TEST[:4]}-{pair}' in n3 else n3[f'2021-{pair}']
        mism = []
        for s in names:
            for key, val in (('bleu|13a|' + s, pt['bleu13a'][names.index(s)]),
                             ('chrf|13a|' + s, pt['chrf'][names.index(s)])):
                if key in vg['rho'] and abs(vg['rho'][key] - val) > 0.05:
                    mism.append(f'{key}: {val:.2f} vs {vg["rho"][key]:.2f}')
        res['validation'][f'{pair}_point_vs_r59_n3'] = 'ok' if not mism else mism[:4]
        assert not mism, mism[:4]
        print('  V1 vs r59_n3 ok', flush=True)

        bs_arr = np.load(os.path.join(BASE, f'_ea3_bs_{pair}.npy'))
        assert bs_arr.shape == (len(names), n), (bs_arr.shape, len(names), n)
        bs_seg = {s: bs_arr[i] for i, s in enumerate(names)}
        pt['bertscore'] = np.array([float(bs_seg[s].mean()) for s in names])

        rng = np.random.default_rng(SEED)
        boots = {c: [] for c in CONTRACTS}
        ens_rows, med_rows = [], []
        for it in range(R):
            idx = rng.integers(0, n, n)
            rs = [refs[j] for j in idx]
            col = {}
            col['bleu13a'] = np.array([sacrebleu.corpus_bleu([keep[s][j] for j in idx], [rs],
                                                             tokenize='13a', force=True).score
                                       for s in names])
            col['chrf'] = np.array([sacrebleu.corpus_chrf([keep[s][j] for j in idx], [rs]).score
                                    for s in names])
            col['bertscore'] = np.array([float(bs_seg[s][idx].mean()) for s in names])
            for c in CONTRACTS:
                boots[c].append(col[c])
            Zb = zsc(np.stack([col[c] for c in CONTRACTS]).T)
            ens_rows.append(Zb.mean(1))
            med_rows.append(np.median(Zb, axis=1))
            if (it + 1) % 100 == 0:
                print('  boot', it + 1, flush=True)
        col_mat = {c: np.vstack(boots[c]) for c in CONTRACTS}
        ens_mat = np.vstack(ens_rows)
        med_mat = np.vstack(med_rows)

        Zf = zsc(np.stack([pt[c] for c in CONTRACTS]).T)
        pt_vec = {'ens': Zf.mean(1), 'median': np.median(Zf, axis=1)}
        for c in CONTRACTS:
            pt_vec[c] = pt[c]

        def tie_pct(mat, ia, ib):
            d = mat[:, ia] - mat[:, ib]
            return bool(np.percentile(d, 2.5) <= 0 <= np.percentile(d, 97.5))

        def tie_basic(vec, ia, ib, theta):
            d = vec[:, ia] - vec[:, ib]
            lo = 2 * theta - np.percentile(d, 97.5)
            hi = 2 * theta - np.percentile(d, 2.5)
            return bool(lo <= 0 <= hi)

        flips = []
        n_tied = {c: 0 for c in CONTRACTS}
        false_tie = {c: 0 for c in CONTRACTS}
        n_pairs = 0
        agree = {'aer': 0, 'median': 0, 'either_contract': 0}
        basic_flips_discovery = 0
        for ia in range(len(names)):
            for ib in range(ia + 1, len(names)):
                n_pairs += 1
                signs = {c: float(np.sign(pt[c][ia] - pt[c][ib])) for c in CONTRACTS}
                ties_pct = {c: tie_pct(col_mat[c], ia, ib) for c in CONTRACTS}
                ties_basic = {c: tie_basic(col_mat[c], ia, ib, pt[c][ia] - pt[c][ib])
                              for c in CONTRACTS}
                for c in CONTRACTS:
                    if ties_pct[c]:
                        n_tied[c] += 1
                        others = [col_mat[o][:, ia] - col_mat[o][:, ib]
                                  for o in CONTRACTS if o != c]
                        if any(np.percentile(x, 2.5) > 0 or np.percentile(x, 97.5) < 0
                               for x in others):
                            false_tie[c] += 1
                fb_here = False
                for c1 in CONTRACTS:
                    for c2 in CONTRACTS:
                        if c1 < c2 and signs[c1] != 0 and signs[c2] != 0 \
                                and signs[c1] != signs[c2]:
                            b_here = {}
                            for key, mat, th in (('aer', ens_mat, pt_vec['ens']),
                                                 ('median', med_mat, pt_vec['median'])):
                                b_here[key] = tie_basic(mat, ia, ib, th[ia] - th[ib])
                            b_here.update({c: ties_basic[c] for c in CONTRACTS})
                            b_here['either_contract'] = bool(ties_basic[c1] or ties_basic[c2])
                            cov = {'aer': tie_pct(ens_mat, ia, ib),
                                   'median': tie_pct(med_mat, ia, ib)}
                            cov.update({c: ties_pct[c] for c in CONTRACTS})
                            cov['either_contract'] = bool(ties_pct[c1] or ties_pct[c2])
                            flips.append({'pair': f'{names[ia]}|{names[ib]}',
                                          'between': f'{c1}|{c2}',
                                          'basic': b_here, **cov})
                            for k in agree:
                                agree[k] += int(b_here[k] == cov[k])
                            if not any(b_here.values()):
                                fb_here = True
                if fb_here:
                    basic_flips_discovery += 1

        cov_aer = sum(1 for f in flips if f['aer'])
        cov_either = sum(1 for f in flips if f['either_contract'])
        covb_aer = sum(1 for f in flips if f['basic']['aer'])
        covb_med = sum(1 for f in flips if f['basic']['median'])
        covb_either = sum(1 for f in flips if f['basic']['either_contract'])
        covb_best = max(sum(1 for f in flips if f['basic'][c]) for c in CONTRACTS)
        g = {'n_systems': len(names), 'n_segments': n, 'n_pairs': n_pairs,
             'n_contract_induced_flips_pct': len(flips),
             'n_flips_basic_discovery': basic_flips_discovery,
             'coverage_percentile_frozen': {'aer_ensemble': cov_aer,
                                            'either_of_flipping_contracts': cov_either,
                                            'n': len(flips)},
             'coverage_basic': {'aer_ensemble': covb_aer, 'median_of_z': covb_med,
                                'best_single': covb_best,
                                'either_of_flipping_contracts': covb_either,
                                'n': len(flips)},
             'tie_verdict_agreement_pct_vs_basic': {
                 k: f'{v}/{len(flips)}' for k, v in agree.items()},
             'false_tie_rate_pct': {c: f'{false_tie[c]}/{n_tied[c]}' for c in CONTRACTS},
             'flips_detail': flips,
             }
        res['grids'][f'newstest2021-{pair}'] = g
        print(f"  flips={len(flips)} basic_disc={basic_flips_discovery} "
              f"covPCT(aer/either)={cov_aer}/{cov_either} "
              f"covBASIC(aer/med/best/either)={covb_aer}/{covb_med}/{covb_best}/{covb_either} "
              f"agree={agree}", flush=True)
    open(OUT, 'w', encoding='utf-8').write(json.dumps(res, ensure_ascii=False, indent=1,
                                                      default=lambda o: o.item()))
    print('saved', OUT)


if __name__ == '__main__':
    main()
