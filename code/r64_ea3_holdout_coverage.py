# -*- coding: utf-8 -*-
"""r64_ea3_holdout_coverage.py — E-A3 (Round-12, DA-C3/R2 minimum evidence).

Hold-out tie coverage. The WMT19-calibrated CI-based tie rule (a pair is tied
when the 95% percentile CI of the resample-level paired difference contains
zero) is applied on newstest2021 de-en / zh-en (frozen system outputs, same
registered contract set {bleu13a, chrf, bertscore}); for every contract-induced
flip we ask whether the pair is declared tied under >=1 of the two contracts,
under the z-ensemble (AER), and under median-of-z. Also false-tie rates and
Wilson CIs. Exploratory: R=1000 (registered WMT19 runs used 2000), disclosed.

Validation gate: point corpus bleu13a/chrf reproduce r59_n3_wmt_scorer_axis.json.
"""
import glob
import json
import os
import re

import numpy as np
import sacrebleu
from bert_score import score as bs_score

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASE, '..', 'wmt21-news-systems-main')
STAT = os.path.join(BASE, '..', '..', 'revision_v5_experiments', 'statistics')
OUT = os.path.join(BASE, 'r64_ea3_holdout_coverage.json')
# per-pair safe save when invoked with a single pair argument
if len(__import__('sys').argv) > 1:
    OUT = OUT.replace('.json', f'_{__import__("sys").argv[1]}.json')
R = 1000
SEED = 7
CONTRACTS = ['bleu13a', 'chrf', 'bertscore']
PAIRS = ['de-en', 'zh-en']
import sys as _sys
if len(_sys.argv) > 1:
    PAIRS = [_sys.argv[1]]
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


def kendall_tau_b(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    conc = disc = tx = ty = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            sx = np.sign(x[i] - x[j]); sy = np.sign(y[i] - y[j])
            if sx == 0 and sy == 0: tx += 1; ty += 1
            elif sx == 0: tx += 1
            elif sy == 0: ty += 1
            elif sx * sy > 0: conc += 1
            else: disc += 1
    den = np.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / den if den else 1.0


def zsc(mat):
    return (mat - mat.mean(0)) / (mat.std(0, ddof=1) + 1e-12)


def main():
    n3 = json.load(open(os.path.join(STAT, 'r59_n3_wmt_scorer_axis.json'), encoding='utf-8'))
    res = {'meta': {
        'experiment': 'E-A3 hold-out tie coverage (DA-C3/R2 minimum evidence)',
        'grids': [f'newstest2021-{p}' for p in PAIRS], 'contracts': CONTRACTS,
        'tie_rule': 'WMT19-calibrated CI rule: 95% percentile CI of resample-level paired diff contains 0',
        'bootstrap': {'reps': R, 'unit': 'segment', 'seed': SEED, 'ci': 'percentile95',
                      'note': 'R=1000 exploratory (registered WMT19 used 2000); corpus bleu13a/chrf '
                              'recomputed per resample; bertscore = mean of cached roberta-large '
                              'segment scores (rescale off)'},
        'tier': 'exploratory, post-review round 12',
        'chrf_note': 'plain sacrebleu chrF (default), matching the r59/N3 hold-out census protocol; the WMT19 registered set used chrf++ (word_order=2) — the transferred object is the CI-based tie rule',
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
        # V1: match r59_n3 (tolerance 0.05 absolute)
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
        bs_seg = {}
        for s in names:
            P, R_, F = bs_score(keep[s], refs, model_type='roberta-large', lang='en',
                                batch_size=128, device='cuda', verbose=False,
                                rescale_with_baseline=False)
            bs_seg[s] = np.asarray(F)
            print('  bertscore', s, flush=True)
        pt['bertscore'] = np.array([float(bs_seg[s].mean()) for s in names])
        np.save(f'_ea3_bs_{pair}.npy', np.stack([bs_seg[s] for s in names]))

        rng = np.random.default_rng(SEED)
        boots = {c: [] for c in CONTRACTS}
        ens_rows, med_rows = [], []
        for it in range(R):
            idx = rng.integers(0, n, n)
            rs = [refs[j] for j in idx]
            col = {}
            col['bleu13a'] = np.array([sacrebleu.corpus_bleu([keep[s][j] for j in idx], [rs],
                                    tokenize='13a', force=True).score for s in names])
            col['chrf'] = np.array([sacrebleu.corpus_chrf([keep[s][j] for j in idx], [rs]).score for s in names])
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

        def tie(mat, ia, ib):
            d = mat[:, ia] - mat[:, ib]
            return bool(np.percentile(d, 2.5) <= 0 <= np.percentile(d, 97.5))

        flips = []
        n_tied = {c: 0 for c in CONTRACTS}
        false_tie = {c: 0 for c in CONTRACTS}
        n_pairs = 0
        for ia in range(len(names)):
            for ib in range(ia + 1, len(names)):
                n_pairs += 1
                signs = {c: float(np.sign(pt[c][ia] - pt[c][ib])) for c in CONTRACTS}
                ties_here = {c: tie(col_mat[c], ia, ib) for c in CONTRACTS}
                for c in CONTRACTS:
                    if ties_here[c]:
                        n_tied[c] += 1
                        others = [col_mat[o][:, ia] - col_mat[o][:, ib] for o in CONTRACTS if o != c]
                        if any(np.percentile(x, 2.5) > 0 or np.percentile(x, 97.5) < 0 for x in others):
                            false_tie[c] += 1
                for c1 in CONTRACTS:
                    for c2 in CONTRACTS:
                        if c1 < c2 and signs[c1] != 0 and signs[c2] != 0 and signs[c1] != signs[c2]:
                            cov = {'aer': tie(ens_mat, ia, ib), 'median': tie(med_mat, ia, ib)}
                            cov.update({c: ties_here[c] for c in CONTRACTS})
                            cov['either_contract'] = bool(ties_here[c1] or ties_here[c2])
                            flips.append({'pair': f'{names[ia]}|{names[ib]}',
                                          'between': f'{c1}|{c2}', **cov})

        def wilson(k, m, z=1.96):
            if not m:
                return None
            p = k / m
            d = 1 + z * z / m
            cc = (p + z * z / (2 * m)) / d
            s = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / d
            return [round((cc - s) * 100, 1), round((cc + s) * 100, 1)]

        cov_aer = sum(1 for f in flips if f['aer'])
        cov_med = sum(1 for f in flips if f['median'])
        cov_either = sum(1 for f in flips if f['either_contract'])
        best_single_cov = max(sum(1 for f in flips if f[c]) for c in CONTRACTS)
        res['grids'][f'newstest2021-{pair}'] = {
            'n_systems': len(names), 'n_segments': n, 'n_pairs': n_pairs,
            'n_contract_induced_flips': len(flips),
            'flip_rate_wilson95': wilson(len(flips), n_pairs),
            'coverage': {'aer_ensemble': cov_aer, 'median_of_z': cov_med,
                         'best_single': best_single_cov,
                         'either_of_flipping_contracts': cov_either, 'n': len(flips),
                         'aer_wilson95': wilson(cov_aer, len(flips)),
                         'median_wilson95': wilson(cov_med, len(flips))},
            'false_tie_rate': {c: f'{false_tie[c]}/{n_tied[c]}' for c in CONTRACTS},
            'flips_detail': flips,
        }
        g = res['grids'][f'newstest2021-{pair}']
        print(f"  flips={g['n_contract_induced_flips']} AER={cov_aer} median={cov_med} "
              f"either={cov_either} false_tie={g['false_tie_rate']}", flush=True)
    open(OUT, 'w', encoding='utf-8').write(json.dumps(res, ensure_ascii=False, indent=1,
                                                      default=lambda o: o.item()))
    print('saved', OUT)


if __name__ == '__main__':
    main()
