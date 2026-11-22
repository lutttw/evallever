# -*- coding: utf-8 -*-
"""_p13_ea12b_false_tie.py — r64/E-A12b: AER ensemble false-tie rate on the
held-out grids at the SAME all-pairs scope and definition as the stored
single-contract rates in r64_ea3 (E-A3 computed false-tie only for single
contracts; the ensemble rate was never computed).

Replicates r64_ea3_holdout_coverage.py exactly: same loading (frozen WMT21
outputs + cached roberta-large segment BERTScore npy), same bootstrap
(R=1000, seed=7, corpus bleu13a/chrf recomputed per resample, bertscore =
cached segment means over the resampled index), same tie rule (95% percentile
CI of resample-level paired difference contains 0, inclusive).

Outputs per grid:
- validation: singles n_tied/false_tie must reproduce the stored rates
  exactly; flip-pair ensemble coverage must reproduce cov_aer;
- new: aer_ensemble and median_of_z n_tied/false_tie over ALL pairs.

Merged into r64_ea12_batch1_analyses.json as E4b (supersedes the
flips-detail-scoped E4 recount, whose scope could not match by construction).
"""
import glob
import json
import os
import re
import sys

import numpy as np
import sacrebleu

BASE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, '..')
REPO = os.path.join(BASE, 'wmt21-news-systems-main')
STAT = os.path.join(OUTDIR, 'evallever_release_v1_submission', 'statistics')
BS_CACHE = os.path.join(BASE, 'wmt_exp')
OUT = os.path.join(BASE, 'r64_ea12b_false_tie_allpairs.json')
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
    return (mat - mat.mean(0)) / (mat.std(0, ddof=1) + 1e-12)


def main():
    ea3 = json.load(open(os.path.join(STAT, 'r64_ea3_holdout_coverage.json'),
                         encoding='utf-8', errors='replace'))
    res = {'meta': {
        'experiment': 'E-A12b ensemble false-tie rate, all-pairs scope, '
                      'replicating E-A3 (R=1000 seed=7; corpus metrics '
                      'recomputed per resample; cached bertscore segment means)',
        'tie_rule': '95% percentile CI of resample-level paired diff contains 0 '
                    '(inclusive), identical to E-A3',
        'definition': 'false tie for rule X = X declares tied while some other '
                      'rule CI-excludes (either direction); scope = all pairs',
    }, 'grids': {}}
    for pair in PAIRS:
        gkey = f'newstest2021-{pair}'
        stored = ea3['grids'][gkey]
        keep, refs = load_pair(pair)
        names = sorted(keep)
        n = len(refs)
        assert len(names) == stored['n_systems'] and n == stored['n_segments'], \
            f'grid shape mismatch: {len(names)}x{n}'
        bs = np.load(os.path.join(BS_CACHE, f'_ea3_bs_{pair}.npy'))
        assert bs.shape == (len(names), n)

        rng = np.random.default_rng(SEED)
        boots = {c: [] for c in CONTRACTS}
        ens_rows, med_rows = [], []
        for it in range(R):
            idx = rng.integers(0, n, n)
            rs = [refs[j] for j in idx]
            col = {}
            col['bleu13a'] = np.array([sacrebleu.corpus_bleu([keep[s][j] for j in idx],
                                            [rs], tokenize='13a', force=True).score
                                       for s in names])
            col['chrf'] = np.array([sacrebleu.corpus_chrf([keep[s][j] for j in idx],
                                    [rs], beta=2).score for s in names])
            col['bertscore'] = np.array([float(bs[i][idx].mean())
                                         for i in range(len(names))])
            for c in CONTRACTS:
                boots[c].append(col[c])
            Zb = zsc(np.stack([col[c] for c in CONTRACTS]).T)
            ens_rows.append(Zb.mean(1))
            med_rows.append(np.median(Zb, axis=1))
            if (it + 1) % 200 == 0:
                print(f'  {pair} boot {it + 1}/{R}', flush=True)
        col_mat = {c: np.vstack(boots[c]) for c in CONTRACTS}
        ens_mat = np.vstack(ens_rows)
        med_mat = np.vstack(med_rows)

        def tie(mat, ia, ib):
            d = mat[:, ia] - mat[:, ib]
            return bool(np.percentile(d, 2.5) <= 0 <= np.percentile(d, 97.5))

        def excl(mat, ia, ib):
            d = mat[:, ia] - mat[:, ib]
            return bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)

        rules = {**{c: col_mat[c] for c in CONTRACTS},
                 'aer_ensemble': ens_mat, 'median_of_z': med_mat}
        n_tied = {r: 0 for r in rules}
        false_tie = {r: 0 for r in rules}
        cov_flip = {'aer_ensemble': 0, 'median_of_z': 0}
        n_flip = 0
        npairs = 0
        for ia in range(len(names)):
            for ib in range(ia + 1, len(names)):
                npairs += 1
                pt = {c: float(np.sign(col_mat[c][:, ia].mean()
                                       - col_mat[c][:, ib].mean()))
                      for c in CONTRACTS}
                ties_here = {r: tie(rules[r], ia, ib) for r in rules}
                for r in rules:
                    if ties_here[r]:
                        n_tied[r] += 1
                        others = [o for o in rules if o != r]
                        if any(excl(rules[o], ia, ib) for o in others):
                            false_tie[r] += 1
                flip = any(pt[c1] != 0 and pt[c2] != 0 and pt[c1] != pt[c2]
                           for c1 in CONTRACTS for c2 in CONTRACTS if c1 < c2)
                if flip:
                    n_flip += 1
                    cov_flip['aer_ensemble'] += int(ties_here['aer_ensemble'])
                    cov_flip['median_of_z'] += int(ties_here['median_of_z'])
        singles_check = {c: f'{false_tie[c]}/{n_tied[c]}' for c in CONTRACTS}
        res['grids'][gkey] = {
            'n_pairs_total': npairs, 'n_contract_induced_flips': n_flip,
            'validation': {
                'singles_false_tie_recount': singles_check,
                'singles_stored': stored['false_tie_rate'],
                'singles_match': singles_check == stored['false_tie_rate'],
                'flip_coverage_ensemble_recount': cov_flip['aer_ensemble'],
                'flip_coverage_ensemble_stored':
                    stored['coverage']['aer_ensemble'],
                'flip_coverage_match':
                    cov_flip['aer_ensemble'] == stored['coverage']['aer_ensemble'],
                'flip_coverage_median_recount': cov_flip['median_of_z'],
                'flip_coverage_median_stored': stored['coverage']['median_of_z'],
            },
            'ensemble_false_tie_allpairs':
                f'{false_tie["aer_ensemble"]}/{n_tied["aer_ensemble"]}',
            'ensemble_false_tie_pct':
                round(100 * false_tie['aer_ensemble'] / n_tied['aer_ensemble'], 1)
                if n_tied['aer_ensemble'] else None,
            'median_false_tie_allpairs':
                f'{false_tie["median_of_z"]}/{n_tied["median_of_z"]}',
            'singles_false_tie_allpairs': singles_check,
        }
        print(f'{pair}: AER {res["grids"][gkey]["ensemble_false_tie_allpairs"]} '
              f'({res["grids"][gkey]["ensemble_false_tie_pct"]}%) | '
              f'match={res["grids"][gkey]["validation"]["singles_match"]}/'
              f'{res["grids"][gkey]["validation"]["flip_coverage_match"]}',
              flush=True)
    json.dump(res, open(OUT, 'w', encoding='utf-8'), indent=1)
    print('saved ->', OUT)


if __name__ == '__main__':
    main()
