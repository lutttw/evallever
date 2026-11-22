# -*- coding: utf-8 -*-
"""r64_ea6_epsilon_tost.py — E-A6: empirical epsilon anchoring + TOST
robustness matrix on the registered 3-contract set (r60 protocol).

part=a (fast, CPU-light): anchor-resampling noise floor for tau_b.
  - split-half (200 draws) and leave-one-segment-out jackknife (n draws) of the
    per-system anchor mean (wmt-raw); tau_b of each contract's point reading
    (and the 3-contract ensemble point) against the perturbed anchor;
    SD(half)/sqrt(2) and jackknife SE => noise floor; calibrated eps = 1x/2x
    max floor; discordant-pair calibration for tau_b steps.
  Output: r64_ea6_epsilon_anchor.json

part=b (heavy; run after ea4 v2 reruns): per-rep bootstrap (R=2000, seed 7,
corpus bleu13a/chrf recomputed per resample, bertscore = cached seg mean) on
the registered set; per-rep tau_ens/tau_best/delta SAVED; verdict matrix
R in {200,1000,2000} x CI in {percentile, basic} x anchor in {wmt-raw, wmt-z}
x eps in {.05,.10}; distinct tau values (degenerate-CI diagnosis);
jackknife-over-systems delta CI as the system-level comparator.
  Output: r64_ea6_tost_matrix.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import sacrebleu

BASE = os.path.dirname(os.path.abspath(__file__))
MT = os.path.join(BASE, '..', '..', 'multi_task', 'mt')
PAIRS = ['de-en', 'zh-en']
C3 = ['bleu13a', 'chrf', 'bertscore']
R = 2000
SEED = 7
EPS = [0.05, 0.10]


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


def zsc(m):
    return (m - m.mean(0)) / (m.std(0, ddof=1) + 1e-12)


def load(pair):
    df = pd.read_csv(os.path.join(MT, 'data', f'metrics_task_{pair}.csv'), sep=';')
    df = df[df['dataset'] == 'wmt19']
    wide = df.pivot_table(index='sent_id', columns='mt_system',
                          values='system_output', aggfunc='first')
    wide = wide.drop(columns=['Human'], errors='ignore')
    ref = df.groupby('sent_id')['ref'].first()
    sysn = list(wide.columns)
    sents = list(wide.index)
    refs = [str(ref[s]) for s in sents]
    hyps = {s: [str(wide.loc[i, s]) if pd.notna(wide.loc[i, s]) else ''
                for i in sents] for s in sysn}
    raw = df.groupby('mt_system')['wmt-raw'].mean()
    wz = df.groupby('mt_system')['wmt-z'].mean()
    rawseg = df.pivot_table(index='sent_id', columns='mt_system',
                            values='wmt-raw', aggfunc='first')
    return sysn, sents, refs, hyps, raw, wz, rawseg


def part_a():
    res = {'meta': {'experiment': 'E-A6a: anchor-resampling noise floor for tau_b '
                                  '(epsilon calibration)', 'draws_split_half': 200,
                    'jackknife': 'leave-one-segment-out', 'tier': 'exploratory, post-review R12'},
           'grids': {}}
    for pair in PAIRS:
        sysn, sents, refs, hyps, raw, wz, rawseg = load(pair)
        anch = np.array([raw[s] for s in sysn])
        pt = {'bleu13a': np.array([sacrebleu.corpus_bleu(hyps[s], [refs], tokenize='13a',
                                                         force=True).score for s in sysn]),
              'chrf': np.array([sacrebleu.corpus_chrf(hyps[s], [refs], word_order=2,
                                                      beta=2).score for s in sysn])}
        bs = json.load(open(os.path.join(MT, 'runs', f'_bs_seg_{pair}.json')))
        pt['bertscore'] = np.array([np.mean(bs[s]) for s in sysn])
        pt['ens3'] = zsc(np.stack([pt[c] for c in C3]).T).mean(1)
        seg = np.stack([rawseg[s].values for s in sysn], 1)     # (n_seg, n_sys)
        n = seg.shape[0]
        rng = np.random.default_rng(20261030)
        # split-half
        half = {k: [] for k in list(pt) + ['anchor_self']}
        for _ in range(200):
            idx = rng.permutation(n)
            h1, h2 = seg[idx[:n // 2]], seg[idx[n // 2:]]
            a1, a2 = np.nanmean(h1, 0), np.nanmean(h2, 0)
            half['anchor_self'].append(kendall_tau_b(a1, a2))
            for k, v in pt.items():
                half[k].append(kendall_tau_b(v, a1))
        # jackknife LOO over segments
        sums = np.nancumsum(seg, 0)[-1]
        tot = np.array([np.nansum(seg[:, j]) for j in range(seg.shape[1])])
        cnt = np.array([np.sum(~np.isnan(seg[:, j])) for j in range(seg.shape[1])])
        jk = {k: [] for k in pt}
        for i in range(n):
            loo = (tot - np.nan_to_num(seg[i])) / np.maximum(cnt - (~np.isnan(seg[i])).astype(int), 1)
            for k, v in pt.items():
                jk[k].append(kendall_tau_b(v, loo))
        floors = {}
        for k in pt:
            sd_half = float(np.std(half[k], ddof=1))
            se_jk = float(np.std(jk[k], ddof=1)) * np.sqrt(n) * np.sqrt((n - 1) / n) ** 0  # raw SD of LOO taus
            floors[k] = {'split_half_sd_at_half_n': round(sd_half, 4),
                         'implied_full_n_floor': round(sd_half / np.sqrt(2), 4),
                         'jackknife_loo_sd_of_tau': round(float(np.std(jk[k], ddof=1)), 4),
                         'point_tau': round(float(kendall_tau_b(pt[k], anch)), 4)}
        n_sys = len(sysn)
        n_pairs = n_sys * (n_sys - 1) // 2
        res['grids'][pair] = {
            'n_systems': n_sys, 'n_pairs': n_pairs, 'n_segments': n,
            'split_half_anchor_self_tau': {'mean': round(float(np.mean(half['anchor_self'])), 4),
                                           'sd': round(float(np.std(half['anchor_self'], ddof=1)), 4)},
            'floors': floors,
            'discordant_pair_calibration': {f'tau={t}': f'~{round(t * n_pairs)} of {n_pairs} pairs'
                                            for t in (0.05, 0.10)},
            'calibrated_eps': {'1x_max_jackknife_floor': round(max(f['jackknife_loo_sd_of_tau']
                                                                   for f in floors.values()), 4),
                               '2x': round(2 * max(f['jackknife_loo_sd_of_tau']
                                                   for f in floors.values()), 4)},
        }
        print(pair, json.dumps(res['grids'][pair], ensure_ascii=False), flush=True)
    out = os.path.join(BASE, 'r64_ea6_epsilon_anchor.json')
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
    print('saved', out)


def part_b():
    res = {'meta': {'experiment': 'E-A6b: TOST robustness matrix (registered 3-contract set)',
                    'R': R, 'seed': SEED, 'ci': 'percentile+basic',
                    'anchors': ['wmt-raw', 'wmt-z'],
                    'tier': 'exploratory, post-review R12'}, 'grids': {}}
    for pair in PAIRS:
        sysn, sents, refs, hyps, raw, wz, rawseg = load(pair)
        anch = {'wmt-raw': np.array([raw[s] for s in sysn]),
                'wmt-z': np.array([wz[s] for s in sysn])}
        bs = json.load(open(os.path.join(MT, 'runs', f'_bs_seg_{pair}.json')))
        bs_seg = {s: np.array(bs[s]) for s in sysn}
        n = len(sents)
        pt = {'bleu13a': np.array([sacrebleu.corpus_bleu(hyps[s], [refs], tokenize='13a',
                                                         force=True).score for s in sysn]),
              'chrf': np.array([sacrebleu.corpus_chrf(hyps[s], [refs], word_order=2,
                                                      beta=2).score for s in sysn]),
              'bertscore': np.array([float(bs_seg[s].mean()) for s in sysn])}
        taus0 = {c: kendall_tau_b(pt[c], anch['wmt-raw']) for c in C3}
        best = max(C3, key=lambda c: taus0[c])
        rng = np.random.default_rng(SEED)
        rows = {'tau_ens': [], 'tau_best': []}
        for it in range(R):
            idx = rng.integers(0, n, n)
            rs = [refs[j] for j in idx]
            col = {'bleu13a': np.array([sacrebleu.corpus_bleu([hyps[s][j] for j in idx], [rs],
                                       tokenize='13a', force=True).score for s in sysn]),
                   'chrf': np.array([sacrebleu.corpus_chrf([hyps[s][j] for j in idx], [rs],
                                       word_order=2, beta=2).score for s in sysn]),
                   'bertscore': np.array([float(bs_seg[s][idx].mean()) for s in sysn])}
            Z = zsc(np.stack([col[c] for c in C3]).T)
            rows['tau_ens'].append(kendall_tau_b(Z.mean(1), anch['wmt-raw']))
            rows['tau_best'].append(kendall_tau_b(col[best], anch['wmt-raw']))
            if (it + 1) % 100 == 0:
                print(f'  {pair} boot {it+1}', flush=True)
        te = np.array(rows['tau_ens']); tb = np.array(rows['tau_best'])
        delta = te - tb
        point_delta = kendall_tau_b(zsc(np.stack([pt[c] for c in C3]).T).mean(1),
                                    anch['wmt-raw']) - taus0[best]
        mat = {}
        for RR in (200, 1000, 2000):
            d = delta[:RR]
            for ci in ('percentile', 'basic'):
                lo, hi = np.percentile(d, [5, 95])
                if ci == 'basic':
                    lo, hi = 2 * point_delta - hi, 2 * point_delta - lo
                mat[f'R={RR}|{ci}'] = {'ci90': [round(float(lo), 4), round(float(hi), 4)],
                                       **{f'eps={e}': bool(lo >= -e and hi <= e) for e in EPS}}
        # anchor sensitivity (point level, from e14-style): raw vs z for ens and best
        ptens_raw = kendall_tau_b(zsc(np.stack([pt[c] for c in C3]).T).mean(1), anch['wmt-raw'])
        ptens_z = kendall_tau_b(zsc(np.stack([pt[c] for c in C3]).T).mean(1), anch['wmt-z'])
        # jackknife over systems (degenerate-CI comparator)
        jk_d = []
        for i in range(len(sysn)):
            keep = [j for j in range(len(sysn)) if j != i]
            Zk = zsc(np.stack([pt[c] for c in C3]).T)[keep]
            jk_d.append(kendall_tau_b(Zk.mean(1), anch['wmt-raw'][keep])
                        - kendall_tau_b(pt[best][keep], anch['wmt-raw'][keep]))
        res['grids'][pair] = {
            'best_single': best, 'point_delta': round(float(point_delta), 4),
            'distinct_tau_ens_values': int(len(np.unique(np.round(te, 6)))),
            'distinct_delta_values': int(len(np.unique(np.round(delta, 6)))),
            'verdict_matrix': mat,
            'anchor_z_point_effect': {'ens_raw': round(float(ptens_raw), 4),
                                      'ens_z': round(float(ptens_z), 4)},
            'jackknife_over_systems_delta_range': [round(float(np.min(jk_d)), 4),
                                                   round(float(np.max(jk_d)), 4)],
            'saved': {'tau_ens': te.tolist(), 'delta': delta.tolist()},
        }
        g = res['grids'][pair]
        print(pair, 'matrix:', json.dumps(mat, ensure_ascii=False), flush=True)
        print(pair, 'distinct tau:', g['distinct_tau_ens_values'],
              '| jk-sys delta range:', g['jackknife_over_systems_delta_range'], flush=True)
    out = os.path.join(BASE, 'r64_ea6_tost_matrix.json')
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
    print('saved', out)


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'a'
    (part_a if part == 'a' else part_b)()
