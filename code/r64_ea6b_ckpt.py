# -*- coding: utf-8 -*-
"""r64_ea6b_ckpt.py — E-A6b per-grid runner with checkpoint/resume.

Same protocol as r64_ea6_epsilon_tost.py part_b (registered 3-contract set,
R=2000, seed 7, corpus bleu13a/chrf recomputed per resample, bertscore =
cached segment mean), split per grid so the two grids can run as parallel
processes. Every 200 reps the per-rep arrays plus the RNG bit-generator
state are checkpointed to _ea6b_ckpt_{pair}.npz — an externally killed
process resumes its exact stream instead of restarting (~9 s/rep).

Usage: python -I r64_ea6b_ckpt.py de-en   (or: zh-en)
Output: r64_ea6b_{pair}.json; once both per-grid JSONs exist, also writes
the merged r64_ea6_tost_matrix.json (schema identical to part_b output).
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
CKPT_EVERY = 200


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
    return sysn, sents, refs, hyps, raw, wz


def run_pair(pair):
    sysn, sents, refs, hyps, raw, wz = load(pair)
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
    te_list, d_list = [], []
    start = 0
    ckpt = os.path.join(BASE, f'_ea6b_ckpt_{pair}.npz')
    if os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=True)
        te_list = z['tau_ens'].tolist()
        d_list = z['delta'].tolist()
        start = len(te_list)
        rng.bit_generator.state = z['rng_state'].item()
        print(f'resume {pair} at rep {start}', flush=True)
    for it in range(start, R):
        idx = rng.integers(0, n, n)
        rs = [refs[j] for j in idx]
        col = {'bleu13a': np.array([sacrebleu.corpus_bleu([hyps[s][j] for j in idx], [rs],
                                   tokenize='13a', force=True).score for s in sysn]),
               'chrf': np.array([sacrebleu.corpus_chrf([hyps[s][j] for j in idx], [rs],
                                   word_order=2, beta=2).score for s in sysn]),
               'bertscore': np.array([float(bs_seg[s][idx].mean()) for s in sysn])}
        Z = zsc(np.stack([col[c] for c in C3]).T)
        te_list.append(kendall_tau_b(Z.mean(1), anch['wmt-raw']))
        d_list.append(te_list[-1] - kendall_tau_b(col[best], anch['wmt-raw']))
        if (it + 1) % CKPT_EVERY == 0 or it + 1 == R:
            np.savez(ckpt, tau_ens=np.array(te_list), delta=np.array(d_list),
                     rng_state=np.array([rng.bit_generator.state], dtype=object))
            print(f'  {pair} ckpt {it+1}', flush=True)
    te = np.array(te_list)
    delta = np.array(d_list)
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
    ptens_raw = kendall_tau_b(zsc(np.stack([pt[c] for c in C3]).T).mean(1), anch['wmt-raw'])
    ptens_z = kendall_tau_b(zsc(np.stack([pt[c] for c in C3]).T).mean(1), anch['wmt-z'])
    jk_d = []
    for i in range(len(sysn)):
        keep = [j for j in range(len(sysn)) if j != i]
        Zk = zsc(np.stack([pt[c] for c in C3]).T)[keep]
        jk_d.append(kendall_tau_b(Zk.mean(1), anch['wmt-raw'][keep])
                    - kendall_tau_b(pt[best][keep], anch['wmt-raw'][keep]))
    g = {'best_single': best, 'point_delta': round(float(point_delta), 4),
         'distinct_tau_ens_values': int(len(np.unique(np.round(te, 6)))),
         'distinct_delta_values': int(len(np.unique(np.round(delta, 6)))),
         'verdict_matrix': mat,
         'anchor_z_point_effect': {'ens_raw': round(float(ptens_raw), 4),
                                   'ens_z': round(float(ptens_z), 4)},
         'jackknife_over_systems_delta_range': [round(float(np.min(jk_d)), 4),
                                                round(float(np.max(jk_d)), 4)],
         'saved': {'tau_ens': te.tolist(), 'delta': delta.tolist()}}
    out = os.path.join(BASE, f'r64_ea6b_{pair}.json')
    json.dump({'meta': {'experiment': 'E-A6b per-grid (r64_ea6b_ckpt.py)',
                        'pair': pair, 'R': R, 'seed': SEED,
                        'tier': 'exploratory, post-review R12'},
               'grid': g},
              open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(pair, 'matrix:', json.dumps(mat, ensure_ascii=False), flush=True)
    print(pair, 'distinct tau:', g['distinct_tau_ens_values'],
          '| jk-sys delta range:', g['jackknife_over_systems_delta_range'], flush=True)
    print('saved', out, flush=True)
    merge()


def merge():
    grids = {}
    for p in PAIRS:
        f = os.path.join(BASE, f'r64_ea6b_{p}.json')
        if not os.path.exists(f):
            return
        grids[p] = json.load(open(f, encoding='utf-8'))['grid']
    merged = {'meta': {'experiment': 'E-A6b: TOST robustness matrix '
                       '(registered 3-contract set)', 'R': R, 'seed': SEED,
                       'ci': 'percentile+basic', 'anchors': ['wmt-raw', 'wmt-z'],
                       'tier': 'exploratory, post-review R12',
                       'runner': 'r64_ea6b_ckpt.py (per-grid, checkpointed)'},
              'grids': grids}
    out = os.path.join(BASE, 'r64_ea6_tost_matrix.json')
    json.dump(merged, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('merged ->', out, flush=True)


if __name__ == '__main__':
    pair = sys.argv[1]
    assert pair in PAIRS, f'pair must be one of {PAIRS}'
    run_pair(pair)
