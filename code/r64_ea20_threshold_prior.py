# -*- coding: utf-8 -*-
"""E-A20: threshold-axis prior, full version — one sort gives every
threshold's audit for free (the exact analogue of E-A17's decomposed prior
on the gold axis, but EXACT: the score is the model's own deterministic
output, so threshold-axis arms/flip points are computable with zero
statistical error; only doc-resampling CIs apply).

Theory: within the fixed-threshold family (predict relation r iff
logit_r > t), the per-doc counts (TP, FP, FN) at any t are cumulative
functions of a single sort of the logits against the gold labels.  Hence:
  - F1_A(t), F1_B(t) curves for every system pair (deterministic)
  - delta(t) = F1_A - F1_B curve; sign changes of delta = threshold-axis
    flip points tau* (where a system pair's ordering flips INSIDE the
    fixed-threshold family)
  - doc-paired bootstrap on the curve gives a flip-uncertainty band.

Also connects to the registry: the fixed-vs-decoded contrast sits at t=0
(fixed family) vs the native decode (a different policy, outside the
scalar family) — noted, per E-A19's space declaration.

Output: stats/r64_ea20_threshold_prior.json
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, r'E:\WorkBuddy\参考文献解读\04_文档级IE\out\multi_task\audit_published')
import audit_published as A  # noqa: E402

TAUS = np.round(np.linspace(-8, 8, 65), 2)
R_BOOT = 200
SEED = 20261114


def load_system(d0, rel2id, goldsD):
    logits = np.asarray(np.load(A.RV5 / d0 / "logits_evaluation.npy",
                                mmap_mode="r"))[:, 1:]          # classes 0..95
    keys = list(A.load_jsonl(A.RV5 / d0 / "keys_evaluation.jsonl"))
    dids = np.array([k['doc_id'] for k in keys])
    h = np.array([int(k['h']) for k in keys])
    t = np.array([int(k['t']) for k in keys])
    # gold01 matrix (N x 96): 1 where (doc,h,t,rel_id) in gD
    gold01 = np.zeros_like(logits, dtype=bool)
    for i, k in enumerate(keys):
        gset = goldsD.get((k['doc_id'], int(k['h']), int(k['t'])), set())
        for r in gset:
            if r in rel2id:
                gold01[i, rel2id[r]] = True
    return logits, gold01, dids, h, t


def counts_at(logits, gold01, dids, tau, doc_flat, n_docs):
    pred = (logits > tau).ravel()
    g = np.broadcast_to(gold01, logits.shape).ravel()
    tp_d = np.bincount(doc_flat[pred & g], minlength=n_docs)
    fp_d = np.bincount(doc_flat[pred & ~g], minlength=n_docs)
    fn_d = np.bincount(doc_flat[(~pred) & g], minlength=n_docs)
    return tp_d, fp_d, fn_d


def f1_curve(logits, gold01, dids, doc_index, n_docs, taus,
                  boot_seed=None, doc_flat=None):
    """Returns per-tau full-pool (F1, tp_d/fp_d/fn_d snapshot)."""
    rows = []
    rng = np.random.default_rng(boot_seed) if boot_seed is not None else None
    for tau in taus:
        tp_d, fp_d, fn_d = counts_at(logits, gold01, dids, tau,
                                    doc_flat, n_docs)
        TP, FP, FN = tp_d.sum(), fp_d.sum(), fn_d.sum()
        P = TP / max(1, TP + FP); R = TP / max(1, TP + FN)
        f1 = 2 * P * R / (P + R) if P + R else 0.0
        rows.append({'tau': float(tau), 'F1': float(f1),
                     'tp_d': tp_d, 'fp_d': fp_d, 'fn_d': fn_d})
    return rows


def f1_from_counts(tp_d, fp_d, fn_d, sel=None):
    if sel is None:
        TP, FP, FN = tp_d.sum(), fp_d.sum(), fn_d.sum()
    else:
        TP, FP, FN = tp_d[sel].sum(), fp_d[sel].sum(), fn_d[sel].sum()
    P = TP / max(1, TP + FP); R = TP / max(1, TP + FN)
    return 2 * P * R / (P + R) if P + R else 0.0


def main():
    A.rel2id = json.load(open(A.PREP / 'relation_to_index.json', encoding='utf-8'))
    goldsD = {}
    for row in A.load_jsonl(A.PREP / 'golds_gD.jsonl'):
        goldsD[(row['doc_id'], int(row['h']), int(row['t']))] = set(row['rels'])

    claims = {'C1': ('dreeam', 'atlop_base'),
              'C2': ('kddocre', 'docunet'),
              'C3': ('ttmre_release', 'kddocre')}
    all_sys = sorted({s for pair in claims.values() for s in pair})

    out = {'meta': {
        'experiment': 'E-A20 threshold-axis prior, full version: one sort '
                      'gives every threshold audit exactly (zero statistical '
                      'error on the curve itself; only doc-resampling CIs '
                      'apply)',
        'family': 'fixed-threshold family: predict r iff logit_r > t',
        'taus': len(TAUS), 'tau_grid': [float(TAUS[0]), float(TAUS[-1])],
        'note': 'native decode (registry decoded side) is a different policy, '
                'outside this scalar family (E-A19 space declaration)'},
        'systems': {}, 'claims': {}}

    curves = {}
    for sysname in all_sys:
        logits, gold01, dids, h, t = load_system(
            A.RUNS[sysname][0], A.rel2id, goldsD)
        uniq = sorted(set(dids))
        doc_index = {d: i for i, d in enumerate(uniq)}
        di = np.array([doc_index[d] for d in dids])
        n_docs = len(uniq)
        doc_flat = np.repeat(di, logits.shape[1])
        rows = f1_curve(logits, gold01, dids, di, n_docs, TAUS, doc_flat=doc_flat)
        curves[sysname] = (rows, logits, gold01, di, n_docs)
        curve_f1 = [round(r['F1'], 4) for r in rows]
        out['systems'][sysname] = {
            'F1_curve': curve_f1,
            'F1_at_0': round(rows[np.argmin(np.abs(TAUS))]['F1'], 4),
            'F1_max': round(max(r['F1'] for r in rows), 4),
            'argmax_tau': float(TAUS[int(np.argmax([r['F1'] for r in rows]))]),
            'note': 'F1(t) deterministic within the fixed-threshold family'}
        print(f'{sysname}: F1@0={out["systems"][sysname]["F1_at_0"]} '
              f'max={out["systems"][sysname]["F1_max"]}'
              f'@t={out["systems"][sysname]["argmax_tau"]}', flush=True)

    # per-claim: delta(t) curve, flip points, flip-uncertainty band
    for cid, (sa, sb) in claims.items():
        ra, _, _, dia, na = curves[sa]
        rb, _, _, dib, nb = curves[sb]
        # common docs for the pair
        common = sorted(set(np.unique(dia)) & set(np.unique(dib)))
        # map doc indices: rebuild per-system doc key lists
        dA = {d: i for i, d in enumerate(sorted(set(dia)))}
        dB = {d: i for i, d in enumerate(sorted(set(dib)))}
        selA = np.array([dA[d] for d in common])
        selB = np.array([dB[d] for d in common])
        deltas = []
        cis = []
        rng = np.random.default_rng(SEED)
        for rA, rB in zip(ra, rb):
            d = f1_from_counts(rA['tp_d'], rA['fp_d'], rA['fn_d'], selA) \
                - f1_from_counts(rB['tp_d'], rB['fp_d'], rB['fn_d'], selB)
            deltas.append(float(d))
            # doc-paired bootstrap CI
            bs = []
            for _ in range(R_BOOT):
                sel = rng.integers(0, len(common), len(common))
                d_b = (f1_from_counts(rA['tp_d'], rA['fp_d'], rA['fn_d'], sel[selA])
                       - f1_from_counts(rB['tp_d'], rB['fp_d'], rB['fn_d'], sel[selB]))
                bs.append(d_b)
            cis.append([float(np.percentile(bs, 2.5)),
                        float(np.percentile(bs, 97.5))])
        # flip points: sign changes of delta along tau
        flips = []
        for i in range(1, len(deltas)):
            if (deltas[i] > 0) != (deltas[i - 1] > 0) and \
                    not (deltas[i] == 0 or deltas[i - 1] == 0):
                # linear interpolation for tau*
                t0, t1 = TAUS[i - 1], TAUS[i]
                d0_, d1_ = deltas[i - 1], deltas[i]
                tau_star = float(t0 + (0 - d0_) * (t1 - t0) / (d1_ - d0_))
                flips.append(round(tau_star, 3))
        # flip-uncertainty band: taus where the bootstrap CI contains 0
        band = [float(TAUS[i]) for i, (lo, hi) in enumerate(cis)
                if lo <= 0 <= hi]
        out['claims'][cid] = {
            'sysA': sa, 'sysB': sb, 'n_common': len(common),
            'delta_curve': [round(x, 4) for x in deltas],
            'ci_curve': [[round(lo, 4), round(hi, 4)] for lo, hi in cis],
            'flip_points_tau': flips,
            'flip_uncertainty_band': [min(band), max(band)] if band else None,
            'note': 'delta(t)=F1_A(t)-F1_B(t) deterministic; flips = sign '
                    'changes inside the fixed-threshold family; band = tau '
                    'where doc-bootstrap CI contains 0'}
        print(f'{cid}: flips at tau={flips} band={out["claims"][cid]["flip_uncertainty_band"]}',
              flush=True)

    o = A.OUT / 'r64_ea20_threshold_prior.json'
    json.dump(out, open(o, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('saved ->', o)


if __name__ == '__main__':
    main()
