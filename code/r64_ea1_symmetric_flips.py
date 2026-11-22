# -*- coding: utf-8 -*-
"""r64_ea1_symmetric_flips.py — E-A1' (Round-12, DA-C1(1) honest adaptation).

FACTUAL CORRECTION (recorded in output): MultiTACRED (Hennig et al., ACL 2023)
is the machine-translated multilingual TACRED released via LDC — it is NOT a
set of multiple sanctioned English gold variants. The DA's "MultiTACRED flip
registry" premise is therefore infeasible as specified. Closest feasible
symmetric (no-supersession) venues with frozen predictions in the bundle:

  VENUE A  G0 (TACRED original) vs G1 (TACREV, +936 same-guideline re-annotation
           labels): same guideline+inventory re-annotation, the least-
           supersession pair available on English TACRED.
  VENUE B  G0 vs G2 (Re-TACRED): guideline-revision venue (41->40 inventory) —
           reported SEPARATELY and relabeled per R3 P1-3 (name-anchored, not a
           pure re-annotation).

All 35 published systems' released test predictions (frozen). Instance-paired
bootstrap (R=2000, seed 20260926), percentile 95%. Flip = point micro-F1
ordering changes between gauges; tie = 95% CI of paired F1 difference contains
zero under that gauge. Reports flip registry, per-flip tie status (coverage),
false-tie rate, Wilson CIs.

Validation gate: per-system point micro-F1 reproduces r59_n1_tacred_gauges.json.
"""
import glob
import itertools
import json
import os
from collections import Counter, defaultdict

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
TACREV = os.path.join(BASE, '..', 'tacrev')
STAT = os.path.join(BASE, '..', '..', 'revision_v5_experiments', 'statistics')
OUT = os.path.join(STAT, 'r64_ea1_symmetric_flips.json')
R = 2000
BOOTSEED = 20260926
EXCLUDE = ('no_relation',)

# ---------- gauges (verbatim n1_tacred_gauges.py protocol) ----------
pred_files = sorted(glob.glob(os.path.join(TACREV, 'test_results', '*.jsonl')))
systems = {}
gold_votes = defaultdict(Counter)
for fp in pred_files:
    name = os.path.basename(fp).split('_test')[0]
    rows = [json.loads(l) for l in open(fp, encoding='utf-8')]
    systems[name] = {r['id']: r['label_pred'] for r in rows}
    for r in rows:
        gold_votes[r['id']][r['label_true']] += 1
g0 = {i: c.most_common(1)[0][0] for i, c in gold_votes.items()}
tacrev_patch = json.load(open(os.path.join(TACREV, 'test_patch.json')))
g1 = dict(g0)
for r_ in tacrev_patch:
    g1[r_['id']] = r_['relation']
rt_patch = json.load(open(os.path.join(BASE, 'test_id2label.json')))
g2 = {i: lab for i, lab in rt_patch.items() if i in g0}

names = sorted(systems)
ids_g0 = [i for i in g0 if all(i in systems[n] for n in names)]
print(f'{len(names)} systems, |G0∩preds|={len(ids_g0)}, |G2|={len(g2)}')


def inst_tpf(pred, gold, ids):
    tp = np.zeros(len(ids))
    fp = np.zeros(len(ids))
    fn = np.zeros(len(ids))
    for k, i in enumerate(ids):
        p, t = systems[pred][i], gold[i]
        if p == t and p not in EXCLUDE:
            tp[k] = 1
        else:
            if p not in EXCLUDE:
                fp[k] = 1
            if t not in EXCLUDE:
                fn[k] = 1
    return tp, fp, fn


def micro_f1_vec(TP, FP, FN):
    den = 2 * TP + FP + FN
    return np.where(den > 0, 2 * TP / np.maximum(den, 1e-9), 0.0)


rng = np.random.default_rng(BOOTSEED)
res = {'meta': {
    'experiment': "E-A1' symmetric-venue flip registry (DA-C1(1) honest adaptation)",
    'correction': 'MultiTACRED (ACL 2023) = multilingual machine-translated TACRED via LDC, '
                  'NOT multiple sanctioned English gold variants; the premise from the panel '
                  'digest is corrected here. Venues used: TACREV same-guideline re-annotation '
                  '(least-supersession) and Re-TACRED (guideline revision, name-anchored, '
                  'reported separately).',
    'reading': 'micro-F1, TACRED protocol (no_relation only as FP/FN)',
    'bootstrap': {'reps': R, 'unit': 'test instance', 'seed': BOOTSEED, 'ci': 'percentile95'},
    'n_systems': len(names),
}, 'validation': {}}

# validation: point micro-F1 vs r59_n1 json
n1 = json.load(open(os.path.join(STAT, 'r59_n1_tacred_gauges.json'), encoding='utf-8'))
mism = []
for gauge, gold, ids in (('G0', g0, ids_g0), ('G1', g1, ids_g0), ('G2', g2, None)):
    gid = [i for i in g2 if all(i in systems[n] for n in names)] if gauge == 'G2' else ids
    for n in names:
        TP, FP, FN = (sum(x) for x in inst_tpf(n, gold, gid))
        f1 = micro_f1_vec(np.array([TP]), np.array([FP]), np.array([FN]))[0]
        ref = n1['systems'][n][f'microF1_{gauge}']
        if abs(f1 - ref) > 6e-5:   # r59 values rounded to 4 decimals
            mism.append(f'{n}|{gauge}|{f1:.4f}!={ref:.4f}')
res['validation']['point_microF1_vs_r59_n1'] = 'exact' if not mism else mism[:5]
assert not mism, mism[:5]
print('validation exact')

brng = np.random.default_rng(BOOTSEED)


GOLDS = {'G0': g0, 'G1': g1, 'G2': g2}


def run_axis(gA, gB, ids, label, note):
    n_inst = len(ids)
    brng = np.random.default_rng(BOOTSEED)
    idx = brng.integers(0, n_inst, (R, n_inst))
    # per system per gauge per-instance vectors
    TPm, FPm, FNm = {}, {}, {}
    for n in names:
        for g in (gA, gB):
            tp, fp, fn = inst_tpf(n, GOLDS[g], ids)
            TPm[(n, g)], FPm[(n, g)], FNm[(n, g)] = tp, fp, fn
    # bootstrap sums: (R, n_sys) per gauge
    boot = {}
    pt = {}
    for g in (gA, gB):
        TPb = np.stack([TPm[(n, g)][idx].sum(1) for n in names], 1).astype(float)
        FPb = np.stack([FPm[(n, g)][idx].sum(1) for n in names], 1).astype(float)
        FNb = np.stack([FNm[(n, g)][idx].sum(1) for n in names], 1).astype(float)
        boot[g] = micro_f1_vec(TPb, FPb, FNb)          # (R, S)
        TPs = np.array([TPm[(n, g)].sum() for n in names], float)
        FP2 = np.array([FPm[(n, g)].sum() for n in names], float)
        FNs = np.array([FNm[(n, g)].sum() for n in names], float)
        pt[g] = micro_f1_vec(TPs, FP2, FNs)            # (S,)
    pos = {n: i for i, n in enumerate(names)}
    flips, tied_A, tied_B, covered = [], 0, 0, 0
    false_tieA = false_tieB = n_tiedA = n_tiedB = 0
    n_pairs = 0
    for a, b in itertools.combinations(names, 2):
        ia, ib = pos[a], pos[b]
        dA = pt[gA][ia] - pt[gA][ib]
        dB = pt[gB][ia] - pt[gB][ib]
        if dA == 0 or dB == 0 or np.sign(dA) == np.sign(dB):
            # still count tie stats on non-flip pairs
            pass
        n_pairs += 1
        bA = boot[gA][:, ia] - boot[gA][:, ib]
        bB = boot[gB][:, ia] - boot[gB][:, ib]
        tieA = np.percentile(bA, 2.5) <= 0 <= np.percentile(bA, 97.5)
        tieB = np.percentile(bB, 2.5) <= 0 <= np.percentile(bB, 97.5)
        n_tiedA += tieA
        n_tiedB += tieB
        exclB = (np.percentile(bB, 2.5) > 0) or (np.percentile(bB, 97.5) < 0)
        exclA = (np.percentile(bA, 2.5) > 0) or (np.percentile(bA, 97.5) < 0)
        false_tieA += bool(tieA and exclB)
        false_tieB += bool(tieB and exclA)
        if dA != 0 and dB != 0 and np.sign(dA) != np.sign(dB):
            tied_A += tieA
            tied_B += tieB
            cov = tieA or tieB
            covered += cov
            flips.append({'pair': f'{a}|{b}',
                          'dA': round(float(dA), 4), 'dB': round(float(dB), 4),
                          'tie_A': bool(tieA), 'tie_B': bool(tieB), 'covered': bool(cov)})

    def wilson(k, n, z=1.96):
        p = k / n if n else 0.0
        d = 1 + z * z / n if n else 1
        c = (p + z * z / (2 * n)) / d if n else 0
        s = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d if n else 0
        return [round((c - s) * 100, 1), round((c + s) * 100, 1)]

    return {
        'venues': f'{label}: {note}',
        'n_pairs': n_pairs, 'n_flips': len(flips),
        'flip_rate_wilson95': wilson(len(flips), n_pairs),
        'coverage_of_flips': {'tied_under_A': tied_A, 'tied_under_B': tied_B,
                              'either': covered, 'n': len(flips),
                              'either_wilson95': wilson(covered, len(flips))},
        'false_tie_rate': {'tiedA_but_exclB': f'{false_tieA}/{n_tiedA}',
                           'tiedB_but_exclA': f'{false_tieB}/{n_tiedB}'},
        'flips_detail': flips,
    }


res['axis_G0_vs_G1_TACREV'] = run_axis(
    'G0', 'G1', ids_g0, 'G0(TACRED) vs G1(TACREV)',
    'same-guideline re-annotation, least-supersession venue on English TACRED')
ids_g2 = [i for i in g2 if all(i in systems[n] for n in names)]
res['axis_G0_vs_G2_ReTACRED_separate'] = run_axis(
    'G0', 'G2', ids_g2, 'G0(TACRED) vs G2(Re-TACRED)',
    'guideline-revision venue (inventory 41->40; name-anchored; asymmetric, relabeled per R3)')

res['summary'] = {
    'venue_A': f"{res['axis_G0_vs_G1_TACREV']['n_flips']} CI-status flips / "
               f"{res['axis_G0_vs_G1_TACREV']['n_pairs']} pairs; coverage "
               f"{res['axis_G0_vs_G1_TACREV']['coverage_of_flips']['either']}/{res['axis_G0_vs_G1_TACREV']['n_flips']}",
    'venue_B': f"{res['axis_G0_vs_G2_ReTACRED_separate']['n_flips']} flips; coverage "
               f"{res['axis_G0_vs_G2_ReTACRED_separate']['coverage_of_flips']['either']}/{res['axis_G0_vs_G2_ReTACRED_separate']['n_flips']}",
    'note': 'Venue A is the symmetric-ambiguous demonstration: flips between a gold and its '
            'same-guideline re-annotation, with tie status from frozen predictions.',
}
open(OUT, 'w', encoding='utf-8').write(json.dumps(
    res, ensure_ascii=False, indent=1, default=lambda o: o.item()
    if hasattr(o, 'item') else int(o)))
print(json.dumps(res['summary'], ensure_ascii=False, indent=1))
print('saved', OUT)
