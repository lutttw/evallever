# -*- coding: utf-8 -*-
"""r64_ea2_dose_response.py — E-A2 (Round-12 review, DA-C1 minimum evidence):

Gold-axis dose-response. Subsample the gR−gD added gold relations at
r ∈ {0, .25, .5, .75, 1} (document-level Bernoulli, seed 20261030, 3 draws per
dose) and recompute Γ = F1_intra − F1_inter for every frozen-prediction run at
its registered decode. Tests Lemma-A scaling: arm(r) monotone in |D_r|,
direction stable. Exploratory (post-review), disclosed as such.

Validation gates (hard asserts):
  V1: docunet_pe8_s20260917 @gR native per-doc counters == archived per_doc file.
  V2: X2 ml_D_r100_s20260917 @gR fixed_0.5 counters == rev5 evidence per_doc file.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'src' / 'jcip_audit'))
from rev5_recount_lib import CNT, PREP, REC, count, goldmap, goldmatrix, jl, load, metrics  # noqa: E402

OUT = HERE / 'r64_ea2_dose_response.json'
REPS = 2000
BOOTSEED = 20260926          # registered convention (r1_rank_flip)
DOSE_SEED = 20261030
DOSES = [0.25, 0.50, 0.75, 1.00]
DRAWS = 3

SUBJECTS = []  # (family, run_id, decode, logits_path, keys_path)
def add(fam, rid, dec, lp, kp):
    if lp.exists() and kp.exists():
        SUBJECTS.append((fam, rid, dec, lp, kp))
    else:
        print('SKIP missing', rid)

# X2 (REC layout: <run>/logits_evaluation/{logits.npy,keys.jsonl}); sigmoid fixed_0.5
for arm in ('D', 'R'):
    for rho in ('r025', 'r100'):
        for s in ('20260917', '20260918', '20260919'):
            rid = f'ml_{arm}_{rho}_s{s}'
            add('X2', rid, 'x2_fixed05', REC / rid / 'logits_evaluation' / 'logits.npy',
                REC / rid / 'logits_evaluation' / 'keys.jsonl')
# trained encoders (layout: runs/<id>/{logits_evaluation.npy,keys_evaluation.jsonl}); atlop_native
FAMS = [
    ('atlop_pe8', 'revision_v5_experiments/independent_model_retrain/runs'),
    ('atlop_full', 'revision_v5_experiments/independent_model/runs'),
    ('docunet_pe8', 'revision_v5_experiments/docunet/runs'),
    ('docunetL', 'revision_v5_experiments/docunet_large/runs'),
    ('dreeam', 'revision_v5_experiments/dreeam/runs'),
    ('kddocre', 'revision_v5_experiments/kddocre/runs'),
    ('ttmre', 'revision_v5_experiments/ttmre/runs'),
    ('atlop_rob_large', 'revision_v5_experiments/_atlop_roberta_large/runs'),
]
_runids = []
for fam, rd in FAMS:
    d = ROOT / 'out' / rd
    for run in sorted(d.glob('*')):
        if run.is_dir() and (run / 'logits_evaluation.npy').exists() \
           and 'smoke' not in run.name and '5090' not in run.name and 'rerun' not in run.name \
           and 'local' not in run.name:
            add(fam, run.name, 'atlop_native', run / 'logits_evaluation.npy',
                run / 'keys_evaluation.jsonl')


def decode_predictions(dec, logits):
    if dec == 'x2_fixed05':
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs >= 0.5
    if dec == 'atlop_native':
        th = logits[:, :1]
        mask = logits > th
        topv = np.sort(logits, axis=1)[:, -4:][:, :1]
        mask &= logits >= topv
        return mask[:, 1:]                      # drop NA column -> (N,96)
    raise ValueError(dec)


def main():
    rel = load(PREP / 'relation_to_index.json')
    keys0 = jl(SUBJECTS[0][4])
    docs = sorted({k['doc_id'] for k in keys0})
    di = {d: i for i, d in enumerate(docs)}
    n_docs = len(docs)
    gold_D = goldmatrix(keys0, goldmap(PREP / 'golds_gD.jsonl', rel), 96)
    gold_R = goldmatrix(keys0, goldmap(PREP / 'golds_gR.jsonl', rel), 96)
    added = gold_R & ~gold_D
    n_added = int(added.sum())
    n_gD, n_gR = int(gold_D.sum()), int(gold_R.sum())
    print(f'gD rels={n_gD} gR rels={n_gR} added={n_added} docs={n_docs}')

    didx = np.array([di[k['doc_id']] for k in keys0])
    grp = np.array([k['group'] for k in keys0])
    # every run must share the same candidate order
    for fam, rid, dec, lp, kp in SUBJECTS[1:]:
        ks = jl(kp)
        assert len(ks) == len(keys0) and all(
            a['doc_id'] == b['doc_id'] and a['h'] == b['h'] and a['t'] == b['t']
            for a, b in zip(ks, keys0)), f'key order mismatch {rid}'

    # dose masks: document-level Bernoulli over added cells, 3 draws per dose
    rng = np.random.default_rng(DOSE_SEED)
    dose_masks = {}                              # (r, draw) -> bool (N,96) added-keep
    for r in DOSES:
        for draw in range(DRAWS):
            keep_docs = rng.random(n_docs) < r
            m = np.zeros_like(added)
            for row in np.flatnonzero(added.any(axis=1)):
                if keep_docs[didx[row]]:
                    m[row] = added[row]
            dose_masks[(r, draw)] = m

    # bootstrap weights (registered convention)
    brng = np.random.default_rng(BOOTSEED)
    idx = brng.integers(0, n_docs, (REPS, n_docs))
    W = np.zeros((REPS, n_docs))
    for b, row in enumerate(idx):
        W[b] = np.bincount(row, minlength=n_docs)

    def gap_stats(a):
        m = metrics(a.sum(0))
        point = m[0, 2] - m[1, 2]                # F1 intra − inter
        bm = metrics((W @ a.reshape(n_docs, -1)).reshape(REPS, 2, len(CNT)))
        boots = bm[:, 0, 2] - bm[:, 1, 2]
        return float(point), np.percentile(boots, [2.5, 97.5]).tolist()

    results = {'meta': {
        'experiment': 'E-A2 dose-response (gold-axis, document-level subsampling of gR−gD added relations)',
        'tier': 'exploratory, post-review round 12 (DA-C1 minimum evidence)',
        'population': f'{len(keys0)} frozen candidate pairs, {n_docs} docs, 498-doc dev subset',
        'doses': DOSES, 'draws_per_dose': DRAWS, 'dose_seed': DOSE_SEED,
        'bootstrap': {'reps': REPS, 'unit': 'document', 'seed': BOOTSEED, 'ci': 'percentile95'},
        'n_gold_gD': n_gD, 'n_gold_gR': n_gR, 'n_added_cells': n_added,
        'reading': 'Γ = F1_intra − F1_inter at registered decode per family',
    }, 'validation': {}, 'subjects': []}

    # V1: docunet reproduction vs archived per_doc (gR native)
    def perdoc_from_logits(lp, kp, dec, gold):
        lg = np.load(lp).astype(np.float64)
        pred = decode_predictions(dec, lg)
        return count(pred, gold, didx, grp, n_docs, np.ones(len(keys0), bool))

    arch = json.load(open(
        ROOT / 'out/revision_v5_experiments/docunet/per_doc_docunet_pe8_s20260917.json',
        encoding='utf-8'))['per_doc']['evaluation|gR|atlop_native|earliest_mention']
    a_chk = perdoc_from_logits(SUBJECTS[[s[1] for s in SUBJECTS].index('docunet_pe8_s20260917')][3],
                               SUBJECTS[[s[1] for s in SUBJECTS].index('docunet_pe8_s20260917')][4],
                               'atlop_native', gold_R)
    diffs = [f'{c}_{g}' for j, g in enumerate(('intra', 'inter')) for z, c in enumerate(CNT)
             if sum(arch[g][c]) != int(a_chk[:, j, z].sum())]
    results['validation']['V1_docunet_gR_native_reproduction'] = 'exact' if not diffs else diffs
    assert not diffs, diffs
    print('V1 exact')

    ev = ROOT / 'out/rev5_recompute/evidence/per_doc_ml_D_r100_s20260917.json'
    if ev.exists():
        arch2 = json.load(open(ev, encoding='utf-8'))['U|fixed_0.5|multi']
        i2 = [s[1] for s in SUBJECTS].index('ml_D_r100_s20260917')
        a2 = perdoc_from_logits(SUBJECTS[i2][3], SUBJECTS[i2][4], 'x2_fixed05', gold_R)
        diffs2 = [f'{c}_{g}' for j, g in enumerate(('intra', 'inter')) for z, c in enumerate(CNT)
                  if sum(arch2[g][c]) != int(a2[:, j, z].sum())]
        results['validation']['V2_x2_gR_fixed05_reproduction'] = 'exact' if not diffs2 else diffs2
        assert not diffs2, diffs2
        print('V2 exact')

    for fam, rid, dec, lp, kp in SUBJECTS:
        lg = np.load(lp).astype(np.float64)
        pred = decode_predictions(dec, lg)
        a0 = count(pred, gold_D, didx, grp, n_docs, np.ones(len(keys0), bool))
        g0, _ = gap_stats(a0)
        entry = {'family': fam, 'run': rid, 'decode': dec, 'gamma_gD': g0, 'doses': {}}
        for r in DOSES:
            draws = []
            for draw in range(DRAWS):
                gold_r = gold_D | dose_masks[(r, draw)]
                ar = count(pred, gold_r, didx, grp, n_docs, np.ones(len(keys0), bool))
                gr, _ = gap_stats(ar)
                dr = gr - g0
                bm_r = metrics((W @ ar.reshape(n_docs, -1)).reshape(REPS, 2, len(CNT)))
                bm_0 = metrics((W @ a0.reshape(n_docs, -1)).reshape(REPS, 2, len(CNT)))
                dboot = (bm_r[:, 0, 2] - bm_r[:, 1, 2]) - (bm_0[:, 0, 2] - bm_0[:, 1, 2])
                draws.append({'gamma': gr, 'delta': dr,
                              'ci95': np.percentile(dboot, [2.5, 97.5]).tolist(),
                              'kept_added': int(dose_masks[(r, draw)].sum())})
            entry['doses'][str(r)] = draws
        # monotonicity + sign stability across draws-mean
        rs = DOSES
        mean_arm = [np.mean([abs(d['delta']) for d in entry['doses'][str(r)]]) for r in rs]
        from scipy.stats import spearmanr
        rho, _ = spearmanr(mean_arm, rs)
        deltas = [d['delta'] for r in rs for d in entry['doses'][str(r)]]
        entry['spearman_arm_vs_dose'] = float(rho)
        entry['sign_consistent'] = bool(len({np.sign(d) for d in deltas}) == 1)
        entry['mean_arm_by_dose'] = mean_arm
        results['subjects'].append(entry)
        print(f'{rid}: g0={g0:+.4f} arms={[round(a,4) for a in mean_arm]} rho={rho:+.2f} sign_ok={entry["sign_consistent"]}', flush=True)

    # aggregate: family-mean normalized arm
    fams = {}
    for e in results['subjects']:
        fams.setdefault(e['family'], []).append(np.array(e['mean_arm_by_dose']))
    results['family_mean_arm_by_dose'] = {f: np.mean(v, 0).tolist() for f, v in fams.items()}
    results['dose_fractions'] = {str(r): float(np.mean([
        e['doses'][str(r)][0]['kept_added'] for e in results['subjects']]) / max(n_added, 1))
        for r in DOSES}
    n_mono = sum(1 for e in results['subjects'] if e['spearman_arm_vs_dose'] >= 0.9)
    n_sign = sum(1 for e in results['subjects'] if e['sign_consistent'])
    results['summary'] = {
        'n_subjects': len(results['subjects']),
        'n_spearman_ge_0.9': n_mono, 'n_sign_consistent': n_sign,
        'verdict_note': 'arm monotone in |D| (Spearman>=0.9) and direction-stable subjects above; '
                        'dose fractions are realized kept-added fractions (doc-level Bernoulli).'}
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    print('saved', OUT)


if __name__ == '__main__':
    main()
