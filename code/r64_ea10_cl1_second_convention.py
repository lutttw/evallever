# -*- coding: utf-8 -*-
"""_p9_cl1_second_convention.py — CL1 under a SECOND independent ATLOP
retraining convention (review item R1-W3 option b).

CL1 audit convention A (current): ATLOP side = _atlop_probe/runs/
atlop_D_s20260917. Convention B (this script): ATLOP side =
independent_model_retrain/runs/atlop_D_s202609{17,18,19} (three seeds),
x the same DREEAM stand-in teacher seeds, both golds.
Also prints convention-A ATLOP F1 for distinctness.
Caliber identical to r56 v2/PS (per-doc counts, doc-paired bootstrap,
R=10000, seed 20260926).
Output: stats/audit_cl1_second_convention.json
"""
import json
from collections import defaultdict

import numpy as np

import audit_published as A

CONVB = ("independent_model_retrain/runs/atlop_D_s20260917,"
         "independent_model_retrain/runs/atlop_D_s20260918,"
         "independent_model_retrain/runs/atlop_D_s20260919")


def per_seed_counts_for(runs, golds_by_g):
    out = {}
    for g, golds in golds_by_g.items():
        fmt = None
        seq = []
        for d0 in runs:
            bd, _n, fmt = A.per_doc_counts(d0, A.rel2id, golds, fmt=fmt)
            seq.append(bd)
        out[g] = seq
    return out


def main():
    A.rel2id = json.load(open(A.PREP / 'relation_to_index.json',
                              encoding='utf-8'))
    golds = {}
    for g in ('gD', 'gR'):
        golds[g] = {}
        for row in A.load_jsonl(A.PREP / ('golds_%s.jsonl' % g)):
            golds[g][(row['doc_id'], int(row['h']), int(row['t']))] = \
                set(row['rels'])

    # convention-B ATLOP: second independent retrain, 3 seeds
    convB_runs = [A.RV5 / p for p in CONVB.split(',')]
    # DREEAM stand-in: 3 seeds (same as audit)
    dreeam_runs = A.RUNS['dreeam']

    counts = {}
    for g in ('gD', 'gR'):
        fmt = None
        seqs = {}
        for label, runs in (('atlop_convB', convB_runs),
                            ('dreeam_standin', dreeam_runs)):
            seq = []
            f = None
            for d0 in runs:
                bd, _n, f = A.per_doc_counts(d0, A.rel2id, golds[g], fmt=f)
                seq.append(bd)
            seqs[label] = seq
        counts[g] = seqs
        print(g, 'done', flush=True)

    common = set(counts['gD']['atlop_convB'][0])
    for g in ('gD', 'gR'):
        for label, seq in counts[g].items():
            for bd in seq:
                common &= set(bd)
    docs_l = sorted(common)
    print('common docs:', len(docs_l), flush=True)

    def one(g, da, db, docs_l):
        fa = A.corpus_f1(da)
        fb = A.corpus_f1(db)
        rng = np.random.default_rng(A.SEED + (0 if g == 'gD' else 1))
        boots = np.zeros(A.BOOT)
        for b in range(A.BOOT):
            idx = rng.integers(0, len(docs_l), len(docs_l))
            sel = set(docs_l[j] for j in idx)
            boots[b] = A.corpus_f1(da, sel) - A.corpus_f1(db, sel)
        ci = [round(float(np.percentile(boots, 2.5)), 4),
              round(float(np.percentile(boots, 97.5)), 4)]
        return round(fa, 4), round(fb, 4), round(fb - fa, 4), ci

    rows = []
    n = min(len(counts['gD']['atlop_convB']),
            len(counts['gR']['atlop_convB']),
            len(counts['gD']['dreeam_standin']),
            len(counts['gR']['dreeam_standin']))
    for si in range(n):
        rec = {'seed_index': si}
        for g in ('gD', 'gR'):
            da = counts[g]['atlop_convB'][si]
            db = counts[g]['dreeam_standin'][si]
            fa, fb, delta, ci = one(g, da, db, docs_l)
            rec['F1_%s_ATLOPconvB' % g] = fa
            rec['F1_%s_DREEAMstandin' % g] = fb
            rec['delta_%s' % g] = delta
            rec['ci_%s' % g] = ci
        # ordering under both golds: HOLDS iff DREEAM > ATLOP on both
        rec['ordering'] = ('HOLDS' if rec['delta_gD'] > 0 and rec['delta_gR'] > 0
                           else 'REVERSES' if rec['delta_gR'] < 0 and rec['delta_gD'] < 0
                           else 'MIXED')
        rows.append(rec)
        print('seed', si, json.dumps(rec), flush=True)

    res = {'meta': {
        'experiment': 'CL1 under a second independent ATLOP retraining '
                      'convention (independent_model_retrain, 3 seeds) x '
                      'DREEAM stand-in teacher seeds; same caliber as r56 v2/PS',
        'atlop_convB_runs': CONVB,
        'population': '%d common docs' % len(docs_l)},
        'per_seed': rows}
    outp = os_path_join_out('audit_cl1_second_convention.json')
    json.dump(res, open(outp, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=1)
    print('saved ->', outp)


def os_path_join_out(name):
    import os
    return os.path.join(A.OUT, name)


if __name__ == '__main__':
    main()
