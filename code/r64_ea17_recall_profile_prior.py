# -*- coding: utf-8 -*-
"""E-A17 (v2): decomposed prior for the remittance h_a — per-class recall
profile x added-region composition (strata = relation class; zero alignment
ambiguity, unlike sentence-distance which needs an unavailable vertex mapping).

Prediction: h_a_hat = sum_c recall_c x |added_c|, where recall_c is the
system's recall on gD relations of class c and |added_c| the added-region
count in that class.  Measured h_a (E-A16 method) minus the prediction =
difficulty premium of the under-annotated stratum.
Output: stats/r64_ea17_recall_profile_prior.json
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, r'E:\WorkBuddy\参考文献解读\04_文档级IE\out\multi_task\audit_published')
import audit_published as A  # noqa: E402


def main():
    A.rel2id = json.load(open(A.PREP / 'relation_to_index.json', encoding='utf-8'))
    golds = {}
    for g in ('gD', 'gR'):
        golds[g] = {}
        for row in A.load_jsonl(A.PREP / ('golds_%s.jsonl' % g)):
            golds[g][(row['doc_id'], int(row['h']), int(row['t']))] = set(row['rels'])

    gd_set = set()
    for (d, h, t), rels in golds['gD'].items():
        for r in rels:
            if r in A.rel2id:
                gd_set.add((d, h, t, A.rel2id[r]))
    gr_set = set()
    for (d, h, t), rels in golds['gR'].items():
        for r in rels:
            if r in A.rel2id:
                gr_set.add((d, h, t, A.rel2id[r]))
    added = gr_set - gd_set
    added_c = defaultdict(int)
    for _, _, _, r in added:
        added_c[r] += 1
    print('added total:', len(added), '| classes:', len(added_c), flush=True)

    all_sys = sorted({s for pair in
                      {'C1': ('dreeam', 'atlop_base'),
                       'C2': ('kddocre', 'docunet'),
                       'C3': ('ttmre_release', 'kddocre')}.values()
                      for s in pair})

    out = {'meta': {
        'experiment': 'E-A17 decomposed prior for the remittance h_a: '
                      'per-class recall profile x added-region composition',
        'strata': 'relation class (rel2id); sentence-distance stratification '
                  'unavailable (prepared space has no vertex mapping)',
        'premium': 'predicted - measured = difficulty premium of the '
                   'under-annotated stratum (non-random gD omissions)',
        'added_total': len(added)},
        'added_composition': {A.rel_by_id.get(r, str(r)) if False else str(r):
                              v for r, v in sorted(added_c.items())},
        'systems': {}}

    rel_name = {v: k for k, v in A.rel2id.items()}
    out['added_composition'] = {rel_name.get(r, str(r)): v
                                for r, v in sorted(added_c.items())}

    for sysname in all_sys:
        per_seed = []
        for d0 in A.RUNS[sysname]:
            cd, _, fmt = A.per_doc_counts(d0, A.rel2id, golds['gD'], fmt=None)
            logits = np.asarray(np.load(A.RV5 / d0 / "logits_evaluation.npy",
                                        mmap_mode="r"))
            keys = list(A.load_jsonl(A.RV5 / d0 / "keys_evaluation.jsonl"))
            pred = A.DECODE_VARIANTS[fmt](logits)
            # per-class recall profile on gD
            tp_c = defaultdict(int)
            fn_c = defaultdict(int)
            sets_d = defaultdict(set)
            for i, k in enumerate(keys):
                did = k['doc_id']
                gset = golds['gD'].get((did, int(k['h']), int(k['t'])), set())
                gids = {A.rel2id[r] for r in gset if r in A.rel2id}
                pids = {int(v) - 1 for v in np.where(pred[i])[0].tolist()}
                for r in gids:
                    (tp_c if r in pids else fn_c)[r] += 1
                for v in pids:
                    sets_d[did].add((int(k['h']), int(k['t']), v))
            rec_c = {r: tp_c[r] / max(1, tp_c[r] + fn_c[r])
                     for r in set(tp_c) | set(fn_c)}
            # predicted h_a
            ha_hat = sum(rec_c.get(r, 0.0) * n for r, n in added_c.items())
            # measured
            h_a_meas = sum(len(sets_d[d] & {c for c in added
                                            if c[:3] == ()} | set())
                           for d in sets_d)  # placeholder, replaced below
            h_a_meas = 0
            added_by_doc = defaultdict(set)
            for d, h, t, r in added:
                added_by_doc[d].add((h, t, r))
            for d, ps in sets_d.items():
                h_a_meas += len(ps & added_by_doc.get(d, set()))
            per_seed.append({
                'n_classes_with_recall': len(rec_c),
                'h_a_hat': round(ha_hat, 1),
                'h_a_measured': int(h_a_meas),
                'premium': round(ha_hat - h_a_meas, 1)})
            print(f'{sysname} seed{len(per_seed)-1}: hat={ha_hat:.0f} '
                  f'meas={h_a_meas} premium={ha_hat - h_a_meas:.0f}', flush=True)
        mean_hat = float(np.mean([r['h_a_hat'] for r in per_seed]))
        mean_meas = float(np.mean([r['h_a_measured'] for r in per_seed]))
        out['systems'][sysname] = {
            'per_seed': per_seed,
            'h_a_hat_mean': round(mean_hat, 1),
            'h_a_measured_mean': round(mean_meas, 1),
            'premium_pct_of_hat': round(100 * (mean_hat - mean_meas)
                                        / max(1, mean_hat), 1)}

    o = A.OUT / 'r64_ea17_recall_profile_prior.json'
    json.dump(out, open(o, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('saved ->', o)
    for s, v in out['systems'].items():
        print(f"{s:14s} hat={v['h_a_hat_mean']:.0f} meas={v['h_a_measured_mean']:.0f} "
              f"premium={v['premium_pct_of_hat']}%")


if __name__ == '__main__':
    main()
