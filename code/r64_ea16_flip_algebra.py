# -*- coding: utf-8 -*-
"""E-A16 (v2): algebraic flip propagation — corrected.
Per-doc algebra: TP_R=TP_D+h_a-h_r; FP_R=FP_D-h_a+h_r;
FN_R=FN_D+(|added_d|-h_a)-(|removed_d|-h_r), with h_a=|pred_d n added_d|,
h_r=|pred_d n removed_d| (4-tuple membership). Identity asserted per doc.
Deltas/break points computed on the r56 common-doc intersection."""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, r'E:\WorkBuddy\参考文献解读\04_文档级IE\out\multi_task\audit_published')
import audit_published as A  # noqa: E402


def pred_pos_sets(run_dir, rel2id, golds_probe, fmt=None):
    logits = np.asarray(np.load(A.RV5 / run_dir / "logits_evaluation.npy",
                                mmap_mode="r"))
    keys = list(A.load_jsonl(A.RV5 / run_dir / "keys_evaluation.jsonl"))
    if fmt is None:
        best, fmt = -1.0, "A"
        for name, fn in A.DECODE_VARIANTS.items():
            pred = fn(logits)
            tp = fp = fn_ = 0
            for i, k in enumerate(keys):
                gset = golds_probe.get((k["doc_id"], int(k["h"]), int(k["t"])), set())
                gids = {rel2id[r] for r in gset if r in rel2id}
                pids = {int(v) - 1 for v in np.where(pred[i])[0].tolist()}
                t = len(pids & gids)
                tp += t; fp += len(pids) - t; fn_ += len(gids) - t
            pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn_)
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
            if f1 > best:
                best, fmt = f1, name
    pred = A.DECODE_VARIANTS[fmt](logits)
    sets = defaultdict(set)
    for i, k in enumerate(keys):
        for v in np.where(pred[i])[0].tolist():
            sets[k["doc_id"]].add((int(k["h"]), int(k["t"]), int(v) - 1))
    return sets, fmt


def gold_keyset(golds):
    out = set()
    for (d, h, t), rels in golds.items():
        for r in rels:
            if r in A.rel2id:
                out.add((d, h, t, A.rel2id[r]))
    return out


def main():
    A.rel2id = json.load(open(A.PREP / 'relation_to_index.json', encoding='utf-8'))
    golds = {}
    for g in ('gD', 'gR'):
        golds[g] = {}
        for row in A.load_jsonl(A.PREP / ('golds_%s.jsonl' % g)):
            golds[g][(row['doc_id'], int(row['h']), int(row['t']))] = set(row['rels'])

    gd_set = gold_keyset(golds['gD'])
    gr_set = gold_keyset(golds['gR'])
    added = gr_set - gd_set
    removed = gd_set - gr_set
    added_d = defaultdict(set)
    removed_d = defaultdict(set)
    for d, h, t, r in added:
        added_d[d].add((h, t, r))
    for d, h, t, r in removed:
        removed_d[d].add((h, t, r))
    print(f'gold diff: added={len(added)} removed={len(removed)}', flush=True)

    claims = {'C1': ('dreeam', 'atlop_base'),
              'C2': ('kddocre', 'docunet'),
              'C3': ('ttmre_release', 'kddocre')}

    systems = {}
    identity_all = True
    for sysname, dirs in A.RUNS.items():
        recs = []
        for si, d0 in enumerate(dirs):
            sets_d, fmt = pred_pos_sets(d0, A.rel2id, golds['gD'])
            cd, _, fmt = A.per_doc_counts(d0, A.rel2id, golds['gD'], fmt=fmt)
            cr, _, _ = A.per_doc_counts(d0, A.rel2id, golds['gR'], fmt=fmt)
            ok_all = True
            ha_t = hr_t = 0
            for did, cnt in cd.items():
                p = sets_d.get(did, set())
                add = added_d.get(did, set())
                rem = removed_d.get(did, set())
                h_a = len(p & add)
                h_r = len(p & rem)
                ha_t += h_a; hr_t += h_r
                tR = cnt['tp'] + h_a - h_r
                fR = cnt['fp'] - h_a + h_r
                nR = (cnt['fn'] + (len(add) - h_a) - (len(rem) - h_r))
                ref = cr.get(did, {'tp': 0, 'fp': 0, 'fn': 0})
                if (tR, fR, nR) != (ref['tp'], ref['fp'], ref['fn']):
                    ok_all = False
            identity_all &= ok_all
            recs.append({'seed': si, 'h_a': int(ha_t), 'h_r': int(hr_t),
                         'identity_ok': bool(ok_all)})
            print(f'{sysname} seed{si}: h_a={ha_t} h_r={hr_t} identity={ok_all}',
                  flush=True)
        systems[sysname] = {'per_seed': recs,
                            'identity_all': all(r['identity_ok'] for r in recs),
                            'h_a_mean': float(np.mean([r['h_a'] for r in recs])),
                            'h_r_mean': float(np.mean([r['h_r'] for r in recs]))}

    def counts_common(sysname):
        recs = []
        for d0 in A.RUNS[sysname]:
            cd, _, fmt = A.per_doc_counts(d0, A.rel2id, golds['gD'], fmt=None)
            cr, _, _ = A.per_doc_counts(d0, A.rel2id, golds['gR'], fmt=fmt)
            recs.append((cd, cr))
        return recs

    sys_recs = {s: counts_common(s) for s in A.RUNS}
    sys_keys = {s: set(recs[0][0]) for s, recs in sys_recs.items()}

    def f1(tp, fp, fn):
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        return 2 * p * r / (p + r) if p + r else 0.0

    def agg(recs, keys, which):
        tp = fp = fn = 0
        for cd, cr in recs:
            src = cd if which == 'gD' else cr
            for k in keys:
                d = src.get(k, {'tp': 0, 'fp': 0, 'fn': 0})
                tp += d['tp']; fp += d['fp']; fn += d['fn']
        n = len(recs)
        return tp / n, fp / n, fn / n

    claims_out = {}
    for cid, (sa, sb) in claims.items():
        keys = sys_keys[sa] & sys_keys[sb]
        blk = {'sysA': sa, 'sysB': sb, 'n_common': len(keys)}
        for g in ('gD', 'gR'):
            tpA, fpA, fnA = agg(sys_recs[sa], keys, g)
            tpB, fpB, fnB = agg(sys_recs[sb], keys, g)
            delta = f1(tpA, fpA, fnA) - f1(tpB, fpB, fnB)

            def dshift(k):
                if g == 'gR':
                    return (f1(tpA + k, fpA - k, fnA - k)
                            - f1(tpB, fpB, fnB))
                return (f1(tpA - k, fpA + k, fnA + k)
                        - f1(tpB, fpB, fnB))
            base_sign = delta > 0
            kA = next((k for k in range(0, 20000)
                       if (dshift(k) > 0) != base_sign), None)
            blk[g] = {'delta': round(delta, 4), 'flip_shift_A_hits': kA}
            print(cid, g, 'delta', round(delta, 4), 'breakA', kA, flush=True)
        claims_out[cid] = blk

    o = A.OUT / 'r64_ea16_flip_algebra.json'
    out = {'meta': {
        'experiment': 'E-A16 algebraic flip propagation (L1 upgrade of the '
                      'mechanism paragraph)',
        'rule': 'TP_R=TP_D+h_a-h_r; FP_R=FP_D-h_a+h_r; '
                'FN_R=FN_D+(|added_d|-h_a)-(|removed_d|-h_r) per doc; '
                'h_a=|pred n added_d|, h_r=|pred n removed_d|',
        'gold_diff': {'added': len(added), 'removed': len(removed)},
        'identity_all_systems': bool(identity_all),
        'break_semantics': 'flip_shift_A_hits = added-region hits side A must '
                           'gain (gR) or lose (gD) to flip the audited sign; '
                           'deterministic analogue of the bootstrap CI'},
        'systems': systems, 'claims': claims_out}
    json.dump(out, open(o, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('saved ->', o)


if __name__ == '__main__':
    main()
