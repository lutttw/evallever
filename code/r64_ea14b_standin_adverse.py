# -*- coding: utf-8 -*-
"""E-A14 adverse direction: degrade side B (the comparator stand-in) and
measure when the audited verdict breaks. Same machinery as E-A14 main."""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, r'E:\WorkBuddy\参考文献解读\04_文档级IE\out\multi_task\audit_published')
import audit_published as A  # noqa: E402

R_DRAW = 200
BOOT = 500
LEVELS = [0.005, 0.01, 0.02, 0.04, 0.08]
SEED = 20261113


def f1_of(tp, fp, fn):
    p = tp / np.maximum(tp + fp, 1e-12)
    r = tp / np.maximum(tp + fn, 1e-12)
    return 2 * p * r / np.maximum(p + r, 1e-12)


def boot_ci(tpA, fpA, fnA, tpB, fpB, fnB, rng):
    n = len(tpA)
    idx = rng.integers(0, n, (BOOT, n))
    ta = tpA[idx].sum(1); fa = fpA[idx].sum(1); na = fnA[idx].sum(1)
    tb = tpB[idx].sum(1); fb = fpB[idx].sum(1); nb = fnB[idx].sum(1)
    d = f1_of(ta, fa, na) - f1_of(tb, fb, nb)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    A.rel2id = json.load(open(A.PREP / 'relation_to_index.json', encoding='utf-8'))
    golds = {}
    for g in ('gD', 'gR'):
        golds[g] = {}
        for row in A.load_jsonl(A.PREP / ('golds_%s.jsonl' % g)):
            golds[g][(row['doc_id'], int(row['h']), int(row['t']))] = set(row['rels'])

    claims = {'C1': ('dreeam', 'atlop_base'),
              'C2': ('kddocre', 'docunet'),
              'C3': ('ttmre_release', 'kddocre')}
    out = {'meta': {'experiment': 'E-A14b adverse-direction stand-in sensitivity: '
                    'TP-removal degradation on side B (comparator stand-in), '
                    'where a flip would erase the audited verdict',
                    'levels_p': LEVELS, 'R_draw': R_DRAW, 'boot': BOOT,
                    'seed': SEED},
           'claims': {}}

    for cid, (sa, sb) in claims.items():
        res = {'claim': cid, 'sysA': sa, 'sysB': sb, 'gold': {}}
        for g in ('gD', 'gR'):
            sa_docs, sb_docs = {}, {}
            fmt = None
            for d0 in A.RUNS[sa]:
                bd, _n, fmt = A.per_doc_counts(d0, A.rel2id, golds[g], fmt=fmt)
                for k, v in bd.items():
                    sa_docs.setdefault(k, []).append(v)
            fmt = None
            for d0 in A.RUNS[sb]:
                bd, _n, fmt = A.per_doc_counts(d0, A.rel2id, golds[g], fmt=fmt)
                for k, v in bd.items():
                    sb_docs.setdefault(k, []).append(v)

            def avg(perk):
                return {k: {f: float(np.mean([x[f] for x in v]))
                            for f in ('tp', 'fp', 'fn')} for k, v in perk.items()}
            da, db = avg(sa_docs), avg(sb_docs)
            common = sorted(set(da) & set(db))

            def arrs(docs):
                tp = np.array([docs[k]["tp"] for k in common], dtype=float)
                fp = np.array([docs[k]["fp"] for k in common], dtype=float)
                fn = np.array([docs[k]["fn"] for k in common], dtype=float)
                return tp, fp, fn

            tpA, fpA, fnA = arrs(da)
            tpB, fpB, fnB = arrs(db)
            base_f1b = float(f1_of(tpB.sum(), fpB.sum(), fnB.sum()))
            base_delta = float(f1_of(tpA.sum(), fpA.sum(), fnA.sum())
                               - base_f1b)
            rng0 = np.random.default_rng(SEED)
            blo, bhi = boot_ci(tpA, fpA, fnA, tpB, fpB, fnB, rng0)
            base_ci_excl = (blo > 0) or (bhi < 0)

            levels = {}
            for p in LEVELS:
                rng = np.random.default_rng(SEED + 7 + int(p * 1e6))
                flips_sign = 0
                flips_ci = 0
                drops = []
                for d in range(R_DRAW):
                    keep = rng.random(len(tpB)) >= p
                    tpB_d = np.where(keep, tpB, 0.0)
                    fnB_d = fnB + (tpB - tpB_d)
                    drop = base_f1b - float(f1_of(tpB_d.sum(), fpB.sum(), fnB_d.sum()))
                    drops.append(drop)
                    dd = float(f1_of(tpA.sum(), fpA.sum(), fnA.sum())
                               - f1_of(tpB_d.sum(), fpB.sum(), fnB_d.sum()))
                    lo, hi = boot_ci(tpA, fpA, fnA, tpB_d, fpB, fnB_d, rng)
                    if (dd > 0) != (base_delta > 0) and dd != 0:
                        flips_sign += 1
                    if base_ci_excl and not (lo > 0 or hi < 0):
                        flips_ci += 1
                levels['p=%.3f' % p] = {
                    'mean_F1_drop_B': round(float(np.mean(drops)), 4),
                    'sign_flip_rate': round(flips_sign / R_DRAW, 3),
                    'ci_exclusion_loss_rate': round(flips_ci / R_DRAW, 3)}
            res['gold'][g] = {'base': {'delta': round(base_delta, 4),
                                       'ci': [round(blo, 4), round(bhi, 4)],
                                       'base_ci_excl': bool(base_ci_excl)},
                              'levels': levels}
            print(cid, g, 'base_delta', round(base_delta, 4), flush=True)
        out['claims'][cid] = res

    o = A.OUT / 'r64_ea14b_standin_adverse.json'
    json.dump(out, open(o, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('saved ->', o)
    for cid in out['claims']:
        for g in ('gD', 'gR'):
            blk = out['claims'][cid]['gold'][g]
            print(cid, g, json.dumps(blk['levels']))


if __name__ == '__main__':
    main()
