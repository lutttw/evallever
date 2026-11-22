# -*- coding: utf-8 -*-
"""E-A14: empirical stand-in sensitivity curves for the published-claims audit
(r6 residual action (a); DA M2 / R-13 follow-through).

Question: at what stand-in degradation does each claim's verdict break?
Method: TP-removal (recall) degradation on the claim-subject side's per-doc
counts (fraction p of TPs randomly moved to FN, Binomial per doc), swept over
p in {0.005, 0.01, 0.02, 0.04, 0.08}; per level, R_draw=200 degraded replicas;
per replica, a 500-resample doc-paired bootstrap CI of delta = F1(A')-F1(B);
readouts: mean corpus-F1 drop of side A, sign-flip rate, CI-exclusion-loss
rate, and gR_behavior flip rate (reversed/holds/tie per r56 rules).
Base (p=0) cross-checked against r56 v2 stored deltas/CIs.

Output: stats/r64_ea14_standin_sensitivity.json (in audit_published/stats).
"""
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


def arrs(docs, docs_l):
    tp = np.array([docs[k]["tp"] for k in docs_l], dtype=float)
    fp = np.array([docs[k]["fp"] for k in docs_l], dtype=float)
    fn = np.array([docs[k]["fn"] for k in docs_l], dtype=float)
    return tp, fp, fn


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

    # rebuild per-system averaged per-doc counts exactly as r56 v2
    sys_docs, fmts = {}, {}
    for sysname, dirs in A.RUNS.items():
        per_seed = []
        fmt = None
        for d0 in dirs:
            bd, _n, fmt = A.per_doc_counts(d0, A.rel2id, golds['gD'], fmt=fmt)
            per_seed.append(bd)
        k0 = set(per_seed[0])
        docs_avg = {k: {f: float(np.mean([bd[k][f] for bd in per_seed]))
                        for f in ('tp', 'fp', 'fn')} for k in k0}
        sys_docs[sysname] = docs_avg
        fmts[sysname] = fmt

    claims = {'C1': ('dreeam', 'atlop_base'),
              'C2': ('kddocre', 'docunet'),
              'C3': ('ttmre_release', 'kddocre')}
    stored = json.load(open(
    'E:/WorkBuddy/参考文献解读/04_文档级IE/out/evallever_release_v1_submission/statistics/r56_published_claims_audit_v2.json',
    encoding='utf-8', errors='replace'))[1:]
    stored_by = {e['claim_id']: e for e in stored}

    out = {'meta': {
        'experiment': 'E-A14 empirical stand-in sensitivity curves: TP-removal '
                      '(recall) degradation on the claim-subject side, verdict '
                      'flip rates vs degradation level',
        'degradation_model': 'per-doc Binomial: each TP moved to FN with prob p; '
                             'side A (claim subject) degraded, side B fixed',
        'levels_p': LEVELS, 'R_draw': R_DRAW, 'boot': BOOT, 'seed': SEED,
        'base_check': 'p=0 deltas/CIs cross-checked against r56 v2 stored values'},
        'claims': {}}

    for cid, (sa, sb) in claims.items():
        e_st = stored_by[cid]
        res = {'claim': e_st['claim'], 'sysA': sa, 'sysB': sb, 'gold': {}}
        docs_l = sorted(set(sys_docs[sa]) & set(sys_docs[sb]))
        for g in ('gD', 'gR'):
            sa_docs, sb_docs = {}, {}
            fmt = fmts[sa]
            for d0 in A.RUNS[sa]:
                bd, _n, fmt = A.per_doc_counts(d0, A.rel2id, golds[g], fmt=fmt)
                for k, v in bd.items():
                    sa_docs.setdefault(k, []).append(v)
            fmt = fmts[sb]
            for d0 in A.RUNS[sb]:
                bd, _n, fmt = A.per_doc_counts(d0, A.rel2id, golds[g], fmt=fmt)
                for k, v in bd.items():
                    sb_docs.setdefault(k, []).append(v)
            def avg(perk):
                return {k: {f: float(np.mean([x[f] for x in v]))
                            for f in ('tp', 'fp', 'fn')} for k, v in perk.items()}
            da, db = avg(sa_docs), avg(sb_docs)
            common = sorted(set(da) & set(db))
            tpA, fpA, fnA = arrs(da, common)
            tpB, fpB, fnB = arrs(db, common)
            base_f1a = float(f1_of(tpA.sum(), fpA.sum(), fnA.sum()))
            base_f1b = float(f1_of(tpB.sum(), fpB.sum(), fnB.sum()))
            base_delta = base_f1a - base_f1b
            rng0 = np.random.default_rng(SEED)
            blo, bhi = boot_ci(tpA, fpA, fnA, tpB, fpB, fnB, rng0)
            st = e_st[f'delta_{g}'], e_st[f'ci_{g}']
            base_ok = (abs(base_delta - st[0]) < 0.01
                       and abs(blo - st[1][0]) < 0.01 and abs(bhi - st[1][1]) < 0.01)

            levels = {}
            for p in LEVELS:
                rng = np.random.default_rng(SEED + int(p * 1e6))
                flips_sign = 0
                flips_ci = 0
                drops = []
                base_ci_excl = (blo > 0) or (bhi < 0)
                for d in range(R_DRAW):
                    keep = rng.random(len(tpA)) >= p
                    tpA_d = np.where(keep, tpA, 0.0)
                    fnA_d = fnA + (tpA - tpA_d)
                    drop = base_f1a - float(f1_of(tpA_d.sum(), fpA.sum(), fnA_d.sum()))
                    drops.append(drop)
                    lo, hi = boot_ci(tpA_d, fpA, fnA_d, tpB, fpB, fnB, rng)
                    dd = float(f1_of(tpA_d.sum(), fpA.sum(), fnA_d.sum())
                               - f1_of(tpB.sum(), fpB.sum(), fnB.sum()))
                    if (dd > 0) != (base_delta > 0) and not (dd == 0):
                        flips_sign += 1
                    if base_ci_excl and not (lo > 0 or hi < 0):
                        flips_ci += 1
                levels['p=%.3f' % p] = {
                    'mean_F1_drop_A': round(float(np.mean(drops)), 4),
                    'sign_flip_rate': round(flips_sign / R_DRAW, 3),
                    'ci_exclusion_loss_rate': round(flips_ci / R_DRAW, 3)}
            res['gold'][g] = {
                'base': {'F1_A': round(base_f1a, 4), 'F1_B': round(base_f1b, 4),
                         'delta': round(base_delta, 4), 'ci': [round(blo, 4), round(bhi, 4)],
                         'stored_delta_ci': [st[0], st[1]],
                         'base_check_ok': bool(base_ok)},
                'levels': levels}
        out['claims'][cid] = res
        print(cid, json.dumps(res['gold']['gR']['levels'], indent=0)[:200], flush=True)

    o = A.OUT / 'r64_ea14_standin_sensitivity.json'
    json.dump(out, open(o, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('saved ->', o)


if __name__ == '__main__':
    main()
