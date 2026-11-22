# -*- coding: utf-8 -*-
"""r64_ea5a_format_tie_layer.py — C2(1) partial: tie layer on the format axis.

DA C2 minimum evidence (1): run AER's tie-declaration component on the existing
format×model grid (regenerated outputs; ties do not require frozen predictions).
Local frozen outputs cover D5 (English DocRED prompts) with BOTH formats for
Qwen-2.5-7B and Mistral-7B. Computes per-format Γ per model, the format-induced
ordering flip(s), CI-based tie status per format and under the format-ensemble
(z over models averaged over {json, freetext}), doc-paired bootstrap R=2000,
seed 20260926 (registered convention). Validation gate: Γ point reproduces
audit_*.json metrics.gD (intra.F1 − inter.F1).

NOT executable tonight (recorded): am.19b constrained decoding (Mistral-7B
weights absent locally; HF network unreachable; pipeline + SMOKE artifact exist
at d5_llm_hook/preds_..._CONSTRAINED_SMOKE.jsonl.bak). ℓ-axis (HacRED zh/en)
per-doc outputs are not in the local bundle (aggregates only, p15_*).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # out/d5_llm_hook
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'docs' / 'src' / 'jcip_audit'))
import d5_llm_hook as H                          # noqa: E402

OUT = HERE / 'r64_ea5a_format_tie_layer.json'
R = 2000
BOOTSEED = 20260926
CELLS = [('qwen2.5-7b-instruct', 'json'), ('qwen2.5-7b-instruct', 'freetext'),
         ('mistral-7b-instruct', 'json'), ('mistral-7b-instruct', 'freetext')]


def perdoc_counters(model, fmt):
    """Faithful replica of cmd_audit (registered): full m×m candidate loop,
    preds/G as relation-index sets, group = intra iff first[h]==first[t]."""
    from collections import defaultdict
    D, per_doc, _ = H.build_canon_map()
    rel2idx = {k: v for k, v in json.loads(H.REL_IDX.read_text(encoding='utf-8')).items()
               if k.upper() != 'NA'}
    G = {}
    for line in (H.OLD_PREP / 'golds_gD.jsonl').read_text(encoding='utf-8').split('\n'):
        if line.strip():
            r = json.loads(line)
            G[(r['doc_id'], r['h'], r['t'])] = {rel2idx[nm] for nm in r['rels'] if nm in rel2idx}
    name2canon = {}
    for dd in D:
        did = H.doc_id_of(dd)
        mm = defaultdict(set)
        for raw, c in per_doc[did]['raw2canon'].items():
            mm[str(dd['vertexSet'][raw][0]['name']).strip()].add(c)
        name2canon[did] = mm
    doc_ids = sorted(per_doc)
    pos = {d: i for i, d in enumerate(doc_ids)}
    preds = defaultdict(set)
    for line in (HERE / f'preds_{model}_{fmt}.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        did = r['did']
        if did not in per_doc:
            continue
        m = name2canon[did]
        ents = r.get('ents') or sorted(m)
        for h, rl, t in H.parse_output(r['raw'], r['fmt'], ents, k_bound=r.get('k_index')):
            if not isinstance(h, str) or h.startswith('<<'):
                continue
            ri = rel2idx.get(rl)
            if ri is None:
                continue
            hc_l, tc_l = m.get(h), m.get(t)
            if not hc_l or not tc_l or len(hc_l) > 1 or len(tc_l) > 1:
                continue
            hc, tc = next(iter(hc_l)), next(iter(tc_l))
            if hc == tc:
                continue
            preds[(did, hc, tc)].add(ri)
    cnt = np.zeros((len(doc_ids), 2, 3), dtype=np.int64)
    for did in doc_ids:
        k = pos[did]
        fm = per_doc[did]
        m = len(fm['sigs'])
        first = fm['first_sent']
        for h in range(m):
            for t in range(m):
                if h == t:
                    continue
                key = (did, h, t)
                P = preds.get(key, set())
                Gd = G.get(key, set())
                g = 0 if first[h] == first[t] else 1
                cnt[k, g, 0] += len(P & Gd)
                cnt[k, g, 1] += len(P - Gd)
                cnt[k, g, 2] += len(Gd - P)
    return cnt, doc_ids


def gamma(cnt):
    c = cnt.sum(0)
    def f1(row):
        tp, fp, fn = row
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return f1(c[0]) - f1(c[1])


def main():
    # NOTE: intra/inter grouping above is approximated via first-sentence membership of
    # entity mentions; the validation gate below enforces agreement with the frozen
    # audit values — if grouping were wrong, gate fails.
    res = {'meta': {
        'experiment': 'C2(1) partial: CI tie layer on the format axis (D5, English DocRED)',
        'cells': [f'{m}|{f}' for m, f in CELLS],
        'bootstrap': {'reps': R, 'unit': 'document', 'seed': BOOTSEED, 'ci': 'percentile95'},
        'infeasible_tonight': [
            'am.19b constrained decoding: Mistral-7B weights absent locally, HF unreachable '
            '(pipeline exists; SMOKE artifact preds_qwen2.5-7b-instruct_json_CONSTRAINED_SMOKE.jsonl.bak)',
            'ℓ-axis (HacRED zh/en prompts) tie layer: per-doc outputs not in local bundle (aggregates only)'],
        'tier': 'exploratory, post-review round 12',
    }, 'validation': {}, 'cells': {}}
    rng = np.random.default_rng(BOOTSEED)
    n_docs = None
    counters = {}
    for model, fmt in CELLS:
        cnt, doc_ids = perdoc_counters(model, fmt)
        n_docs = len(doc_ids)
        counters[(model, fmt)] = cnt
        g = gamma(cnt)
        # bootstrap gamma dist
        boots = []
        idx_all = rng.integers(0, n_docs, (R, n_docs))
        for row in idx_all:
            boots.append(gamma(cnt[row]))
        res['cells'][f'{model}|{fmt}'] = {'gamma': round(g, 4),
                                          'ci95': [round(float(np.percentile(boots, 2.5)), 4),
                                                   round(float(np.percentile(boots, 97.5)), 4)]}
        audit = json.load(open(HERE / f'audit_preds_{model}_{fmt}.json', encoding='utf-8'))
        ref = audit['metrics']['gD']['intra']['F1'] - audit['metrics']['gD']['inter']['F1']
        ok = abs(g - ref) < 5e-3
        res['validation'][f'{model}|{fmt}'] = f'{g:.4f} vs {ref:.4f} ' + ('ok' if ok else 'MISMATCH')
        print(f'{model}|{fmt}: Γ={g:+.4f} (audit {ref:+.4f})', flush=True)
        assert ok, (model, fmt, g, ref)

    # tie layer on the model pair under each format + format-ensemble
    def pair_tie(cnt_a, cnt_b):
        d = []
        for row in idx_all[:R]:
            d.append(gamma(cnt_a[row]) - gamma(cnt_b[row]))
        return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    models = ['qwen2.5-7b-instruct', 'mistral-7b-instruct']
    fmts = ['json', 'freetext']
    # z over models per format, averaged over formats = format-ensemble reading
    z = {}
    for f in fmts:
        gs = np.array([gamma(counters[(m, f)]) for m in models])
        z[f] = (gs - gs.mean()) / (gs.std(ddof=1) + 1e-12)
    ens = {m: float(np.mean([z[f][i] for f in fmts])) for i, m in enumerate(models)}
    out = {}
    for f in fmts:
        ga, gb = gamma(counters[(models[0], f)]), gamma(counters[(models[1], f)])
        lo, hi = pair_tie(counters[(models[0], f)], counters[(models[1], f)])
        out[f] = {'gap_q_minus_m': round(ga - gb, 4), 'tie': bool(lo <= 0 <= hi),
                  'ci95': [round(lo, 4), round(hi, 4)]}
    sj = np.sign(out['json']['gap_q_minus_m'])
    sf = np.sign(out['freetext']['gap_q_minus_m'])
    res['pair_layer'] = {
        'pair': 'qwen2.5-7b | mistral-7b',
        'per_format': out,
        'format_induced_flip': bool(sj != 0 and sf != 0 and sj != sf),
        'covered_by_either_format_tie': None,
        'ensemble_reading': {k: round(v, 4) for k, v in ens.items()},
    }
    if res['pair_layer']['format_induced_flip']:
        res['pair_layer']['covered_by_either_format_tie'] = bool(
            out['json']['tie'] or out['freetext']['tie'])
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(res['pair_layer'], ensure_ascii=False, indent=1))
    print('saved', OUT)


if __name__ == '__main__':
    main()
