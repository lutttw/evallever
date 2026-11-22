# -*- coding: utf-8 -*-
"""r64_ea5b_language_tie_layer.py — C2(1) ℓ-axis: tie layer on prompt language.

Frozen per-doc outputs found locally (docs/out/p15_hacred_icl): Qwen-2.5-7B with
all four cells {H-zh, H-en} × {json, freetext}. Rebuilds the char-layer space
exactly as p15_hacred_icl.cmd_audit, audits each cell (validation gate: Γ point
vs frozen audit jsons), then runs the CI tie layer: model pair Q7B is the only
model with all four cells — the ℓ-axis contrast here is *within-model across
languages*; we report per-cell Γ, per-language Γ gaps json-vs-freetext (format
arm under each prompt language), ℓ-flips (Γ ordering across cells), and CI tie
status. Doc-paired bootstrap R=2000, seed 20260926.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                     # .../04_文档级IE
sys.path.insert(0, str(ROOT / 'docs' / 'src' / 'jcip_audit'))
import p15_hacred_icl as P                 # noqa: E402
from d5_llm_hook import parse_output       # noqa: E402

CELLS = [('zh', 'json'), ('zh', 'freetext'), ('en', 'json'), ('en', 'freetext')]
FNAME = {('zh', 'json'): 'preds_qwen2.5-7b-instruct_json.jsonl',
         ('zh', 'freetext'): 'preds_qwen2.5-7b-instruct_freetext.jsonl',
         ('en', 'json'): 'preds_qwen2.5-7b-instruct_en_json.jsonl',
         ('en', 'freetext'): 'preds_qwen2.5-7b-instruct_en_freetext.jsonl'}
R = 2000
BOOTSEED = 20260926


def build_space():
    char = json.loads(P.DEV.read_text(encoding='utf-8'))
    golds_c = {}
    for g in P.jl(P.PREP / 'candidates' / '../golds_evaluation.jsonl') if False else []:
        pass
    for g in P.jl(P.PREP / 'golds_evaluation.jsonl'):
        golds_c.setdefault(g['doc_id'], {})[(g['h'], g['t'])] = g['rels']
    space = {}
    for dd in char:
        did = P.doc_id_of(dd)
        first = [min((m['sent_id'] for m in ent), default=0) for ent in dd['vertexSet']]
        n2ix = defaultdict(set)
        for i, ent in enumerate(dd['vertexSet']):
            n2ix[ent[0]['name']].add(i)
        space[did] = {'n': len(dd['vertexSet']), 'first': first,
                      'name2ix': dict(n2ix), 'gold': golds_c.get(did, {})}
    docs = {P.doc_id_of(dd): dd for dd in char}
    return docs, space


def gamma_from_cnt(cnt):
    c = cnt.sum(0)
    def f1(row):
        tp, fp, fn = row
        den = 2 * tp + fp + fn
        return 2 * tp / den if den else 0.0
    return f1(c[0]) - f1(c[1])


def main():
    docs, space = build_space()
    rng = np.random.default_rng(BOOTSEED)
    doc_ids = sorted(space)
    n = len(doc_ids)
    res = {'meta': {'experiment': 'C2(1) ℓ-axis tie layer (HacRED H-zh vs H-en prompts)',
                    'model': 'qwen2.5-7b-instruct', 'cells': [f'{l}|{f}' for l, f in CELLS],
                    'bootstrap': {'reps': R, 'unit': 'document', 'seed': BOOTSEED,
                                  'ci': 'percentile95'},
                    'tier': 'exploratory, post-review round 12'}, 'validation': {}, 'cells': {}}
    cnts = {}
    for cell in CELLS:
        lang, fmt = cell
        preds_name = defaultdict(set)
        for r in P.jl(HERE / FNAME[cell]):
            ents = r['ents']
            for h, rl, t in parse_output(r['raw'], r['fmt'], ents):
                preds_name[r['did']].add((h, rl, t))
        for did in space:
            preds_name.setdefault(did, set())
        out = P.audit_space(preds_name, docs, space)
        doc_ids_r, cnt, _ = out['_per_doc_arrays']
        assert doc_ids_r == doc_ids
        cnts[cell] = cnt
        g = gamma_from_cnt(cnt)
        boots = [gamma_from_cnt(cnt[row]) for row in rng.integers(0, n, (R, n))]
        res['cells'][f'{lang}|{fmt}'] = {'gamma': round(g, 4),
                                         'ci95': [round(float(np.percentile(boots, 2.5)), 4),
                                                  round(float(np.percentile(boots, 97.5)), 4)]}
        # validation vs frozen audit if exists
        af = HERE / ('audit_preds_qwen2.5-7b-instruct_json.json' if lang == 'zh' and fmt == 'json'
                     else f'audit_preds_qwen2.5-7b-instruct_{fmt}.json' if lang == 'zh'
                     else f'audit_preds_qwen2.5-7b-instruct_en_{fmt}.json')
        if af.exists():
            a = json.load(open(af, encoding='utf-8'))
            met = a.get('metrics', a)
            node = met.get('char', met)
            try:
                ref = node['intra']['F1'] - node['inter']['F1']
                res['validation'][f'{lang}|{fmt}'] = f'{g:.4f} vs {ref:.4f} ' + (
                    'ok' if abs(g - ref) < 5e-3 else 'MISMATCH')
            except Exception:
                res['validation'][f'{lang}|{fmt}'] = 'audit structure differs (see json)'
        print(f'{lang}|{fmt}: Γ={g:+.4f} ci={res["cells"][f"{lang}|{fmt}"]["ci95"]}', flush=True)

    def pair_ci(ca, cb):
        d = [gamma_from_cnt(ca[row]) - gamma_from_cnt(cb[row])
             for row in rng.integers(0, n, (R, n))]
        return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    arms = {}
    for lang in ('zh', 'en'):
        lo, hi = pair_ci(cnts[(lang, 'json')], cnts[(lang, 'freetext')])
        arms[f'{lang}_format_arm_json_minus_freetext'] = {
            'point': round(gamma_from_cnt(cnts[(lang, 'json')])
                           - gamma_from_cnt(cnts[(lang, 'freetext')]), 4),
            'ci95': [round(lo, 4), round(hi, 4)],
            'tie': bool(lo <= 0 <= hi)}
    for fmt in ('json', 'freetext'):
        lo, hi = pair_ci(cnts[('zh', fmt)], cnts[('en', fmt)])
        arms[f'{fmt}_language_arm_zh_minus_en'] = {
            'point': round(gamma_from_cnt(cnts[('zh', fmt)])
                           - gamma_from_cnt(cnts[('en', fmt)]), 4),
            'ci95': [round(lo, 4), round(hi, 4)],
            'tie': bool(lo <= 0 <= hi)}
    res['arms'] = arms
    g = {f'{l}|{f}': gamma_from_cnt(cnts[(l, f)]) for l in ('zh', 'en') for f in ('json', 'freetext')}
    res['summary'] = {
        'gamma_map': {k: round(v, 4) for k, v in g.items()},
        'format_arm_sign_stable': bool(np.sign(arms['zh_format_arm_json_minus_freetext']['point'])
                                       == np.sign(arms['en_format_arm_json_minus_freetext']['point'])),
        'language_arm_sign_stable': bool(np.sign(arms['json_language_arm_zh_minus_en']['point'])
                                         == np.sign(arms['freetext_language_arm_zh_minus_en']['point'])),
        'all_arms_tied': bool(all(a['tie'] for a in arms.values())),
    }
    (HERE / 'r64_ea5b_language_tie_layer.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(res['summary'], ensure_ascii=False))
    print('saved r64_ea5b_language_tie_layer.json')


if __name__ == '__main__':
    main()
