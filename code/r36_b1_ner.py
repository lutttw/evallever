# -*- coding: utf-8 -*-
"""r36_b1_ner.py — 修订69:B1 边界实验证实(冻结设计)。

NER(CoNLL-2003)天然双金标:strict span vs. relaxed span(同为合法评估
口径)。构造分歧集 D(strict, relaxed)(逐句:strict 错而 relaxed 对的
句子)与三模型误差集,重算 disagreement-referenced leverage
arm_D = |Y∩D|/|D|(句级口径,与 P1 存储的 per-sentence error flags 对齐)。

终点(纯描述性):三模型的 arm_D 与 D 密度;对照 B1 预测。
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src' / 'jcip_audit'))
from p1_cross_task import spans_from_bio, evaluate_ner  # noqa: F401

DATA = ROOT / 'out/revision_v5_experiments/data'
PRED = DATA / 'p1_preds'
OUTD = ROOT / 'out/revision_v5_experiments/statistics/r36_b1_ner.json'

rows = [json.loads(l) for l in open(DATA / 'p1_conll03_test.jsonl', encoding='utf-8')]
gold_tags = [r['tags'] for r in rows]
gold_spans = [spans_from_bio(t) for t in gold_tags]
n_sent = len(rows)
print(f'[r36] sentences={n_sent}', flush=True)

MODELS = {
    'bert-base (dslim)': 'ner__dslim__bert-base-NER.json',
    'roberta-large (Jean-Baptiste)': 'ner__Jean-Baptiste__roberta-large-ner-english.json',
    'distilbert (elastic)': 'ner__elastic__distilbert-base-cased-finetuned-conll03-english.json',
}

results = {}
per_model_armD = {}
for label, fn in MODELS.items():
    d = json.load(open(PRED / fn, encoding='utf-8'))
    s_err = d['strict_error_flags']
    r_err = d['relaxed_error_flags']
    assert len(s_err) == n_sent
    # D_M 按模型定义:{句 : 该模型预测 strict 判错而 relaxed 判对}
    d_flags = [bool(s) and not bool(r) for s, r in zip(s_err, r_err)]
    inter = sum(1 for s, dd in zip(s_err, d_flags) if s and dd)
    nD = sum(d_flags)
    arm_d = inter / nD if nD else float('nan')
    results[label] = {
        'strict_err_rate': round(sum(s_err) / n_sent, 4),
        'relaxed_err_rate': round(sum(r_err) / n_sent, 4),
        'D_size': nD,
        'D_density': round(nD / n_sent, 4),
        'Y_intersect_D': inter,
        'arm_D': round(arm_d, 4),
        'strict_f1': d['strict_f1'],
        'relaxed_f1': d['relaxed_f1'],
    }
    per_model_armD[label] = arm_d
    print(f"[r36] {label}: D={nD} arm_D={arm_d:.4f} "
          f"strictErr={results[label]['strict_err_rate']} "
          f"relaxedErr={results[label]['relaxed_err_rate']}", flush=True)

arms = list(per_model_armD.values())
spread = max(arms) - min(arms)
verdict = {
    'D_size_note': 'D is per-model (see per_model D_size)',
    'arm_D_values': {k: round(v, 4) for k, v in per_model_armD.items()},
    'arm_D_spread': round(spread, 4),
    'arm_D_mean': round(float(np.mean(arms)), 4),
    'b1_reading': ('disagreement set exists and arm_D varies across generators: '
                   'B1 narrows to single-gold tasks (SST-2 class); NER does '
                   'supply a lever after all'
                   if (spread > 0.05 and min(arms) > 0.02) else
                   'arm_D near-zero or generator-invariant: B1 confirmed, '
                   'NER-style dual-gold does not supply a usable lever'),
}
res = {
    'meta': {
        'amendment': 69,
        'design': 'B1 boundary verification: NER (CoNLL-2003) natural dual gold '
                  '(strict vs relaxed span), disagreement set per sentence = '
                  'strict-wrong and relaxed-correct; arm_D = |Y intersect D|/|D| '
                  'per model from P1 stored per-sentence error flags (no new '
                  'inference)',
        'models': list(MODELS.keys()),
        'n_sentences': n_sent,
        'b1_reference': 'p1_cross_task.json results.ner (B1: leverage transfer '
                        'fails, r=-1.0/-0.26)',
    },
    'per_model': results,
    'verdict': verdict,
}
OUTD.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8', newline='\n')
print(json.dumps(verdict, ensure_ascii=False, indent=1))
print('saved ->', OUTD)
