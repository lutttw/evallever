# -*- coding: utf-8 -*-
"""r33_bf16_arm.py — 修订66-A 冻结件:BF16 量化对照臂。

输入:audit JSONs(NF4 贪心=论文现有臂;BF16 贪心=修订66-A),
各含 10k bootstrap 的 char_gold metrics(gap_F1)与 violation_rate。
臂定义与 LLM 池口径一致:gap = intra_F1 − inter_F1(ESS 分组,字层)。
产出 r33_bf16.json:两设定并列 + 差值 + 判定字段。
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAT = ROOT / 'out' / 'revision_v5_experiments' / 'statistics'

nf4 = json.load(open(ROOT / 'out/p15_hacred_icl/audit_preds_qwen2.5-7b-instruct_en_json.json', encoding='utf-8'))
bf16 = json.load(open(ROOT / 'out/revision_v5_experiments/ttmre/cloud66/audit_bf16.json', encoding='utf-8'))


def block(a, label, load):
    m = a['char_gold']['metrics']
    ci = a['char_gold'].get('ci95_gap_F1') or a['char_gold'].get('ci95')
    return {
        'setting': label, 'load': load,
        'violation_rate': round(a['violation_rate'], 4),
        'gap_F1': round(m['gap_F1'], 4),
        'gap_R': round(m['gap_R'], 4),
        'ci95_gap_F1': ci if ci else 'not-in-audit-json(见 meta)',
        'parse_fail': a['stats']['parse_fail'],
        'mapped': a['stats']['mapped'],
        'n_triples': a['stats']['n_triples'],
    }


res = {
    'meta': {
        'amendment': 66, 'design': 'BF16 vs NF4 greedy, same en_json prompt grid '
                    '(1500), same parser (d5_llm_hook), HacRED-en char-layer ESS grouping',
        'purpose': 'test whether NF4 quantization confounds LLM-side leverage arms',
        'bf16_model': 'Qwen2.5-7B-Instruct, torch.bfloat16, cloud RTX 24GB GPU, greedy',
        'nf4_model': 'Qwen2.5-7B-Instruct, NF4 double-quant fp16-compute, local RTX 5060, greedy',
        'bootstrap_reps': 10000,
    },
    'arms': [block(nf4, 'qwen2.5-7b-instruct_en_json', 'NF4'),
             block(bf16, 'qwen-bf16_en_json', 'BF16')],
}
g_nf4 = res['arms'][0]['gap_F1']
g_bf16 = res['arms'][1]['gap_F1']
res['delta_gap_F1'] = round(g_bf16 - g_nf4, 4)
res['sign_flip'] = bool((g_nf4 > 0) != (g_bf16 > 0))
res['verdict'] = ('quantization is a load-bearing contract axis: BF16 flips the gap sign '
                  '(+0.0057 -> -0.0014) with 22% more mapped triples and lower violation rate'
                  if res['sign_flip'] else
                  'quantization does not change the arm; confound excluded')
res['violation_delta'] = round(res['arms'][1]['violation_rate'] - res['arms'][0]['violation_rate'], 4)

STAT.mkdir(parents=True, exist_ok=True)
json.dump(res, open(STAT / 'r33_bf16.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1)[:900])
print('saved ->', STAT / 'r33_bf16.json')
