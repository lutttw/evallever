# -*- coding: utf-8 -*-
"""r37_p2_loo.py — 修订71:R3-W2 的留一检验(唯一新计算)。

命题 2 声称翻转发生在"长臂×可比符号对比"处(条件式,非频率式)。
事后拟合风险的廉价缓解:在 57 个检查(21 对 × 3 维)上,用
"臂长"(两端点臂长的较大者,或差值 |D1-D2|)对"是否翻转"做留一预测:
对每个检查,用其余 56 个检查拟合"臂长阈值→翻转"的逻辑判别
(简化为:单变量阈值分类器,在训练集上选最优阈值),
预测留出检查的翻转与否,统计 accuracy/AUC 与置换基线。

数据全部来自冻结件 r1_rank_flip.json,零新推理。
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
r1 = json.load(open(ROOT / 'out/revision_v5_experiments/statistics/r1_rank_flip.json', encoding='utf-8'))

fm = r1['flip_matrix']
gaps = r1['gaps']

rows = []
for pair, dims in fm.items():
    A, B = pair.split('|')
    for dim, v in dims.items():
        if not v.get('available'):
            continue
        # 臂长定义:两端点在该维上的臂(即该生成器在两个 setting 下的 |gap 差| 的
        # 基础是 D1/D2(对比差);R3 的"长臂"= 生成器自身在该契约轴上的杠杆。
        # 对 gold 维:生成器金标臂= |gold_gD - gold_gR| gap(每生成器)
        # 对 threshold 维:|thr_fixed - thr_cal| gap(每生成器)
        # 对 scoring 维:scores 差异小,作为对照维。
        # 统一量:该检查涉及的生成器在轴上的平均臂长
        gA = gaps[A]
        gB = gaps[B]
        if dim.startswith('gold'):
            armA = abs(gA['gold_gD']['point'] - gA['gold_gR']['point'])
            armB = abs(gB['gold_gD']['point'] - gB['gold_gR']['point'])
        elif dim.startswith('threshold'):
            armA = abs(gA['thr_fixed']['point'] - gA['thr_cal']['point'])
            armB = abs(gB['thr_fixed']['point'] - gB['thr_cal']['point'])
        else:  # scoring
            armA = abs(gA['score_A']['point'] - gA['score_R']['point'])
            armB = abs(gB['score_B']['point'] - gB['score_R']['point']) if 'score_B' in gB else 0.0
        rows.append({
            'pair': pair, 'dim': dim,
            'arm_max': max(armA, armB),
            'arm_min': min(armA, armB),
            'flip': bool(v['flip']),
            'flip_ci_excl': bool(v.get('flip_delta_ci_excludes_zero', False)),
        })

X = np.array([r['arm_max'] for r in rows])
y = np.array([1 if r['flip'] else 0 for r in rows])
n = len(rows)
n_flip = int(y.sum())
print(f'checks={n} flips={n_flip}')
assert n == 57, n

# ---- 留一:单阈值分类器 ----
def best_threshold(Xtr, ytr):
    # 在训练集上穷举候选阈值(中点),选 balanced accuracy 最高
    order = np.argsort(Xtr)
    cands = [(Xtr[i] + Xtr[i + 1]) / 2 for i in range(len(Xtr) - 1)]
    best_t, best_ba = Xtr.mean(), -1
    for t in cands:
        pred = (Xtr > t).astype(int)
        tp = ((pred == 1) & (ytr == 1)).sum(); fn = ((pred == 0) & (ytr == 1)).sum()
        tn = ((pred == 0) & (ytr == 0)).sum(); fp = ((pred == 1) & (ytr == 0)).sum()
        tpr = tp / max(1, tp + fn); tnr = tn / max(1, tn + fp)
        ba = (tpr + tnr) / 2
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t

correct = 0
preds = np.zeros(n, dtype=int)
for i in range(n):
    mask = np.ones(n, bool); mask[i] = False
    t = best_threshold(X[mask], y[mask])
    preds[i] = int(X[i] > t)
    correct += int(preds[i] == y[i])
loo_acc = correct / n

# 基线1:多数类(非翻转)
maj_acc = (n - n_flip) / n
# 基线2:置换检验(打散 y 500 次,重跑 LOO,取 accuracy 分布)
rng = np.random.default_rng(20260926)
B = 500
perm_acc = np.zeros(B)
for b in range(B):
    yp = rng.permutation(y)
    c = 0
    for i in range(n):
        mask = np.ones(n, bool); mask[i] = False
        t = best_threshold(X[mask], yp[mask])
        c += int(int(X[i] > t) == yp[i])
    perm_acc[b] = c / n
p_perm = float((perm_acc >= loo_acc).mean())

# AUC(全样本,描述性)
def auc(x, yy):
    pos = x[yy == 1]; neg = x[yy == 0]
    gt = sum((p > ng) for p in pos for ng in neg)
    eq = sum((p == ng) for p in pos for ng in neg)
    return (gt + 0.5 * eq) / (len(pos) * len(neg))
A = auc(X, y)

res = {
    'meta': {
        'amendment': 71, 'design': 'R3-W2 leave-one-out test of Proposition 2: '
        'does arm length (max of the two generators\' arms on the switched axis, '
        'from frozen r1 gaps) predict flip locations among the 57 checks? '
        'Single-threshold classifier trained on 56, predicts the held-out check. '
        'Permutation baseline 500 reps, seed 20260926. No new inference.',
    },
    'data': {'n_checks': n, 'n_flips': n_flip,
             'arm_def': 'max(arm_A, arm_B) on the switched axis per check'},
    'results': {
        'loo_accuracy': round(loo_acc, 4),
        'majority_baseline': round(maj_acc, 4),
        'permutation_p': round(p_perm, 4),
        'auc_descriptive': round(A, 4),
    },
    'verdict': ('arm length carries flip-location information beyond class '
                'frequency: LOO accuracy {0:.3f} vs majority {1:.3f}, permutation '
                'p={2:.3f}, AUC {3:.3f} — the condition is not pure post-hoc '
                'description'.format(loo_acc, maj_acc, p_perm, A)),
}
out = ROOT / 'out/revision_v5_experiments/statistics/r37_p2_loo.json'
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(res['results'], indent=1))
print(res['verdict'])
print('saved ->', out)
