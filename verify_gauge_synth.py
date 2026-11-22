# -*- coding: utf-8 -*-
# 规范不变性合成检验(rev162, T8):验证 §3.1 两个断言
#   (a) 对称聚合计分(accuracy/micro-F1/macro-F1)在纯标签改名(双射)下严格不变;
#   (b) 名称锚定权重计分在权重表未随规范迁移时塌缩(不穿过联合划分)。
# 随机种子固定;结果写入 stdout,供发布包留档。
import random

random.seed(20261022)
L = [f"L{k}" for k in range(6)]
gold = [random.choice(L) for _ in range(300)]
pred = [random.choice(L) for _ in range(300)]
r = {l: f"M{k}" for k, l in enumerate(L)}
gold2 = [r[g] for g in gold]
pred2 = [r[p] for p in pred]

def acc(p, g):
    return sum(a == b for a, b in zip(p, g)) / len(g)

def f1s(p, g, labels):
    out = []
    for c in labels:
        tp = sum(a == c and b == c for a, b in zip(p, g))
        fp = sum(a == c and b != c for a, b in zip(p, g))
        fn = sum(a != c and b == c for a, b in zip(p, g))
        out.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return out

micro = acc(pred, gold)
macro = sum(f1s(pred, gold, L)) / len(L)
micro2 = acc(pred2, gold2)
macro2 = sum(f1s(pred2, gold2, list(r.values()))) / len(r)

print(f"accuracy  old={micro:.12f} new={micro2:.12f} diff={abs(micro-micro2):.2e}")
print(f"macro-F1  old={macro:.12f} new={macro2:.12f} diff={abs(macro-macro2):.2e}")
assert abs(micro - micro2) < 1e-12 and abs(macro - macro2) < 1e-12
print("=> (a) 对称聚合计分在改名下严格不变 ✓")

w = {l: random.random() for l in L}
wa_old = sum(w[b] for a, b in zip(pred, gold) if a == b) / 300
hits_new = sum(1 for a, b in zip(pred2, gold2) if a == b and b in w)
print(f"name-anchored weights: old={wa_old:.6f}, new-space hits={hits_new}/300")
print("=> (b) 名称锚定计分不穿过划分:权重表不迁移时在新规范下无定义/塌缩 ✓")
