# EvalLever / 评测杠杆工具

NLP 评测的三个诊断命令:**力臂(arm)**、**种子处方(seed)**、**违规审计(audit)**。
配套论文:句内—跨句差距的评价杠杆理论与预注册范式(以结构化预测为验证场)。

## 安装(开发模式)

```bash
pip install -e .
# 或直接:python -m evallever.cli --help
```

## 三命令

### 1) `eval-lever arm` — 两模型力臂 + 排名翻转风险

输入两模型预测 JSONL 与金标 JSONL,输出:逐模型分数、错误集 Jaccard 距离(力臂)、
排名翻转风险旗标(启发式:分数差 ≤ tol 且力臂 ≥ jd_min)。

```bash
eval-lever arm --pred-a pred_a.jsonl --pred-b pred_b.jsonl --gold gold.jsonl
```

预测 JSONL 每行:`{"did": "...", "labels": [...]}`(标签集合,与金标同键空间)。
金标 JSONL 每行:`{"did": "...", "labels": [...]}`。

**counts 模式**(复现 r1 口径):输入项目 per_doc 计数文件/目录,
`--key-template` 含 `{gold}` 占位;两模型键约定不同时用
`--key-template-a/--key-template-b`:

```bash
eval-lever arm --counts-mode \
  --counts-a per_doc_atlop_D_s*.json \
  --counts-b per_doc_x3_p3m_s20260917 \
  --key-template-a "evaluation|{gold}|atlop_native|earliest_mention" \
  --key-template-b "evaluation|{gold}|native|p3m_native|earliest_mention"
```

### 2) `eval-lever seed` — 符号判定所需最小种子数处方

输入逐种子 gap 点估计(可选逐种子文档级 bootstrap σ 与跨种子相关 ρ),
输出每个 n 的 P(池化 95% CI 排除 0) 曲线与 n*(95% 判据)。

```bash
eval-lever seed --gaps gaps.json
# gaps.json: {"gaps": [-0.0291, 0.0281, ...], "sigmas": [0.0095, ...], "rho": 0.3, "grid": [2,3,5,8,12,15]}
```

口径:r2 轨道 B(子集枚举 ≤10000;SE² 含 ρ 跨种子相关项;Z=1.959964)。

### 3) `eval-lever audit` — LLM 输出违规扫描

对 d5 预测 JSONL 逐记录解析,按六类违规计数(parse_fail / oob_index /
rel_unmapped / self_pair / ambiguous / value_hallucination),输出违规率与
unverifiable 旗标(>5% 门槛,预注册纪律)。

```bash
eval-lever audit --pred preds.jsonl --dev-docs dev_revised.json \
    --redocred-dev redocred_dev.json --re2-dev re2_dev.json --rel2idx relation_to_index.json
```

## 自检(兼容性验证,±0.005)

```bash
python -m evallever.selftest
```

三项:T1 从 per_doc 计数**数值级重算**翻转量并与 r1 对账(差 <0.005);T2 种子处方
n*=8 对账 r2;T3 违规率 4.57% 对账 14B 审计。自检需项目数据在默认相对路径。

### 5) `eval-lever predict` — 命题 2 翻转条件的前置检验

对一条登记预测给出命题 2 条件判定:翻转 = 长臂 **且** 能力差距相近。
阈值语料相关,须使用者自备(论文用留一法拟合,LOO acc 0.860):

```bash
eval-lever predict --arm-file spec.json
# spec.json: {"arm_abs_delta": 0.048, "margin_point": 0.01, "tau_arm": 0.03, "margin_tol": 0.02}
```

## 来源与口径

实现自包含(不依赖项目内部代码);统计口径与来源论文一致并已在
`src/jcip_audit/{r1_rank_flip,r2_seed_convergence,d5_llm_hook}.py` 中独立验证。
"""
