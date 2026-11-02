# -*- coding: utf-8 -*-
"""r41_violation_regression.py — 修订76(D 类):R4 违规预测因子(修订11 冻结设计的执行)。

冻结因子表(修订11,一次冻死):文档 token 数(对数)、实体表大小、目标关系训练集频率
  (对数)、输出格式(两水平)、模型(三水平:qwen-7B/mistral-7B/qwen-14B)。
对象:D5 五设置(Qwen-7B/Mistral-7B × json/freetext + Qwen-14B json)全部三元级记录。
模型:逻辑回归主表(sklearn,无正则);推断=按文档聚类的 bootstrap CI(500 次,
  与全文 doc-bootstrap 惯例一致;修订11 的"混合效应附录"以此替代,差异如实披露);
  不做交互项筛选。报告:系数/OR/AME。
关系频率源:data/docred_official_hf/train_annotated.json.gz(官方人工训练集)。
输出:statistics/r41_violation_regression.json
"""
import gzip
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "jcip_audit"))
import d5_llm_hook as D5                                   # noqa: E402

STAT = ROOT / "out/revision_v5_experiments/statistics"
BOOT, SEED = 500, 20260926
ARMS = [  # (模型水平, 格式, preds 文件)
    ("qwen7b", "json", "preds_qwen2.5-7b-instruct_json.jsonl"),
    ("qwen7b", "freetext", "preds_qwen2.5-7b-instruct_freetext.jsonl"),
    ("mistral7b", "json", "preds_mistral-7b-instruct_json.jsonl"),
    ("mistral7b", "freetext", "preds_mistral-7b-instruct_freetext.jsonl"),
    ("qwen14b", "json", "preds_qwen2.5-14b-instruct_json.jsonl"),
]
FEATURES = ["log_doc_tokens", "n_ents", "log_rel_freq", "fmt_json",
            "model_mistral", "model_qwen14b"]


def jl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from sklearn.linear_model import LogisticRegression

    # 训练集关系频率(官方人工标注 train)
    rel_freq = Counter()
    with gzip.open(ROOT / "data/docred_official_hf/train_annotated.json.gz", "rt",
                   encoding="utf-8") as fh:
        for dd in json.load(fh):
            for lab in dd["labels"]:
                rel_freq[lab["r"]] += 1
    print(f"train relations counted: {sum(rel_freq.values())} labels, "
          f"{len(rel_freq)} types")

    # 实体映射(与 r24 同构)
    _, per_doc, _ = D5.build_canon_map()
    rel2idx = {k: v for k, v in json.loads(D5.REL_IDX.read_text(encoding="utf-8")).items()
               if k.upper() != "NA"}
    dev = json.loads(D5.DEV.read_text(encoding="utf-8"))
    meta = {}
    for dd in dev:
        did = D5.doc_id_of(dd)
        meta[did] = {"n_ents": len(dd["vertexSet"]),
                     "doc_tokens": sum(len(s) for s in dd["sents"]),
                     "names": [str(v[0]["name"]).strip() for v in dd["vertexSet"]]}

    rows = []          # 每三元组一条
    arm_stats = {}
    # 预构建 did → {实体名: {候选索引集}}(与 r24 同构,一次构建)
    name2canon_all = {}
    for dd in dev:
        did = D5.doc_id_of(dd)
        m = defaultdict(set)
        pd_map = per_doc[did]["raw2canon"]
        for raw_i, v in enumerate(dd["vertexSet"]):
            name = str(v[0]["name"]).strip()
            if raw_i in pd_map:
                m[name].add(pd_map[raw_i])
        name2canon_all[did] = m
    for model, fmt, fname in ARMS:
        n_tr = n_v = 0
        for r in jl(D5.OUT / fname):
            did = r["did"]
            if did not in meta:
                continue
            m = meta[did]
            n2c = name2canon_all[did]
            doc_tok_log = float(np.log1p(m["doc_tokens"]))
            n_ents = m["n_ents"]
            for h, rl, t in D5.parse_output(r["raw"], r["fmt"], r["ents"],
                                            k_bound=r.get("k_index")):
                n_tr += 1
                if h in ("<<PARSE_FAIL>>", "<<VIOLATION>>"):
                    viol = 1
                elif rel2idx.get(rl) is None:
                    viol = 1
                else:
                    hl, tl = n2c.get(h), n2c.get(t)
                    if hl is None or tl is None or len(hl) > 1 or len(tl) > 1:
                        viol = 1
                    elif next(iter(hl)) == next(iter(tl)):
                        viol = 1
                    else:
                        viol = 0
                freq = rel_freq.get(rl, 0)
                rows.append({
                    "did": did, "y": viol, "model": model,
                    "log_doc_tokens": doc_tok_log, "n_ents": n_ents,
                    "log_rel_freq": float(np.log1p(freq)),
                    "fmt_json": 1.0 if fmt == "json" else 0.0,
                    "model_mistral": 1.0 if model == "mistral7b" else 0.0,
                    "model_qwen14b": 1.0 if model == "qwen14b" else 0.0})
                n_v += viol
        arm_stats[f"{model}_{fmt}"] = {"n_triples": n_tr, "n_violations": n_v,
                                       "rate": round(n_v / max(1, n_tr), 4)}
        print(f"[{model}_{fmt}] triples={n_tr} viol={n_v} rate={n_v/max(1,n_tr):.4f}")

    # ---- 主拟合(无正则 LogisticRegression) ----
    X = np.array([[row[f] for f in FEATURES] for row in rows])
    y = np.array([row["y"] for row in rows], dtype=int)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    lr.fit(X, y)
    beta = lr.coef_[0]
    p = lr.predict_proba(X)[:, 1]
    ame = []
    for j, f in enumerate(FEATURES):
        if f in ("fmt_json", "model_mistral", "model_qwen14b"):
            X1 = X.copy(); X1[:, j] = 1.0
            X0 = X.copy(); X0[:, j] = 0.0
            ame.append(float((lr.predict_proba(X1)[:, 1]
                              - lr.predict_proba(X0)[:, 1]).mean()))
        else:
            ame.append(float((beta[j] * p * (1 - p)).mean()))

    # ---- doc-cluster bootstrap CI ----
    dids = np.array([row["did"] for row in rows])
    uniq = np.unique(dids)
    doc_idx = {d: np.where(dids == d)[0] for d in uniq}
    rng = np.random.default_rng(SEED)
    boots = np.empty((BOOT, len(FEATURES)))
    for b in range(BOOT):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([doc_idx[d] for d in pick])
        bl = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
        bl.fit(X[sel], y[sel])
        boots[b] = bl.coef_[0]
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b+1}/{BOOT}")

    res = {"meta": {"amendment": 76, "design_source": "prereg 修订11 (R4)",
                    "n_triples": len(rows), "n_docs": len(uniq),
                    "n_violations": int(y.sum()),
                    "inference": "doc-cluster bootstrap 500 reps (seed 20260926); "
                                 "substitutes the frozen mixed-effects appendix item, "
                                 "disclosed",
                    "ts": datetime.now().isoformat(timespec="seconds")},
           "arm_rates": arm_stats,
           "coef": {}}
    for j, f in enumerate(FEATURES):
        lo, hi = np.percentile(boots[:, j], [2.5, 97.5])
        res["coef"][f] = {"beta": round(float(beta[j]), 4),
                          "OR": round(float(np.exp(beta[j])), 4),
                          "ci95_OR": [round(float(np.exp(lo)), 4),
                                      round(float(np.exp(hi))), 4] and
                                      [round(float(np.exp(lo)), 4),
                                       round(float(np.exp(hi)), 4)],
                          "AME": round(ame[j], 4),
                          "ci_excludes_zero": bool(lo > 0 or hi < 0)}
        print(f"{f:16s} beta={beta[j]:+.4f} OR={np.exp(beta[j]):.3f} "
              f"CI[{np.exp(lo):.3f},{np.exp(hi):.3f}] AME={ame[j]:+.4f}")

    out = STAT / "r41_violation_regression.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
