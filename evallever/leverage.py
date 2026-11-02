# -*- coding: utf-8 -*-
"""evallever.leverage —— 力臂(arm)计算 / leverage-arm computation.

两种模式 / two modes
---------------------
1) predictions 模式(通用简单格式):
   两个预测 JSONL + 金标 JSONL → 逐模型分数(P/R/F1,句内/跨句分组可选)、
   错误集 Jaccard 距离、排名翻转风险旗标(启发式筛查)。

2) counts 模式(项目 per_doc 计数格式):
   输入项目落盘的 per_doc 逐文档计数文件/目录(与 r1_rank_flip.py 相同格式),
   按金标契约两端(gD/gR)计算逐模型 Gap_F1 点估计、换金标力臂 |Δ|,
   并做与 r1 一致的符号翻转(sign flip)判定。
   —— 该模式用于复现 out/revision_v5_experiments/statistics/r1_rank_flip.json。

口径来源(已验证,本文件为自包含复制实现,不 import 项目代码):
  * 池化 F1 = 2*tp / (2*tp + fp + fn)(label-level,同 rev5_recount_lib.metrics 的 F1 列);
  * Gap_F1 = F1(intra 句内) − F1(inter 跨句),分组规则=earliest_mention(计数文件已含分组);
  * 逐模型 gap 点估计 = 跨 run 平均(r1 口径:重采样内跨 run 平均的点估计部分);
  * 翻转判定:D1 = A_gD − B_gD,D2 = A_gR − B_gR,sign(D1) ≠ sign(D2)(均非零)即翻转。

仅用标准库 / stdlib only;确定性 / deterministic;无网络 / no network。
"""
from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------- 通用小件

FP_LABEL_METRIC = "f1"          # 标签级 label-level(r1/r2 主口径)
PAIR_LABEL_METRIC = "pair_f1"   # 配对级 pair-level(eval_spec.pair_level 同义)


def _div(x: float, y: float) -> float:
    """安全除法 / safe division(分母 0 → 0.0,与项目 metrics 的空组口径一致)。"""
    return x / y if y else 0.0


def load_jsonl(path) -> list:
    """读 JSONL(容忍 BOM / 跳过空行)。"""
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def _pid(x) -> str:
    """实体标识规范化:整数 → 十进制串;其余 → 原样字符串。

    预测与金标只要彼此一致(都用索引或都用实体名)即可匹配。
    """
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    return x if isinstance(x, str) else str(x)


# ---------------------------------------------------------------- predictions 模式

def build_gold_map(gold_records) -> dict:
    """金标 JSONL → {(did, h, t): {"group": 0/1/None, "labels": set}}。

    金标格式(每行一个候选):
      {"did": "...", "h": <索引或实体名>, "t": ..., "group": 0|1, "labels": ["rel", ...]}
    group 可省略(此时无法算 gap);labels 可为空(负候选)。
    推荐金标列出**全部候选**(含空标签),否则未列出的预测对将并入整体 FP(见 README)。
    """
    gmap = {}
    for r in gold_records:
        did = _pid(r["did"])
        key = (did, _pid(r["h"]), _pid(r["t"]))
        grp = r.get("group", None)
        if grp is not None:
            grp = int(grp)
            if grp not in (0, 1):
                raise ValueError(f"gold group 必须为 0/1,得到 {grp!r}(did={did})")
        labels = set(str(x) for x in (r.get("labels") or []))
        if key in gmap:
            raise ValueError(f"gold 重复候选:{key}")
        gmap[key] = {"group": grp, "labels": labels}
    return gmap


def build_pred_map(pred_records) -> dict:
    """预测 JSONL → {(did, h, t): set(rels)}(同对多行/多条目取并集)。

    预测格式(每行一个文档,labels 为三元组列表):
      {"did": "...", "labels": [["h", "rel", "t"], ...]}
    三元组亦兼容字典形式 {"h": ..., "r": ..., "t": ...}。
    """
    pmap = {}
    n_docs = 0
    for r in pred_records:
        n_docs += 1
        did = _pid(r["did"])
        for tri in (r.get("labels") or []):
            if isinstance(tri, dict):
                h, rel, t = tri["h"], tri["r"], tri["t"]
            else:
                if len(tri) != 3:
                    raise ValueError(f"pred 三元组长度必须为 3(did={did}):{tri!r}")
                h, rel, t = tri[0], tri[1], tri[2]
            pmap.setdefault((did, _pid(h), _pid(t)), set()).add(str(rel))
    return pmap, n_docs


def _counts_for_metric(pmap, gmap, metric):
    """按候选累加计数,返回 (overall, per_group, error_items, n_pred_outside)。

    * metric=f1      : tp/fp/fn 为**标签级**(P 分母=预测标签数,R 分母=金标标签数);
    * metric=pair_f1 : tp=命中金标候选数,fp=无金标但有预测的候选数,fn=有金标未命中数;
    * error_items    : 错误集 = {(did,h,t,rel): 预测与金标在该关系上不一致(双向)};
    * 未列入金标的候选上的预测:并入 overall 的 fp(pair_f1 同理),单独计数返回
      (金标应列出全部候选;此处保证 P 不被悄悄抬高)。
    """
    overall = {"tp": 0, "fp": 0, "fn": 0}
    per_group = {0: {"tp": 0, "fp": 0, "fn": 0}, 1: {"tp": 0, "fp": 0, "fn": 0}}
    error_items = set()
    n_pred_outside = 0

    if metric == FP_LABEL_METRIC:
        # 金标侧:逐候选、逐标签
        for key, g in gmap.items():
            pred = pmap.get(key, set())
            gold = g["labels"]
            tp = len(pred & gold)
            fp = len(pred - gold)
            fn = len(gold - pred)
            overall["tp"] += tp; overall["fp"] += fp; overall["fn"] += fn
            if g["group"] in per_group:
                cg = per_group[g["group"]]
                cg["tp"] += tp; cg["fp"] += fp; cg["fn"] += fn
            for rel in pred - gold:
                error_items.add(key + (rel,))
            for rel in gold - pred:
                error_items.add(key + (rel,))
        # 预测侧:未列入金标的候选(整体 fp,无分组)
        for key, pred in pmap.items():
            if key not in gmap:
                n_pred_outside += len(pred)
                overall["fp"] += len(pred)
                for rel in pred:
                    error_items.add(key + (rel,))
    elif metric == PAIR_LABEL_METRIC:
        for key, g in gmap.items():
            pred = pmap.get(key, set())
            has_gold = bool(g["labels"])
            hit = has_gold and bool(pred & g["labels"])
            if has_gold:
                overall["tp"] += 1 if hit else 0
                overall["fn"] += 0 if hit else 1
                if g["group"] in per_group:
                    cg = per_group[g["group"]]
                    if hit:
                        cg["tp"] += 1
                    else:
                        cg["fn"] += 1
            elif pred:
                overall["fp"] += 1
                if g["group"] in per_group:
                    per_group[g["group"]]["fp"] += 1
            # 错误集(配对口径仍以关系标签为粒度登记)
            for rel in pred - g["labels"]:
                error_items.add(key + (rel,))
            for rel in g["labels"] - pred:
                error_items.add(key + (rel,))
        for key, pred in pmap.items():
            if key not in gmap and pred:
                n_pred_outside += len(pred)
                overall["fp"] += 1
    else:
        raise ValueError(f"未定义 metric:{metric}(可选 f1 / pair_f1)")
    return overall, per_group, error_items, n_pred_outside


def _prf(tp, fp, fn, metric):
    if metric == FP_LABEL_METRIC:
        p = _div(tp, tp + fp)
        r = _div(tp, tp + fn)
    else:  # pair_f1:P = tp/(tp+fp),R = tp/(tp+fn)(fp 含无金标误报候选)
        p = _div(tp, tp + fp)
        r = _div(tp, tp + fn)
    return {"P": p, "R": r, "F1": _div(2 * tp, 2 * tp + fp + fn)}


def _jaccard_distance(set_a: set, set_b: set) -> float:
    """Jaccard 距离 = 1 − |A∩B|/|A∪B|(两集皆空 → 0.0,视为完全一致)。"""
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return 1.0 - len(set_a & set_b) / union


def arm_predictions(pred_a_path, pred_b_path, gold_path, metric=FP_LABEL_METRIC,
                    tol=0.01, jd_min=0.5) -> dict:
    """predictions 模式主入口:双模型 + 金标 → 力臂报告(见 README 输出示例)。

    flip_risk 为**启发式筛查旗标**(非 r1 的 bootstrap 翻转检验):
      排名量 q(gap 或整体分数)相差 ≤ tol 且错误集 Jaccard 距离 ≥ jd_min
      → 两模型"读数接近但在不同样本上犯错",换合法契约后排序易翻转,标记风险。
    """
    pred_a, n_docs_a = build_pred_map(load_jsonl(pred_a_path))
    pred_b, n_docs_b = build_pred_map(load_jsonl(pred_b_path))
    gmap = build_gold_map(load_jsonl(gold_path))

    report_models = {}
    error_sets = {}
    for name, pmap, n_docs in (("model_a", pred_a, n_docs_a), ("model_b", pred_b, n_docs_b)):
        overall, per_group, errs, n_out = _counts_for_metric(pmap, gmap, metric)
        has_group = all(g["group"] is not None for g in gmap.values())
        m = {"n_docs": n_docs, "n_pred_pairs": len(pmap), "n_errors": len(errs),
             "pred_pairs_not_in_gold": n_out,
             "overall": _prf(overall["tp"], overall["fp"], overall["fn"], metric)}
        if has_group:
            m["intra"] = _prf(per_group[0]["tp"], per_group[0]["fp"], per_group[0]["fn"], metric)
            m["inter"] = _prf(per_group[1]["tp"], per_group[1]["fp"], per_group[1]["fn"], metric)
            m["gap_f1"] = m["intra"]["F1"] - m["inter"]["F1"]
        else:
            m["intra"] = m["inter"] = None
            m["gap_f1"] = None
        report_models[name] = m
        error_sets[name] = errs

    jd = _jaccard_distance(error_sets["model_a"], error_sets["model_b"])
    rank_qty = "gap_f1" if report_models["model_a"]["gap_f1"] is not None else "overall_F1"
    qa = report_models["model_a"]["gap_f1"] if rank_qty == "gap_f1" \
        else report_models["model_a"]["overall"]["F1"]
    qb = report_models["model_b"]["gap_f1"] if rank_qty == "gap_f1" \
        else report_models["model_b"]["overall"]["F1"]
    delta = qa - qb
    flag = bool(abs(delta) <= tol and jd >= jd_min)

    return {
        "mode": "predictions",
        "metric": metric,
        "gold_n_candidates": len(gmap),
        "model_a": report_models["model_a"],
        "model_b": report_models["model_b"],
        "leverage_arm": {
            "error_set_jaccard_distance": jd,
            "error_set_intersection": len(error_sets["model_a"] & error_sets["model_b"]),
            "error_set_union": len(error_sets["model_a"] | error_sets["model_b"]),
            "error_set_definition": "{(did,h,t,rel): 预测与金标不一致(缺标或多标)}",
        },
        "rank_quantity": rank_qty,
        "rank_flip_risk": {
            "flag": flag,
            "rule": f"|Δ{rank_qty}| ≤ tol 且 Jaccard 距离 ≥ jd_min(启发式筛查,非推断检验)",
            "delta_rank_quantity": delta,
            "tol": tol, "jd_min": jd_min,
        },
    }


# ---------------------------------------------------------------- counts 模式(项目 per_doc 计数)

def _iter_count_files(source: str) -> list:
    """展开一个计数源:文件 → [文件];目录 → 目录下全部 evaluation__*.json(排序)。"""
    p = Path(source)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(p.glob("evaluation__*.json"))
        if not files:
            raise FileNotFoundError(f"目录 {p} 下没有 evaluation__*.json 计数文件")
        return files
    raise FileNotFoundError(f"计数源不存在:{p}")


def _load_count_keys(path: Path) -> dict:
    """读一个 per_doc 计数文件 → {配置键: {"docs","intra","inter"}}。

    兼容两种落位:X2/P3M 顶层即配置键;ATLOP 为 {run_id, ts, per_doc:{...}} 需剥一层。
    (与 r1_rank_flip.load_counts 的剥层逻辑一致。)
    """
    d = json.loads(path.read_text(encoding="utf-8-sig"))
    if "per_doc" in d and isinstance(d.get("per_doc"), dict):
        d = d["per_doc"]
    return d


def pooled_gap_f1(entry: dict) -> float:
    """单个 (配置键, run) 的池化 Gap_F1 点估计 = F1(intra) − F1(inter)。

    entry = {"docs": [...], "intra": {cnt: [逐文档]}, "inter": {...}};
    池化 = 对文档求和(r1 口径:metrics(a.sum(0)) 的 F1 列 = 2tp/(2tp+fp+fn))。
    文档求和与文档顺序无关,故无需规范序。
    """
    def pooled_f1(side):
        tp = float(sum(side["tp"]))
        fp = float(sum(side["fp"]))
        fn = float(sum(side["fn"]))
        return _div(2.0 * tp, 2.0 * tp + fp + fn)
    return pooled_f1(entry["intra"]) - pooled_f1(entry["inter"])


def _mean(xs) -> float:
    return float(sum(xs) / len(xs))


def arm_counts(sources_a, sources_b, key_template, golds=("gD", "gR")) -> dict:
    """counts 模式主入口:复现 r1 的换金标力臂与翻转载荷。

    * sources_a / sources_b:per_doc 计数文件或目录的列表(目录展开 evaluation__*.json);
    * key_template:配置键模板;字符串=两模型同键;二元组/双元素列表=(A 键模板, B 键模板)
      (如 ATLOP "evaluation|{gold}|atlop_native|earliest_mention" 与
      P3M "evaluation|{gold}|native|p3m_native|earliest_mention" 异键情形);
    * 对每个 (源文件, gold):若该文件含对应键 → 记一个 run 的池化 Gap_F1;
      缺键则跳过(P3M 的 gD/gR 分文件落盘即为此情形);同一键出现在同源多个文件 → 报错。
    * 逐模型 gap 点 = 跨 run 平均;Δ = gap(golds[0]) − gap(golds[1]);
      D1 = A_g0 − B_g0,D2 = A_g1 − B_g1;sign(D1) ≠ sign(D2)(均非零)→ flip=True。
    """
    if len(golds) != 2:
        raise ValueError("golds 必须恰为契约两端,如 ('gD','gR')")
    templates = (key_template, key_template) if isinstance(key_template, str) \
        else tuple(key_template)
    if len(templates) != 2:
        raise ValueError("key_template 为序列时必须恰为 (A 模板, B 模板)")

    def side_report(sources, label, key_template):
        runs = {g: [] for g in golds}   # gold → [(源名, gap)]
        for src in sources:
            for f in _iter_count_files(src):
                keys = _load_count_keys(f)
                for g in golds:
                    key = key_template.replace("{gold}", g)
                    if key not in keys:
                        continue
                    runs[g].append((str(f), pooled_gap_f1(keys[key])))
        out = {"label": label, "delta_point": None, "leverage_abs_delta": None}
        for g in golds:
            if not runs[g]:
                raise ValueError(f"[{label}] 无任何源提供 gold={g} 的键 {key_template}")
            out[f"gap_{g}_point"] = _mean([x[1] for x in runs[g]])
            out[f"n_runs_{g}"] = len(runs[g])
            out[f"per_run_gap_{g}"] = {s: v for s, v in runs[g]}
        out["delta_point"] = out[f"gap_{golds[0]}_point"] - out[f"gap_{golds[1]}_point"]
        out["leverage_abs_delta"] = abs(out["delta_point"])
        return out

    a = side_report(list(sources_a), "A", templates[0])
    b = side_report(list(sources_b), "B", templates[1])
    d1 = a[f"gap_{golds[0]}_point"] - b[f"gap_{golds[0]}_point"]
    d2 = a[f"gap_{golds[1]}_point"] - b[f"gap_{golds[1]}_point"]
    s1, s2 = (d1 > 0) - (d1 < 0), (d2 > 0) - (d2 < 0)
    flip = bool(s1 != s2 and s1 != 0 and s2 != 0)
    return {
        "mode": "perdoc_counts",
        "counts_key_template": key_template,
        "golds": list(golds),
        "model_a": a,
        "model_b": b,
        "pair_flip": {
            "D1_point": d1, "D2_point": d2,
            "flip_delta_point": d1 - d2,
            "flip": flip,
            "rule": "sign(D1) ≠ sign(D2)(均非零)即排名翻转;与 r1_rank_flip 同判据",
        },
    }
