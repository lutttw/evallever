# -*- coding: utf-8 -*-
"""evallever.audit_report —— LLM 输出违规审计 / violation scan for LLM outputs.

对 d5_llm_hook.py 已验证审计口径的**自包含薄封装**(逐字复制解析与计数逻辑,
不 import 项目代码):解析 LLM 原始输出 → 六类违规计数 → 违规率 + unverifiable 旗标。

违规分类(修订 23b taxonomy,与 d5_llm_hook.cmd_audit 逐一对应):
  * parse_fail      : 无法解析出合法三元组结构(json 无有效 {"triples": [...]} /
                      freetext 无 "A -> r -> B" 且非 "(none)");
  * oob_index       : 实体索引越界/格式非法(E<i> 超出实体表,或非实体 ID);
  * rel_unmapped    : 关系名不在关系表(relation_to_index,排除 NA);
  * value_hallucination(=hallucinated): 头/尾实体名不在该文档公共实体名表;
  * ambiguous       : 实体名对应多个 canon 索引(多名同实体,无法回连);
  * self_pair       : 头尾回连到同一 canon 索引;
  * mapped          : 合法回连到公共键集 (did, hc, tc) 的三元组。
  违规率 = (n_triples − mapped) / n_triples;> 5% → unverifiable(该设置不可核验)。

仅用标准库 / stdlib only;确定性 / deterministic;无网络 / no network。
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------- 文档指纹与规范键集
# 以下三个函数逐字取自 src/jcip_audit/d5_llm_hook.py(已验证),仅将路径参数化。


def canon(o) -> str:
    return json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def doc_id_of(d) -> str:
    return hashlib.sha256(canon({"title": str(d.get("title", "")).strip(),
                                 "sents": d["sents"]}).encode("utf-8")).hexdigest()


def sig_of(v) -> tuple:
    return tuple(sorted({(m["sent_id"], m["pos"][0], m["pos"][1]) for m in v}))


def build_canon_map(dev_docs, redocred_dev, re2_dev):
    """三来源公共键集(与 x3_bridge_pred 逐字一致):{doc_id: (raw2canon, first_sent, sigs)}。

    需要三个对齐后的 dev 文件(docred / redocred / re2,各 498 篇,句子序列一致)。
    """
    D = json.loads(Path(dev_docs).read_text(encoding="utf-8"))
    R = json.loads(Path(redocred_dev).read_text(encoding="utf-8"))
    R2 = json.loads(Path(re2_dev).read_text(encoding="utf-8"))
    assert len(D) == len(R) == len(R2) == 498
    per_doc, title_map = {}, {}
    for dd, dr, d2 in zip(D, R, R2):
        assert dd["sents"] == dr["sents"] == d2["sents"]
        did = doc_id_of(dd)
        title_map[str(dd.get("title", "")).strip()] = did
        ms = []
        for d in (dd, dr, d2):
            m = defaultdict(list)
            for i, v in enumerate(d["vertexSet"]):
                m[sig_of(v)].append(i)
            ms.append(m)
        common = sorted(s for s in set(ms[0]) & set(ms[1]) & set(ms[2])
                        if all(len(m[s]) == 1 for m in ms))
        raw2canon = {ms[0][s][0]: k for k, s in enumerate(common)}
        first = [min(x[0] for x in s) for s in common]
        per_doc[did] = {"raw2canon": raw2canon, "first_sent": first, "sigs": common,
                        "n_common": len(common)}
    return D, per_doc, title_map


# ---------------------------------------------------------------- 输出解析(逐字复制)

TRI_JSON = re.compile(r'\[\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*\]')
TRI_FREE = re.compile(r"([^;|\n]+?)\s+->\s+([^;|\n]+?)\s+->\s+([^;|\n]+)")


EID = re.compile(r"E\d+")


def _resolve(x, ents, k_bound=None):
    """实体 ID(如 E3)→ 实体名;越界/格式错误返回 None(计违规)。

    k_bound(约束解码版记录的索引上界)优先于 ents 长度——多名同实体时
    ents 长度 < 索引上限,按 ents 判会把合法高位索引误计为违规。
    """
    if isinstance(x, str) and EID.fullmatch(x.strip()):
        i = int(x.strip()[1:])
        if k_bound is not None:
            if 0 <= i < k_bound:
                return ents[i] if i < len(ents) else None
            return None
        if 0 <= i < len(ents):
            return ents[i]
    return None


def parse_output(raw: str, fmt: str, ents: list, k_bound=None) -> list:
    """返回 [(h_name, rel, t_name), ...];解析失败/越界索引以哨兵元组返回。"""
    if fmt == "json":
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and isinstance(obj.get("triples"), list):
                    out = []
                    for x in obj["triples"]:
                        if (isinstance(x, (list, tuple)) and len(x) == 3
                                and all(isinstance(s, str) for s in x)):
                            h, t = _resolve(x[0], ents, k_bound), _resolve(x[2], ents, k_bound)
                            if h is None or t is None:
                                out.append(("<<VIOLATION>>",) * 3)
                            else:
                                out.append((h, x[1].strip(), t))
                    return out
            except json.JSONDecodeError:
                pass
        return [("<<PARSE_FAIL>>",) * 3] if raw.strip() else []
    out = []
    for h, r, t in TRI_FREE.findall(raw):
        h, r, t = h.strip(), r.strip(), t.strip()
        if "Relations:" in h:
            h = h.split("Relations:")[-1].strip()
        hr, tr = _resolve(h, ents, k_bound), _resolve(t, ents, k_bound)
        if hr is None or tr is None:
            out.append(("<<VIOLATION>>",) * 3)
        else:
            out.append((hr, r, tr))
    if not out and "(none)" in raw:
        return []
    return out or ([("<<PARSE_FAIL>>",) * 3] if raw.strip() else [])


# ---------------------------------------------------------------- 违规扫描

TAXONOMY = {
    "parse_fail": "无法解析出合法三元组结构",
    "oob_index": "实体索引越界或非实体 ID",
    "rel_unmapped": "关系名不在关系表(排除 NA)",
    "hallucinated": "头/尾实体名不在该文档公共实体名表(value_hallucination)",
    "ambiguous": "实体名对应多个 canon 索引(多名同实体)",
    "self_pair": "头尾回连同一 canon 索引",
    "mapped": "合法回连到公共键集 (did, hc, tc)",
}


def scan(pred_path, dev_docs, redocred_dev, re2_dev, rel2idx_path,
         schema_format=None) -> dict:
    """审计主入口:逐记录解析 + 六类违规计数。

    pred_path : d5 预测 JSONL,每行 {"did", "raw", "fmt"?, "ents"?, "k_index"?}
    schema_format : "json" | "freetext" | None;给定则覆盖全部记录的 fmt,
                    None(默认)按逐记录 "fmt" 字段(d5 原口径)。
    dev_docs / redocred_dev / re2_dev : 三源对齐 dev 数据(构建公共键集与实体名表);
    rel2idx_path : relation_to_index.json(排除 NA 后为合法关系名表)。
    """
    D, per_doc, _ = build_canon_map(dev_docs, redocred_dev, re2_dev)
    rel2idx = {k: v for k, v in json.loads(Path(rel2idx_path).read_text(encoding="utf-8")).items()
               if k.upper() != "NA"}
    # 实体名 → canon(精确匹配;多名歧义计违规;与 d5 同源同法)
    name2canon = {}
    for dd in D:
        did = doc_id_of(dd)
        m = defaultdict(set)
        for raw_i, c in per_doc[did]["raw2canon"].items():
            m[str(dd["vertexSet"][raw_i][0]["name"]).strip()].add(c)
        name2canon[did] = m

    stats = {"n_triples": 0, "parse_fail": 0, "oob_index": 0, "hallucinated": 0,
             "ambiguous": 0, "self_pair": 0, "rel_unmapped": 0, "mapped": 0, "docs_missing": 0}
    n_records = 0
    for line in Path(pred_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n_records += 1
        r = json.loads(line)
        did = r["did"]
        if did not in per_doc:
            stats["docs_missing"] += 1
            continue
        m = name2canon[did]
        ents = r.get("ents") or sorted(m)
        k_bound = r.get("k_index")
        fmt = schema_format if schema_format is not None else r["fmt"]
        for h, rl, t in parse_output(r["raw"], fmt, ents, k_bound=k_bound):
            stats["n_triples"] += 1
            if h == "<<PARSE_FAIL>>":
                stats["parse_fail"] += 1
                continue
            if h == "<<VIOLATION>>":
                stats["oob_index"] += 1
                continue
            ri = rel2idx.get(rl)
            if ri is None:
                stats["rel_unmapped"] += 1
                continue
            hc_l, tc_l = m.get(h), m.get(t)
            if hc_l is None or tc_l is None:
                stats["hallucinated"] += 1
                continue
            if len(hc_l) > 1 or len(tc_l) > 1:
                stats["ambiguous"] += 1
                continue
            hc, tc = next(iter(hc_l)), next(iter(tc_l))
            if hc == tc:
                stats["self_pair"] += 1
                continue
            stats["mapped"] += 1

    viol = stats["n_triples"] - stats["mapped"]
    viol_rate = viol / stats["n_triples"] if stats["n_triples"] else 0.0
    return {
        "pred": str(pred_path),
        "schema_format": schema_format if schema_format is not None else "per-record fmt 字段",
        "n_records_scanned": stats["n_triples"] and None or None,  # 占位,下面覆盖
        "stats": stats,
        "n_violations": viol,
        "violation_rate": viol_rate,
        "unverifiable_threshold": 0.05,
        "unverifiable": bool(viol_rate > 0.05),
        "taxonomy": TAXONOMY,
    }
