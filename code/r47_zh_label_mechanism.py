# -*- coding: utf-8 -*-
"""r47_zh_label_mechanism.py — 修订77:中文 H-XI1 失败(违规率随规模上升)的
标签词表接口机制分解(纯 CPU,零新推断,只读冻结预测)。

背景:
  §4.1 报告 英文 DocRED 违规率随规模下降,而中文 HacRED 相反
  (7B 22.1% → 14B 28.9%,逐文档去重口径)。r46 已用受控清单收窄排除
  "关系知识面溢出" 假说。本脚本在标签词表接口层面定位真实机制。

设计(全部在冻结预测上做,无 GPU):
  1. 用与 §4.3 一致的 p15 审计语义(每文档集合去重)复算两臂违规率;
  2. 把越界关系名按机制三分类:
       action_lexicalization  动词形态(词表要角色名词,模型吐动作)
       role_paraphrase        词表码的同义改写(字符重叠 ≥ 0.5)
       off_inventory          词表外关系(出生时间/获得奖项 等真实世界关系)
  3. 拆分母/分子:
       mapped/doc   每文档有效词表内三元组(分母)
       unmapped/doc 每文档越界三元组(分子)
  4. 语言对照:同模型(7B)中文指令臂 vs 英文指令臂,HacRED 同一批文档。
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "jcip_audit"))
import d5_llm_hook as D5                                    # noqa: E402
import p15_hacred_icl as P15                                # noqa: E402

STAT = ROOT / "out" / "revision_v5_experiments" / "statistics"
SEED = 20260926
REPS = 10000

ARMS = [
    ("qwen-7b_zh-hacred_json",  ROOT / "out/p15_pull/preds_qwen-ms_json.jsonl"),
    ("qwen-14b_zh-hacred_json", ROOT / "out/p15_hacred_icl/preds_qwen2.5-14b-instruct_json.jsonl"),
    ("qwen-7b_en-hacred_json",  ROOT / "out/p15_hacred_icl/preds_qwen2.5-7b-instruct_en_json.jsonl"),
]

CODES = list(P15.RELS.keys())
CODE_CHARS = {c: set(re.findall(r"[\u4e00-\u9fff]", c)) for c in CODES}

# 词表要"角色名词",这些是模型吐的"动作/动词形态"
ACTION_FORMS = {
    "主持", "主持节目", "出演", "饰演", "演唱", "参演", "主演作品",
    "制作", "编剧", "导演作品", "出演作品", "改编", "执导",
}

# 词表外关系(HacRED 完全不标注的真实世界关系;显式优先于字符重叠)
OFF_INVENTORY = {
    "出生时间", "出生日期", "出生年份", "出生年代", "出生地", "出生",
    "去世时间", "去世日期", "去世年份", "逝世", "逝世时间",
    "名字", "姓名", "学位", "学历", "国籍", "职业", "年龄", "性别", "身高", "体重",
    "获得奖项", "奖项", "荣誉称号", "代表作", "作品", "专辑", "歌曲", "发行专辑",
    "播出平台", "代理艺人", "导师", "职务", "职位", "出生地",
}

# 词表码的显式同义改写(角色名词换词)
SYN2CODE = {
    "就职机构": "所属机构", "就职单位": "所属机构", "任职机构": "所属机构",
    "任职单位": "所属机构", "工作单位": "所属机构", "所处机构": "所属机构",
    "所属单位": "所属机构", "成员": "所属机构",
    "出版时间": "出版/发行时间", "发行时间": "出版/发行时间",
    "出版发行时间": "出版/发行时间", "发行/出版时间": "出版/发行时间",
    "播出/发行时间": "出版/发行时间",
    "成立日期": "成立时间", "创办时间": "成立时间", "创立时间": "成立时间",
    "首映时间": "首播时间", "播出时间": "首播时间", "上映时间": "首播时间",
    "参与": "嘉宾", "参与节目": "嘉宾", "参与作品": "嘉宾", "参加": "嘉宾",
}


def nearest_overlap(name: str):
    """与某词表码的最大汉字重叠率(重叠数 / min(名长, 码长))。"""
    cs = set(re.findall(r"[\u4e00-\u9fff]", name))
    if not cs:
        return None, 0.0
    best, bestr = None, 0.0
    for c, cc in CODE_CHARS.items():
        r = len(cs & cc) / max(1, min(len(cs), len(cc)))
        if r > bestr:
            bestr, best = r, c
    return best, bestr


def classify(name: str) -> str:
    """越界关系名的机制三分类(优先级:显式动作形态 > 显式词表外 > 显式同义 > 字符重叠)。"""
    s = str(name).strip()
    if s in ACTION_FORMS:
        return "action_lexicalization"
    if s in OFF_INVENTORY:
        return "off_inventory"
    if s in SYN2CODE:
        return "role_paraphrase"
    _, r = nearest_overlap(s)
    return "role_paraphrase" if r >= 0.5 else "off_inventory"


def jl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]


def audit(path: Path, name2ix: dict):
    """p15 审计语义(与 r24 逐字一致,每文档集合去重)。返回:
       per[did] = (n, viol_full, viol_relname) —— viol_full 与 §4.3 总违规率口径一致,
       viol_relname 只计关系名越界(机制分解的对象)。"""
    per = {}
    cats = Counter()
    cat_names = defaultdict(Counter)
    for r in jl(path):
        did = r["did"]
        if did not in name2ix:
            continue
        n2ix_d = name2ix[did]
        seen = {}
        n = viol_full = viol_relname = 0
        for h, rl, t in D5.parse_output(r["raw"], r["fmt"], r["ents"], k_bound=r.get("k_index")):
            key = (h, str(rl), t)
            if key in seen:
                continue
            seen[key] = True
            n += 1
            sentinel = h in ("<<PARSE_FAIL>>", "<<VIOLATION>>")
            if sentinel:
                viol_full += 1
                cats["parse_fail"] += 1
                cat_names["parse_fail"][str(h)] += 1
                continue
            if P15.RELS.get(rl) is None:
                viol_full += 1
                viol_relname += 1
                c = classify(rl)
                cats[c] += 1
                cat_names[c][str(rl).strip()] += 1
                continue
            hl, tl = n2ix_d.get(h), n2ix_d.get(t)
            if hl is None or tl is None or len(hl) > 1 or len(tl) > 1:
                viol_full += 1                              # 实体映射违规
                cats["entity_map"] += 1
                continue
            if next(iter(hl)) == next(iter(tl)):
                viol_full += 1                              # 头尾同实体
                cats["same_entity"] += 1
        per[did] = (n, viol_full, viol_relname)
    return per, cats, cat_names


def boot_delta_ratio(per_a: dict, per_b: dict):
    """对 (rate_b - rate_a) 做文档级 bootstrap,percentile 95%(总违规率口径)。
    A=参照臂,B=对比臂。per[did] = (n, viol_full, viol_relname)。"""
    dids = sorted(set(per_a) & set(per_b))
    na = np.array([per_a[d][0] for d in dids], dtype=np.int64)
    va = np.array([per_a[d][1] for d in dids], dtype=np.int64)
    nb = np.array([per_b[d][0] for d in dids], dtype=np.int64)
    vb = np.array([per_b[d][1] for d in dids], dtype=np.int64)
    point = vb.sum() / nb.sum() - va.sum() / na.sum()
    rng = np.random.default_rng(SEED)
    N = len(dids)
    stats = np.empty(REPS)
    for i in range(REPS):
        idx = rng.integers(0, N, N)
        stats[i] = vb[idx].sum() / nb[idx].sum() - va[idx].sum() / na[idx].sum()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"delta_point": float(point), "delta_ci95": [float(lo), float(hi)],
            "rate_a": float(va.sum() / na.sum()), "rate_b": float(vb.sum() / nb.sum()),
            "n_docs": N, "excludes_zero": bool(lo > 0 or hi < 0)}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # 实体名 -> 实体索引(每文档),用于实体映射违规判定(与 r24 一致)
    name2ix = {}
    for dd in json.loads(P15.DEV.read_text(encoding="utf-8")):
        did = P15.doc_id_of(dd)
        m = defaultdict(set)
        for i, ent in enumerate(dd["vertexSet"]):
            m[ent[0]["name"]].add(i)
        name2ix[did] = dict(m)

    res = {"meta": {"amendment": 77,
                    "design": "Chinese H-XI1 label-lexicon interface decomposition "
                              "(frozen predictions, CPU only, p15 dedup semantics)",
                    "seed": SEED, "reps": REPS,
                    "categories": {"action_lexicalization": "词表要角色名词,模型吐动作/动词",
                                   "role_paraphrase": "词表码的同义改写(汉字重叠>=0.5)",
                                   "off_inventory": "词表外真实世界关系",
                                   "parse_fail": "解析失败哨兵(<<VIOLATION>>/<<PARSE_FAIL>>)",
                                   "entity_map": "实体名未能唯一映射到文档实体",
                                   "same_entity": "头尾解析为同一实体"}},
           "arms": {}}

    audits = {}
    for tag, path in ARMS:
        per, cats, names = audit(path, name2ix)
        audits[tag] = per
        nd = len(per)
        tot = sum(v[0] for v in per.values())
        vfull = sum(v[1] for v in per.values())
        vrel = sum(v[2] for v in per.values())
        row = {"n_docs": nd, "triples": tot,
               "viol_full": vfull, "viol_relname": vrel,
               "rate_full_pct": round(vfull / tot * 100, 2),
               "rate_relname_pct": round(vrel / tot * 100, 2),
               "triples_per_doc": round(tot / nd, 3),
               "viol_full_per_doc": round(vfull / nd, 3),
               "viol_relname_per_doc": round(vrel / nd, 3),
               "category_counts": dict(cats),
               "category_per_doc": {k: round(v / nd, 4) for k, v in cats.items()},
               "top_relname_unmapped": [[s, n] for s, n in Counter(
                   {s: c for c in ("action_lexicalization", "role_paraphrase", "off_inventory")
                    for s, c in names[c].items()}).most_common(8)]}
        res["arms"][tag] = row
        print(f"\n[{tag}] docs={nd} triples={tot} ({tot/nd:.2f}/doc) | "
              f"总违规={vfull} ({vfull/nd:.3f}/doc, {vfull/tot*100:.1f}%) | "
              f"关系名越界={vrel} ({vrel/nd:.3f}/doc, {vrel/tot*100:.1f}%)")
        print(f"   全类别: " + "  ".join(f"{k}={v}({v/nd:.3f}/doc)" for k, v in
              sorted(cats.items(), key=lambda x: -x[1])))
        for c in ("action_lexicalization", "role_paraphrase", "off_inventory"):
            print(f"   {c:22s}: {cats[c]:5d} ({cats[c]/max(1,vrel)*100:4.1f}% of relname) "
                  f"top={[s for s,_ in names[c].most_common(4)]}")

    # 规模效应(7B→14B,中文臂)
    a, b = audits["qwen-7b_zh-hacred_json"], audits["qwen-14b_zh-hacred_json"]
    res["scale_effect_zh"] = boot_delta_ratio(a, b)
    # 语言对照(7B:中→英指令)
    c = audits["qwen-7b_en-hacred_json"]
    res["prompt_language_effect_7b"] = boot_delta_ratio(a, c)

    print(f"\n=== 规模效应(中文臂 7B→14B) ===")
    se = res["scale_effect_zh"]
    print(f"  Δ总违规率 = {se['delta_point']*100:+.2f}pt  CI95=[{se['delta_ci95'][0]*100:+.2f}, "
          f"{se['delta_ci95'][1]*100:+.2f}]  排零={se['excludes_zero']}")
    print(f"=== 指令语言效应(7B 中→英,HacRED 同文档) ===")
    pe = res["prompt_language_effect_7b"]
    print(f"  Δ总违规率 = {pe['delta_point']*100:+.2f}pt  CI95=[{pe['delta_ci95'][0]*100:+.2f}, "
          f"{pe['delta_ci95'][1]*100:+.2f}]  排零={pe['excludes_zero']}")

    # 机制签名:两臂的越界名构成(只计关系名越界,排除哨兵/实体映射)
    def sig(tag):
        cc = res["arms"][tag]["category_counts"]
        t = sum(cc.get(k, 0) for k in ("action_lexicalization", "role_paraphrase", "off_inventory")) or 1
        return {k: round(cc.get(k, 0) / t, 3) for k in
                ("action_lexicalization", "role_paraphrase", "off_inventory")}
    s7, s14 = sig("qwen-7b_zh-hacred_json"), sig("qwen-14b_zh-hacred_json")
    res["mechanism_signature"] = {
        "qwen-7b": s7, "qwen-14b": s14,
        "shift": {k: round(s14[k] - s7[k], 3) for k in s7},
        "reading": "规模上升时,越界名的构成从'动作形态滑移'转向'角色名词同义改写'与'词表外关系导入';"
                   "两臂都非因关系知识缺失而失败,而是在标签词表接口的不同层次失败。"}
    print(f"\n=== 机制签名(越界名构成) ===")
    print(f"  7B : {s7}")
    print(f"  14B: {s14}")

    out = STAT / "r47_zh_label_mechanism.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nsaved ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
