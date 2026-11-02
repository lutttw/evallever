# -*- coding: utf-8 -*-
"""p15_hacred_icl.py —— 预注册修订 17:P1.5 X-A 中文 ICL 钩子(HacRED)。

设计冻结于 docs/ACL_主会执行方案_20261003/预注册_v1_20261003.md 修订 17(X-A),要点:
  * 数据:HacRED 评测拆分(test_revised.json,1500 篇,字符层冻结契约,与 15 种子同源);
    候选/分组/金标全部复用 out/jcip_audit_v1/hacred_prepared 既有审计基础设施
    (评测候选总体 253,732,对应 D5 的 189,556 式全候选审计);
  * 提示:中文模板重写,4-shot;示例选择沿用 D5 的 sha256 哈希最近邻规则
    (对评测前 2 篇标题,各取 sha256(train_title|dev_title) 最小的 2 篇训练文档);
    关系表 = 26 个中文官方关系名(hacred_rel2id.json 剔除 NA,与 hacred_prepare 同源);
    输出沿用修订 8 索引化格式(E{i} 引用逐文档实体表);
  * 违规冒烟门槛升级为 20–50 篇抽样(修订 8 的 3 篇教训),按占比分层:文档字符长度
    五分位分层,层内取 sha256(seed|doc_id) 最小者,共 40 篇;违规率 >5% 该设置带旗,
    与 D5 同纪律,不调门槛;
  * 模型:Qwen2.5-7B-Instruct(中文系,主)+ Mistral-7B-Instruct-v0.3(英文系,交互对照,
    云端执行);NF4 + greedy,与 D5 零改动;
  * 对照矩阵(全部在**同一批 LLM 预测**上审计):
      L1 换计分:Gap 的 A(hit-any)/ R(逐金标)/ F1 三计分读数(H-XL1:三计分符号一致性);
      L2 换金标:字层金标(labels_char,既有 golds_evaluation)vs 词层金标(labels_word,
        在词层索引空间独立重建候选/分组/金标)——HacRED 特有的"切分粒度金标对"
        (H-XL2:Δgap_F1 = 字层 − 词层,配对整文档 bootstrap 10000 次,种子 20260926;
        预注册预期:E6 已证词/字金标按名三元组零差异、分组翻转率 0,故 |Δ| 可能 ≈0,
        若如此,如实报告为"切分粒度契约在中文词/字双层下不构成有效杠杆"——边界发现);
      L3 提示契约:json vs freetext 输出格式(探索,与 D5 L3 同轴)。
  * 解析纪律:与 d5_llm_hook.py 逐字一致(直接复用其 parse_output/_resolve);
    关系名不在 26 关系表 / 越界索引 / 歧义名(一名多实体)均计契约违规。

用法:
  python src/jcip_audit/p15_hacred_icl.py build                 # 构建中文提示(json+freetext)+冒烟子集
  python src/jcip_audit/p15_hacred_icl.py mock-audit            # 金标自喂审计自检(期望 F1=1)
  python src/jcip_audit/p15_hacred_icl.py infer --model <路径或HF名> \
      --format json --subset smoke --load 4bit                  # 推断(smoke|full)
  python src/jcip_audit/p15_hacred_icl.py audit \
      --pred out/p15_hacred_icl/preds_<tag>_json.jsonl          # 解析+双金标审计
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "jcip_audit"))

import hacred_runtime as HR                              # noqa: E402  (canon/读 jsonl 纪律)
from d5_llm_hook import parse_output, select_demos       # noqa: E402  (解析零改动)

DEV = ROOT / "data" / "hacred" / "test_revised.json"     # 评测拆分(字符层,1500 篇)
TRAIN = ROOT / "data" / "hacred" / "train_revised.json"
REPO = ROOT / "data" / "hacred_repo" / "test.json"       # 官方双层标注(JSONL 内容 .json 名)
PREP = ROOT / "out" / "jcip_audit_v1" / "hacred_prepared"
OUT = ROOT / "out" / "p15_hacred_icl"
STAT = ROOT / "out" / "revision_v5_experiments" / "statistics"
SEED = "20260926"
BOOT = 10000
MAX_DEMO_SETS, MAX_TGT_SETS, MAX_DEMO_TRIPLES = 10, 24, 12   # 与 D5 相同(HacRED ≤21 句,无截断)
MAX_NEW_TOKENS, MAX_PROMPT_TOKENS = 768, 7000
# 修订 7 的 batch=2 系本地 8GB 卡约束;云端 24GB GPU(24GB)可经 P15_BATCH 提高吞吐。
# 显存核算:NF4 权重 ≈5.5G + batch8 极长提示(7k token)KV ≈3.2G ≈ 9G,24GB 充裕。
BATCH = int(os.environ.get("P15_BATCH", "2"))
SMOKE_N_PER_STRATUM = 8                                  # 5 分位 × 8 = 40 篇(修订 17:20–50 篇)
# H-XL3 英文 D5 参照(读自 out/d5_llm_hook/audit_preds_*.json,2026-10 落盘值)
D5_EN_REF = {"qwen2.5-14b-instruct_json": 0.0457, "qwen2.5-7b-instruct_freetext": 0.1164,
             "mistral-7b-instruct_json": 0.1884, "mistral-7b-instruct_freetext": 0.0705}


def jl(p: Path) -> list:
    """HaCRED 正文含 U+2028,必须 split("\\n")(hacred_runtime 纪律)。"""
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]


def doc_id_of(d) -> str:
    """与 hacred_prepare/hacred_runtime 一致:sha256(canon({title, sents})),title=id 串。"""
    return hashlib.sha256(HR.canon({"title": str(d.get("title", "")).strip(),
                                    "sents": d["sents"]}).encode("utf-8")).hexdigest()


def load_rels() -> dict:
    """26 个中文官方关系名 → 审计索引(hacred_prepare 同源)。"""
    return json.loads((PREP / "relation_to_index.json").read_text(encoding="utf-8"))


def sents_text(sents, cap) -> list:
    """中文句子按字符直接拼接(字符层冻结契约;D5 的空格拼接是英文习惯,中文不适用)。"""
    return ["[S%d] %s" % (k, "".join(s)) for k, s in enumerate(sents[:cap])]


# ---------------------------------------------------------------- 构建
# 修订 33(C1 语言去混杂):en 模板为 zh 的逐句直译;输出空间不变
# (26 个中文官方关系名、E{i} 索引、中文实体名),仅换指令/字段语言。
L = {
    "zh": {"doc": "文档:", "ent": "实体表: ", "rel": "关系表: ", "relsep": "、",
           "out_json": "关系(JSON): ", "out_free": "关系: ", "none": "(无)"},
    "en": {"doc": "Document:", "ent": "Entity table: ", "rel": "Relation table: ",
           "relsep": ", ", "out_json": "Relations(JSON): ", "out_free": "Relations: ",
           "none": "(none)"},
}


def doc_block(title, sents, ent_line, out_line, cap, lang="zh") -> str:
    s = f"### {title}\n{ent_line}\n{L[lang]['doc']}\n" + "\n".join(sents_text(sents, cap))
    if out_line:
        s += f"\n{out_line}"
    return s


def make_prompt(rel_line, demo_blocks, tgt_block, fmt, lang="zh") -> str:
    if lang == "en":
        if fmt == "json":
            instr = ('Output strict JSON only: {"triples": [["E0", "作者", "E5"], ...]}. '
                     'The head and tail entity of every triple must be an entity index '
                     'from the entity table above (of the form "E0"), and the relation '
                     'must be copied verbatim from the relation table. If unsure, omit '
                     'the triple. Do not output dates, quantities, or any string that is '
                     'not an entity index.')
        else:
            instr = ("Output a single line: 'Relations: E0 -> 作者 -> E5; ...' "
                     "(if there are no relations, output 'Relations: (none)'). "
                     "The head and tail entity of every triple must be an entity index "
                     "from the entity table above, and the relation must be copied "
                     "verbatim from the relation table. If unsure, omit the triple.")
        return rel_line + "\n\n" + "\n\n".join(demo_blocks) + f"\n\n{tgt_block}\n{instr}"
    if fmt == "json":
        instr = ('仅输出严格 JSON:{"triples": [["E0", "作者", "E5"], ...]}。'
                 '每条三元组的头实体和尾实体必须使用上方实体表中的实体编号(形如 "E0"),'
                 '关系必须从关系表中原样选取。若不确定,请省略该三元组。'
                 '禁止输出日期、数量或任何不是实体编号的字符串。')
    else:
        instr = ("仅输出一行:'关系: E0 -> 作者 -> E5; ...'(若无关系则输出 '关系: (无)')。"
                 "每条三元组的头实体和尾实体必须使用上方实体表中的实体编号,"
                 "关系必须从关系表中原样选取。若不确定,请省略该三元组。")
    return rel_line + "\n\n" + "\n\n".join(demo_blocks) + f"\n\n{tgt_block}\n{instr}"


def out_line_of(d, fmt, ents_ix, lang="zh") -> str:
    tri = [[ents_ix[d["vertexSet"][lb["h"]][0]["name"]], lb["r"],
            ents_ix[d["vertexSet"][lb["t"]][0]["name"]]]
           for lb in d.get("labels", []) if lb["r"] in RELS and lb["h"] != lb["t"]]
    if fmt == "json":
        return L[lang]["out_json"] + HR.canon({"triples": tri[:MAX_DEMO_TRIPLES]})
    body = "; ".join(f"{h} -> {r} -> {t}" for h, r, t in tri[:MAX_DEMO_TRIPLES])
    return L[lang]["out_free"] + (body if body else L[lang]["none"])


RELS = load_rels()


def cmd_build(a):
    lang = getattr(a, "lang", "zh")
    suf = "" if lang == "zh" else "en_"          # zh 文件名不变(字节回归);en 加前缀
    OUT.mkdir(parents=True, exist_ok=True)
    dev = json.loads(DEV.read_text(encoding="utf-8"))
    train = json.loads(TRAIN.read_text(encoding="utf-8"))
    rel_line = L[lang]["rel"] + L[lang]["relsep"].join(sorted(RELS))
    # 演示选择:与 D5 同一冻结规则(评测前 2 篇标题 × sha256 最近邻各 2 篇训练文档)
    demo_ids = select_demos(train, [str(dev[0].get("title", "")).strip(),
                                    str(dev[1].get("title", "")).strip()])
    demos = []
    for k in demo_ids:
        td = train[k]
        ents = sorted({v[0]["name"] for v in td["vertexSet"]})
        ix = {n: f"E{i}" for i, n in enumerate(ents)}
        demos.append({"title": str(td.get("title", "")).strip(), "sents": td["sents"],
                      "ent_line": L[lang]["ent"]
                      + ", ".join(f"E{i}={n}" for i, n in enumerate(ents)),
                      "tri_json": out_line_of(td, "json", ix, lang),
                      "tri_free": out_line_of(td, "freetext", ix, lang)})
    rep = {"lang": lang, "demo_ids": demo_ids,
           "demo_titles": [d["title"] for d in demos],
           "demo_rule": "sha256(train_title|dev_title) 最小的 2x2(评测前 2 篇;与 D5 同一冻结规则)",
           "n_relations": len(RELS), "relations": sorted(RELS),
           "output_format": "entity-index (E{i}),修订 8 索引化",
           "note_title": "HacRED 无标题字段,运行时契约以 id 串为 title(hacred_adapt 口径)",
           "note_lang": ("修订 33:en=仅指令/字段语言英译;关系表/实体名/E{i} 输出空间不变"
                         if lang == "en" else "中文冻结模板(修订 17)"),
           "docs": 0, "chars": []}
    # 冒烟子集:文档字符长度五分位分层 × 各 8 篇(sha256(seed|doc_id) 最小)——文档级,与语言无关
    doclen = [(doc_id_of(d), sum(len("".join(s)) for s in d["sents"])) for d in dev]
    ranked = sorted(doclen, key=lambda x: x[1])
    qs = [ranked[int(q / 5 * len(ranked))][1] for q in range(1, 5)]
    smoke = []
    for did, ln in doclen:
        s = sum(ln >= q for q in qs)                       # 0..4 层
        smoke.append((s, hashlib.sha256((SEED + "|" + did).encode("utf-8")).hexdigest(), did))
    smoke_ids = set()
    for s in range(5):
        pick = sorted(x for x in smoke if x[0] == s)[:SMOKE_N_PER_STRATUM]
        smoke_ids |= {p[2] for p in pick}
    smoke_file = OUT / "smoke_dids.json"
    if lang == "zh":
        smoke_file.write_text(HR.canon(sorted(smoke_ids)), encoding="utf-8")
    else:                                                  # en 复用同一冒烟子集,不重写
        assert set(json.loads(smoke_file.read_text(encoding="utf-8"))) == smoke_ids, \
            "smoke_dids.json 与重算不一致(数据漂移?)"
    for fmt in ("json", "freetext"):
        with (OUT / f"prompts_{suf}{fmt}.jsonl").open("w", encoding="utf-8") as f:
            for dd in dev:
                did = doc_id_of(dd)
                ents = sorted({v[0]["name"] for v in dd["vertexSet"]})
                trunc = len(dd["sents"]) > MAX_TGT_SETS     # HacRED ≤21 句,应为 0
                rep["docs"] += 1
                rep["chars"].append(sum(len("".join(s)) for s in dd["sents"]))
                blocks = []
                for d in demos:
                    ol = d["tri_json"] if fmt == "json" else d["tri_free"]
                    blocks.append(doc_block(d["title"], d["sents"], d["ent_line"], ol,
                                            MAX_DEMO_SETS, lang))
                tl = L[lang]["ent"] + ", ".join(f"E{i}={n}" for i, n in enumerate(ents))
                tgt = doc_block(str(dd.get("title", "")).strip(), dd["sents"], tl, None,
                                MAX_TGT_SETS, lang)
                f.write(HR.canon({"did": did, "title": str(dd.get("title", "")).strip(),
                                  "fmt": fmt, "truncated": trunc, "ents": ents,
                                  "smoke": did in smoke_ids,
                                  "prompt": make_prompt(rel_line, blocks, tgt, fmt, lang)}) + "\n")
    rep["smoke_n"] = len(smoke_ids)
    rep["doc_chars"] = {"mean": sum(rep["chars"]) / len(rep["chars"]),
                        "max": max(rep["chars"])}
    del rep["chars"]
    report = OUT / ("build_report.json" if lang == "zh" else "build_report_en.json")
    report.write_text(HR.canon(rep), encoding="utf-8")
    print(f"[build:{lang}] {rep['docs']} 篇 × 2 格式 → {OUT}(前缀 '{suf or '(无)'}');"
          f"冒烟 {len(smoke_ids)} 篇;演示={rep['demo_titles']}")


# ---------------------------------------------------------------- 推断
def cmd_infer(a):
    import torch
    from d5_llm_hook import load_model                     # NF4/greedy 与 D5 零改动
    if getattr(a, "sample", False):
        torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    lang = getattr(a, "lang", "zh")
    suf = "" if lang == "zh" else "en_"
    rows = jl(OUT / f"prompts_{suf}{a.format}.jsonl")
    if a.subset == "smoke":
        keep = set(json.loads((OUT / "smoke_dids.json").read_text(encoding="utf-8")))
        rows = [r for r in rows if r["did"] in keep]
        assert len(rows) == len(keep), f"冒烟子集不全:{len(rows)}/{len(keep)}"
    tok, model = load_model(a.model, a.load)
    tag = Path(a.model).name or a.model.split("/")[-1]   # 本地路径取目录名(剔除盘符冒号)
    n_trunc_tok, t0, n_done = 0, time.time(), 0
    with (OUT / f"preds_{tag}_{suf}{a.format}{a.run_suffix}.jsonl").open("w", encoding="utf-8") as fo:
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            texts = []
            for r in batch:
                msgs = [{"role": "user", "content": r["prompt"]}]
                t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                ids = tok(t, add_special_tokens=False)["input_ids"]
                if len(ids) > MAX_PROMPT_TOKENS:           # HacRED 短文档,预期 0 次
                    n_trunc_tok += 1
                    t = tok.decode(ids[:MAX_PROMPT_TOKENS])
                    t = t.rsplit("[S", 1)[0] + "\n(输出截断。)"
                texts.append(t)
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gkw = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=bool(a.sample),
                           num_beams=1, pad_token_id=tok.pad_token_id)
                if a.sample:                       # 修订 44:运行间方差复验臂
                    gkw.update(temperature=a.temperature, top_p=1.0)
                out = model.generate(**enc, **gkw)
            for r, inp, o in zip(batch, enc["input_ids"], out):
                fo.write(HR.canon({"did": r["did"], "title": r["title"], "model": tag,
                                   "fmt": a.format, "load": a.load, "ents": r["ents"],
                                   "raw": tok.decode(o[len(inp):], skip_special_tokens=True)})
                         + "\n")
            n_done += len(batch)
            dt = (time.time() - t0) / n_done
            print(f"[infer] {n_done}/{len(rows)}  {dt:.1f}s/篇  "
                  f"全量1500篇预计 {dt * 1500 / 3600:.1f}h", flush=True)
    print(f"[infer] 完成;超 token 截断 {n_trunc_tok} 篇;"
          f"均速 {(time.time() - t0) / max(n_done, 1):.1f}s/篇")


# ---------------------------------------------------------------- 词层金标(词层索引空间独立重建)
def build_word_space() -> dict:
    """逐文档独立重建词层空间;did 与字层管线一致(官方 README:sents_char ↔ sents)。"""
    char = json.loads(DEV.read_text(encoding="utf-8"))
    repo = jl(REPO)
    assert len(char) == len(repo) == 1500
    rel_ix = RELS
    space = {}
    map_stats = {"gold_triple_char_only": set(), "gold_triple_word_only": set()}
    for rec_c, rec_o in zip(char, repo):
        assert str(rec_c["id"]) == str(rec_o["id"])
        did = doc_id_of(rec_c)
        vw = rec_o["vertex_word"]
        sent_sets = [{m["sent_id"] for m in ent} for ent in vw]
        first = [min(s) if s else 0 for s in sent_sets]
        name2ix = defaultdict(set)
        for i, ent in enumerate(vw):
            name2ix[ent[0]["name"]].add(i)
        per_pair = defaultdict(set)
        n_self = 0
        for lb in rec_o["labels_word"]:
            if lb["r"] not in rel_ix:
                continue
            if lb["h"] == lb["t"]:
                n_self += 1
                continue
            per_pair[(lb["h"], lb["t"])].add(rel_ix[lb["r"]])
        # 金标按名三元组与字层对账(E6 口径;差集应为空)
        cn = {i: ent[0]["name"] for i, ent in enumerate(rec_c["vertexSet"])}
        wn = {i: ent[0]["name"] for i, ent in enumerate(vw)}
        gc = {(cn[lb["h"]], cn[lb["t"]], lb["r"]) for lb in rec_c["labels"]
              if lb["h"] != lb["t"] and lb["r"] in rel_ix}
        gw = {(wn[lb["h"]], wn[lb["t"]], lb["r"]) for lb in rec_o["labels_word"]
              if lb["h"] != lb["t"] and lb["r"] in rel_ix}
        map_stats["gold_triple_char_only"] |= (gc - gw)
        map_stats["gold_triple_word_only"] |= (gw - gc)
        space[did] = {"n": len(vw), "first": first, "name2ix": dict(name2ix),
                      "gold": {k: sorted(v) for k, v in per_pair.items()},
                      "selfloop_dropped": n_self}
    return {"space": space, "gold_delta": {
        "char_only": sorted(map_stats["gold_triple_char_only"]),
        "word_only": sorted(map_stats["gold_triple_word_only"])}}


# ---------------------------------------------------------------- 审计
def audit_space(preds_name: dict, docs: dict, space: dict) -> dict:
    """在一个索引空间(字层或词层)上审计:全候选总体计数 + 命名解析违规。

    preds_name: did -> {(h_name, rel, t_name)};docs: did -> 文档记录(取顶点名)。
    返回与 d5 相同风格的计数与指标。
    """
    stats = {"n_triples": 0, "parse_fail": 0, "oob_index": 0, "hallucinated": 0,
             "ambiguous": 0, "self_pair": 0, "rel_unmapped": 0, "mapped": 0}
    preds = defaultdict(set)                                # (did, h, t) -> {rel_idx}
    for did, triples in preds_name.items():
        sp = space[did]
        n2i = sp["name2ix"]
        for h, rl, t in triples:
            stats["n_triples"] += 1
            if h == "<<PARSE_FAIL>>":
                stats["parse_fail"] += 1
                continue
            if h == "<<VIOLATION>>":
                stats["oob_index"] += 1
                continue
            ri = RELS.get(rl)
            if ri is None:
                stats["rel_unmapped"] += 1
                continue
            hl, tl = n2i.get(h), n2i.get(t)
            if hl is None or tl is None:
                stats["hallucinated"] += 1                  # E 表内名字必可解析,防御性计数
                continue
            if len(hl) > 1 or len(tl) > 1:
                stats["ambiguous"] += 1                     # D5 同纪律:歧义名计违规
                continue
            hc, tc = next(iter(hl)), next(iter(tl))
            if hc == tc:
                stats["self_pair"] += 1
                continue
            preds[(did, hc, tc)].add(ri)
            stats["mapped"] += 1
    # 全候选总体
    import numpy as np
    doc_ids = sorted(space)
    pos = {d: k for k, d in enumerate(doc_ids)}
    cnt = np.zeros((len(doc_ids), 2, 3), dtype=np.int64)    # (doc, intra/inter, tp/fp/fn)
    pairo = np.zeros((len(doc_ids), 2, 3), dtype=np.int64)  # hit / fp_pair / n_gold_pairs
    n_cand = 0
    for did in doc_ids:
        sp = space[did]
        k = pos[did]
        first = sp["first"]
        gold = {kk: set(v) for kk, v in sp["gold"].items()}
        for h in range(sp["n"]):
            for t in range(sp["n"]):
                if h == t:
                    continue
                n_cand += 1
                P = preds.get((did, h, t), set())
                G = gold.get((h, t), set())
                g = 0 if first[h] == first[t] else 1
                tp, fp, fn = len(P & G), len(P - G), len(G - P)
                c = cnt[k, g]
                c[0] += tp; c[1] += fp; c[2] += fn
                if G:
                    pairo[k, g, 0] += 1 if tp > 0 else 0
                    pairo[k, g, 2] += 1
                elif P:
                    pairo[k, g, 1] += 1
    met = metrics_from(cnt, pairo)
    return {"stats": stats, "n_candidates": n_cand, "metrics": met,
            "_per_doc_arrays": (doc_ids, cnt, pairo)}


def metrics_from(cnt, pairo) -> dict:
    out = {}
    for g, gn in ((0, "intra"), (1, "inter")):
        tp, fp, fn = int(cnt[:, g, 0].sum()), int(cnt[:, g, 1].sum()), int(cnt[:, g, 2].sum())
        n_gold = tp + fn
        hit, fp_pair, n_gold_p = int(pairo[:, g, 0].sum()), int(pairo[:, g, 1].sum()), int(pairo[:, g, 2].sum())
        F1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        R = tp / n_gold if n_gold else 0.0
        A = hit / n_gold_p if n_gold_p else 0.0
        pF1 = 2 * hit / (2 * hit + fp_pair + (n_gold_p - hit)) if (2 * hit + fp_pair + (n_gold_p - hit)) else 0.0
        out[gn] = {"F1": F1, "R": R, "A": A, "pair_F1": pF1, "tp": tp, "fp": fp, "fn": fn}
    out["gap_F1"] = out["intra"]["F1"] - out["inter"]["F1"]
    out["gap_R"] = out["intra"]["R"] - out["inter"]["R"]
    out["gap_A"] = out["intra"]["A"] - out["inter"]["A"]
    out["gap_pair_F1"] = out["intra"]["pair_F1"] - out["inter"]["pair_F1"]
    return out


def cmd_mock_audit(_):
    """自检:把字层金标当作"模型输出"喂回审计(规则化名字解析),期望违规=0、F1=1。

    注意:E 表按排序唯一名构造,歧义名(一名多顶点)无法经名字通道无损表达——
    这类金标对会落入 ambiguous(与真实 LLM 预测同纪律),自检阈值相应放宽并如实记录。
    """
    char = json.loads(DEV.read_text(encoding="utf-8"))
    preds_name = {}
    n_amb_gold = 0
    for dd in char:
        did = doc_id_of(dd)
        ents = {v[0]["name"] for v in dd["vertexSet"]}
        amb = {dd["vertexSet"][i][0]["name"] for i in range(len(dd["vertexSet"]))}
        n_multi = len(dd["vertexSet"]) - len(ents)
        per = set()
        for lb in dd["labels"]:
            if lb["h"] == lb["t"] or lb["r"] not in RELS:
                continue
            hn, tn = dd["vertexSet"][lb["h"]][0]["name"], dd["vertexSet"][lb["t"]][0]["name"]
            per.add((hn, lb["r"], tn))
        n_amb_gold += n_multi
        preds_name[did] = per
    ws = build_word_space()
    space_c = {}
    for dd in char:
        did = doc_id_of(dd)
        first = [min((m["sent_id"] for m in ent), default=0) for ent in dd["vertexSet"]]
        n2ix = defaultdict(set)
        for i, ent in enumerate(dd["vertexSet"]):
            n2ix[ent[0]["name"]].add(i)
        gold = defaultdict(set)
        for lb in dd["labels"]:
            if lb["h"] != lb["t"] and lb["r"] in RELS:
                gold[(lb["h"], lb["t"])].add(RELS[lb["r"]])
        space_c[did] = {"n": len(dd["vertexSet"]), "first": first, "name2ix": dict(n2ix),
                        "gold": {k: sorted(v) for k, v in gold.items()}}
    res_c = audit_space(preds_name, char, space_c)
    print(json.dumps({"char": {"stats": res_c["stats"], "metrics": res_c["metrics"]}},
                     ensure_ascii=False, indent=1, default=float))
    (OUT / "mock_audit.json").write_text(HR.canon(
        {"char": {"stats": res_c["stats"], "metrics": res_c["metrics"]},
         "n_docs_with_dup_names": sum(1 for dd in char
                                      if len({v[0]["name"] for v in dd["vertexSet"]})
                                      != len(dd["vertexSet"]))}), encoding="utf-8")


def cmd_audit(a):
    import numpy as np
    OUT.mkdir(parents=True, exist_ok=True)
    char = json.loads(DEV.read_text(encoding="utf-8"))
    docs = {doc_id_of(dd): dd for dd in char}
    # 字层空间(与 hacred_prepare 冻结契约对账)
    cand = jl(PREP / "candidates" / "evaluation.jsonl")
    golds_c = {}
    for g in jl(PREP / "golds_evaluation.jsonl"):
        golds_c.setdefault(g["doc_id"], {})[(g["h"], g["t"])] = g["rels"]
    space_c = {}
    for dd in char:
        did = doc_id_of(dd)
        first = [min((m["sent_id"] for m in ent), default=0) for ent in dd["vertexSet"]]
        n2ix = defaultdict(set)
        for i, ent in enumerate(dd["vertexSet"]):
            n2ix[ent[0]["name"]].add(i)
        space_c[did] = {"n": len(dd["vertexSet"]), "first": first, "name2ix": dict(n2ix),
                        "gold": golds_c.get(did, {})}
    # 对账:重建空间 vs 冻结候选文件
    frozen_first = sum(1 for c in cand if c["group"] == 0)
    recon_first = sum(1 for sp in space_c.values()
                      for h in range(sp["n"]) for t in range(sp["n"])
                      if h != t and sp["first"][h] == sp["first"][t])
    assert len(cand) == 253732, len(cand)
    # 词层空间(独立重建)
    ws = build_word_space()
    space_w = ws["space"]
    # 预测(名字三元组,解析与 D5 逐字一致)
    preds_name = defaultdict(set)
    parse_extra = {"n_recs": 0}
    for r in jl(Path(a.pred)):
        parse_extra["n_recs"] += 1
        ents = r["ents"]
        for h, rl, t in parse_output(r["raw"], r["fmt"], ents):
            preds_name[r["did"]].add((h, rl, t))
    for did in space_c:
        preds_name.setdefault(did, set())
    res_c = audit_space(preds_name, docs, space_c)
    res_w = audit_space(preds_name, docs, space_w)
    st = res_c["stats"]
    n_tri = st["n_triples"]
    viol = n_tri - st["mapped"]
    viol_rate = viol / n_tri if n_tri else 0.0
    # L2:Δgap_F1(字层 − 词层),配对整文档 bootstrap(种子 20260926)
    doc_ids = res_c["_per_doc_arrays"][0]
    assert doc_ids == res_w["_per_doc_arrays"][0]
    cnt_c, pairo_c = res_c["_per_doc_arrays"][1], res_c["_per_doc_arrays"][2]
    cnt_w, pairo_w = res_w["_per_doc_arrays"][1], res_w["_per_doc_arrays"][2]
    N = len(doc_ids)
    rng = np.random.default_rng(int(SEED))
    deltas = np.empty(BOOT)
    for b in range(BOOT):
        idx = rng.integers(0, N, N)

        def gap(cnt, pairo):
            f = []
            for g in (0, 1):
                tp, fp, fn = cnt[idx, g, 0].sum(), cnt[idx, g, 1].sum(), cnt[idx, g, 2].sum()
                f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
            return f[0] - f[1]

        deltas[b] = gap(cnt_c, pairo_c) - gap(cnt_w, pairo_w)
    point = res_c["metrics"]["gap_F1"] - res_w["metrics"]["gap_F1"]
    ci = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))
    # L1(H-XL1):三计分符号一致性(字层主金标;词层并列记录)
    def signs(m):
        s = {k: (1 if m[k] > 0 else (-1 if m[k] < 0 else 0)) for k in ("gap_F1", "gap_R", "gap_A")}
        vals = {v for v in s.values()}
        return {"signs": s, "consistent": len(vals - {0}) <= 1}
    rep = {"pred": a.pred, "n_pred_recs": parse_extra["n_recs"],
           "seed": SEED, "bootstrap_reps": BOOT,
           "stats": st, "violation_rate": viol_rate,
           "violation_gate": {"threshold": 0.05, "unverifiable": bool(viol_rate > 0.05),
                              "rule": "违规率 >5% 该设置带旗(修订 2/17 同纪律,不调门槛)"},
           "n_candidates_char": res_c["n_candidates"], "n_candidates_word": res_w["n_candidates"],
           "char_gold": {"metrics": res_c["metrics"], "L1_signs": signs(res_c["metrics"])},
           "word_gold": {"metrics": res_w["metrics"], "L1_signs": signs(res_w["metrics"])},
           "H_XL1": {"claim": "同一批中文预测在 A/R/F1 三计分下差距读数符号不一致(预期成立)",
                     "char_consistent": signs(res_c["metrics"])["consistent"],
                     "word_consistent": signs(res_w["metrics"])["consistent"],
                     "held_char": not signs(res_c["metrics"])["consistent"]},
           "H_XL2": {"claim": "词层 vs 字层金标切换使 F1 差距移动 |Δ|>0(方向/量级开放;"
                              "若 ≈0,如实报告为切分粒度契约不构成有效杠杆——边界发现)",
                     "delta_gap_F1": {"point": float(point), "ci95": list(ci)},
                     "gap_F1_char": res_c["metrics"]["gap_F1"],
                     "gap_F1_word": res_w["metrics"]["gap_F1"],
                     "gold_delta_word_vs_char": ws["gold_delta"]},
           "H_XL3": {"claim": "中文提示违规率与英文 D5 量级可比(探索)",
                     "zh_violation_rate": viol_rate, "en_d5_reference": D5_EN_REF},
           "frozen_contract_crosscheck": {"frozen_candidates": len(cand),
                                          "frozen_intra": frozen_first,
                                          "recomputed_intra_char": recon_first}}
    stem = Path(a.pred).stem
    (OUT / f"audit_{stem}.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    STAT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({k: rep[k] for k in ("stats", "violation_rate")}, ensure_ascii=False,
                     default=float))
    for tag, r in (("字层", res_c), ("词层", res_w)):
        m = r["metrics"]
        print(f"[{tag}] gap_F1={m['gap_F1']:+.4f} gap_R={m['gap_R']:+.4f} "
              f"gap_A={m['gap_A']:+.4f}(候选 {r['n_candidates']:,})")
    print(f"[L2] Δgap_F1(字−词) = {point:+.6f}  CI95 {ci[0]:+.6f}..{ci[1]:+.6f}")
    print(f"[L1] 字层三计分符号一致? {signs(res_c['metrics'])['consistent']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    fb = sub.add_parser("build")
    fb.add_argument("--lang", choices=("zh", "en"), default="zh",
                    help="修订 33:en=仅指令/字段语言英译,输出空间不变")
    sub.add_parser("mock-audit")
    fi = sub.add_parser("infer")
    fi.add_argument("--model", required=True)
    fi.add_argument("--format", choices=("json", "freetext"), required=True)
    fi.add_argument("--subset", choices=("smoke", "full"), default="smoke")
    fi.add_argument("--load", choices=("fp16", "4bit"), default="4bit")
    fi.add_argument("--lang", choices=("zh", "en"), default="zh")
    fi.add_argument("--sample", action="store_true", help="修订44:抽样解码")
    fi.add_argument("--temperature", type=float, default=0.7)
    fi.add_argument("--seed", type=int, default=20260926)
    fi.add_argument("--run-suffix", default="", help="产物文件名后缀")
    fa = sub.add_parser("audit")
    fa.add_argument("--pred", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "infer": cmd_infer, "audit": cmd_audit,
     "mock-audit": cmd_mock_audit}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
