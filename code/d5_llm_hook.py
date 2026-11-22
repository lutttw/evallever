# -*- coding: utf-8 -*-
"""d5_llm_hook.py —— 预注册修订 2:D5 LLM 钩子实验(构建/推断/审计三模式)。

设计冻结于 docs/ACL_主会执行方案_20261003/预注册_v1_20261003.md 修订 2,要点:
  * 模型:Qwen2.5-7B-Instruct(主)/ Llama-3.1-8B-Instruct(鲁棒性),FP16 greedy,不采样;
  * 数据:DocRED dev 498 篇,冻结契约(三来源公共键集,gD/gR 金标,earliest-mention 分组);
  * 4-shot,示例 = 训练拆分中标题哈希与开发前 2 篇最近的 4 篇(确定性);
  * 对照(全部在**同一批 LLM 预测**上审计):
      L1 计分单位:A(hit-any)/ R(逐金标)/ F1;
      L2 换金标:Gap_F1(gD) − Gap_F1(gR),配对整文档 bootstrap 10000 次,种子 20260926;
      L3 提示契约:json 三元组 vs freetext 两种输出格式(探索);
  * 解析纪律:实体名精确匹配(strip 后,不折叠大小写)回连公共键集;候选全集之外的
    三元组计为契约违规,违规率 >5% 该设置标 unverifiable。

用法:
  python src/jcip_audit/d5_llm_hook.py build                    # 本地:构建提示(两格式)
  python src/jcip_audit/d5_llm_hook.py infer --model Qwen/Qwen2.5-7B-Instruct --format json
  python src/jcip_audit/d5_llm_hook.py audit --pred out/d5_llm_hook/preds_Qwen2.5-7B-Instruct_json.jsonl
                                                                 # 本地:解析+审计
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "jcip_audit"))

import eval_spec as S                                     # noqa: E402

DEV = ROOT / "data" / "docred_aligned" / "dev_revised.json"
TRAIN = ROOT / "data" / "docred_aligned" / "train_revised.json"
OLD_PREP = ROOT / "out" / "lre_rebuttal_v2" / "prepared"
REL_IDX = OLD_PREP / "relation_to_index.json"
OUT = ROOT / "out" / "d5_llm_hook"
PD_KEYS = ("n_cand", "n_pred", "n_gold", "tp", "fp", "fn", "n_gold_pairs", "hit", "exact", "fp_pair")
MAX_DEMO_SETS, MAX_TGT_SETS, MAX_DEMO_TRIPLES = 10, 24, 12
MAX_NEW_TOKENS, MAX_PROMPT_TOKENS, BATCH = 768, 7000, 2   # 修订 7:8GB 本地卡,batch 2 控 KV 显存


def canon(o) -> str:
    return json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def doc_id_of(d) -> str:
    return hashlib.sha256(canon({"title": str(d.get("title", "")).strip(),
                                 "sents": d["sents"]}).encode("utf-8")).hexdigest()


def sig_of(v) -> tuple:
    return tuple(sorted({(m["sent_id"], m["pos"][0], m["pos"][1]) for m in v}))


def build_canon_map():
    """三来源公共键集(与 x3_bridge_pred 逐字一致):{doc_id: (raw2canon, first_sent, sigs)}"""
    D = json.loads(DEV.read_text(encoding="utf-8"))
    R = json.loads((ROOT / "data" / "redocred_aligned" / "dev_revised.json").read_text(encoding="utf-8"))
    R2 = json.loads((ROOT / "data" / "re2_aligned" / "dev_revised.json").read_text(encoding="utf-8"))
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


def sents_text(sents, cap) -> list:
    return ["[S%d] %s" % (k, " ".join(s)) for k, s in enumerate(sents[:cap])]


def select_demos(train_docs, dev_titles, n_each=2):
    """冻结规则:对开发前 2 篇的标题,各取 sha256(train_title|dev_title) 最小的 2 篇训练文档。"""
    picked = []
    for dt in dev_titles[:2]:
        scored = sorted((hashlib.sha256((str(t.get("title", "")).strip() + "|" + dt)
                                        .encode("utf-8")).hexdigest(), k)
                        for k, t in enumerate(train_docs))
        picked += [k for _, k in scored[:n_each]]
    return picked


def fmt_doc_block(title, sents, ent_line, out_line, cap):
    """修订 8:实体一律以 E{i}=name 索引引用;out_line 为 None 表示目标文档。"""
    s = f"### {title}\n{ent_line}\nDocument:\n" + "\n".join(sents_text(sents, cap))
    if out_line:
        s += f"\n{out_line}"
    return s


def cmd_build(_):
    OUT.mkdir(parents=True, exist_ok=True)
    D0, per_doc, title_map = build_canon_map()
    train_docs = json.loads(TRAIN.read_text(encoding="utf-8"))
    rel2idx = {k: v for k, v in json.loads(REL_IDX.read_text(encoding="utf-8")).items()
               if k.upper() != "NA"}
    assert len(rel2idx) == 96, len(rel2idx)
    rel_line = "Relation list: " + ", ".join(sorted(rel2idx))
    demo_ids = select_demos(train_docs, [str(D0[0].get("title", "")).strip(),
                                         str(D0[1].get("title", "")).strip()])
    demos = []
    for k in demo_ids:
        td = train_docs[k]
        ents = sorted({v[0]["name"] for v in td["vertexSet"]})
        ix = {n: f"E{i}" for i, n in enumerate(ents)}
        tri = [[ix[td["vertexSet"][lb["h"]][0]["name"]], lb["r"],
                ix[td["vertexSet"][lb["t"]][0]["name"]]]
               for lb in td.get("labels", []) if lb["r"] in rel2idx]
        demos.append({"title": str(td.get("title", "")).strip(), "sents": td["sents"],
                      "ents": ents, "tri": tri[:MAX_DEMO_TRIPLES]})
    rep = {"demo_titles": [d["title"] for d in demos],
           "demo_rule": "sha256(train_title|dev_title) 最小的 2x2(开发前 2 篇)",
           "n_relations": len(rel2idx), "output_format": "entity-index (E{i})",
           "truncated_docs": 0, "docs": 0, "word_len": []}
    for fmt in ("json", "freetext"):
        with (OUT / f"prompts_{fmt}.jsonl").open("w", encoding="utf-8") as f:
            for dd in D0:
                did = doc_id_of(dd)
                fm = per_doc[did]
                ents = sorted({dd["vertexSet"][r][0]["name"] for r in fm["raw2canon"]})
                trunc = len(dd["sents"]) > MAX_TGT_SETS
                rep["truncated_docs"] += int(trunc)
                wc = sum(len(" ".join(s).split()) for s in dd["sents"][:MAX_TGT_SETS])
                rep["word_len"].append(wc)
                blocks = []
                for d in demos:
                    dl = "Entities: " + ", ".join(f"E{i}={n}" for i, n in enumerate(d["ents"]))
                    if fmt == "json":
                        ol = 'Relations (JSON): ' + canon({"triples": d["tri"]})
                    else:
                        ol = ("Relations: " + ("; ".join(f"{h} -> {r} -> {t}" for h, r, t in d["tri"])
                                               if d["tri"] else "(none)"))
                    blocks.append(fmt_doc_block(d["title"], d["sents"], dl, ol, MAX_DEMO_SETS))
                tl = "Entities: " + ", ".join(f"E{i}={n}" for i, n in enumerate(ents))
                tgt = fmt_doc_block(str(dd.get("title", "")).strip(), dd["sents"], tl, None, MAX_TGT_SETS)
                if fmt == "json":
                    instr = ('Output strict JSON only: {"triples": [["E0", "rel", "E5"], ...]}. '
                             "Every head and tail MUST be an entity ID (like \"E0\") whose name is in the "
                             "Entities list above. If unsure, omit the triple. Never use dates, quantities, "
                             "or any string that is not an entity ID from the list.")
                else:
                    instr = ("Output one line: 'Relations: E0 -> rel -> E5; ...' (or 'Relations: (none)'). "
                             "Every head and tail MUST be an entity ID from the Entities list above. "
                             "If unsure, omit that triple.")
                user = f"{rel_line}\n\n" + "\n\n".join(blocks) + f"\n\n{tgt}\n{instr}"
                f.write(canon({"did": did, "title": str(dd.get("title", "")).strip(),
                               "fmt": fmt, "truncated": trunc, "ents": ents, "prompt": user}) + "\n")
                rep["docs"] += 1
    rep["word_len"] = {"mean": sum(rep["word_len"]) / len(rep["word_len"]),
                       "max": max(rep["word_len"])}
    (OUT / "build_report.json").write_text(canon(rep), encoding="utf-8")
    print(f"[build] 498 docs × 2 格式 → {OUT};示例={rep['demo_titles']};截断 {rep['truncated_docs']} 篇;"
          f"输出格式=entity-index")


def load_model(name: str, load: str = "fp16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if load == "4bit":                       # 修订 7:8GB 显存本地执行,NF4 + greedy
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config=q,
                                                     device_map="auto", trust_remote_code=True)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16,
                                                         device_map="auto", trust_remote_code=True)
        except TypeError:                    # transformers 4.x 用 torch_dtype
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                                         device_map="auto", trust_remote_code=True)
    model.eval()
    return tok, model


def _build_constrained_fn(tok, rel_names, k_by_row):
    """修订 19:RegexParser 路线的约束解码(替代三版失败的手搓状态机)。

    每行按其实体表大小 k 与全局关系名集合构建正则,字符级解析器 + TokenEnforcer
    逐步放行合法 token;JSON 结构/索引上限/关系名/自对(头=尾同索引)全部编码进正则。
    """
    import torch
    # transformers 5.x 兼容 shim:PreTrainedTokenizerBase 移到 tokenization_utils_base,
    # lm-format-enforcer 0.11.3 仍从旧路径 import——先补别名再加载集成。
    import transformers.tokenization_utils_base as _tub
    import transformers.tokenization_utils as _tu
    if not hasattr(_tu, "PreTrainedTokenizerBase"):
        _tu.PreTrainedTokenizerBase = _tub.PreTrainedTokenizerBase
    from lmformatenforcer import RegexParser, TokenEnforcer
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data

    tok_data = build_token_enforcer_tokenizer_data(tok)
    parsers_by_k = {}

    rel_alt = "|".join(re.escape(r) for r in sorted(rel_names))

    def _parser_for(k):
        if k not in parsers_by_k:
            # 🔴 转义教学(2026-10-06 深夜定位):pattern 必须用 raw string——
            # 普通串 '\\[' 发给 intregex 的是"转义反斜杠+字符类",嵌套 [ 从未放行,
            # 而坏字符类恰好匹配扁平序列(此前"扁平 OK"的假象根源)。
            triple = r'"E[0-9]{1,2}", "(?:' + rel_alt + r')", "E[0-9]{1,2}"'
            pattern = r'\{"triples": \[' + triple + r'(, ' + triple + r')*\]'
            parsers_by_k[k] = RegexParser(pattern)
        return parsers_by_k[k]

    ens = {}

    def fn(batch_id, input_ids):
        k = k_by_row[batch_id]
        if k not in ens:
            ens[k] = TokenEnforcer(tok_data, _parser_for(k))
        tl = ens[k].get_allowed_tokens(input_ids.tolist())
        allowed = [t for t in range(tok.vocab_size) if tl.is_token_allowed(t)]
        if not allowed:
            allowed = [tok.eos_token_id]
        return torch.tensor(sorted(allowed), dtype=torch.long, device=input_ids.device)

    return fn


def cmd_infer(a):
    import torch
    fmt = a.format
    rows = [json.loads(l) for l in (OUT / f"prompts_{fmt}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if getattr(a, "limit", 0):
        rows = rows[:a.limit]
    tok, model = load_model(a.model, getattr(a, "load", "fp16"))
    tag = a.model.split("/")[-1]
    n_trunc_tok = 0
    constrain = getattr(a, "constrained", False)
    paf = None
    k_by_row = None
    if constrain:
        rel_names = sorted(k for k, v in json.loads(REL_IDX.read_text(encoding="utf-8")).items()
                           if k.upper() != "NA")
    with (OUT / f"preds_{tag}_{fmt}.jsonl").open("w", encoding="utf-8") as fo:
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            if constrain:
                ks = []
                for r in batch:
                    ms = re.findall(r'E(\d+)=', r["prompt"])
                    assert ms, f"prompt missing entity list: {r['title'][:30]}"
                    ks.append(max(int(x) + 1 for x in ms))
                k_by_row = ks
                paf = _build_constrained_fn(tok, rel_names, ks)
            texts = []
            for r in batch:
                msgs = [{"role": "user", "content": r["prompt"]}]
                t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                ids = tok(t, add_special_tokens=False)["input_ids"]
                if len(ids) > MAX_PROMPT_TOKENS:        # 超长:按记录截断目标句数,重试一次
                    n_trunc_tok += 1
                    t = tok.decode(ids[:MAX_PROMPT_TOKENS])
                    t = t.rsplit("[S", 1)[0] + "\n(Output truncated.)"
                texts.append(t)
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen_kw = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                              num_beams=1, pad_token_id=tok.pad_token_id)
                if constrain and fmt == "json":
                    gen_kw["prefix_allowed_tokens_fn"] = paf
                out = model.generate(**enc, **gen_kw)
            for ri, (r, inp, o) in enumerate(zip(batch, enc["input_ids"], out)):
                gen = o[len(inp):]
                rec = {"did": r["did"], "title": r["title"], "model": tag, "fmt": fmt,
                       "load": getattr(a, "load", "fp16"), "ents": r.get("ents", []),
                       "raw": tok.decode(gen, skip_special_tokens=True)}
                if constrain and fmt == "json":
                    # 修订 19:记录索引上界 k(约束口径)。审计判 oob 用 k 而非 ents
                    # 长度(多名同实体时 ents 长度 < 索引上限,会造成假 oob)
                    rec["constrained"] = True
                    rec["k_index"] = k_by_row[ri]
                fo.write(canon(rec) + "\n")
            print(f"[infer] {i + len(batch)}/{len(rows)}", flush=True)
    print(f"[infer] 完成;超 token 截断 {n_trunc_tok} 篇")


TRI_JSON = re.compile(r'\[\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*\]')
TRI_FREE = re.compile(r"([^;|\n]+?)\s+->\s+([^;|\n]+?)\s+->\s+([^;|\n]+)")


EID = re.compile(r"E\d+")


def _resolve(x, ents, k_bound=None):
    """修订 8:实体 ID(如 E3)→ 实体名;越界/格式错误返回 None(计违规)。
    修订 19:k_bound(约束版记录的索引上界)优先于 ents 长度——多名同实体时
    ents 长度 < 索引上限,按 ents 判会把合法高位索引误计为违规。"""
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
        # 修订 35:freetext 模板的标签前缀属合法输出(zh 提示规定 '关系: E0 -> …',
        # en 规定 'Relations: E0 -> …'),必须剥掉而非计违规;合规空输出 '关系: (无)'
        # 与 'Relations: (none)' 同样放行。此前 zh 臂每行首三元组被误计 oob、
        # 合规空答案被误计 parse_fail(云端 Oct-5 旧版则是静默丢弃首三元组——同错)。
        for lab in ("Relations:", "关系:"):
            if lab in h:
                h = h.split(lab)[-1].strip()
        hr, tr = _resolve(h, ents, k_bound), _resolve(t, ents, k_bound)
        if hr is None or tr is None:
            out.append(("<<VIOLATION>>",) * 3)
        else:
            out.append((hr, r, tr))
    if not out and ("(none)" in raw or "(无)" in raw):
        return []
    return out or ([("<<PARSE_FAIL>>",) * 3] if raw.strip() else [])


def cmd_audit(a):
    OUT.mkdir(parents=True, exist_ok=True)
    D, per_doc, _ = build_canon_map()
    rel2idx = {k: v for k, v in json.loads(REL_IDX.read_text(encoding="utf-8")).items()
               if k.upper() != "NA"}
    # 金标(关系名 → 审计索引;与 x3_bridge_pred 同源同法)
    golds = {}
    for tag in ("gD", "gR"):
        G = {}
        for line in (OLD_PREP / f"golds_{tag}.jsonl").read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            r = json.loads(line)
            G[(r["doc_id"], r["h"], r["t"])] = {rel2idx[nm] for nm in r["rels"] if nm in rel2idx}
        golds[tag] = G
        print(f"[gold] {tag}: {len(G):,} 对")
    # 实体名 → canon(精确匹配;多名歧义计违规)
    name2canon = {}
    for dd in D:
        did = doc_id_of(dd)
        m = defaultdict(set)
        for raw, c in per_doc[did]["raw2canon"].items():
            m[str(dd["vertexSet"][raw][0]["name"]).strip()].add(c)
        name2canon[did] = m
    # 读预测
    preds = defaultdict(set)                                # (did,hc,tc) -> {rel_idx}
    stats = {"n_triples": 0, "parse_fail": 0, "oob_index": 0, "hallucinated": 0,
             "ambiguous": 0, "self_pair": 0, "rel_unmapped": 0, "mapped": 0, "docs_missing": 0}
    for line in Path(a.pred).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        did = r["did"]
        if did not in per_doc:
            stats["docs_missing"] += 1
            continue
        m = name2canon[did]
        ents = r.get("ents") or sorted(m)                    # 修订 8:E 下标 → 名称
        # 修订 19:约束版记有 k_index(权威索引上界);无约束版 ents 长度口径不变。
        # 约束版 oob 理论为 0(掩码保证),parse_output 的 ents 长度判定只会把
        # 合法高位索引误判为 VIOLATION——对约束版放行高位索引校验:
        k_bound = r.get("k_index")
        for h, rl, t in parse_output(r["raw"], r["fmt"], ents,
                                     k_bound=k_bound):
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
            preds[(did, hc, tc)].add(ri)
            stats["mapped"] += 1
    viol = stats["n_triples"] - stats["mapped"]
    viol_rate = viol / stats["n_triples"] if stats["n_triples"] else 0.0
    # 全候选总体审计(189,556)
    import numpy as np
    doc_ids = sorted(per_doc)
    pos = {d: k for k, d in enumerate(doc_ids)}
    per_gold = {}
    for tag in ("gD", "gR"):
        G = golds[tag]
        cnt = np.zeros((len(doc_ids), 2, 3), dtype=np.int64)      # (doc, intra/inter, tp/fp/fn)
        aux = np.zeros((len(doc_ids), 2, 3), dtype=np.int64)      # n_pred / has_gold / n_pred*has_gold?
        pairo = np.zeros((len(doc_ids), 2, 3), dtype=np.int64)    # hit / fp_pair / n_gold_pairs
        for did in doc_ids:
            k = pos[did]
            fm = per_doc[did]
            m = len(fm["sigs"])
            first = fm["first_sent"]
            for h in range(m):
                for t in range(m):
                    if h == t:
                        continue
                    key = (did, h, t)
                    P = preds.get(key, set())
                    Gd = G.get(key, set())
                    g = 0 if first[h] == first[t] else 1
                    tp, fp, fn = len(P & Gd), len(P - Gd), len(Gd - P)
                    has = 1 if Gd else 0
                    c = cnt[k, g]
                    c[0] += tp; c[1] += fp; c[2] += fn
                    aux[k, g, 0] += len(P); aux[k, g, 1] += has; aux[k, g, 2] += (len(P) > 0 and not has)
                    if has:
                        pairo[k, g, 0] += 1 if tp > 0 else 0
                        pairo[k, g, 2] += 1
                    elif len(P) > 0:
                        pairo[k, g, 1] += 1
        per_gold[tag] = {"cnt": cnt, "aux": aux, "pairo": pairo}

    def metrics(pg):
        out = {}
        for g, gn in ((0, "intra"), (1, "inter")):
            tp, fp, fn = pg["cnt"][:, g, 0].sum(), pg["cnt"][:, g, 1].sum(), pg["cnt"][:, g, 2].sum()
            n_pred, n_gold = pg["aux"][:, g, 0].sum(), pg["cnt"][:, g, 0].sum() + pg["cnt"][:, g, 2].sum()
            hit, fp_pair, n_gold_p = pg["pairo"][:, g, 0].sum(), pg["pairo"][:, g, 1].sum(), pg["pairo"][:, g, 2].sum()
            F1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            R = tp / n_gold if n_gold else 0.0
            A = hit / n_gold_p if n_gold_p else 0.0
            pF1_tp, pF1_fp, pF1_fn = hit, fp_pair, n_gold_p - hit
            pF1 = 2 * pF1_tp / (2 * pF1_tp + pF1_fp + pF1_fn) if (2 * pF1_tp + pF1_fp + pF1_fn) else 0.0
            out[gn] = {"F1": F1, "R": R, "A": A, "pair_F1": pF1}
        out["gap_F1"] = out["intra"]["F1"] - out["inter"]["F1"]
        out["gap_R"] = out["intra"]["R"] - out["inter"]["R"]
        out["gap_A"] = out["intra"]["A"] - out["inter"]["A"]
        out["gap_pair_F1"] = out["intra"]["pair_F1"] - out["inter"]["pair_F1"]
        return out

    met = {tag: metrics(per_gold[tag]) for tag in ("gD", "gR")}
    # L2 配对整文档 bootstrap:Gap(gD) − Gap(gR)
    rng = np.random.default_rng(20260926)
    N = len(doc_ids)
    deltas = np.empty(10000)
    for b in range(10000):
        idx = rng.integers(0, N, N)
        gs = {}
        for tag in ("gD", "gR"):
            f = []
            for g in (0, 1):
                tp = per_gold[tag]["cnt"][idx, g, 0].sum()
                fp = per_gold[tag]["cnt"][idx, g, 1].sum()
                fn = per_gold[tag]["cnt"][idx, g, 2].sum()
                f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
            gs[tag] = f[0] - f[1]
        deltas[b] = gs["gD"] - gs["gR"]
    point = met["gD"]["gap_F1"] - met["gR"]["gap_F1"]
    ci = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))
    rep = {"pred": a.pred, "stats": stats, "violation_rate": viol_rate,
           "unverifiable": bool(viol_rate > 0.05), "n_candidates_expected": 189556,
           "metrics": met, "L2_delta_gapF1": {"point": point, "ci95": ci},
           "L1_note": "gap_F1/gap_R/gap_A/gap_pair_F1 为同预测四计分读数(L1)"}
    (OUT / f"audit_{Path(a.pred).stem}.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("stats", "violation_rate", "unverifiable")}, ensure_ascii=False))
    for tag in ("gD", "gR"):
        m = met[tag]
        print(f"[{tag}] gap_F1={m['gap_F1']:+.4f} gap_R={m['gap_R']:+.4f} "
              f"gap_A={m['gap_A']:+.4f} gap_pair_F1={m['gap_pair_F1']:+.4f}")
    print(f"[L2] Δgap_F1 = {point:+.4f}  CI95 {ci[0]:+.4f}..{ci[1]:+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    fi = sub.add_parser("infer")
    fi.add_argument("--model", required=True)
    fi.add_argument("--format", choices=("json", "freetext"), required=True)
    fi.add_argument("--limit", type=int, default=0)
    fi.add_argument("--load", choices=("fp16", "4bit"), default="fp16")
    fi.add_argument("--constrained", action="store_true",
                    help="修订 19:三重掩码约束解码(仅 json 格式)")
    fa = sub.add_parser("audit")
    fa.add_argument("--pred", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "infer": cmd_infer, "audit": cmd_audit}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
