# -*- coding: utf-8 -*-
"""phase0_build_multi.py — 路径B: ar/cs/hi 三语言的 Phase 0.

自含 build_pair 副本(唯一差异: doc 分组按语言定制——
  ar: sid '/' 前 = 真实文档; cs: '/' 前+段落段; hi: v1.4 sid 首段 = 源文档域 a_tree..);
其余与 EN/ZH 完全同构(同一映射表 M、同一对齐规则、同一产物格式).
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase0_build as p0  # noqa: E402
from phase0_build import map_deprel, map_upos  # noqa: E402

P = HERE.parent / "ud" / "pathB"
V14 = P / "v14" / "ud-treebanks-v1.4"
V20 = P / "v20" / "ud-treebanks-v2.0"


def parse_robust(path: Path):
    """兼容 '# sent_id = x' 与 '# sent_id x'."""
    sents, cur, sid = [], [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# sent_id"):
            seg = line.split("sent_id", 1)[1].lstrip()
            sid = seg[1:].strip() if seg.startswith("=") else seg.strip()
            continue
        if line.startswith("#"):
            continue
        if not line.strip():
            if cur:
                sents.append((sid, cur))
                cur, sid = [], None
            continue
        c = line.split("\t")
        if "-" in c[0] or "." in c[0]:
            continue
        cur.append((c[1], c[3], c[7], int(c[6])))
    if cur:
        sents.append((sid, cur))
    return sents


def doc_for(lang, sid_a, sid_b):
    if lang == "ar":
        return sid_b.split("/")[0]
    if lang == "cs":
        return sid_b.split("/")[0].rsplit("-", 1)[0]
    if lang == "hi":
        return sid_a.split("-")[0]          # a_tree/b_tree/... 源文档域
    return sid_b.rsplit("-", 1)[0]


def build_pair_multi(name, pa, pb, lang):
    sa, sb = parse_robust(pa), parse_robust(pb)

    def k(toks):
        forms = [t[0] for t in toks]
        return "".join(forms).lower() if lang == "zh" else " ".join(forms).lower()

    ba, bb = {}, {}
    for sid, toks in sa:
        ba.setdefault(k(toks), []).append((sid, toks))
    for sid, toks in sb:
        bb.setdefault(k(toks), []).append((sid, toks))

    out, stats = [], Counter()
    for key_, lista in ba.items():
        listb = bb.get(key_)
        if not listb:
            continue
        for (sid_a, tx), (sid_b, ty) in zip(lista, listb):
            if len(tx) != len(ty):
                stats["drift"] += 1
                continue
            case_child = set()
            for i, (_, d, _, h) in enumerate(tx):
                if d.split(":")[0] == "case":
                    case_child.add(h - 1)
            rows, ok = [], True
            for i, (a, b) in enumerate(zip(tx, ty)):
                fa, ua, da, ha = a
                fb, ub, db, hb = b
                u1m = map_upos(ua)
                d1m = map_deprel(da, i in case_child)
                mappable = (u1m is not None) and (d1m is not None) and \
                           (ua != "_") and (da != "_")
                ok &= mappable
                stats["tokens"] += 1
                stats["mappable"] += mappable
                if ua != ub:
                    stats["upos_diff"] += 1
                if da != db or ha != hb:
                    stats["dep_or_head_diff"] += 1
                rows.append({"f": fa, "u2": ub, "u1m": u1m if mappable else None,
                             "d2": db, "d1m": d1m if mappable else None,
                             "h2": hb, "h1": ha,
                             "ren_u": ua != ub, "ren_d": da != db})
            out.append({"sid": sid_b, "doc": doc_for(lang, sid_a, sid_b),
                        "mappable_all": ok, "toks": rows})
    stats["sentences"] = len(out)
    return out, dict(stats)


PAIRS = [
    ("ar_dev", V14 / "UD_Arabic/ar-ud-dev.conllu",
     V20 / "UD_Arabic/ar-ud-dev.conllu", "ar"),
    ("cs_dev", V14 / "UD_Czech/cs-ud-dev.conllu",
     V20 / "UD_Czech/cs-ud-dev.conllu", "cs"),
    ("hi_dev", V14 / "UD_Hindi/hi-ud-dev.conllu",
     V20 / "UD_Hindi/hi-ud-dev.conllu", "hi"),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    (HERE / "align").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    udmap = {"upos_ren": {"PRT": "PART", "CONJ": "CCONJ"},
             "deprel_ren": {"dobj": "obj", "neg": "advmod",
                            "auxpass": "aux:pass", "nsubjpass": "nsubj:pass",
                            "mwe": "fixed", "name": "flat", "remnant": "orphan",
                            "poss": "nmod:poss", "nmod:tmod": "obl:tmod",
                            "nmod:npmod": "obl:npmod", "det:predet": "det",
                            "vmod": "advcl"},
             "context_rules": ["nmod->obl iff case child else nmod",
                               "dative->obl iff case child else iobj"],
             "source": "UD v1.4->v2.0 official migration; doc-driven, no fitting; "
                       "same map as EN/ZH suite"}
    (HERE / "udmap_pathb.json").write_text(json.dumps(udmap, ensure_ascii=False,
                                                      indent=1), encoding="utf-8")
    docs_report = {}
    for name, pa, pb, lang in PAIRS:
        rows, stats = build_pair_multi(name, pa, pb, lang)
        f = HERE / "align" / f"{name}.jsonl"
        with f.open("w", encoding="utf-8") as fo:
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        ndocs = len({r["doc"] for r in rows})
        docs_report[name] = {"sentences": stats["sentences"], "documents": ndocs,
                             "drift": stats.get("drift", 0),
                             "mappable_pct": round(100 * stats["mappable"] /
                                                   max(1, stats["tokens"]), 2),
                             "upos_diff_pct": round(100 * stats.get("upos_diff", 0) /
                                                    max(1, stats["tokens"]), 2),
                             "dep_or_head_diff_pct": round(100 * stats.get("dep_or_head_diff", 0) /
                                                           max(1, stats["tokens"]), 2)}
        print(name, docs_report[name])
    manifest = {"udmap_pathb.json": sha(HERE / "udmap_pathb.json"),
                "align": {f"{n}.jsonl": sha(HERE / "align" / f"{n}.jsonl")
                          for n, *_ in PAIRS},
                "diagnostics": docs_report,
                "note": "Path B (rev113): ar/cs/hi replication of the UD gold-axis "
                        "mechanism suite; v1.4->v2.0, official migration map only"}
    (HERE / "results" / "manifest_pathb.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[ok] Path B Phase 0 -> results/manifest_pathb.json")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
