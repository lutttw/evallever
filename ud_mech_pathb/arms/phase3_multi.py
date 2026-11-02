# -*- coding: utf-8 -*-
"""phase3_multi.py — 路径B C 臂: stanza 公共解析器 cs/hi/ar 三语言冻结预测.

协议镜像 phase3_stanza.py(预分词 tokenize_pretokenized, 输出逐句 upos/head/deprel);
模型为各语言默认主树库包 nocharlm(经 hf-mirror 下载),纯 CPU(CUDA 屏蔽).
产物: arms/stanza_C_{lang}.json (与 stanza_C_en.json 同构).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["STANZA_RESOURCES_DIR"] = str(HERE / "stanza_models")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ARMS = HERE / "arms"
LANGS = ["cs", "hi", "ar"]
FILES = {  # align -> 输出名
    "cs": ("cs_dev", "stanza_C_cs"),
    "hi": ("hi_dev", "stanza_C_hi"),
    "ar": ("ar_dev", "stanza_C_ar"),
}
M = {"cs": "pdt_nocharlm", "hi": "hdtb_nocharlm", "ar": "padt_nocharlm"}


def run(lang):
    from stanza import Pipeline
    sm = HERE / "stanza_models" / lang
    t0 = time.time()
    nlp = Pipeline(lang=lang, processors="tokenize,lemma,pos,depparse",
                   download_method=None, tokenize_pretokenized=True,
                   use_gpu=False, logging_level="WARN",
                   tokenize_model_path=str(sm / "tokenize" / f"{M[lang]}.pt"),
                   pos_model_path=str(sm / "pos" / f"{M[lang]}.pt"),
                   lemma_model_path=str(sm / "lemma" / f"{M[lang]}.pt"),
                   depparse_model_path=str(sm / "depparse" / f"{M[lang]}.pt"),
                   pretrain_path=str(sm / "pretrain" / "conll17.pt"))
    print(f"[C:{lang}] pipeline loaded {time.time()-t0:.0f}s", flush=True)
    align, out_name = FILES[lang]
    sents = [json.loads(l) for l in
             (HERE / "align" / f"{align}.jsonl").open(encoding="utf-8")]
    docs = [[t["f"] for t in s["toks"]] for s in sents]
    pred = nlp(docs)
    recs = []
    skip = 0
    for s, d in zip(sents, pred.sentences):
        toks = list(d.tokens)
        if len(toks) != len(s["toks"]):
            skip += 1
            continue
        words = [w for t in toks for w in t.words]
        recs.append({"sid": s["sid"], "doc": s["doc"],
                     "upos": [w.upos for w in words],
                     "head": [w.head for w in words],
                     "deprel": [w.deprel for w in words]})
    (ARMS / f"{out_name}.json").write_text(json.dumps(recs, ensure_ascii=False),
                                           encoding="utf-8")
    print(f"[C:{lang}] {len(recs)} sentences (skip {skip}), "
          f"{time.time()-t0:.0f}s total", flush=True)


def main():
    ARMS.mkdir(exist_ok=True)
    for lang in LANGS:
        out = ARMS / f"stanza_C_{lang}.json"
        if out.exists() and "--force" not in sys.argv:
            print(f"[C:{lang}] exists, skip")
            continue
        run(lang)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
