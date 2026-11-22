# -*- coding: utf-8 -*-
"""phase2_multi.py — 路径B B/B′ 族:mBERT 冻结特征 + 线性探针,15 种子 × ar/cs/hi.

B  训练于 v2.0 train 金标;B′ 训练契约臂:v1.4 train 标签经 M 映射.
优化:v1/v2 train 的 token 流已证 100% 孪生——若顺序亦逐位一致,特征只抽一次
(B′ 复用 X,仅换标签数组);顺序不一致则退回分别抽取.
全部 CPU(特征提取 + 线性探针),不占 GPU 主链(作业16 独占 GPU).
"""
from __future__ import annotations

import os
_HERE_F = __import__("pathlib").Path(__file__).resolve()
ROOT = _HERE_F.parents[3]
os.environ["HF_HUB_CACHE"] = str(ROOT / ".hf_cache" / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = _HERE_F.parent
sys.path.insert(0, str(HERE))
import phase2_bert as p2  # noqa: E402

UD = HERE.parent / "ud"
PB = UD / "pathB"
V20 = PB / "v20" / "ud-treebanks-v2.0"
V14 = PB / "v14" / "ud-treebanks-v1.4"
ARMS = HERE / "arms"
MB = "google-bert/bert-base-multilingual-cased"
LANGS = ["ar", "cs", "hi"]
MODELS = {lang: MB for lang in LANGS}
TRAIN = {("B", lg): (V20 / f"{td}/{lg}-ud-train.conllu", False)
         for lg, td in [("ar", "UD_Arabic"), ("cs", "UD_Czech"), ("hi", "UD_Hindi")]}
TRAIN.update({("B'", lg): (V14 / f"{td}/{lg}-ud-train.conllu", True)
              for lg, td in [("ar", "UD_Arabic"), ("cs", "UD_Czech"), ("hi", "UD_Hindi")]})


def load_align_multi(lang):
    sents = []
    for ln in (HERE / "align" / f"{lang}_dev.jsonl").open(encoding="utf-8"):
        r = json.loads(ln)
        sents.append({"sid": r["sid"], "doc": r["doc"],
                      "u2": [p2.T2I[t["u2"]] for t in r["toks"]],
                      "u1m": [p2.T2I[t["u1m"]] if t["u1m"] else None
                              for t in r["toks"]]})
    return sents


def forms_order_identity(path_a, path_b):
    """逐句逐 token 检查两 train 文件的 form 流是否同序同形."""
    def forms(path):
        out = []
        toks = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                continue
            if not line.strip():
                if toks:
                    out.append(toks)
                    toks = []
                continue
            c = line.split("\t")
            if "-" in c[0] or "." in c[0]:
                continue
            toks.append(c[1].lower())
        if toks:
            out.append(toks)
        return out
    fa, fb = forms(path_a), forms(path_b)
    if len(fa) != len(fb):
        return False
    for sa, sb in zip(fa, fb):
        if sa != sb:
            return False
    return True


def main():
    t0 = time.time()
    ARMS.mkdir(exist_ok=True)
    align = {lang: load_align_multi(lang) for lang in LANGS}
    for lang in LANGS:
        name_b, name_bp = f"B_{lang}", f"Bp_{lang}"
        tp_b, map_b = TRAIN[("B", lang)]
        tp_bp, map_bp = TRAIN[("B'", lang)]
        tr_b = p2.read_conllu(tp_b, map_b)
        ident = forms_order_identity(tp_b, tp_bp)
        print(f"[multi {lang}] train order-identity: {ident}", flush=True)
        # 测试特征
        tag_test = f"test_{lang}"
        f_test = p2.FEATS / f"feats_{tag_test}_{MB.split('/')[-1]}.npz"
        if not f_test.exists():
            test_sents = []
            for ln in (HERE / "align" / f"{lang}_dev.jsonl").open(encoding="utf-8"):
                r = json.loads(ln)
                test_sents.append(([t["f"] for t in r["toks"]],
                                   [0] * len(r["toks"])))
            print(f"[multi] extracting test features {lang}", flush=True)
            p2.extract(MB, test_sents, tag_test)
        # B 特征(v2 标签)
        Xt, Yt, lt, _ = p2.extract(MB, tr_b, name_b)
        FX, FY, _ = p2.flatten(Xt, Yt, lt)
        # B′ 特征:同 token 流(已证孪生),仅换标签
        tr_bp = p2.read_conllu(tp_bp, map_bp)
        if ident and len(tr_bp) == len(tr_b):
            same = all(len(a[1]) == len(b[1]) for a, b in zip(tr_b, tr_bp))
        else:
            same = False
        if same:
            Ybp = np.full_like(Yt, -1)
            for i, (sents_b, sents_bp) in enumerate(zip(tr_b, tr_bp)):
                L = len(sents_b[1])
                Ybp[i, :L] = np.array(sents_bp[1][:L], dtype=np.int16)
            FX_bp, FY_bp = FX, torch.tensor(
                np.concatenate([Ybp[i, :int(lt[i])] for i in range(len(lt))])).long()
            print(f"[multi {lang}] B′ reuses B features (labels swapped)", flush=True)
        else:
            Xt2, Yt2, lt2, _ = p2.extract(MB, tr_bp, name_bp)
            FX_bp, FY_bp, _ = p2.flatten(Xt2, Yt2, lt2)
        # 测试张量(kept 掩码对齐)
        d = np.load(f_test)
        Xte, lte = d["X"], d["lens"]
        kte = d["kept"] if "kept" in d.files else np.tile(
            np.arange(Xte.shape[1], dtype=np.int32), (Xte.shape[0], 1))
        tes = []
        for si, s in enumerate(align[lang]):
            L = int(lte[si])
            kept = [k for k in kte[si][:L] if 0 <= k < len(s["u2"])]
            tes.append({"sid": s["sid"], "doc": s["doc"],
                        "X": torch.tensor(Xte[si, :L], dtype=torch.float32),
                        "u2": [s["u2"][k] for k in kept],
                        "u1m": [s["u1m"][k] for k in kept]})
        for seed in p2.SEEDS:
            print(f"[multi {time.time()-t0:6.0f}s] "
                  + p2.probe_once("B", lang, seed, FX, FY, None, tes), flush=True)
            print(f"[multi {time.time()-t0:6.0f}s] "
                  + p2.probe_once("B'", lang, seed, FX_bp, FY_bp, None, tes),
                  flush=True)
        del FX, FY, FX_bp, FY_bp
        print(f"[multi] {lang} done {time.time()-t0:.0f}s", flush=True)
    print(f"[multi] ALL DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
