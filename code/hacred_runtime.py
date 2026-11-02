# -*- coding: utf-8 -*-
"""hacred_runtime.py —— X4（HaCRED 中文，单来源）训练运行时。

★ 与 X2 的关系：**完全不改 `core_runtime.py`**（X2 路径保持逐字不变）。
   本文件复用其与数据集无关的部分（`build_input` / `full_state_hash` / `tuple_tree`），
   并针对 X4 的三处差异实现：

   ① **单来源索引空间**：HaCRED 只有一套标注 ⇒ canonical 下标 = `vertexSet` 下标，
      无跨来源签名求交（X2 的 per-source manifest / filter_units 均不适用）。
   ② **tokenizer 标记注册**（AMENDMENT 05）：`bert-base-chinese` 的基本切分会在 `[`/`]` 处断开
      `[unused1]`，必须注册为 additional_special_tokens ⇒ 各 1 token、词表不变、无需 resize。
   ③ **epoch 式训练**（方案 §10 与 X2 的停止规则不同）：
      每个 epoch 遍历**完整训练集合**，保留最后不足 8 条的**尾批**并按实际样本数归一 loss，
      **不跨 epoch 拼接**；样本暴露量 = 3·N_H；更新数 B_H = 3·⌈N_H/8⌉（不套用英文 45,462）。

用法（云端）：
  python hacred_runtime.py train --run-id hacred_s20260917 --device cuda \
      --effective-batch 8 --max-updates 20559 [--resume] [--limit-units 40]
  python hacred_runtime.py plumbing --run-id hacred_s20260917     # ★ 不依赖 torch 的数据流自检
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "jcip_audit"))

from core_runtime import build_input, full_state_hash, tuple_tree   # noqa: E402  复用（数据集无关）

PREP = ROOT / "out" / "jcip_audit_v1" / "hacred_prepared"
DATA = ROOT / "data" / "hacred"
MODEL = ROOT / "models" / "bert-base-chinese"
OUT = ROOT / "out" / "jcip_audit_v1"
MARKERS = ("[unused1]", "[unused2]", "[unused3]", "[unused4]")
EXPECTED_UPDATES = 20_559          # B_H（3·⌈54,820/8⌉）；用于 report 的 smoke/正式判定
EPOCHS = 3
MAXLEN = 512


def canon(o):
    return json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def load_jsonl(p: Path):
    """★ 必须 split("\\n")，不可用 splitlines()（HaCRED 正文含 U+2028）。"""
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]


def make_tokenizer():
    """② AMENDMENT 05：注册 4 个标记（已在词表内 ⇒ 无需 resize）。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    v0 = len(tok)
    tok.add_special_tokens({"additional_special_tokens": list(MARKERS)})
    lens = {m: len(tok(m)["input_ids"][1:-1]) for m in MARKERS}
    if any(v != 1 for v in lens.values()) or len(tok) != v0:
        raise SystemExit(f"[tokenizer] 标记注册失败：{lens}，词表 {v0}→{len(tok)}")
    return tok, {"vocab_before": v0, "vocab_after": len(tok), "marker_token_lens": lens}


def signatures_single_source(recs: list) -> dict:
    """① 单来源索引空间：下标 = vertexSet 下标，签名 = 该实体全部提及的 (sent_id,start,end)。"""
    out = {}
    for rec in recs:
        did = hashlib.sha256(canon({"title": str(rec.get("title", "")).strip(),
                                    "sents": rec["sents"]}).encode("utf-8")).hexdigest()
        out[did] = [tuple(sorted((m["sent_id"], m["pos"][0], m["pos"][1]) for m in ent))
                    for ent in rec["vertexSet"]]
    return out


def docs_by_id(recs: list) -> dict:
    return {hashlib.sha256(canon({"title": str(r.get("title", "")).strip(),
                                 "sents": r["sents"]}).encode("utf-8")).hexdigest(): r for r in recs}


def load_units_for_seed(seed: str) -> tuple[list, dict]:
    """训练单元 = 全部唯一正实体对 + floor(ρ·Q_pos) 条确定性未标注样本（方案 §10）。"""
    golds = load_jsonl(PREP / "golds_train.jsonl")
    pos = [{"doc_id": g["doc_id"], "h": g["h"], "t": g["t"], "is_positive": True} for g in golds]
    unl_path = PREP / "sampling" / f"seed_{seed}.jsonl"
    unl = [{"doc_id": r["doc_id"], "h": r["h"], "t": r["t"], "is_positive": False}
           for r in load_jsonl(unl_path)]
    units = pos + unl
    # 规范序（保证跨 seed 可复现），epoch 内再洗牌
    units.sort(key=lambda u: (u["doc_id"], u["h"], u["t"]))
    meta = {"Q_positive": len(pos), "n_unlabeled": len(unl), "N_H": len(units),
            "units_sha256": hashlib.sha256(canon(units).encode()).hexdigest(),
            "sampling_sha256": hashlib.sha256(unl_path.read_bytes()).hexdigest()}
    return units, meta


def epoch_batches(n: int, bs: int, seeds_iter):
    """③ epoch 式批次计划：每 epoch 洗牌后顺序切批，**尾批保留实际大小**、不跨 epoch 拼接。"""
    plan = []
    for epoch, rng in enumerate(seeds_iter):
        order = list(range(n))
        rng.shuffle(order)
        batches = [order[i:i + bs] for i in range(0, n, bs)]
        plan.append({"epoch": epoch, "n_batches": len(batches),
                     "tail_size": len(batches[-1]) if batches else 0,
                     "order_sha256": hashlib.sha256((",".join(map(str, order))).encode()).hexdigest()})
        yield epoch, batches
    # 供 plumbing 只用
    return plan


def cmd_plumbing(args) -> int:
    """不依赖 torch 的数据流自检：单来源签名 / 单元与预算 / 批次计划 / 标记与输入构造。"""
    seed = args.run_id.rsplit("s", 1)[1]
    recs = read_json(DATA / "train_revised.json")
    docs = docs_by_id(recs)
    sigs = signatures_single_source(recs)
    units, meta = load_units_for_seed(seed)
    rel_idx = read_json(PREP / "relation_to_index.json")
    golds = {(g["doc_id"], g["h"], g["t"]): g["rels"] for g in load_jsonl(PREP / "golds_train.jsonl")}

    rngs = [random.Random(int(seed) + e) for e in range(EPOCHS)]
    plan = []
    total_updates = 0
    for epoch, batches in epoch_batches(len(units), args.effective_batch, rngs):
        total_updates += len(batches)
        plan.append({"epoch": epoch, "n_batches": len(batches), "tail_size": len(batches[-1])})
    # 逐批检查：单元可解析、金标可查、build_input 可构造
    rngs = [random.Random(int(seed) + e) for e in range(EPOCHS)]
    checked = 0
    bad = 0
    for epoch, batches in epoch_batches(len(units), args.effective_batch, rngs):
        for idxs in batches[:2] + batches[-1:]:
            for i in idxs:
                u = units[i]
                if u["doc_id"] not in docs or u["doc_id"] not in sigs:
                    bad += 1
                    continue
                sig = sigs[u["doc_id"]]
                if not (0 <= u["h"] < len(sig) and 0 <= u["t"] < len(sig)):
                    bad += 1
                    continue
                if not sig[u["h"]] or not sig[u["t"]]:
                    bad += 1
                    continue
                txt = build_input(docs[u["doc_id"]]["sents"], sig, u["h"], u["t"])
                if not txt or txt.count(MARKERS[0]) < 1 or txt.count(MARKERS[2]) < 1:
                    bad += 1
                rels = golds.get((u["doc_id"], u["h"], u["t"]))
                if u["is_positive"] and not rels:
                    bad += 1                      # 正例必须有金标
                if (not u["is_positive"]) and rels:
                    bad += 1                      # 未标注不得带金标
                checked += 1
    tok, tok_meta = make_tokenizer()
    sample = units[0]
    text = build_input(docs[sample["doc_id"]]["sents"], sigs[sample["doc_id"]], sample["h"], sample["t"])
    rep = {"mode": "plumbing", "run_id": args.run_id, "seed": seed,
           "units": meta, "expected_B_H": EPOCHS * ((meta["N_H"] + args.effective_batch - 1) // args.effective_batch),
           "computed_updates_3ep": total_updates, "epoch_plan": plan,
           "batch_checked": checked, "batch_bad": bad,
           "tokenizer": tok_meta, "marker_counts_in_sample": {m: text.count(m) for m in MARKERS},
           "sample_token_len_no_trunc": len(tok(text, truncation=False)["input_ids"]),
           "n_relations": len(rel_idx)}
    rep["MATCH_EXPECTED"] = (rep["expected_B_H"] == EXPECTED_UPDATES == total_updates)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    outdir = OUT / "runs"; outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{args.run_id}.plumbing.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
    return 0 if (bad == 0 and rep["MATCH_EXPECTED"]) else 1


def train(args) -> int:
    import numpy as np
    import torch

    seed = args.run_id.rsplit("s", 1)[1]
    rel_idx = read_json(PREP / "relation_to_index.json")
    recs = read_json(DATA / "train_revised.json")
    docs = docs_by_id(recs)
    sigs = signatures_single_source(recs)
    units, meta = load_units_for_seed(seed)
    golds = {(g["doc_id"], g["h"], g["t"]): g["rels"] for g in load_jsonl(PREP / "golds_train.jsonl")}
    if args.limit_units:
        units = units[:args.limit_units]
        meta["limited_to"] = args.limit_units
    if len(units) < args.effective_batch:
        raise SystemExit("[x4] 单元数不足一个有效 batch")

    torch.manual_seed(int(seed)); np.random.seed(int(seed) % (2**32)); random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    tokenizer, tok_meta = make_tokenizer()
    from transformers import AutoModel
    encoder = AutoModel.from_pretrained(str(MODEL), local_files_only=True)
    head = torch.nn.Sequential(torch.nn.Dropout(0.1),
                               torch.nn.Linear(encoder.config.hidden_size, len(rel_idx)))
    device = torch.device(args.device)
    encoder.to(device); head.to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()),
                                  lr=2e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")     # 尾批按**实际样本数**归一
    run_dir = OUT / "runs" / (args.artifact_id or args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_path = run_dir / "checkpoint_resume.pt"
    step = 0
    initial_hash = full_state_hash(encoder, head)
    losses, tails = [], []

    epoch_rngs = [random.Random(int(seed) + e) for e in range(EPOCHS)]
    state = {"epoch": 0, "cursor_in_epoch": 0}
    if args.resume and resume_path.exists():
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(st["encoder"]); head.load_state_dict(st["head"])
        optimizer.load_state_dict(st["optimizer"])
        torch.set_rng_state(st["torch_rng"]); np.random.set_state(st["numpy_rng"])
        random.setstate(tuple_tree(st["python_rng"]))
        if torch.cuda.is_available() and st["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        step = st["step"]
        state = {"epoch": st["epoch"], "cursor_in_epoch": st["cursor_in_epoch"]}
        epoch_rngs = [random.Random(int(seed) + e) for e in range(EPOCHS)]
        for e in range(state["epoch"]):          # 重放已过 epoch 的洗牌以保持后续顺序一致
            order = list(range(len(units))); epoch_rngs[e].shuffle(order)
        print(f"[x4] resume from step={step} epoch={state['epoch']} cursor={state['cursor_in_epoch']}")

    # 逐 epoch、逐批；尾批不补齐、不跨 epoch 拼接
    done = False
    for epoch in range(state["epoch"], EPOCHS):
        order = list(range(len(units))); epoch_rngs[epoch].shuffle(order)
        start = state["cursor_in_epoch"] if epoch == state["epoch"] else 0
        for b in range(start, (len(order) + args.effective_batch - 1) // args.effective_batch):
            if step >= args.max_updates:
                done = True
                break
            idxs = order[b * args.effective_batch:(b + 1) * args.effective_batch]
            tails.append(len(idxs))
            texts, targets = [], []
            for i in idxs:
                u = units[i]
                texts.append(build_input(docs[u["doc_id"]]["sents"], sigs[u["doc_id"]], u["h"], u["t"]))
                tgt = [0.0] * len(rel_idx)
                for r in golds.get((u["doc_id"], u["h"], u["t"]), []):
                    tgt[r] = 1.0
                targets.append(tgt)
            enc = tokenizer(texts, truncation=True, max_length=MAXLEN, padding=True,
                            return_tensors="pt").to(device)
            tt = torch.tensor(targets, dtype=torch.float32, device=device)
            encoder.train(); head.train(); optimizer.zero_grad(set_to_none=True)
            logits = head(encoder(**enc).last_hidden_state[:, 0])
            loss = loss_fn(logits, tt); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.0)
            optimizer.step(); step += 1; losses.append(float(loss.detach().cpu()))
            state["epoch"] = epoch; state["cursor_in_epoch"] = b + 1
        if done:
            break

    ckpt = {"run_id": args.run_id, "step": step, "epoch": state["epoch"],
            "cursor_in_epoch": state["cursor_in_epoch"], "effective_batch": args.effective_batch,
            "unit_count": len(units), "sample_exposures": int(sum(tails)),
            "encoder": encoder.state_dict(), "head": head.state_dict(), "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    for attempt in range(3):
        try:
            torch.save(ckpt, resume_path); break
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(10)
    report = {"run_id": args.run_id, "dataset": "hacred", "source": "single",
              "status": "smoke_only" if args.max_updates < EXPECTED_UPDATES else "trained_unverified",
              "device": str(device), "updates": step, "expected_updates": EXPECTED_UPDATES,
              "epochs": EPOCHS, "effective_batch": args.effective_batch,
              "unit_count": len(units), "units_meta": meta,
              "sample_exposures": int(sum(tails)), "tail_sizes_hist": {str(k): tails.count(k) for k in sorted(set(tails))},
              "losses": losses, "last_loss": losses[-1] if losses else None,
              "tokenizer": tok_meta,
              "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
              "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
              "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
              "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
              "initial_state_sha256": initial_hash, "final_state_sha256": full_state_hash(encoder, head)}
    (run_dir / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "plumbing"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--artifact-id", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--effective-batch", type=int, default=8)
    ap.add_argument("--max-updates", type=int, default=EXPECTED_UPDATES)
    ap.add_argument("--limit-units", type=int)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    return cmd_plumbing(a) if a.command == "plumbing" else train(a)


if __name__ == "__main__":
    raise SystemExit(main())
