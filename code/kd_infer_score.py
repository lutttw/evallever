# -*- coding: utf-8 -*-
"""kd_infer_score.py —— KD-DocRE 第六生成器的推断与契约计分。

infer:载入作者发布 ckpt(roberta-continue-trained-best.pt,RoBERTa-large 编码器 +
  AxialTransformer + Bilinear,473 keys),对 evaluation 总体(189,556 冻结候选行序)
  前向取 logits(97),保存 logits/keys/报告(ckpt+代码 sha256,确定性);
  运行目录=out/revision_v5_experiments/kddocre/runs/<run_id>/。
  注意:KD forward 输出全实体对 ne×ne 网格 logits,此处按行号 h*n+t 映射回候选行序。
score:复用 e3_atlop_infer_score.run_score(native/sigmoid05 解码 × gR/gD/gR2 ×
  earliest_mention),产出 per_doc_<run_id>.json → 可被 r1_rank_flip 作为第 6 生成器消费。

用法(cwd 任意):
  python kd_infer_score.py --mode infer --limit-docs 3   # 冒烟
  python kd_infer_score.py --mode infer                  # 全量
  python kd_infer_score.py --mode score --run-id kddocre_release
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "data" / "kddocre_repo" / "tonytan48-KD-DocRE-2714f6b"
CKPT = ROOT / "data" / "kddocre_models" / "roberta-continue-trained-best.pt"
OUTD = ROOT / "out" / "revision_v5_experiments" / "kddocre"


def log(*a):
    print(f"[KDDOCRE {datetime.now().strftime('%H:%M:%S')}]", *a, flush=True)


def load_tokenizer():
    from transformers import AutoTokenizer
    snap = sorted((ROOT / ".hf_cache/hub/models--Jean-Baptiste--roberta-large-ner-english"
                   / "snapshots").iterdir())[0]
    tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
    # transformers 5.x 移除该方法;roberta 特殊序列 = <s> ids </s></s>
    tok.build_inputs_with_special_tokens = (
        lambda ids, *a, **k: [tok.cls_token_id] + list(ids) + [tok.sep_token_id, tok.sep_token_id])
    return tok


def run_infer(a):
    import torch
    from transformers import AutoConfig, AutoModel

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import long_seq as _lseqlib                # noqa: E402  先补丁再导入 model
    import torch.nn.functional as _F           # noqa: E402

    def _pli_kd(model, input_ids, attention_mask, start_tokens, end_tokens):
        """与原版 long_seq 语义一致(末层 attention),但走 transformers 5.x return_dict。"""
        n, c = input_ids.size()
        st = torch.tensor(start_tokens).to(input_ids)
        et = torch.tensor(end_tokens).to(input_ids)
        ls, le = st.size(0), et.size(0)
        if c <= 512:
            o = model(input_ids=input_ids, attention_mask=attention_mask,
                      output_attentions=True, return_dict=True)
            return o.last_hidden_state, o.attentions[-1]
        ni, nm, seg = [], [], []
        sl = attention_mask.sum(1).cpu().numpy().astype("int32").tolist()
        for i, l_i in enumerate(sl):
            if l_i <= 512:
                ni.append(input_ids[i, :512]); nm.append(attention_mask[i, :512]); seg.append(1)
            else:
                i1 = torch.cat([input_ids[i, :512 - le], et], dim=-1)
                i2 = torch.cat([st, input_ids[i, (l_i - 512 + ls): l_i]], dim=-1)
                ni.extend([i1, i2])
                nm.extend([attention_mask[i, :512], attention_mask[i, (l_i - 512): l_i]])
                seg.append(2)
        input_ids = torch.stack(ni, dim=0)
        attention_mask = torch.stack(nm, dim=0)
        o = model(input_ids=input_ids, attention_mask=attention_mask,
                  output_attentions=True, return_dict=True)
        so, at = o.last_hidden_state, o.attentions[-1]
        i = 0
        no, na = [], []
        for (n_s, l_i) in zip(seg, sl):
            if n_s == 1:
                no.append(_F.pad(so[i], (0, 0, 0, c - 512)))
                na.append(_F.pad(at[i], (0, c - 512, 0, c - 512)))
            else:
                o1 = _F.pad(so[i][:512 - le], (0, 0, 0, c - 512 + le))
                m1 = _F.pad(attention_mask[i][:512 - le], (0, c - 512 + le))
                a1 = _F.pad(at[i][:, :512 - le, :512 - le],
                            (0, c - 512 + le, 0, c - 512 + le))
                o2 = _F.pad(so[i + 1][ls:], (0, 0, l_i - 512 + ls, c - l_i))
                a2 = _F.pad(at[i + 1][:, ls:, ls:],
                            [l_i - 512 + ls, c - l_i, l_i - 512 + ls, c - l_i])
                m2 = _F.pad(attention_mask[i + 1][ls:], (l_i - 512 + ls, c - l_i))
                m = m1 + m2 + 1e-10
                no.append((o1 + o2) / m.unsqueeze(-1))
                aa = a1 + a2
                na.append(aa / (aa.sum(-1, keepdim=True) + 1e-10))
            i += n_s
        return torch.stack(no, dim=0), torch.stack(na, dim=0)

    _lseqlib.process_long_input = _pli_kd
    from model import DocREModel_KD            # noqa: E402
    import e3_atlop_common as C                # noqa: E402

    sys.path.insert(0, str(ROOT / "src" / "jcip_audit"))

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if getattr(a, "encoder", "roberta") == "bert":
        from transformers import AutoTokenizer as _AT
        BERT_SNAP = ROOT / ".hf_cache/hub/models--google-bert--bert-base-cased/snapshots"
        snap = sorted(BERT_SNAP.iterdir())[0]
        tok = _AT.from_pretrained(str(snap), local_files_only=True)
        if not hasattr(tok, "build_inputs_with_special_tokens"):
            tok.build_inputs_with_special_tokens = (
                lambda ids, *a2, **k: [tok.cls_token_id] + list(ids) + [tok.sep_token_id])
        ttype, emb = "bert", 768
    else:
        snap = sorted((ROOT / ".hf_cache/hub/models--Jean-Baptiste--roberta-large-ner-english"
                       / "snapshots").iterdir())[0]
        tok = load_tokenizer()
        ttype, emb = "roberta", 1024
    config = AutoConfig.from_pretrained(str(snap), local_files_only=True, num_labels=97)
    encoder = AutoModel.from_pretrained(str(snap), local_files_only=True,
                                        attn_implementation="eager")
    config.transformer_type = ttype
    config.cls_token_id = tok.cls_token_id
    config.sep_token_id = tok.sep_token_id
    args = SimpleNamespace(gamma_pos=0, gamma_neg=0)
    model = DocREModel_KD(args, config, encoder, emb_size=emb, block_size=64, num_labels=97)
    ckpt = Path(getattr(a, "ckpt", "") or CKPT)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd.pop("model.embeddings.position_ids", None)   # HF 新版自行注册 buffer
    model.load_state_dict(sd, strict=True)
    model.to(0).eval()
    log(f"ckpt loaded (strict): {ckpt.name}, transformer={ttype}, emb={emb}")

    run_dir = Path(a.run_id and (OUTD / "runs" / a.run_id))
    feats, meta, rows = C.build_population("evaluation", tok, a.max_len,
                                           limit_docs=a.limit_docs)
    log(f"population: {len(feats)} docs, {len(rows)} candidate rows")
    outs = []
    with torch.no_grad():
        for k, f in enumerate(feats):
            ids = torch.tensor([f["input_ids"]], dtype=torch.long).to(0)
            mask = torch.ones_like(ids, dtype=torch.float).to(0)
            n_x = len(f["entity_pos"])
            grid = [[h, t] for h in range(n_x) for t in range(n_x)]  # KD forward 按 n×n 全网格算
            _, logits = model(input_ids=ids, attention_mask=mask,
                              entity_pos=[f["entity_pos"]], hts=[grid])
            logits = logits.view(n_x * n_x, -1)   # 保持展平,行号 = h*n_x + t
            idx = torch.tensor([h * n_x + t for h, t in f["hts"]], dtype=torch.long).to(0)
            outs.append(logits[idx].detach().float().cpu().numpy())
            if (k + 1) % 50 == 0:
                log(f"  {k+1}/{len(feats)} docs")
    logits = np.concatenate(outs, 0).astype(np.float32)
    assert len(logits) == len(rows), (len(logits), len(rows))
    # 列重映射:KD 官方序(Na=0, DocRED official 1..96)→ 我们的约定(Na=0, 字母序 1..96)
    # 语义:remapped[:, our_col] = raw[:, kd_col] ⇒ perm[our_col] = kd_col
    kd_rel2id = json.loads((REPO / "meta" / "rel2id.json").read_text(encoding="utf-8"))
    our_rel2id = json.loads((C.PREP / "relation_to_index.json").read_text(encoding="utf-8"))
    perm = np.zeros(97, dtype=np.int64)
    for rel, kid in kd_rel2id.items():
        our_col = 0 if kid == 0 else our_rel2id[rel] + 1
        perm[our_col] = kid
    assert sorted(perm[1:].tolist()) == list(range(1, 97))
    assert perm[0] == 0
    logits = logits[:, perm]
    assert np.isfinite(logits).all()
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "logits_evaluation.npy", logits)
    with (run_dir / "keys_evaluation.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"doc_id": r["doc_id"], "h": r["h"], "t": r["t"],
                                 "group": int(r["group"])}, ensure_ascii=False) + "\n")
    rep = {"n": int(len(logits)), "n_docs": len(meta),
           "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
           "ckpt_sha256": hashlib.sha256(CKPT.read_bytes()).hexdigest(),
           "code_sha256": hashlib.sha256((REPO / "model.py").read_bytes()).hexdigest(),
           "encoder": "roberta-large(ner-english snapshot 作架构/分词器,权重全部被 ckpt 覆盖)",
           "deterministic": bool(torch.are_deterministic_algorithms_enabled()),
           "ts": datetime.now().isoformat(timespec="seconds")}
    (run_dir / "infer_evaluation.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
    log(f"infer: n={rep['n']} docs={rep['n_docs']} sha={rep['logits_sha256'][:12]}")


def run_score(a):
    sys.path.insert(0, str(ROOT / "src" / "jcip_audit"))
    import e3_atlop_infer_score as E3          # noqa: E402
    shim = argparse.Namespace(dir=str(OUTD), run_id=a.run_id)
    E3.run_score(shim)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["infer", "score"], required=True)
    ap.add_argument("--run-id", default="kddocre_release")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--encoder", choices=["roberta", "bert"], default="roberta")
    ap.add_argument("--ckpt", default="")
    a = ap.parse_args()
    if a.mode == "infer":
        run_infer(a)
    else:
        run_score(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
