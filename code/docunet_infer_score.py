# -*- coding: utf-8 -*-
"""docunet_infer_score.py —— DocuNet 第六生成器的推断与契约计分。

infer:载入本地 PE8 训练的 docunet_pe8_s<seed>/best.ckpt,对 evaluation 总体
  (189,556 冻结候选行序)前向取 logits(97),保存 logits/keys/报告;
  运行目录=out/revision_v5_experiments/docunet/runs/<run_id>/。
  注意:DocuNet(context-based 通道)按 hts 行序输出,候选行天然 1:1 对齐;
  get_label 只在训练 loss 路径,这里手动复刻 forward 到 bilinear 取原始 logits。
score:复用 e3_atlop_infer_score.run_score(native/sigmoid05 解码 × gR/gD/gR2 ×
  earliest_mention),产出 per_doc_<run_id>.json → 可被 r1_rank_flip 作为第 6 生成器消费。

用法(cwd 任意):
  python docunet_infer_score.py --mode infer --limit-docs 3   # 冒烟
  python docunet_infer_score.py --mode infer                  # 全量
  python docunet_infer_score.py --mode score --run-id docunet_pe8_s20260917
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data" / "docunet_work"
OUTD = ROOT / "out" / "revision_v5_experiments" / "docunet"


def log(*a):
    print(f"[DOCUNET {datetime.now().strftime('%H:%M:%S')}]", *a, flush=True)


def load_model(ckpt_path, seed_tag: str):
    import torch
    from types import SimpleNamespace
    os.chdir(WORK)
    sys.path.insert(0, str(WORK))
    import overrides as _ov
    _ov.overrides = lambda f: f                      # noqa: E731
    import long_seq as _ls
    import torch.nn.functional as _F
    import numpy as _np
    sys.path.insert(0, str(ROOT / "src" / "jcip_audit"))

    def _pli_v5(model, input_ids, attention_mask, start_tokens, end_tokens):
        """与训练/DREEAM 同构的 v5 长输入(batch=1 路径)。"""
        n, c = input_ids.size()
        st = torch.tensor(start_tokens).to(input_ids)
        et = torch.tensor(end_tokens).to(input_ids)
        ls, le = st.size(0), et.size(0)
        if c <= 512:
            o = model(input_ids=input_ids, attention_mask=attention_mask,
                      output_attentions=True, output_hidden_states=True, return_dict=True)
            return (torch.stack(o.hidden_states[-3:], dim=1).mean(dim=1),
                    torch.stack(o.attentions[-3:], dim=1).mean(dim=1))
        ni, nm, seg = [], [], []
        sl = attention_mask.sum(1).cpu().numpy().astype(_np.int32).tolist()
        for i, li in enumerate(sl):
            if li <= 512:
                ni.append(input_ids[i, :512]); nm.append(attention_mask[i, :512]); seg.append(1)
            else:
                i1 = torch.cat([input_ids[i, :512 - le], et], dim=-1)
                i2 = torch.cat([st, input_ids[i, (li - 512 + ls):li]], dim=-1)
                ni.extend([i1, i2])
                nm.extend([attention_mask[i, :512], attention_mask[i, (li - 512):li]])
                seg.append(2)
        input_ids = torch.stack(ni, dim=0); attention_mask = torch.stack(nm, dim=0)
        o = model(input_ids=input_ids, attention_mask=attention_mask,
                  output_attentions=True, output_hidden_states=True, return_dict=True)
        so = torch.stack(o.hidden_states[-3:], dim=1).mean(dim=1)
        at = torch.stack(o.attentions[-3:], dim=1).mean(dim=1)
        i = 0; no, na = [], []
        for (n_s, li) in zip(seg, sl):
            if n_s == 1:
                no.append(_F.pad(so[i], (0, 0, 0, c - 512)))
                na.append(_F.pad(at[i], (0, c - 512, 0, c - 512)))
            else:
                o1 = _F.pad(so[i][:512 - le], (0, 0, 0, c - 512 + le))
                m1 = _F.pad(attention_mask[i][:512 - le], (0, c - 512 + le))
                a1 = _F.pad(at[i][:, :512 - le, :512 - le], (0, c - 512 + le, 0, c - 512 + le))
                o2 = _F.pad(so[i + 1][ls:], (0, 0, li - 512 + ls, c - li))
                a2 = _F.pad(at[i + 1][:, ls:, ls:], [li - 512 + ls, c - li, li - 512 + ls, c - li])
                m2 = _F.pad(attention_mask[i + 1][ls:], (li - 512 + ls, c - li))
                m = m1 + m2 + 1e-10
                no.append((o1 + o2) / m.unsqueeze(-1))
                aa = a1 + a2; na.append(aa / (aa.sum(-1, keepdim=True) + 1e-10))
            i += n_s
        return torch.stack(no, dim=0), torch.stack(na, dim=0)

    _ls.process_long_input = _pli_v5
    from model_balanceloss import DocREModel          # noqa: E402
    from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402
    from docunet_local_train import BERT_DIR          # noqa: E402

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    args = SimpleNamespace(unet_in_dim=768, unet_out_dim=256, down_dim=128, max_height=42,
                           channel_type="context-based", drop_prob=0.0, num_labels=4)
    config = AutoConfig.from_pretrained(BERT_DIR, num_labels=97)
    tok = AutoTokenizer.from_pretrained(BERT_DIR)
    if not hasattr(tok, "build_inputs_with_special_tokens"):
        tok.build_inputs_with_special_tokens = (
            lambda ids, *a, **k: [tok.cls_token_id] + list(ids) + [tok.sep_token_id])
    enc = AutoModel.from_pretrained(BERT_DIR, config=config, attn_implementation="eager")
    config.transformer_type = "bert"
    config.cls_token_id = tok.cls_token_id
    config.sep_token_id = tok.sep_token_id
    model = DocREModel(config, args, enc, num_labels=args.num_labels)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.to(0).eval()
    log(f"ckpt loaded (strict, {len(sd)} tensors, {seed_tag})")
    return model, tok


def forward_logits(model, f):
    """复刻 context-based forward 到 bilinear,返回候选行序原始 logits。"""
    import torch
    from opt_einsum import contract  # noqa: F401  (get_channel_map 内部使用)
    device = next(model.parameters()).device
    ids = torch.tensor([f["input_ids"]], dtype=torch.long).to(device)
    mask = torch.ones_like(ids, dtype=torch.float).to(device)
    seq, att = model.encode(ids, mask, [f["entity_pos"]])
    hs, ts, _, entity_as = model.get_hrt(seq, att, [f["entity_pos"]], [f["hts"]])
    feature_map = model.get_channel_map(seq, entity_as)
    attn_input = model.liner(feature_map).permute(0, 3, 1, 2).contiguous()
    attn_map = model.segmentation_net(attn_input)
    h_t = model.get_ht(attn_map, [f["hts"]])
    hs = torch.tanh(model.head_extractor(torch.cat([hs, h_t], dim=1)))
    ts = torch.tanh(model.tail_extractor(torch.cat([ts, h_t], dim=1)))
    b1 = hs.view(-1, model.emb_size // model.block_size, model.block_size)
    b2 = ts.view(-1, model.emb_size // model.block_size, model.block_size)
    bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, model.emb_size * model.block_size)
    return model.bilinear(bl)


def run_infer(a):
    import torch
    import e3_atlop_common as C
    sys.path.insert(0, str(ROOT / "src" / "jcip_audit"))

    ckpt = Path(a.ckpt)
    model, tok = load_model(ckpt, a.run_id)
    feats, meta, rows = C.build_population("evaluation", tok, a.max_len,
                                           limit_docs=a.limit_docs)
    log(f"population: {len(feats)} docs, {len(rows)} candidate rows")
    outs = []
    with torch.no_grad():
        for k, f in enumerate(feats):
            outs.append(forward_logits(model, f).detach().float().cpu().numpy())
            if (k + 1) % 100 == 0:
                log(f"  {k+1}/{len(feats)} docs")
    logits = np.concatenate(outs, 0).astype(np.float32)
    assert len(logits) == len(rows), (len(logits), len(rows))
    # 列重映射:官方序(Na=0)→ 我们的约定(Na=0, 字母序 1..96)
    kd_rel2id = json.loads((WORK / "meta" / "rel2id.json").read_text(encoding="utf-8"))
    our_rel2id = json.loads((C.PREP / "relation_to_index.json").read_text(encoding="utf-8"))
    perm = np.zeros(97, dtype=np.int64)
    for rel, kid in kd_rel2id.items():
        perm[kid] = 0 if kid == 0 else our_rel2id[rel] + 1
    assert sorted(perm[1:].tolist()) == list(range(1, 97))
    logits = logits[:, perm]
    assert np.isfinite(logits).all()

    run_dir = OUTD / "runs" / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "logits_evaluation.npy", logits)
    with (run_dir / "keys_evaluation.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"doc_id": r["doc_id"], "h": r["h"], "t": r["t"],
                                 "group": int(r["group"])}, ensure_ascii=False) + "\n")
    rep = {"n": int(len(logits)), "n_docs": len(meta),
           "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
           "ckpt_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
           "ckpt": str(ckpt), "deterministic": bool(torch.are_deterministic_algorithms_enabled()),
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
    ap.add_argument("--run-id", default="docunet_pe8_s20260917")
    ap.add_argument("--ckpt", default=str(ROOT / "data" / "docunet_work" /
                                          "docunet_pe8_s20260917" / "best.ckpt"))
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit-docs", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "infer":
        run_infer(a)
    else:
        run_score(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
