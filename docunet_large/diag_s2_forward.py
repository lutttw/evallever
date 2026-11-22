# -*- coding: utf-8 -*-
"""diag_s2_forward.py — 修订77B 决定性诊断:同一 s2 权重,两个前向,同一批 eval 文档。

A 侧=训练脚本自己的前向(DocREModel.__forward__ → get_label 0/1 矩阵,num_labels=4);
B 侧=docunet_large_infer_score.forward_logits 复刻(bilinear 裸 logits + native 解码)。
若 A 有预测、B 为零 ⇒ 两个前向不等价(复刻有误);若都为零 ⇒ ckpt 在保存/加载后退化。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else \
    "data/docunet_work"
os.chdir(HERE)
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

import json

import numpy as np
import torch
from types import SimpleNamespace

from transformers import AutoConfig, AutoModel, AutoTokenizer

import overrides as _ov
_ov.overrides = lambda f: f
import long_seq as _ls
import torch.nn.functional as _F


def pli_v5(model, input_ids, attention_mask, start_tokens, end_tokens):
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
    sl = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
    for i, li in enumerate(sl):
        if li <= 512:
            ni.append(input_ids[i, :512]); nm.append(attention_mask[i, :512]); seg.append(1)
        else:
            i1 = torch.cat([input_ids[i, :512 - le], et], dim=-1)
            i2 = torch.cat([st, input_ids[i, (li - 512 + ls):li]], dim=-1)
            ni.extend([i1, i2]); nm.extend([attention_mask[i, :512],
                                            attention_mask[i, (li - 512):li]])
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


_ls.process_long_input = pli_v5
from model_balanceloss import DocREModel          # noqa: E402
from docunet_large_train import BERT_DIR, _read_cached   # noqa: E402
from torch.utils.data import DataLoader           # noqa: E402
from utils_sample import collate_fn               # noqa: E402


def log(*a):
    print("[DIAG]", *a, flush=True)


def main() -> int:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(HERE, "docunetL_pe8_s20260918", "best.ckpt")
    tag = os.path.basename(os.path.dirname(ckpt))
    devf = os.path.join(ROOT, "data", "docred_local", "dev.json")

    tok = AutoTokenizer.from_pretrained(BERT_DIR)
    if not hasattr(tok, "build_inputs_with_special_tokens"):
        tok.build_inputs_with_special_tokens = lambda ids, *a, **k: [tok.cls_token_id] + list(ids) + [tok.sep_token_id]
    args = SimpleNamespace(num_labels=4, num_class=97, unet_in_dim=1024,
                           unet_out_dim=256, down_dim=128, max_height=42,
                           channel_type="context-based", drop_prob=0.0)
    config = AutoConfig.from_pretrained(BERT_DIR, num_labels=97)
    enc = AutoModel.from_pretrained(BERT_DIR, config=config, attn_implementation="eager")
    config.transformer_type = "bert"
    config.cls_token_id = tok.cls_token_id
    config.sep_token_id = tok.sep_token_id
    model = DocREModel(config, args, enc, num_labels=4)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.to(0).eval()
    log(f"{tag}: ckpt loaded strict")

    feats = _read_cached(tok, devf, "dev.json_dev.pkl")
    log(f"dev feats: {len(feats)}")
    dl = DataLoader(feats[:8], batch_size=2, shuffle=False,
                    collate_fn=collate_fn, drop_last=False)
    nA = 0
    with torch.no_grad():
        for batch in dl:
            inputs = {"input_ids": batch[0].to(0), "attention_mask": batch[1].to(0),
                      "entity_pos": batch[3], "hts": batch[4]}
            out = model(**inputs)
            pred = out if isinstance(out, torch.Tensor) else out[0]
            nA += int(pred.sum())
            log(f"  A侧 get_label 预测数: {int(pred.sum())} (shape {tuple(pred.shape)})")
    log(f"A侧(训练前向/get_label) 8 docs 总预测: {nA}")

    # B 侧:同权重的 bilinear 裸 logits + native 解码
    nB = 0
    with torch.no_grad():
        for batch in dl:
            inputs = {"input_ids": batch[0].to(0), "attention_mask": batch[1].to(0),
                      "entity_pos": batch[3], "hts": batch[4]}
            seq, att = model.encode(inputs["input_ids"], inputs["attention_mask"],
                                    inputs["entity_pos"])
            hs, ts, _, entity_as = model.get_hrt(seq, att, inputs["entity_pos"], inputs["hts"])
            feature_map = model.get_channel_map(seq, entity_as)
            attn_input = model.liner(feature_map).permute(0, 3, 1, 2).contiguous()
            attn_map = model.segmentation_net(attn_input)
            h_t = model.get_ht(attn_map, inputs["hts"])
            hs = torch.tanh(model.head_extractor(torch.cat([hs, h_t], dim=1)))
            ts = torch.tanh(model.tail_extractor(torch.cat([ts, h_t], dim=1)))
            b1 = hs.view(-1, model.emb_size // model.block_size, model.block_size)
            b2 = ts.view(-1, model.emb_size // model.block_size, model.block_size)
            bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, model.emb_size * model.block_size)
            logits = model.bilinear(bl).float().cpu()       # (pairs, 97) 原始 logits
            th = logits[:, :1]
            mask = logits > th
            topv = np.sort(logits.numpy(), axis=1)[:, -4][:, None]
            mask = mask & (logits.numpy() >= topv)
            nB += int(mask[:, 1:].sum())
            log(f"  B侧 native 解码预测数: {int(mask[:, 1:].sum())} "
                f"| logits max={float(logits.max()):.2f} NA列max={float(logits[:,0].max()):.2f}")
    log(f"B侧(复刻前向/native) 8 docs 总预测: {nB}")
    print(f"VERDICT {tag}: A={nA} B={nB} -> "
          + ("两前向等价且均退化" if nA == 0 and nB == 0 else
             "前向不等价(复刻有误)" if nA > 0 and nB == 0 else
             "仅复刻侧有预测(训练get_label退化)" if nA == 0 else "两侧均有预测"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
