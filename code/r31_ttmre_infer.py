# -*- coding: utf-8 -*-
"""修订62 执行:TTM-RE v0.2 官方权重本地推理 → pair-level logits 冻结件。

设计冻结于预注册 修订62(sha 52f063a8)。官方 evaluate() 同构路径:
logits = model(**inputs)[1](isrank=1:每对 98 维,第 0 列=NA 排序基)。
priors 仅进 loss(权重 weight 仅在 get_label 内使用),不影响 logits 与解码;
train_revised 不可得,用占位 priors(披露于 r31 元数据)。

产出 data/ttmre_work/ttmre_dev_logits.npz:
  titles (498,), pair_counts, hts/ent_pairs 序列化 npz + logits 连续存储
  logit_dim=98 (col0=NA), max_seq_length=1024, checkpoint sha256
"""
import argparse, os, sys, json, hashlib, pickle, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
REPO = os.path.join(ROOT, 'data', 'ttmre_work', 'repo')
sys.path.insert(0, REPO)

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

CKPT = os.path.join(ROOT, 'data', 'ttmre_work', 'finetune_state_dict.pth')
DEV = os.path.join(ROOT, 'data', 'docred_local', 'dev.json')
OUT = os.path.join(ROOT, 'data', 'ttmre_work', 'ttmre_dev_logits.npz')

# 本地 roberta-large 快照(与 docunet 臂同源)
SNAP = os.path.join(ROOT, '.hf_cache', 'hub', 'models--FacebookAI--roberta-large', 'snapshots')
BERT_DIR = os.path.join(SNAP, sorted(os.listdir(SNAP))[0]) if os.path.isdir(SNAP) else 'roberta-large'


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    # ── 官方 args.json(v0.2 released)逐字复刻 ──
    args = types.SimpleNamespace(
        transformer_type='roberta', model_name_or_path=BERT_DIR,
        model_type='ttmre', max_seq_length=1024, test_batch_size=4,
        seed=74, num_class=97, isrank=1, m_tag='S-PU',
        beta=0.0, gamma=1.0, m=1.0, e=1.0, num_layers=4, memory_size=200,
        data_dir=os.path.join(ROOT, 'data', 'ttmre_work'),
    )
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.n_gpu = 1

    import prepro as repo_prepro
    read_docred = repo_prepro.read_docred

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_class)
    config._attn_implementation = 'eager'
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if not hasattr(tokenizer, 'build_inputs_with_special_tokens'):
        # transformers 5.x 移除;roberta = <s> ids </s>(与 docunet_large_train 同款守卫)
        tokenizer.build_inputs_with_special_tokens = lambda ids, *a, **k: \
            [tokenizer.cls_token_id] + list(ids) + [tokenizer.sep_token_id]
    config.transformer_type = args.transformer_type
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id

    dev_features, priors = read_docred(args, DEV, tokenizer, max_seq_length=args.max_seq_length)
    print(f'[data] dev feats={len(dev_features)} priors={len(priors)}', flush=True)

    import model2
    enc = AutoModel.from_pretrained(args.model_name_or_path, config=config,
                                    attn_implementation='eager')
    # transformers 5.11: return_dict 非显式 True 时 attentions 被丢弃(实测);
    # 统一注入 return_dict=True,官方 process_long_input 的 output[0]/output[-1] 语义不变
    _orig_fwd = enc.forward
    def _fwd(*a, **k):
        k['return_dict'] = True
        return _orig_fwd(*a, **k)
    enc.forward = _fwd
    enc.to(args.device)
    priors_t = torch.tensor(np.asarray(priors, dtype=np.float64) + 1e-9).to(args.device)
    model = model2.DocREModel(args, config, priors_t, enc, tokenizer)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[ckpt] missing={len(missing)} unexpected={len(unexpected)}', flush=True)
    if missing:
        print('  missing head:', missing[:6], flush=True)
    model.to(args.device)
    model.eval()

    from torch.utils.data import DataLoader
    from train2 import collate_fn
    dl = DataLoader(dev_features, batch_size=args.test_batch_size, shuffle=False,
                    collate_fn=collate_fn, drop_last=False)

    titles = [ft['title'] for ft in dev_features]
    # collate 将 hts 拼成整 batch 张量;逐文档对以 features 为准(shuffle=False 保序)
    pair_counts = [len(ft['hts']) for ft in dev_features]
    hts_flat = [ft['hts'] for ft in dev_features]
    all_logits = []
    t0 = __import__('time').time()
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            inputs = {'input_ids': batch[0].to(args.device),
                      'attention_mask': batch[1].to(args.device),
                      'labels': batch[2],
                      'entity_pos': batch[3],
                      'hts': batch[4]}
            output = model(**inputs)
            logits = output[1].float().cpu().numpy()
            all_logits.append(logits.astype(np.float32))
            if bi % 10 == 0:
                el = (__import__('time').time() - t0) / 60
                print(f'[infer b{bi}/{len(dl)}] {el:.1f}min', flush=True)

    logits = np.concatenate(all_logits, axis=0)
    hts_arr = np.zeros((sum(pair_counts), 2), dtype=np.int32)
    off = 0
    for x in hts_flat:
        arr = np.asarray(x, dtype=np.int32)
        hts_arr[off:off + len(arr)] = arr
        off += len(arr)
    assert logits.shape[0] == sum(pair_counts) == len(hts_arr), 'pair mismatch'
    assert len(pair_counts) == len(titles), 'doc mismatch'

    np.savez_compressed(
        OUT, logits=logits, hts=hts_arr, pair_counts=np.array(pair_counts),
        titles=np.array(titles),
        logit_dim=np.array(logits.shape[1]), ckpt_sha256=sha256_file(CKPT),
        dev_sha256=sha256_file(DEV), max_seq_length=np.array(args.max_seq_length),
    )
    print(f'[done] {OUT} logits={logits.shape} docs={len(titles)}', flush=True)


if __name__ == '__main__':
    main()
