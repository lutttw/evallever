# -*- coding: utf-8 -*-
"""KD-DocRE(BERT-base 自训版)本地训练入口 — 修订 51 候选。

架构 = 官方 DocREModel_KD(RoBERTa 论文配置降到 BERT-base:emb_size=hidden=768,
block=64,AxialTransformer 6 层 8 头),监督训练 = 官方流水线的教师阶段(不用蒸馏),
PE8 分割(2748 训练篇,排除 305 校准篇)。损失 = AFLoss(gamma_pos=gamma_neg=1.0,
官方 train.py 默认)。早停 = dev F1 连续 3 epoch 不再新高。

用法:
  python kddocre_local_train.py --smoke          # 300 篇×1 epoch 冒烟
  python kddocre_local_train.py                  # 全量
"""
import argparse, os, sys, json, hashlib, time, random, math
# 训练链不需要 scipy;屏蔽以防安全软件对 scipy DLL 的间歇拦截
sys.modules['scipy'] = None
sys.modules['scipy.special'] = None
sys.modules['scipy.stats'] = None
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # → <ws>
REPO = os.path.join(ROOT, "data", "kddocre_repo", "tonytan48-KD-DocRE-2714f6b")
os.chdir(REPO)                                           # prepro/evaluation 读相对路径 meta/
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(ROOT, "src", "jcip_audit"))

BERT_SNAP = os.path.join(ROOT, ".hf_cache", "hub", "models--google-bert--bert-base-cased", "snapshots")
BERT_DIR = os.path.join(BERT_SNAP, sorted(os.listdir(BERT_SNAP))[0])
TRAIN_FILE = os.path.join(ROOT, "data", "docred_local", "train_annotated_pe8.json")
DEV_FILE = os.path.join(ROOT, "data", "docred_local", "dev.json")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collate(batch):
    max_len = max(len(f["input_ids"]) for f in batch)
    ids = torch.tensor([f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch], dtype=torch.long)
    mask = torch.tensor([[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch], dtype=torch.float)
    return (ids, mask, [f["labels"] for f in batch], [f["entity_pos"] for f in batch],
            [f["hts"] for f in batch], [f["title"] for f in batch])


def dev_f1(official_results, dev_path):
    truth = json.load(open(dev_path, encoding="utf-8"))
    gold = {}
    for x in truth:
        s = set()
        for lb in x.get("labels", []):
            s.add((lb["h"], lb["t"], lb["r"]))
        gold[x["title"]] = s
    preds_by = {}
    for row in official_results:
        preds_by.setdefault(row["title"], set()).add((row["h_idx"], row["t_idx"], row["r"]))
    tp = fp = fn = 0
    for title, pr in preds_by.items():
        g = gold.get(title, set())
        tp += len(pr & g); fp += len(pr - g); fn += len(g - pr)
    for title, g in gold.items():
        if title not in preds_by:
            fn += len(g)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return prec, rec, f1


def evaluate(model, feats, dev_path, device, batch_size=2):
    dl = DataLoader(feats, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for ids, mask, labels, epos, hts, titles in dl:
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                        entity_pos=epos, hts=hts)
            pred = out[0]                      # labels=None → (get_label 二值矩阵, logits)
            preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds, 0)
    res = []
    h_idx, t_idx, title = [], [], []
    for f in feats:
        for h, t in f["hts"]:
            h_idx.append(h); t_idx.append(t); title.append(f["title"])
    id2rel = {v: k for k, v in json.load(open(os.path.join(REPO, "meta", "rel2id.json"), encoding="utf-8")).items()}
    for i in range(preds.shape[0]):
        for r in np.nonzero(preds[i])[0].tolist():
            if r != 0:
                res.append({"title": title[i], "h_idx": h_idx[i], "t_idx": t_idx[i], "r": id2rel[r]})
    return dev_f1(res, dev_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--eval-batch-size", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=3e-5)
    ap.add_argument("--classifier-lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.limit_train, a.epochs, a.patience = 300, 1, 99
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    random.seed(a.seed); np.random.seed(a.seed)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    import long_seq as _ls
    import torch.nn.functional as _F

    def _pli(model, input_ids, attention_mask, start_tokens, end_tokens):
        """与原版 long_seq 语义一致(末层 attention),走 transformers 5.x return_dict。"""
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
        for i, li in enumerate(sl):
            if li <= 512:
                ni.append(input_ids[i, :512]); nm.append(attention_mask[i, :512]); seg.append(1)
            else:
                i1 = torch.cat([input_ids[i, :512 - le], et], dim=-1)
                i2 = torch.cat([st, input_ids[i, (li - 512 + ls):li]], dim=-1)
                ni.extend([i1, i2])
                nm.extend([attention_mask[i, :512], attention_mask[i, (li - 512):li]])
                seg.append(2)
        input_ids = torch.stack(ni, 0); attention_mask = torch.stack(nm, 0)
        o = model(input_ids=input_ids, attention_mask=attention_mask,
                  output_attentions=True, return_dict=True)
        so, at = o.last_hidden_state, o.attentions[-1]
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
                aa = a1 + a2
                na.append(aa / (aa.sum(-1, keepdim=True) + 1e-10))
            i += n_s
        return torch.stack(no, 0), torch.stack(na, 0)

    _ls.process_long_input = _pli

    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from model import DocREModel_KD
    from prepro import read_docred
    from types import SimpleNamespace

    tok = AutoTokenizer.from_pretrained(BERT_DIR, local_files_only=True)
    if not hasattr(tok, "build_inputs_with_special_tokens"):
        tok.build_inputs_with_special_tokens = (
            lambda ids, *a, **k: [tok.cls_token_id] + list(ids) + [tok.sep_token_id])
    print("[data] preprocessing (first run ~minutes)", flush=True)
    train_feats = read_docred(TRAIN_FILE, tok, max_seq_length=1024)
    if a.limit_train:
        train_feats = train_feats[:a.limit_train]
    dev_feats = read_docred(DEV_FILE, tok, max_seq_length=1024)
    print(f"[data] train={len(train_feats)} dev={len(dev_feats)}", flush=True)

    config = AutoConfig.from_pretrained(BERT_DIR, local_files_only=True, num_labels=97)
    enc = AutoModel.from_pretrained(BERT_DIR, local_files_only=True, attn_implementation="eager")
    config.transformer_type = "bert"
    config.cls_token_id = tok.cls_token_id
    config.sep_token_id = tok.sep_token_id
    args = SimpleNamespace(gamma_pos=1.0, gamma_neg=1.0)   # 官方 train.py 默认
    model = DocREModel_KD(args, config, enc, emb_size=768, block_size=64, num_labels=97)
    model.to(device)

    new_layer = ["extractor", "bilinear", "classifier", "projection", "axial_transformer"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)]},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)],
         "lr": a.classifier_lr}]
    opt = torch.optim.AdamW(grouped, lr=a.learning_rate, eps=1e-6)
    total_steps = len(train_feats) * a.epochs // a.batch_size
    warmup = int(total_steps * 0.06)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, warmup))
                                              if s < warmup else
                                              max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    save_dir = os.path.join(HERE, f"kddocre_pe8_s{a.seed}")
    os.makedirs(save_dir, exist_ok=True)
    log_fh = open(os.path.join(save_dir, "train_log.txt"), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        log_fh.write(s + "\n"); log_fh.flush()

    scaler = torch.amp.GradScaler("cuda")
    dl = DataLoader(train_feats, batch_size=a.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    best_f1, bad, best_epoch = -1.0, 0, -1
    t0 = time.time()
    history = []
    for epoch in range(a.epochs):
        model.train()
        for step, (ids, mask, labels, epos, hts, titles) in enumerate(dl):
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                             labels=labels, entity_pos=epos, hts=hts)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            if step % 500 == 0:
                log(f"[e{epoch} s{step}/{len(dl)}] loss={float(loss):.4f} ({(time.time()-t0)/60:.1f}min)")
        p, r, f1 = evaluate(model, dev_feats, DEV_FILE, device, a.eval_batch_size)
        history.append({"epoch": epoch, "dev_p": p, "dev_r": r, "dev_f1": f1})
        log(f"[e{epoch}] dev P={p:.4f} R={r:.4f} F1={f1:.4f} best={best_f1:.4f} "
            f"elapsed={(time.time()-t0)/60:.1f}min")
        if f1 > best_f1:
            best_f1, best_epoch, bad = f1, epoch, 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best.ckpt"))
            log(f"[e{epoch}] new best -> best.ckpt (F1={f1:.4f})")
        else:
            bad += 1
            if bad >= a.patience:
                log(f"[early-stop] epoch={epoch} best e{best_epoch} F1={best_f1:.4f}")
                break

    report = {"seed": a.seed, "epochs_run": len(history), "best_epoch": best_epoch,
              "best_dev_f1": best_f1, "history": history, "bad_epochs": bad,
              "wall_hours": (time.time()-t0)/3600,
              "train_file": TRAIN_FILE, "train_sha256": sha256_file(TRAIN_FILE),
              "dev_file": DEV_FILE, "arch": "DocREModel_KD bert-base emb768 block64 axial6x8",
              "gamma": [1.0, 1.0], "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(report, open(os.path.join(save_dir, "train_report.json"), "w"), indent=1)
    log(f"[done] {save_dir} wall={report['wall_hours']:.1f}h best_F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
