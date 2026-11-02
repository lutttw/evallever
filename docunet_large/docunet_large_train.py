# -*- coding: utf-8 -*-
"""DocuNet 本地训练入口 — 修订 50。PE8, BERT-base-cased, 3 seeds。"""
import argparse, os, sys, json, hashlib, time, math, random
# 训练链不需要 scipy;火绒对 scipy DLL 的间歇拦截会经由 transformers 间接触发,直接屏蔽
# 火绒间歇拦 torch 首导;transformers 会缓存失败判定——清缓存重试
import transformers.utils.import_utils as _iu
for _ in range(6):
    try:
        _iu.clear_import_cache()
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        break
    except ModuleNotFoundError as _e:
        print('[import retry] cause:', repr(_e.__cause__)[:250], flush=True)
        import time as _t; _t.sleep(2)
else:
    raise RuntimeError("transformers/torch import blocked by AV after retries")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # → <ws>
sys.path.insert(0, HERE)

# ── overrides stub ──
import overrides as _ov
_ov.overrides = lambda f: f

# ── long_seq v5 适配 ──
import long_seq as _ls
import torch.nn.functional as _F
import numpy as _np

def _pli_v5(model, input_ids, attention_mask, start_tokens, end_tokens):
    n, c = input_ids.size()
    st = torch.tensor(start_tokens).to(input_ids)
    et = torch.tensor(end_tokens).to(input_ids)
    ls, le = st.size(0), et.size(0)
    if c <= 512:
        o = model(input_ids=input_ids, attention_mask=attention_mask,
                  output_attentions=True, output_hidden_states=True, return_dict=True)
        seq = torch.stack(o.hidden_states[-3:], dim=1).mean(dim=1)
        att = torch.stack(o.attentions[-3:], dim=1).mean(dim=1)
    else:
        ni, nm, seg = [], [], []
        sl = attention_mask.sum(1).cpu().numpy().astype(_np.int32).tolist()
        for i, li in enumerate(sl):
            if li <= 512:
                ni.append(input_ids[i, :512]); nm.append(attention_mask[i, :512]); seg.append(1)
            else:
                i1 = torch.cat([input_ids[i, :512-le], et], dim=-1)
                i2 = torch.cat([st, input_ids[i, (li-512+ls):li]], dim=-1)
                ni.extend([i1, i2])
                nm.extend([attention_mask[i, :512], attention_mask[i, (li-512):li]])
                seg.append(2)
        input_ids = torch.stack(ni, dim=0); attention_mask = torch.stack(nm, dim=0)
        o = model(input_ids=input_ids, attention_mask=attention_mask,
                  output_attentions=True, output_hidden_states=True, return_dict=True)
        so = torch.stack(o.hidden_states[-3:], dim=1).mean(dim=1)
        at = torch.stack(o.attentions[-3:], dim=1).mean(dim=1)
        i = 0; no, na = [], []
        for (n_s, li) in zip(seg, sl):
            if n_s == 1:
                no.append(_F.pad(so[i], (0, 0, 0, c-512)))
                na.append(_F.pad(at[i], (0, c-512, 0, c-512)))
            else:
                o1 = _F.pad(so[i][:512-le], (0, 0, 0, c-512+le))
                m1 = _F.pad(attention_mask[i][:512-le], (0, c-512+le))
                a1 = _F.pad(at[i][:, :512-le, :512-le], (0, c-512+le, 0, c-512+le))
                o2 = _F.pad(so[i+1][ls:], (0, 0, li-512+ls, c-li))
                a2 = _F.pad(at[i+1][:, ls:, ls:], [li-512+ls, c-li, li-512+ls, c-li])
                m2 = _F.pad(attention_mask[i+1][ls:], (li-512+ls, c-li))
                m = m1 + m2 + 1e-10
                no.append((o1 + o2) / m.unsqueeze(-1))
                aa = a1 + a2; na.append(aa / (aa.sum(-1, keepdim=True) + 1e-10))
            i += n_s
        seq = torch.stack(no, dim=0); att = torch.stack(na, dim=0)
    return seq, att
_ls.process_long_input = _pli_v5

from model_balanceloss import DocREModel
from utils_sample import set_seed, collate_fn
from prepro import read_docred
from transformers import AutoConfig, AutoModel, AutoTokenizer

BERT_SNAP = os.path.join(ROOT, ".hf_cache", "hub", "models--FacebookAI--roberta-large", "snapshots")
BERT_DIR = os.path.join(BERT_SNAP, sorted(os.listdir(BERT_SNAP))[0]) if os.path.isdir(BERT_SNAP) else "bert-base-cased"
TRAIN_FILE = os.path.join(ROOT, "data", "docred_local", "train_annotated_pe8.json")
DEV_FILE = os.path.join(ROOT, "data", "docred_local", "dev.json")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def dev_f1(official_results, dev_path):
    truth = json.load(open(dev_path, encoding="utf-8"))
    gold = {}
    for x in truth:
        s = set()
        for lb in x.get("labels", []): s.add((lb["h"], lb["t"], lb["r"]))
        gold[x["title"]] = s
    tp = fp = fn = 0
    preds_by = {}
    for row in official_results:
        preds_by.setdefault(row["title"], set()).add((row["h_idx"], row["t_idx"], row["r"]))
    for title, pr in preds_by.items():
        g = gold.get(title, set())
        tp += len(pr & g); fp += len(pr - g); fn += len(g - pr)
    for title, g in gold.items():
        if title not in preds_by: fn += len(g)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return prec, rec, f1


def _read_cached(tok, file_in, cache_name, max_seq=1024):
    """read_docred 会 pickle 到 save_file;统一走 cache 目录。"""
    cache = os.path.join(HERE, "cache", cache_name)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    from prepro import read_docred as _rd
    return _rd(tok, file_in, cache, tok, max_seq_length=max_seq)


def evaluate(model, tok, dev_path, device, batch_size=4):
    feats = _read_cached(tok, dev_path, os.path.basename(dev_path).replace(".json", "") + "_dev.pkl")
    dl = DataLoader(feats, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            inputs = {"input_ids": batch[0].to(device),
                      "attention_mask": batch[1].to(device),
                      "entity_pos": batch[3], "hts": batch[4]}
            out = model(**inputs)
            # labels=None 时 forward 返回 (get_label 二值矩阵,) —— 直接是 0/1 预测
            pred = out if isinstance(out, torch.Tensor) else out[0]
            preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    from evaluation import to_official
    results = to_official(preds, feats)   # 本仓库 evaluation.to_official 只返回 res
    return dev_f1(results, dev_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--train_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--classifier_lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=2, help="dev F1 连续 N 个 epoch 不再新高即早停")
    ap.add_argument("--limit-train", type=int, default=0, help=">0 时只取前 N 篇(冒烟)")
    ap.add_argument("--dev-file", default="", help="覆盖默认 dev(冒烟用 dev_smoke.json)")
    args = ap.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    args.max_seq_length = 512
    args.num_labels = 4
    args.num_class = 97
    args.unet_in_dim = 1024
    args.unet_out_dim = 256
    args.down_dim = 128
    args.max_height = 42
    args.channel_type = "context-based"
    args.drop_prob = 0.0
    args.warmup_ratio = 0.06
    args.max_grad_norm = 1.0
    args.adam_epsilon = 1e-6

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    config = AutoConfig.from_pretrained(BERT_DIR, num_labels=args.num_class)
    tok = AutoTokenizer.from_pretrained(BERT_DIR)
    if not hasattr(tok, "build_inputs_with_special_tokens"):
        tok.build_inputs_with_special_tokens = lambda ids, *a, **k: [tok.cls_token_id] + list(ids) + [tok.sep_token_id]
    enc = AutoModel.from_pretrained(BERT_DIR, config=config, attn_implementation="eager")
    enc.gradient_checkpointing_enable()
    config.transformer_type = "roberta"
    config.cls_token_id = tok.cls_token_id
    config.sep_token_id = tok.sep_token_id
    model = DocREModel(config, args, enc, num_labels=args.num_labels)
    model.to(args.device)

    print(f"[data] train={TRAIN_FILE}", flush=True)
    train_feats = _read_cached(tok, TRAIN_FILE, "train_pe8.pkl", max_seq=args.max_seq_length)
    if args.limit_train:
        train_feats = train_feats[:args.limit_train]
    print(f"[data] train feats={len(train_feats)}", flush=True)

    new_layer = ["extractor", "bilinear", "classifier"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)]},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)],
         "lr": args.classifier_lr}]
    opt = torch.optim.AdamW(grouped, lr=args.learning_rate, eps=1e-6)
    total_steps = int(len(train_feats) / args.train_batch_size * args.epochs // args.grad_accum)
    warmup = int(total_steps * 0.06)
    sched = torch.optim.lr_scheduler.LambdaLR(opt,
        lambda step: min(1.0, step / max(1, warmup)) if step < warmup
        else max(0.0, (total_steps - step) / max(1, total_steps - warmup)))
    print(f"[train] steps={total_steps} warmup={warmup}", flush=True)

    save_dir = os.path.join(HERE, f"docunetL_pe8_s{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    log_fh = open(os.path.join(save_dir, "train_log.txt"), "a", encoding="utf-8")

    def log(s):
        line = f"{s}"
        print(line, flush=True)
        log_fh.write(line + "\n"); log_fh.flush()

    num_steps, t0 = 0, time.time()
    scaler = torch.amp.GradScaler("cuda")
    train_dl = DataLoader(train_feats, batch_size=args.train_batch_size, shuffle=True,
                          collate_fn=collate_fn, drop_last=True)
    model.zero_grad()
    best_f1, bad_epochs, best_epoch = -1.0, 0, -1
    dev_path = args.dev_file or os.path.join(ROOT, "data", "docred_local", "dev.json")
    history = []
    for epoch in range(args.epochs):
        model.train()
        ep_loss, n_ep = 0.0, 0
        for step, batch in enumerate(train_dl):
            opt.zero_grad(); model.train()
            inputs = {"input_ids": batch[0].to(args.device),
                      "attention_mask": batch[1].to(args.device),
                      "labels": batch[2],
                      "entity_pos": batch[3], "hts": batch[4]}
            with torch.autocast("cuda", dtype=torch.float16):
                loss, _ = model(**inputs)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(opt); scaler.update(); sched.step()
            num_steps += 1
            ep_loss += float(loss); n_ep += 1
            if step % 500 == 0:
                log(f"[e{epoch} s{step}/{len(train_dl)}] loss={float(loss):.4f} "
                    f"({(time.time()-t0)/60:.1f}min)")
        # ── 每 epoch dev 评估 + 早停 ──
        p, r, f1 = evaluate(model, tok, dev_path, args.device)
        history.append({"epoch": epoch, "dev_p": p, "dev_r": r, "dev_f1": f1})
        log(f"[e{epoch}] dev P={p:.4f} R={r:.4f} F1={f1:.4f} "
            f"best={best_f1:.4f} elapsed={(time.time()-t0)/60:.1f}min")
        if f1 > best_f1:
            best_f1, best_epoch, bad_epochs = f1, epoch, 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best.ckpt"))
            log(f"[e{epoch}] new best -> best.ckpt (F1={f1:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log(f"[early-stop] epoch={epoch} no improvement for {args.patience} epochs "
                    f"(best e{best_epoch} F1={best_f1:.4f})")
                break

    torch.save(model.state_dict(), os.path.join(save_dir, "last.ckpt"))
    report = {"seed": args.seed, "epochs_run": len(history), "best_epoch": best_epoch,
              "best_dev_f1": best_f1, "history": history,
              "early_stopped": bad_epochs >= args.patience,
              "wall_hours": (time.time()-t0)/3600,
              "train_file": TRAIN_FILE, "train_sha256": sha256_file(TRAIN_FILE),
              "dev_file": dev_path, "generated": datetime.datetime.now().isoformat(timespec="seconds")}
    json.dump(report, open(os.path.join(save_dir, "train_report.json"), "w"), indent=1)
    log(f"[done] {save_dir} wall={report['wall_hours']:.1f}h best_F1={best_f1:.4f}")


if __name__ == "__main__":
    import datetime
    main()
