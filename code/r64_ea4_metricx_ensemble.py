# -*- coding: utf-8 -*-
"""r64_ea4_metricx_ensemble.py — E-A4: add a learned metric (MetricX-23-Large,
MQM-trained, ref-based) to the registered WMT19 contract set and re-run the
canonical AER pipeline (r60 protocol).

Stage 1 (GPU): score all (pair, system) segments with MetricX-23-Large
       (google/metricx-23-large-v2p0, local snapshot; input
       "candidate: {hyp} reference: {ref}", max_len 1024, batch 32;
       prediction = MQM error count, lower=better -> quality = -pred).
Stage 2 (CPU): contracts = {bleu13a, chrf, bertscore, metricx23}; point layer
       z-ensemble (r60 protocol); bootstrap R=1000 (exploratory; r60 used 2000),
       corpus bleu/chrf recomputed per resample, bertscore/metricx = mean of
       cached segment scores; anchor = wmt-raw. Reports: per-contract tau,
       best-single identity (3-contract vs 4-contract), TOST verdicts at
       eps .05/.10, coverage of the 14 registered flips under ensemble/median/
       singles. Output: r64_ea4_metricx_ensemble.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import sacrebleu

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_OFFLINE', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

BASE = os.path.dirname(os.path.abspath(__file__))
MX = r'E:\WorkBuddy\metricx-main'
MT = os.path.join(BASE, '..', '..', 'multi_task', 'mt')
FLIPS = json.load(open(os.path.join(MT, 'stats', '_e2_flip_pairs.json')))
OUT = os.path.join(BASE, 'r64_ea4_metricx_ensemble.json')
SEG_CACHE = os.path.join(BASE, '_ea4_mx_seg.json')
R = 1000
SEED = 7
EPS = [0.05, 0.10]
PAIRS = ['de-en', 'zh-en']
C4 = ['bleu13a', 'chrf', 'bertscore', 'metricx23']


def kendall_tau_b(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    conc = disc = tx = ty = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            sx = np.sign(x[i] - x[j]); sy = np.sign(y[i] - y[j])
            if sx == 0 and sy == 0: tx += 1; ty += 1
            elif sx == 0: tx += 1
            elif sy == 0: ty += 1
            elif sx * sy > 0: conc += 1
            else: disc += 1
    den = np.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / den if den else 1.0


def zsc(m):
    return (m - m.mean(0)) / (m.std(0, ddof=1) + 1e-12)


def stage1():
    import torch
    import transformers
    sys.path.insert(0, MX)
    import transformers.models.mt5.modeling_mt5 as _m5
    for _sym in ('__HEAD_MASK_WARNING_MSG',):
        if not hasattr(_m5, _sym):
            setattr(_m5, _sym, '')
    _OrigStack = _m5.MT5Stack
    class _MT5StackCompat(_OrigStack):
        def __init__(self, config, embed_tokens=None, **kw):
            try:
                super().__init__(config, embed_tokens=embed_tokens, **kw)
            except TypeError:
                super().__init__(config, **kw)

        def forward(self, *a, **kw):
            import torch as _t
            ii = kw.get('input_ids')
            am = kw.get('attention_mask')
            if ii is None and a:
                ii = a[0]
            if ii is not None and am is not None and am.dim() == 2 and am.shape[-1] != ii.shape[-1]:
                L = ii.shape[-1]
                if am.shape[-1] < L:
                    pad = _t.ones(*am.shape[:-1], L - am.shape[-1], dtype=am.dtype, device=am.device)
                    am = _t.cat([am, pad], -1)
                else:
                    am = am[..., :L]
                kw['attention_mask'] = am
            return super().forward(*a, **kw)
    _m5.MT5Stack = _MT5StackCompat
    from metricx23 import models
    _tk = r'C:\Users\luhao\.cache\huggingface\hub\models--google--t5-v1_1-base\snapshots\b5fc947a416ea3cb079532cb3c2bbadeb7f800fc'
    import transformers as _tf
    tok = _tf.T5TokenizerFast.from_pretrained(_tk, legacy=False)
    ckpt = r'C:\Users\luhao\.cache\huggingface\hub\models--google--metricx-23-large-v2p0\snapshots'
    ckpt = os.path.join(ckpt, os.listdir(ckpt)[0])
    model = models.MT5ForRegression.from_pretrained(ckpt)
    model.to('cuda'); model.eval()
    seg = {}
    for pair in PAIRS:
        df = pd.read_csv(os.path.join(MT, 'data', f'metrics_task_{pair}.csv'), sep=';')
        df = df[df['dataset'] == 'wmt19']
        wide = df.pivot_table(index='sent_id', columns='mt_system',
                              values='system_output', aggfunc='first')
        wide = wide.drop(columns=['Human'], errors='ignore')
        ref = df.groupby('sent_id')['ref'].first()
        sents = list(wide.index)
        for s in wide.columns:
            hyps = [str(wide.loc[i, s]) if pd.notna(wide.loc[i, s]) else '' for i in sents]
            refs = [str(ref[i]) for i in sents]
            scores = []
            B = 32
            for k in range(0, len(hyps), B):
                texts = [f'candidate: {h} reference: {r}' for h, r in zip(hyps[k:k+B], refs[k:k+B])]
                enc = tok(texts, max_length=1024, truncation=True, padding=False)
                ids_l = [x[:-1] for x in enc['input_ids']]
                am_l = [y[:-1] for y in enc['attention_mask']]
                ids = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(z) for z in ids_l], batch_first=True,
                    padding_value=tok.pad_token_id).to('cuda')
                am = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(z) for z in am_l], batch_first=True,
                    padding_value=0).to('cuda')
                with torch.no_grad():
                    out = model(input_ids=ids, attention_mask=am)
                raw = getattr(out, 'predictions', None)
                if raw is None and isinstance(out, tuple):
                    raw = out[-1]
                if raw is None:
                    raw = out[0]
                raw = raw.view(raw.shape[0], -1)[:, 0] if raw.dim() > 1 else raw.view(-1)
                pred = raw.float().cpu().tolist()
                scores.extend(pred)
                if (k // B) % 10 == 0:
                    print(f'  mx {pair} {s} {k}/{len(hyps)}', flush=True)
            seg[f'{pair}|{s}'] = [-float(x) for x in scores]   # quality = -MQM
            print(f'metricx {pair}|{s}: mean={np.mean(seg[f"{pair}|{s}"]):.3f}', flush=True)
    json.dump(seg, open(SEG_CACHE, 'w'))
    return seg


def main():
    if os.path.exists(SEG_CACHE):
        seg = json.load(open(SEG_CACHE))
    else:
        seg = stage1()
    res = {'meta': {
        'experiment': 'E-A4: MetricX-23-Large (learned, MQM-trained) added to the '
                      'registered contract set; r60 canonical protocol re-run',
        'contracts': C4,
        'bootstrap': {'reps': R, 'unit': 'segment', 'seed': SEED, 'ci': 'percentile',
                      'note': 'R=1000 exploratory (r60 used 2000); corpus bleu13a/chrf '
                              'recomputed per resample; bertscore+metricx = mean of cached '
                              'segment scores; metricx quality = -predicted MQM'},
        'tier': 'exploratory, post-review round 12',
    }, 'grids': {}}
    for pair in PAIRS:
        df = pd.read_csv(os.path.join(MT, 'data', f'metrics_task_{pair}.csv'), sep=';')
        df = df[df['dataset'] == 'wmt19']
        wide = df.pivot_table(index='sent_id', columns='mt_system',
                              values='system_output', aggfunc='first')
        wide = wide.drop(columns=['Human'], errors='ignore')
        ref = df.groupby('sent_id')['ref'].first()
        anchor = df.groupby('mt_system')['wmt-raw'].mean().to_dict()
        systems = [s for s in wide.columns if s in anchor]
        sents = list(wide.index)
        refs = [str(ref[s]) for s in sents]
        hyps = {s: [str(wide.loc[i, s]) if pd.notna(wide.loc[i, s]) else ''
                    for i in sents] for s in systems}
        anch = np.array([anchor[s] for s in systems])
        bs_seg = json.load(open(os.path.join(MT, 'runs', f'_bs_seg_{pair}.json')))
        n = len(sents)
        pt = {
            'bleu13a': np.array([sacrebleu.corpus_bleu(hyps[s], [refs], tokenize='13a',
                                                       force=True).score for s in systems]),
            'chrf': np.array([sacrebleu.corpus_chrf(hyps[s], [refs], word_order=2,
                                                    beta=2).score for s in systems]),
            'bertscore': np.array([np.mean(bs_seg[s]) for s in systems]),
            'metricx23': np.array([np.mean(seg[f'{pair}|{s}']) for s in systems]),
        }
        taus = {c: kendall_tau_b(pt[c], anch) for c in C4}
        best3 = max(['bleu13a', 'chrf', 'bertscore'], key=lambda c: taus[c])
        best4 = max(C4, key=lambda c: taus[c])
        rng = np.random.default_rng(SEED)
        boots = {c: [] for c in C4}
        ens_rows, med_rows = [], []
        for it in range(R):
            idx = rng.integers(0, n, n)
            rs = [refs[j] for j in idx]
            col = {}
            col['bleu13a'] = np.array([sacrebleu.corpus_bleu([hyps[s][j] for j in idx], [rs],
                                    tokenize='13a', force=True).score for s in systems])
            col['chrf'] = np.array([sacrebleu.corpus_chrf([hyps[s][j] for j in idx], [rs],
                                    word_order=2, beta=2).score for s in systems])
            col['bertscore'] = np.array([np.mean([bs_seg[s][j] for j in idx]) for s in systems])
            col['metricx23'] = np.array([np.mean([seg[f'{pair}|{s}'][j] for j in idx])
                                         for s in systems])
            for c in C4:
                boots[c].append(col[c])
            Z = zsc(np.stack([col[c] for c in C4]).T)
            ens_rows.append(Z.mean(1))
            med_rows.append(np.median(Z, axis=1))
            if (it + 1) % 100 == 0:
                print(f'  boot {pair} {it+1}', flush=True)
        col_mat = {c: np.vstack(boots[c]) for c in C4}
        ens = np.vstack(ens_rows); med = np.vstack(med_rows)
        Zp = zsc(np.stack([pt[c] for c in C4]).T)
        ens_pt = Zp.mean(1)
        tau_ens = kendall_tau_b(ens_pt, anch)
        taus_ens_boot = np.array([kendall_tau_b(ens[r], anch) for r in range(R)])
        taus_best_boot = np.array([kendall_tau_b(col_mat[best4][r], anch) for r in range(R)])
        d = taus_ens_boot - taus_best_boot
        lo90, hi90 = np.percentile(d, [5, 95])
        tost4 = {f'eps={e}': bool(lo90 >= -e and hi90 <= e) for e in EPS}
        pos = {s: i for i, s in enumerate(systems)}
        def tie(mat, ia, ib):
            dd = mat[:, ia] - mat[:, ib]
            return bool(np.percentile(dd, 2.5) <= 0 <= np.percentile(dd, 97.5))
        cov = {'ens4': 0, 'median4': 0, 'n': 0}
        for c in C4:
            cov[c] = 0
        for a, b in FLIPS[pair]:
            ia, ib = pos[a], pos[b]
            cov['n'] += 1
            cov['ens4'] += tie(ens, ia, ib)
            cov['median4'] += tie(med, ia, ib)
            for c in C4:
                cov[c] += tie(col_mat[c], ia, ib)
        res['grids'][pair] = {
            'n_systems': len(systems),
            'tau_single': {c: round(taus[c], 4) for c in C4},
            'best_single_3contract': best3, 'best_single_4contract': best4,
            'tau_ens4': round(tau_ens, 4),
            'delta_ens4_vs_best4_ci90': [round(float(lo90), 4), round(float(hi90), 4)],
            'tost_ens4_vs_best4': tost4,
            'flip_coverage': cov,
            'note': 'compare with r60 3-contract canonical: does best-single identity or '
                    'TOST verdict change when the learned metric enters the set?',
        }
        g = res['grids'][pair]
        print(pair, json.dumps(g, ensure_ascii=False), flush=True)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('saved', OUT)


if __name__ == '__main__':
    main()
