# -*- coding: utf-8 -*-
"""_p11_ea11_ood_tost.py — E-A11: out-of-sample margin confirmation for
review item DA-m3 (epsilon=0.10 introduced post-freeze from an exploratory
scan; convention-stability is a disclosed but post-hoc selection rule with
no out-of-sample grid confirmation).

Design:
- Target grid: newstest2021 EN->DE (English source, German target). This
  grid is out-of-sample on two axes relative to the WMT19 de-en/zh-en
  calibration grids: translation direction AND evaluation year.
- Anchor: WMT21 regular human DA judgments, eng*deu rows (all of which
  are src=eng, tgt=deu; verified). Official parsing follows the shipped
  make_rankings.py: keep class==TGT and doc_score==False; z-score per
  annotator; segment score = mean z per (system, doc, seg); system score
  = mean over segments.
- Segment alignment: DA segment ids are document-relative; the SGM test
  files provide the (doc id, seg-in-doc) -> global line mapping.
- Contracts: sentence BLEU-13a, sentence chrF (sacrebleu), BERTScore F
  (bert-base-multilingual-cased, lang=de, GPU) on DA-covered segments.
- AER: z-ensemble of the three contracts (z per segment across systems,
  mean z per system). Comparison: tau_b(AER, anchor) vs tau_b(best single
  contract, anchor); bootstrap over DA-covered segments, R=2000,
  seed 20260926; TOST on delta tau at eps in {0.05, 0.10}.

Output: r64_ea11_ood_tost.json (this directory).
"""
import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import sacrebleu

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASE, '..', 'wmt21-news-systems-main')
SGM_DIR = os.path.join(BASE, '..', '_sgm_wmt21', 'test')
PAIR = 'en-de'          # English source -> German target
LP_TAG = 'ende'         # WMT21 SGM naming
SEED = 20260926
R = 2000
EPS_LIST = (0.05, 0.10)
OUT = os.path.join(BASE, 'r64_ea11_ood_tost.json')


# ---------------------------------------------------------------- map
def build_docmap():
    """(doc id, seg-in-doc 0-based) -> global 0-based line index, from the
    official WMT21 test XML. The news subset (non-tsuite docs) line order is
    validated line-for-line against the shipped txt sources file; DA segment
    ids verified 0-based (1002/1002 hit, 1-based only 934)."""
    from xml.sax.saxutils import unescape
    xml_path = os.path.join(SGM_DIR, f'newstest2021.{PAIR}.xml')
    xml = open(xml_path, encoding='utf-8').read()
    doc_re = re.compile(r'<doc ([^>]*)>')
    seg_re = re.compile(r'<seg id="(\d+)">([^<]*)</seg>')
    src_re = re.compile(r'<src lang="en">(.*?)</src>', re.S)
    lines = []
    docmap = {}
    for m in doc_re.finditer(xml):
        idm = re.search(r'id="([^"]+)"', m.group(1))
        if not idm or idm.group(1).startswith('tsuite'):
            continue
        end = xml.find('</doc>', m.end())
        body = xml[m.end():end]
        sm2 = src_re.search(body)
        if not sm2:
            continue
        for sm in seg_re.finditer(sm2.group(1)):
            docmap[(idm.group(1), int(sm.group(1)) - 1)] = len(lines)
            lines.append(unescape(sm.group(2)).strip())
    src_file = os.path.join(REPO, 'txt', 'sources', f'newstest2021.{PAIR}.src.en')
    src = [unescape(l).strip()
           for l in open(src_file, encoding='utf-8') if l.strip()]
    same = sum(1 for a, b in zip(lines, src) if a == b)
    print(f'docmap: {len(docmap)} pairs; line order check {same}/{len(src)}',
          flush=True)
    if same != len(src) or len(lines) != len(src):
        print('FATAL: XML/txt line order mismatch', file=sys.stderr)
        sys.exit(2)
    return docmap


# ------------------------------------------------------------ auto scores
def load_pair():
    so = os.path.join(REPO, 'txt', 'system-outputs')
    rf = os.path.join(REPO, 'txt', 'references')
    tgt = PAIR.split('-')[1]
    ref_files = sorted(glob.glob(os.path.join(rf, f'newstest2021*.{PAIR}.ref.*.{tgt}')))
    refs = [l.rstrip('\n') for l in open(ref_files[0], encoding='utf-8')]
    keep = {}
    for p in glob.glob(os.path.join(so, f'newstest2021*.{PAIR}.hyp.*')):
        b = os.path.basename(p)
        m = re.match(rf'newstest2021[a-z0-9]*[.]{re.escape(PAIR)}[.]hyp[.](.+)[.][a-z]{{2}}$', b)
        if not m:
            continue
        lines = [l.rstrip('\n') for l in open(p, encoding='utf-8')]
        if len(lines) == len(refs) and all(x.strip() for x in lines):
            keep[m.group(1)] = lines
    return keep, refs


def bertscore_f(hyps, refs, device='cuda'):
    from bert_score import score as bs_score
    P, R_, F = bs_score(hyps, refs, lang='de', verbose=True, device=device,
                        batch_size=64)
    return F.numpy()


# ------------------------------------------------------------ DA parsing
def parse_da(keep, docmap):
    """Returns (ann_judg, n_ann): ann_judg[annotator] = list of
    (system, global_seg, z). Official recipe: TGT only, segment-level only
    (doc_score False), z per annotator. DA coverage per system is a sparse
    subset of segments (WMT design); system scores average over each
    system's own judged segments."""
    da_path = os.path.join(REPO, 'humaneval', 'newstest2021-EX_and_XY',
                           'wmt21-regular.20210930.csv')
    rows = []
    n_raw = n_miss = 0
    with open(da_path, encoding='utf-8') as f:
        for r in csv.reader(f):
            if len(r) < 11 or not r[0].startswith('engdeu'):
                continue
            n_raw += 1
            if r[3] != 'TGT' or r[8] == 'True':   # keep TGT, drop doc scores
                continue
            sysname = r[1]
            if sysname not in keep:
                continue
            g = (r[7], int(r[2]))                # (doc id, seg-in-doc 0-based)
            if g not in docmap:
                n_miss += 1
                continue
            rows.append((r[0], sysname, docmap[g], float(r[6])))
    print(f'DA rows: raw {n_raw} kept {len(rows)} unmapped {n_miss}', flush=True)

    by_ann = defaultdict(list)
    for ann, s, g, v in rows:
        by_ann[ann].append(v)
    stats = {a: (float(np.mean(v)), float(np.std(v))) for a, v in by_ann.items()}
    ann_judg = defaultdict(list)
    for ann, s, g, v in rows:
        mu, sd = stats[ann]
        if sd <= 0:
            continue
        ann_judg[ann].append((s, g, (v - mu) / sd))
    n_ann = len(ann_judg)
    print('annotators:', n_ann, 'z-judgments:',
          sum(len(v) for v in ann_judg.values()), flush=True)
    return ann_judg, n_ann


# ----------------------------------------------------------------- stats
def tau_b(x, y):
    n = len(x)
    nc = nd = nx = ny = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = x[i] - x[j]
            b = y[i] - y[j]
            s = int(a > 0) - int(a < 0)
            t = int(b > 0) - int(b < 0)
            if s * t > 0:
                nc += 1
            elif s * t < 0:
                nd += 1
            if a == 0:
                nx += 1
            if b == 0:
                ny += 1
    den = math.sqrt((nc + nd + nx) * (nc + nd + ny))
    return (nc - nd) / den if den else 0.0


def main():
    docmap = build_docmap()
    keep, refs = load_pair()
    names = sorted(keep)
    n_seg = len(refs)
    print(f'newstest2021 {PAIR}: {len(names)} systems, {n_seg} segments',
          flush=True)

    ann_judg, n_ann = parse_da(keep, docmap)
    judged = sorted(set(s for v in ann_judg.values() for s, _, _ in v))
    print('DA-judged systems:', len(judged), judged, flush=True)
    cov = {s: len(set(g for _, g, _ in (x for v in ann_judg.values()
                                        for x in v if x[0] == s)))
           for s in judged}
    print('per-system DA segment coverage:', cov, flush=True)

    # ---- anchor (official recipe): system score = mean over its judged
    # segments of segment-mean z
    def anchor_from(judg_lists):
        seg_acc = defaultdict(list)
        for lst in judg_lists:
            for s, g, z in lst:
                seg_acc[(s, g)].append(z)
        sys_acc = defaultdict(list)
        for (s, g), v in seg_acc.items():
            sys_acc[s].append(float(np.mean(v)))
        return {s: float(np.mean(v)) for s, v in sys_acc.items()}

    all_lists = list(ann_judg.values())
    anchor = anchor_from(all_lists)
    anchor_vec = np.array([anchor[s] for s in judged])

    # ---- contract scores on the FULL grid (the reading a practitioner
    # would report); AER = z-ensemble over the full grid
    hyp_mat = {s: keep[s] for s in judged}
    contracts = {}
    contracts['bleu13a'] = np.array([[sacrebleu.sentence_bleu(hyp_mat[s][j],
                                    [refs[j]], tokenize='13a').score
                                    for j in range(n_seg)] for s in judged])
    contracts['chrf'] = np.array([[sacrebleu.sentence_chrf(hyp_mat[s][j],
                                  [refs[j]], beta=2).score
                                  for j in range(n_seg)] for s in judged])
    bs_path = os.path.join(BASE, '_ea11_bs_en-de.npy')
    if os.path.exists(bs_path):
        bs = np.load(bs_path)
        print('bertscore cache hit', bs.shape, flush=True)
    else:
        flat_h, flat_r = [], []
        for s in judged:
            flat_h.extend(keep[s])
            flat_r.extend(refs)
        f = bertscore_f(flat_h, flat_r)
        bs = f.reshape(len(judged), n_seg)
        np.save(bs_path, bs)
    contracts['bertscore'] = bs
    print('contracts ready:', {k: v.shape for k, v in contracts.items()},
          flush=True)

    def zseg(mat):
        mu = mat.mean(0, keepdims=True)
        sd = mat.std(0, keepdims=True)
        sd[sd == 0] = 1.0
        return (mat - mu) / sd

    zc = {c: zseg(contracts[c]) for c in contracts}
    ens = np.mean(np.stack([zc[c] for c in zc]), 0)   # (systems, segments)
    aer = zseg(ens).mean(1)
    singles = {c: zc[c].mean(1) for c in contracts}

    tau_aer = tau_b(aer, anchor_vec)
    tau_single = {c: tau_b(singles[c], anchor_vec) for c in singles}
    best = max(tau_single, key=tau_single.get)
    print('tau(AER)=%.4f | singles %s | best=%s %.4f'
          % (tau_aer, {k: round(v, 4) for k, v in tau_single.items()},
             best, tau_single[best]), flush=True)

    # ---- dual bootstrap: auto side resampled over segments, anchor side
    # resampled over annotators (independent resamples per iteration)
    rng = np.random.default_rng(SEED)
    n_ann_l = len(all_lists)
    deltas = np.empty(R)
    for it in range(R):
        idx = rng.integers(0, n_seg, n_seg)
        res = {c: zc[c][:, idx].mean(1) for c in contracts}
        ra = zseg(ens[:, idx]).mean(1)
        pick = rng.integers(0, n_ann_l, n_ann_l)
        anc_b = anchor_from([all_lists[i] for i in pick])
        anc_vec = np.array([anc_b[s] for s in judged])
        rb = max((res[c] for c in res), key=lambda v: tau_b(v, anc_vec))
        deltas[it] = tau_b(ra, anc_vec) - tau_b(rb, anc_vec)
        if (it + 1) % 200 == 0:
            print('  boot', it + 1, flush=True)
    lo = float(np.percentile(deltas, 5))
    hi = float(np.percentile(deltas, 95))
    tost = {'%.2f' % e: bool(lo > -e and hi < e) for e in EPS_LIST}

    out = {
        'meta': {
            'experiment': 'E-A11 out-of-sample margin confirmation (DA-m3): '
                          'TOST of AER vs best single contract on '
                          'newstest2021 en-de',
            'out_of_sample_axes': ['translation direction (en->de vs WMT19 '
                                   'de-en/zh-en calibration)',
                                   'year (2021 vs 2019)'],
            'anchor': 'WMT21 regular human DA (engdeu rows, src=eng tgt=deu '
                      'verified); official make_rankings parsing: TGT only, '
                      'segment-level only, z per annotator, system score = '
                      'mean over the system own judged segments (sparse '
                      'coverage per WMT design)',
            'aer_reading': 'z-ensemble (per-segment z across systems, mean '
                           'z per system) over the full 1002-segment grid',
            'alignment': '(doc id, seg-in-doc, 0-based) -> global line via '
                         'official WMT21 test XML (test.tgz); line order '
                         'validated 1002/1002 against shipped txt sources',
            'contracts': ['bleu13a', 'chrf', 'bertscore'],
            'bertscore': 'bert-base-multilingual-cased, lang=de, GPU',
            'systems': len(judged),
            'system_names': judged,
            'da_coverage_per_system': cov,
            'annotators': n_ann,
            'z_judgments': sum(len(v) for v in ann_judg.values()),
            'R': R, 'seed': SEED,
            'bootstrap': 'auto side: resample 1002 segments; anchor side: '
                         'resample annotators; independent per iteration',
            'tost_rule': 'CI90(delta tau) within (-eps, +eps)'},
        'tau_aer': round(tau_aer, 4),
        'tau_single': {k: round(v, 4) for k, v in tau_single.items()},
        'best_single': best,
        'delta_full': round(tau_aer - tau_single[best], 4),
        'bootstrap_delta_ci90': [round(lo, 4), round(hi, 4)],
        'bootstrap_delta_mean': round(float(deltas.mean()), 4),
        'tost_pass_by_eps': tost,
    }
    json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1)
    print('saved ->', OUT)
    print(json.dumps(out, indent=1)[:1400])


if __name__ == '__main__':
    main()
