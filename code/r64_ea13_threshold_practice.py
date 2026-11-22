# -*- coding: utf-8 -*-
"""B2 step 2: download + classify from direct anthology/AAAI/arXiv IDs."""
import json
import os
import re
import subprocess
import urllib.request

OUTDIR = r'E:\WorkBuddy\参考文献解读\paper4_meta_aggregation\out\b2_threshold_practice'
PDFDIR = os.path.join(OUTDIR, 'pdfs')
os.makedirs(PDFDIR, exist_ok=True)

PAPERS = [
    ('ATLOP', 'https://arxiv.org/pdf/2010.11304'),
    ('DREEAM', 'https://aclanthology.org/2023.eacl-main.145.pdf'),
    ('KD-DocRE', 'https://aclanthology.org/2022.findings-acl.132.pdf'),
    ('DocuNet', 'https://www.ijcai.org/proceedings/2021/0551.pdf'),
    ('TTM-RE', 'https://aclanthology.org/2024.acl-long.26.pdf'),
    ('Global-RE', 'https://aclanthology.org/2025.findings-acl.1002.pdf'),
    ('Re-DocRED', 'https://aclanthology.org/2022.emnlp-main.580.pdf'),
    ('SSAN', 'https://arxiv.org/pdf/2102.10249'),
    ('SAIS', 'https://aclanthology.org/2022.naacl-main.359.pdf'),
    ('SIRE', 'https://aclanthology.org/2021.findings-acl.47.pdf'),
    ('Zhang2022-Better', 'https://aclanthology.org/2022.emnlp-main.568.pdf'),
]
# ATLOP is EMNLP 2021 main: D21-1527? verify by content check after download


def download(url, path):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as r, open(path, 'wb') as f:
            f.write(r.read())
        return os.path.getsize(path) > 20000
    except Exception as e:
        print('  dl fail', url[-40:], str(e)[:60])
        return False


def text_of(pdf):
    txt = pdf[:-4] + '.txt'
    if not os.path.exists(txt):
        subprocess.run(['pdftotext', pdf, txt], capture_output=True)
    if os.path.exists(txt):
        return open(txt, encoding='utf-8', errors='ignore').read()
    return ''


def classify(text):
    tl = text.lower()
    snips = []
    for m in re.finditer(r'threshold', tl):
        i = m.start()
        s = re.sub(r'\s+', ' ', tl[max(0, i - 140):i + 140])
        if s not in snips:
            snips.append(s)
    adaptive = bool(re.search(r'adaptive threshold', tl)) or \
        bool(re.search(r'thresholds? (are|is) learn|learned threshold|threshold learning', tl))
    fixed05 = bool(re.search(r'threshold of 0\.5|threshold.*0\.5|0\.5 as the threshold|greater than 0\.5|score of at least 0\.5', tl))
    tuned = bool(re.search(r'threshold .{0,40}(tuned|searched|selected) on (the )?(dev|validation)|grid search.{0,40}threshold', tl))
    if adaptive and not fixed05:
        pol = 'adaptive/learned'
    elif fixed05 and not adaptive:
        pol = 'fixed 0.5'
    elif adaptive and fixed05:
        pol = 'both mentioned'
    elif tuned:
        pol = 'tuned'
    else:
        pol = 'unspecified'
    return pol, snips[:8], {'adaptive': adaptive, 'fixed05': fixed05, 'tuned': tuned}


def main():
    out = []
    for tag, url in PAPERS:
        print('==', tag, flush=True)
        pdf = os.path.join(PDFDIR, tag + '.pdf')
        if not os.path.exists(pdf):
            if not download(url, pdf):
                out.append({'tag': tag, 'policy': 'download-fail', 'url': url})
                continue
        pol, snips, flags = classify(text_of(pdf))
        title_ok = None
        out.append({'tag': tag, 'url': url, 'policy': pol, 'flags': flags,
                    'snippets': snips})
        print('  ->', pol, '| adaptive:', flags['adaptive'], 'fixed05:', flags['fixed05'])
    json.dump(out, open(os.path.join(OUTDIR, 'b2_direct.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)
    print('saved b2_direct.json')


if __name__ == '__main__':
    main()
