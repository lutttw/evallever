# -*- coding: utf-8 -*-
"""phase1_multi.py — 路径B: A/A′ 感知机臂 × ar/cs/hi × 15 种子(复用 phase1 机制).

幂等:已存在的 arms/perc_{arm}_{lang}_s{seed}.json 跳过.
CPU 多进程(3 workers),与 GPU 主链无冲突.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase1_perceptron as p1  # noqa: E402

PB = HERE.parent / "ud" / "pathB"
V20 = PB / "v20" / "ud-treebanks-v2.0"
V14 = PB / "v14" / "ud-treebanks-v1.4"

EXT = {
    ("A", "ar"): (V20 / "UD_Arabic/ar-ud-train.conllu", "ar_dev", False),
    ("A'", "ar"): (V14 / "UD_Arabic/ar-ud-train.conllu", "ar_dev", True),
    ("A", "cs"): (V20 / "UD_Czech/cs-ud-train.conllu", "cs_dev", False),
    ("A'", "cs"): (V14 / "UD_Czech/cs-ud-train.conllu", "cs_dev", True),
    ("A", "hi"): (V20 / "UD_Hindi/hi-ud-train.conllu", "hi_dev", False),
    ("A'", "hi"): (V14 / "UD_Hindi/hi-ud-train.conllu", "hi_dev", True),
}
p1.CFG.update(EXT)

JOBS = [(arm, lang, seed)
        for (arm, lang) in EXT
        for seed in p1.SEEDS
        if not (ARMS := HERE / "arms").exists()
        or not (ARMS / f"perc_{arm}_{lang}_s{seed}.json").exists()]


def main():
    print(f"[multi] jobs todo: {len(JOBS)} / 90")
    if not JOBS:
        return
    with Pool(3) as pool:
        for msg in pool.imap_unordered(p1.train_seed, JOBS):
            print("[multi]", msg, flush=True)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
