# -*- coding: utf-8 -*-
"""r46_inventory_narrowing.py — 修订77:受控清单收窄实验(纯 CPU,零新推断)。

假设:大模型关系知识面更宽 → 在窄清单下 rel_unmapped 越多 → 违规率越高。
设计:同一批 Qwen-7B/14B 英文 DocRED 预测,人为收窄接受清单
  (取训练频率最高的 top-W 类),计违规率随 W 变化:
  96(全) → 48 → 26 → 16 → 10
关键指标:delta(W) = rate_14B(W) − rate_7B(W);若 |delta| 随 W 减小而增大 ⇒ 证实机制。
"""
import json
import gzip
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D5 = ROOT / "out" / "d5_llm_hook"
STAT = ROOT / "out" / "revision_v5_experiments" / "statistics"
REL_IDX = ROOT / "out" / "lre_rebuttal_v2" / "prepared" / "relation_to_index.json"
TRAIN = ROOT / "data" / "docred_official_hf" / "train_annotated.json.gz"

WIDTHS = [96, 64, 48, 40, 32, 26, 16, 10]
MODELS = {
    "Qwen-7B": "preds_qwen2.5-7b-instruct_json.jsonl",
    "Qwen-14B": "preds_qwen2.5-14b-instruct_json.jsonl",
}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8").read().split("\n") if l.strip()]


def main() -> int:
    # 1. 训练集关系频率(决定 top-W 收窄的排序)
    freq = Counter()
    with gzip.open(str(TRAIN), "rt", encoding="utf-8") as f:
        for dd in json.load(f):
            for lab in dd["labels"]:
                freq[lab["r"]] += 1
    by_freq = [r for r, _ in freq.most_common()]
    print(f"train relation types: {len(freq)}, total labels: {sum(freq.values())}")

    # 2. 加载两模型的全部预测关系码
    model_rels = {}
    model_n = {}
    for label, fname in MODELS.items():
        rel_codes = []
        n = 0
        for r in jl(D5 / fname):
            for h, rl, t in _parse(r):
                rel_codes.append(rl)
                n += 1
        model_rels[label] = rel_codes
        model_n[label] = n
        print(f"{label}: {n} triples, {len(set(rel_codes))} unique relations")

    # 3. 受控收窄:每个 W 下重新计违规率
    results = {"meta": {"amendment": 77,
                        "design": "controlled inventory narrowing on English DocRED predictions; "
                                  "accepted set = top-W relations by training frequency",
                        "widths": WIDTHS, "ts": __import__("datetime").datetime.now().isoformat(timespec="seconds")},
               "per_width": {}, "summary": {}}

    for W in WIDTHS:
        accepted = set(by_freq[:W])
        row = {"inventory_width": W}
        rates = {}
        for label in MODELS:
            total = model_n[label]
            viol = sum(1 for rl in model_rels[label] if rl not in accepted)
            rate = viol / total if total else 0
            rates[label] = rate
            row[f"rate_{label.replace('-','_')}"] = round(rate * 100, 2)
        delta = rates["Qwen-14B"] - rates["Qwen-7B"]
        row["delta_14B_minus_7B_pt"] = round(delta * 100, 2)
        results["per_width"][str(W)] = row
        print(f"W={W:3d} | 7B: {rates['Qwen-7B']*100:5.1f}% | 14B: {rates['Qwen-14B']*100:5.1f}% "
              f"| delta(14B−7B) = {delta*100:+.1f}pt")

    # 4. 汇总:delta 是否随 W 减小而增大(即窄清单下大模型更差)
    ds = [(W, results["per_width"][str(W)]["delta_14B_minus_7B_pt"]) for W in WIDTHS]
    trend = all(ds[i + 1][1] >= ds[i][1] for i in range(len(ds) - 1))
    results["summary"]["delta_monotonically_increasing_as_W_narrows"] = bool(trend)
    results["summary"]["delta_at_96"] = ds[0][1]
    results["summary"]["delta_at_26"] = [x[1] for x in ds if x[0] == 26][0]
    results["summary"]["mechanism_confirmed"] = bool(trend)
    print(f"\nmonotone: {trend} | delta@96={ds[0][1]:+.1f}pt → delta@26={ds[-1][1]:+.1f}pt")

    out = STAT / "r46_inventory_narrowing.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved ->", out)
    return 0


def _parse(r):
    """从一条预测记录解析 (h, rel, t) 三元组(仅映射+违规,不区分类型)。"""
    sys_p = __import__("sys")
    sys_p.path.insert(0, str(ROOT / "src" / "jcip_audit"))
    # 简化:直接用正则从 raw 提取关系码(json 格式)
    import re
    codes = re.findall(r'"(P\d+)"', r.get("raw", ""))
    return [(None, c, None) for c in codes]


if __name__ == "__main__":
    sys.exit(main())
