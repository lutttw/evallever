# -*- coding: utf-8 -*-
"""phase4_multi.py — 路径B分析: ar/cs/hi 三语言的 UD 金标轴机制全套.

(1) 逐臂金标交换 Δ: acc(g2) − acc(g1m), 文档配对 bootstrap CI (10k, seed 20260926);
(2) 训练契约符号反转: A/A′ 逐种子 sign(γ_g2 − γ_g1m) 计数;
(3) 家族对翻转: 每对系统在 g2 vs g1m 下的排序符号;
(4) gauge 切片(C 臂 token 级): 改名位/重标注位/未变位的移动定位.
判定(rev113 冻结): 主判据=各臂配对 Δ CI 排零; 阴性同报.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = HERE / "arms"
BOOT = 10_000
SEED = 20260926
LANGS = ["ar", "cs", "hi"]


PREFIX = {"A": "perc_A", "A'": "perc_A'", "B": "bert_B", "B'": "bert_B'"}


def load_arm(lang, arm):
    out = []
    for f in sorted(ARMS.glob(f"{PREFIX[arm]}_{lang}_s*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out.append(d["per_sentence"])
    return out


def boot_ci(diffs, rng):
    n = len(diffs)
    pt = sum(diffs) / n
    st = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                for _ in range(BOOT))
    return pt, st[int(0.025 * BOOT)], st[int(0.975 * BOOT) - 1]


def arm_delta(lang, arm):
    """同臂预测在 g2 vs g1m 下的逐文档读数差 (acc_g1m − acc_g2)."""
    seeds = load_arm(lang, arm)
    per_doc = {}
    for recs in seeds:
        for r in recs:
            d = r["doc"]
            a2 = r["c2"] / r["n"]
            a1 = r["c1"] / r["nm"] if r["nm"] else None
            per_doc.setdefault(d, []).append((a2, a1))
    docs = sorted(d for d, v in per_doc.items() if all(a1 is not None for _, a1 in v))
    da = {d: (sum(x[1] for x in per_doc[d]) / len(per_doc[d])) for d in docs}
    db = {d: (sum(x[0] for x in per_doc[d]) / len(per_doc[d])) for d in docs}
    diffs = [db[d] - da[d] for d in docs]
    rng = random.Random(SEED)
    pt = sum(diffs) / len(diffs)
    st = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))) /
                len(diffs) for _ in range(BOOT))
    return {"mean": round(pt, 4), "ci": [round(st[250], 4), round(st[9749], 4)],
            "n_docs": len(docs),
            "excl_zero": bool(st[250] > 0 or st[9749] < 0)}


def sign_reversal(lang):
    """A/A′ 逐种子: g2 训练的臂在双金标下的符号 vs g1m 训练的臂."""
    def seed_signs(arm):
        out = {}
        pre = PREFIX[arm]
        for f in sorted(ARMS.glob(f"{pre}_{lang}_s*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            t2 = sum(r["c2"] for r in d["per_sentence"])
            n2 = sum(r["n"] for r in d["per_sentence"])
            t1 = sum(r["c1"] for r in d["per_sentence"])
            n1 = sum(r["nm"] for r in d["per_sentence"])
            out[d["seed"]] = (t2 / n2) - (t1 / n1)
        return out
    sa, sap = seed_signs("A"), seed_signs("A'")
    seeds = sorted(set(sa) & set(sap))
    rev = sum(1 for s in seeds if (sa[s] > 0) != (sap[s] > 0))
    return {"n_seeds": len(seeds), "reversals": rev,
            "A_mean": round(sum(sa[s] for s in seeds) / len(seeds), 4),
            "Ap_mean": round(sum(sap[s] for s in seeds) / len(seeds), 4)}


def gauge_slices_v2(lang):
    """切片正解: 同一预测, g2 金标正确率 vs g1m 金标正确率, 按位分类."""
    align = {json.loads(l)["sid"]: json.loads(l)
             for l in (HERE / "align" / f"{lang}_dev.jsonl").open(encoding="utf-8")}
    pred = json.loads((ARMS / f"stanza_C_{lang}.json").read_text(encoding="utf-8"))
    sl = {"ren": [0, 0, 0], "reann": [0, 0, 0], "unch": [0, 0, 0]}
    for p in pred:
        a = align.get(p["sid"])
        if a is None or len(a["toks"]) != len(p["upos"]):
            continue
        for i, t in enumerate(a["toks"]):
            ok2 = (p["head"][i] == t["h2"] and p["deprel"][i] == t["d2"])
            ok1 = (p["head"][i] == t["h1"] and p["deprel"][i] == t["d1m"])
            if t["ren_u"] or t["ren_d"]:
                k = "ren"
            elif (t["u2"] != t["u1m"] or t["d2"] != t["d1m"] or t["h2"] != t["h1"]):
                k = "reann"
            else:
                k = "unch"
            sl[k][0] += 1
            sl[k][1] += ok2
            sl[k][2] += ok1
    return {k: {"n": v[0], "acc_g2": round(v[1] / v[0], 4) if v[0] else None,
                "acc_g1m": round(v[2] / v[0], 4) if v[0] else None}
            for k, v in sl.items()}


def main():
    res = {"boot": {"n": BOOT, "seed": SEED}, "langs": {}}
    for lang in LANGS:
        L = {}
        for arm in ["A", "A'", "B", "B'", "C"]:
            if arm == "C":
                continue
            try:
                L[f"delta_{arm}"] = arm_delta(lang, arm)
            except Exception as e:
                L[f"delta_{arm}"] = {"error": str(e)}
        L["sign_reversal"] = sign_reversal(lang)
        L["gauge_slices_C"] = gauge_slices_v2(lang)
        res["langs"][lang] = L
    out = HERE / "results" / "phase4_multi.json"
    res["status"] = "FROZEN"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
