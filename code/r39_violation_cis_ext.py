# -*- coding: utf-8 -*-
"""r39_violation_cis_ext.py — 修订75:为 p15_violation_cis.json 补 3 臂 CI。

补算臂(均为 p15 审计语义:每文档集合去重):
  mistral-7b_en-hacred_json/freetext  (out/p15_pull_en/,此前 A3 无 CI 的两行)
  qwen-7b-bf16_en-hacred_json         (cloud66,修订65 后审臂)
门:逐臂重算点估计必须与正典审计件 violation_rate 一致(atol 1e-9),否则终止;
原 11 臂冻结值逐位不动。
协议与 r24 逐字一致:10,000 次文档 bootstrap,seed 20260926,percentile 95%。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "jcip_audit"))

import d5_llm_hook as D5                                   # noqa: E402
import p15_hacred_icl as P15                               # noqa: E402

SEED, REPS = 20260926, 10000
STAT = ROOT / "out/revision_v5_experiments/statistics/p15_violation_cis.json"

NEW_ARMS = [  # (展示名, preds jsonl, 正典审计件)
    ("mistral-7b_en-hacred_json",
     ROOT / "out/p15_pull_en/preds_mistral-ms_en_json.jsonl",
     ROOT / "out/p15_pull_en/audit_preds_mistral-ms_en_json.json"),
    ("mistral-7b_en-hacred_freetext",
     ROOT / "out/p15_pull_en/preds_mistral-ms_en_freetext.jsonl",
     ROOT / "out/p15_pull_en/audit_preds_mistral-ms_en_freetext.json"),
    ("qwen-7b-bf16_en-hacred_json",
     ROOT / "out/revision_v5_experiments/ttmre/cloud66/preds_qwen-bf16_en_json.jsonl",
     ROOT / "out/revision_v5_experiments/ttmre/cloud66/audit_bf16.json"),
]


def jl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    frozen = json.loads(STAT.read_text(encoding="utf-8"))
    before = json.dumps(frozen["arms"], sort_keys=True)

    name2ix = {}
    for dd in json.loads(P15.DEV.read_text(encoding="utf-8")):
        did = P15.doc_id_of(dd)
        m = defaultdict(set)
        for i, ent in enumerate(dd["vertexSet"]):
            m[ent[0]["name"]].add(i)
        name2ix[did] = dict(m)

    for label, preds_path, audit_path in NEW_ARMS:
        if label in frozen["arms"]:
            print(f"[skip] {label} 已存在")
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        per = {}
        for r in jl(preds_path):
            did = r["did"]
            if did not in name2ix:
                continue
            n2ix_d = name2ix[did]
            seen = {}
            n = viol = 0
            for h, rl, t in D5.parse_output(r["raw"], r["fmt"], r["ents"],
                                            k_bound=r.get("k_index")):
                key = (h, rl, t)
                if key in seen:
                    continue
                seen[key] = True
                n += 1
                if h in ("<<PARSE_FAIL>>", "<<VIOLATION>>"):
                    viol += 1
                    continue
                if P15.RELS.get(rl) is None:
                    viol += 1
                    continue
                hl, tl = n2ix_d.get(h), n2ix_d.get(t)
                if hl is None or tl is None or len(hl) > 1 or len(tl) > 1:
                    viol += 1
                    continue
                if next(iter(hl)) == next(iter(tl)):
                    viol += 1
            per[did] = (n, viol)
        tot_n = sum(v[0] for v in per.values())
        tot_v = sum(v[1] for v in per.values())
        rate = tot_v / tot_n
        if abs(rate - audit["violation_rate"]) > 1e-9:
            print(f"[GATE FAIL] {label}: 重算 {rate:.10f} != 审计 {audit['violation_rate']:.10f}")
            return 1
        dids = sorted(per)
        n = np.array([per[d][0] for d in dids], dtype=np.int64)
        v = np.array([per[d][1] for d in dids], dtype=np.int64)
        rng = np.random.default_rng(SEED)
        N = len(dids)
        stats = np.empty(REPS)
        for b in range(REPS):
            idx = rng.integers(0, N, N)
            stats[b] = v[idx].sum() / n[idx].sum()
        lo, hi = np.percentile(stats, [2.5, 97.5])
        frozen["arms"][label] = {"point": float(rate), "ci95": [float(lo), float(hi)],
                                 "n_docs": N, "n_triples": int(tot_n),
                                 "n_violations": int(tot_v)}
        a = frozen["arms"][label]
        print(f"[CI+] {label:34s} {a['point']:.4f} [{a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}]"
              f"  n={tot_n}")

    after_old = json.dumps({k: frozen["arms"][k] for k in frozen["arms"]
                            if k not in {x[0] for x in NEW_ARMS}}, sort_keys=True)
    assert json.loads(after_old) == json.loads(before), "原 11 臂被改动!"
    frozen["meta"]["extension_revision75"] = {
        "added": [x[0] for x in NEW_ARMS],
        "note": "3 臂追补(2×Mistral H-en + BF16);原 11 臂逐位未动;"
                "协议与 r24 一致(10k doc bootstrap, seed 20260926);"
                "逐臂门=重算点估计==正典审计 violation_rate(1e-9)"}
    STAT.write_text(json.dumps(frozen, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[out]", STAT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
