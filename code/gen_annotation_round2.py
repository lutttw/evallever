# -*- coding: utf-8 -*-
"""gen_annotation_round2.py — 修订76:第二标注员双标材料包(盲测版)。

输入=冻结 50 条样本(audit_protocol/violation_sample_50.json,r16 seed 20260926)
  + d5 四臂 preds jsonl(逐条 raw)+ DocRED dev 实体表 + 96 关系清单。
输出=docs/ACL_主会执行方案_20261003/annotation_round2/:
  blind_标注表_v2.csv   盲序打乱(seed 20261013),不含第一轮判定/类别
  标注人说明.md         任务书(共享类码本,不含频次与首轮发现)
  管理员说明.md         盲测流程+分析计划(κ/一致率/仲裁)
  blind_key.csv         盲ID→样本ID 对照(仅管理员持有)
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "jcip_audit"))
import d5_llm_hook as D5                                   # noqa: E402

SAMPLE = ROOT / "out/revision_v5_experiments/audit_protocol/violation_sample_50.json"
DST = ROOT / "docs/ACL_主会执行方案_20261003/annotation_round2"
SHUFFLE_SEED = 20261013

CLASSES = ("parse_fail", "oob_index", "self_pair", "rel_unmapped", "ambiguous")


def jl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    DST.mkdir(parents=True, exist_ok=True)
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))["sample"]

    # did → {实体名: {候选索引集}}(与 r24 同一构建)
    _, per_doc, _ = D5.build_canon_map()
    ents_of = {}
    for dd in json.loads(D5.DEV.read_text(encoding="utf-8")):
        did = D5.doc_id_of(dd)
        m = defaultdict(set)
        for raw_i, v in enumerate(dd["vertexSet"]):
            m[str(v[0]["name"]).strip()].add(raw_i)
        ents_of[did] = m
    rel2idx = {k: v for k, v in json.loads(D5.REL_IDX.read_text(encoding="utf-8")).items()
               if k.upper() != "NA"}
    rel_list = " | ".join(f"{v}:{k}" for k, v in sorted(rel2idx.items(), key=lambda kv: kv[1]))

    # preds 行缓存
    rows_by_file = {}
    for r in sample:
        f = r["src_file"]
        if f not in rows_by_file:
            rows_by_file[f] = jl(D5.OUT / f)

    out_rows, key_rows = [], []
    for r in sample:
        line = rows_by_file[r["src_file"]][r["line_no"] - 1]
        did = r["did"]
        named = ents_of[did]
        ent_rows = []
        for name, idxs in sorted(named.items(), key=lambda kv: sorted(kv[1])):
            ent_rows.append(f"{sorted(idxs)[0]}={name}")
        tri = r["triple"]
        tri_s = " | ".join(str(x) for x in tri)
        out_rows.append({
            "盲ID": None,  # 打乱后填
            "提示格式": r["fmt"],
            "文档标题": r["title"],
            "实体表(索引=名称)": " | ".join(ent_rows),
            "允许关系清单(码=名)": rel_list,
            "被标记三元组(头,关系,尾)": tri_s,
            "模型输出原文": line["raw"],
            "你的判定(真违规/边界/误判)": "",
            "若真违规,类别": "",
            "备注": "",
        })
        key_rows.append({"样本ID": r["样本ID"], "violation_class_first_round": r["violation_class"],
                         "src_file": r["src_file"], "line_no": r["line_no"]})

    rng = np.random.default_rng(SHUFFLE_SEED)
    order = rng.permutation(len(out_rows))
    for new_i, old_i in enumerate(order, 1):
        out_rows[old_i]["盲ID"] = f"B{new_i:02d}"
        key_rows[old_i]["盲ID"] = f"B{new_i:02d}"

    cols = list(out_rows[0].keys())
    with (DST / "blind_标注表_v2.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(out_rows)
    with (DST / "blind_key.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key_rows[0].keys()), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(key_rows)

    (DST / "标注人说明.md").write_text("""# 标注任务说明(第二标注员)

## 任务背景(1 分钟)
信息抽取评测中,模型被要求读一篇文档,输出(头实体, 关系, 尾实体)三元组,
其中头/尾实体**必须用实体表里的编号**,关系**必须用给定清单里的关系码**。
我们让若干模型做了这个任务,并用程序扫描了输出中违反规则的片段。
你的任务:独立判断程序标出的片段**是否真的是违规**。

## 你会拿到什么
`blind_标注表_v2.csv`,每行一个被标记片段,含:提示格式(json/freetext)、
文档实体表(索引=名称)、允许关系清单(码=名)、被标记三元组、模型输出原文。

## 你要填什么(三列)
1. **你的判定**:`真违规` / `边界` / `误判`(三选一)
   - 真违规:按下方规则确属违规;
   - 边界:规则不能明确判定(说明原因放备注);
   - 误判:模型输出其实合规,程序标错了。
2. **若真违规,类别**(单选):
   - `parse_fail`:输出整体不是可解析的 json 三元组列表/自由文本无法抽取;
   - `oob_index`:头或尾实体编号超出实体表范围(对照实体表最大编号判断);
   - `self_pair`:头=尾(同一实体自己与自己发生关系);
   - `rel_unmapped`:关系码不在允许清单里(对照清单判断,注意相近名称不算匹配);
   - `ambiguous`:头或尾名称对应实体表中**多个**编号,无法唯一化;
   - `其他`:以上都不符合,在备注写明(发现新类别也算贡献)。
3. **备注**:任何不确定、需要补充上下文的地方。

## 判定纪律
- 只判**被标记的三元组**,但可以看整段输出作为上下文;
- json 格式里实体通常写成编号,freetext 里可能写实体名——都对照实体表判断;
- 实在无法判断就选 `边界`,不要硬选;
- 请独立完成,完成前不要与作者讨论判例;预计 1.5–2 小时,不限时。
""", encoding="utf-8")

    (DST / "管理员说明.md").write_text("""# 双标管理说明(作者自留,勿交标注员)

## 材料
- `blind_标注表_v2.csv`:50 行,盲序打乱(seed 20261013);**不含**第一轮判定与类别,
  也不含任何频次信息——标注员对第一轮结果全盲。
- `blind_key.csv`:盲ID→样本ID→第一轮类别对照,**仅管理员持有**。

## 流程
1. 把 `blind_标注表_v2.csv` + `标注人说明.md` 交给第二标注员;不提供其他项目材料。
2. 收回填写后的 CSV;**在此之前不透露第一轮任何结果**(含"程序真值")。
3. 回收后由管理员(你)用 blind_key 对齐,计算下面三个数。

## 分析计划(预先登记,修订76)
- **检出一致率**:50 条上"真违规/非真违规"的二分一致率(第一轮口径:检测 50/50);
  其中 oob_index 29 条同时对照**程序真值**(canon 上界),报告第二标注员与程序真值的一致率;
- **分类一致率 + Cohen's κ**:在第一轮的 4 个人工可判类
  (parse_fail/self_pair/rel_unmapped/ambiguous,n=21)上,合并"oob/其他"为对比类后
  计算 κ,同时报原始一致率(第一轮口径:分类 48/50=96%);
- **分歧仲裁**:先记录逐条分歧,再向第二标注员揭示第一轮判定,讨论后定稿;
  仲裁不改原始数字,只产出"定稿标签"列;
- **入稿落位**:§6 人工核验句升级为双标注者(报一致率+κ),Limitations(iv) 从
  "single non-blind annotator" 改为 "two annotators (one non-blind), κ=…";
  A5 "Human verification" 行同步;第二轮发现的**新类别**(若有)按修订23b 先例登记。
- **最低可行版本**:若时间不够,只标前 20 行(盲表已随机排序,前 20 即随机子集),
  报告口径同上,n=20 注明。
""", encoding="utf-8")
    print(f"[done] {DST}")
    print(f"  blind rows: {len(out_rows)} | classes covered: "
          f"{sorted(set(r['violation_class_first_round'] for r in key_rows))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
