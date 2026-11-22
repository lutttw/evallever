# -*- coding: utf-8 -*-
"""evallever.cli —— 四命令命令行入口。

用法:
  eval-lever arm   --pred-a PRED_A --pred-b PRED_B --gold GOLD [--metric f1]
                   [--counts-mode --counts-a SRC_A --counts-b SRC_B
                    --key-template TPL | --key-template-a TA --key-template-b TB]
  eval-lever seed  --gaps GAP_FILE [--criterion 0.95]
  eval-lever audit --pred PRED.jsonl --dev-docs DEV --redocred-dev RED --re2-dev R2 --rel2idx IDX
  eval-lever predict --arm-file SPEC.json   # 命题2 翻转条件对登记预测的前置检验
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_arm(a):
    if a.counts_mode:
        from .leverage import arm_counts
        tpl = (a.key_template_a, a.key_template_b) if a.key_template_a else a.key_template
        res = arm_counts(a.counts_a, a.counts_b, tpl,
                         golds=tuple(x.strip() for x in a.golds.split(",")))
    else:
        from .leverage import arm_predictions
        res = arm_predictions(a.pred_a, a.pred_b, a.gold, a.metric)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


def _cmd_seed(a):
    from .seedrx import prescription
    spec = json.loads(Path(a.gaps).read_text(encoding="utf-8-sig"))
    if isinstance(spec, dict):
        gaps = spec.get("gaps")
        sigmas = spec.get("sigmas")
        rho = spec.get("rho")
        grid = spec.get("grid")
        criterion = spec.get("criterion", 0.95)
        reps = spec.get("reps", 10000)
        seed = spec.get("seed", 20260926)
    else:
        gaps, sigmas, rho, grid = spec, None, None, None
        criterion, reps, seed = 0.95, 10000, 20260926
    res = prescription(gaps, sigmas=sigmas, rho=rho, grid=grid,
                       criterion=criterion, reps=reps, seed=seed)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


def _cmd_audit(a):
    from .audit_report import scan
    res = scan(a.pred, a.dev_docs, a.redocred_dev, a.re2_dev, a.rel2idx,
               schema_format=None if a.fmt == "auto" else a.fmt)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


def _cmd_predict(a):
    """命题2(翻转条件)的登记预测前置检验。

    输入 JSON:{"arm_abs_delta": x, "margin_point": m, "tau_arm": t, "margin_tol": d}
      → predicted_flip = (x >= t) and (|m| <= d):长臂 × 能力差距相近 才预测翻转。
    阈值由使用者按 r37 的留一法程序在自己数据上拟合(论文 LOO acc 0.860),
    工具不做默认值——避免把某一语料的拟合阈值伪装成普适常数。
    """
    spec = json.loads(Path(a.arm_file).read_text(encoding="utf-8-sig"))
    missing = [k for k in ("arm_abs_delta", "margin_point", "tau_arm", "margin_tol")
               if spec.get(k) is None]
    if missing:
        print(json.dumps({"error": f"缺少字段:{missing}",
                          "required": ["arm_abs_delta", "margin_point", "tau_arm",
                                       "margin_tol"]}, ensure_ascii=False))
        return 1
    arm, margin = float(spec["arm_abs_delta"]), float(spec["margin_point"])
    tau, tol = float(spec["tau_arm"]), float(spec["margin_tol"])
    long_arm = arm >= tau
    close = abs(margin) <= tol
    res = {"mode": "predict_prop2",
           "inputs": {"arm_abs_delta": arm, "margin_point": margin,
                      "tau_arm": tau, "margin_tol": tol},
           "condition": {"long_arm": long_arm, "comparable_margins": close},
           "predicted_flip": bool(long_arm and close),
           "basis": "Proposition 2: flip requires a long arm AND comparable capability "
                    "margins; thresholds are corpus-specific (fit as in the paper's "
                    "leave-one-out check; LOO accuracy 0.860 vs majority 0.772)"}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eval-lever",
                                 description="评测杠杆工具:力臂/种子处方/违规审计/翻转预测")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("arm", help="两模型力臂 + 排名翻转风险(启发式)")
    pa.add_argument("--pred-a", required=False)
    pa.add_argument("--pred-b", required=False)
    pa.add_argument("--gold", required=False)
    pa.add_argument("--metric", default="f1")
    pa.add_argument("--counts-mode", action="store_true",
                    help="per_doc 计数模式(复现 r1 口径:池化 Gap_F1 + 换金标翻转)")
    pa.add_argument("--counts-a", nargs="+", help="A 侧 per_doc 计数文件/目录")
    pa.add_argument("--counts-b", nargs="+", help="B 侧 per_doc 计数文件/目录")
    pa.add_argument("--key-template", default=None,
                    help="配置键模板(两模型同键),含 {gold} 占位")
    pa.add_argument("--key-template-a", default=None, help="A 侧键模板(异键时用)")
    pa.add_argument("--key-template-b", default=None, help="B 侧键模板(异键时用)")
    pa.add_argument("--golds", default="gD,gR", help="契约两端,逗号分隔")

    ps = sub.add_parser("seed", help="符号判定所需最小种子数处方")
    ps.add_argument("--gaps", required=True, help="逐种子 gap 输入 JSON(格式见 README)")

    pu = sub.add_parser("audit", help="LLM 输出违规扫描(d5 预测格式)")
    pu.add_argument("--pred", required=True)
    pu.add_argument("--dev-docs", required=True, help="docred_aligned dev_revised.json")
    pu.add_argument("--redocred-dev", required=True)
    pu.add_argument("--re2-dev", required=True)
    pu.add_argument("--rel2idx", required=True, help="relation_to_index.json")
    pu.add_argument("--format", dest="fmt", choices=("json", "freetext", "auto"), default="auto")

    pp = sub.add_parser("predict", help="命题2 翻转条件对登记预测的前置检验")
    pp.add_argument("--arm-file", required=True, help="输入 JSON(见 README)")

    a = ap.parse_args(argv)
    return {"arm": _cmd_arm, "seed": _cmd_seed, "audit": _cmd_audit,
            "predict": _cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
