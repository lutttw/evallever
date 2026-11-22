# -*- coding: utf-8 -*-
"""r45_cross_gpu_diag.py — 修订77 收尾:跨 GPU 推理一致性诊断。

目的:实证证明不同 GPU(5060→5090 或 24GB GPU→5090)产生的 logits 差异
  远小于论文臂值的 CI 宽度,从而消除"是否需要在统一硬件上重跑"的疑虑。

方法:取 dreeam(本地 5060 上推断,冻结 logits 在盘)在 5090 上重跑前
  50 篇的推理,对比:
  1. L∞ 距离(max |logit_new − logit_old|)
  2. native 解码预测翻转数(0→1 或 1→0 的候选-关系对)
  3. 臂值影响上限(翻转数 / 189,556)
判定:L∞ < 1e-3 且翻转 < 50/189,556 ⇒ 跨硬件差异可忽略。
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "out" / "revision_v5_experiments" / "dreeam" / "runs" / "dreeam_pe8_s20260917_l01"
CKPT = ROOT / "data" / "dreeam_repo" / "dreeam" / "dreeam_pe8_s20260917"
STAT = ROOT / "out" / "revision_v5_experiments" / "statistics"
N_DOCS = 50


def log(*a):
    print(f"[XGPU {datetime.now().strftime('%H:%M:%S')}]", *a, flush=True)


def main() -> int:
    old_logits = np.load(RUN_DIR / "logits_evaluation.npy").astype(np.float64)[:N_DOCS]
    keys = [json.loads(l) for l in
            (RUN_DIR / "keys_evaluation.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    log(f"old logits: {old_logits.shape} | docs: {N_DOCS}")

    # 加载 dreeam 模型(与 s1 推断同路径)
    dreeam_repo = ROOT / "data" / "dreeam_repo" / "dreeam"
    sys.path.insert(0, str(dreeam_repo))
    sys.path.insert(0, str(ROOT / "src" / "jcip_audit"))

    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # DREEAM 用 roberta-large
    bert_dir = ROOT / ".hf_cache" / "hub" / "models--FacebookAI--roberta-large" / "snapshots"
    bert_snap = sorted(bert_dir.iterdir())[0]
    config = AutoConfig.from_pretrained(bert_snap, num_labels=97)
    tok = AutoTokenizer.from_pretrained(bert_snap)

    # 加载 DREEAM 自定义模型
    os_chdir = dreeam_repo
    sys.path.insert(0, str(os_chdir))
    os_pwd = os_chdir
    import importlib
    prev = os.getcwd()
    os.chdir(str(os_chdir))
    try:
        # DREEAM 的模型类
        from dreeam.model import DREEAMModel
        model = DREEAMModel.from_pretrained(bert_snap, config=config, num_labels=97)
    except Exception as e:
        log(f"DREEAM import fail: {e}")
        return 1
    finally:
        os.chdir(prev)

