# -*- coding: utf-8 -*-
"""EvalLever —— 评测杠杆工具包(最小可用版,三个 CLI 命令)。

EvalLever — a minimal evaluation-leverage CLI toolkit (three commands).

本包是对本项目已验证分析口径的**独立封装**(自包含,不 import src/jcip_audit,
零第三方依赖,确定性输出,无网络调用):

  * eval-lever arm   : 力臂计算 —— 双模型错误集 Jaccard 距离 / 项目 per_doc
                       计数口径的换金标力臂(gD→gR)与排名翻转检测
                       (口径源自 src/jcip_audit/r1_rank_flip.py,已验证)
  * eval-lever seed  : 种子数收敛处方 —— 给定逐种子 gap 值,输出每 n 的
                       P(95%CI 排除 0) 曲线与"符号判定所需最小种子数"
                       (口径源自 src/jcip_audit/r2_seed_convergence.py,已验证)
  * eval-lever audit : LLM 输出违规审计 —— parse_fail / oob_index /
                       rel_unmapped / self_pair / ambiguous / value_hallucination
                       分类计数 + 违规率 + unverifiable 旗标
                       (口径源自 src/jcip_audit/d5_llm_hook.py,已验证)

兼容性自检(±0.005):python evallever/selftest.py
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
