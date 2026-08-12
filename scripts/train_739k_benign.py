#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基座重训（良性扩表后 cache 813,098，新 7:1:2 划分）。

复用 train_739k_full 训练逻辑，仅改输出目录（避免覆盖旧模型）与实验名。
- 数据：data/.cache（813,098，含新增 ~74k 良性）
- 划分：7:1:2（seed 42，train 569,170）
- 输出：models/full_739k_benign/ + reports/full_739k_benign/
- 续训：默认自动从 models/full_739k_benign/best_model_739k.pt 续跑
        （last_epoch+1 起，恢复 model/optimizer/scheduler/early_stopping）；
       传 --fresh 则忽略已有 checkpoint 从头训练。

运行：python -u scripts/train_739k_benign.py          （续训）
      python -u scripts/train_739k_benign.py --fresh  （全新训练，GPU ~30h）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train_739k_full as T  # noqa: E402

T.OUTPUT_DIR = PROJECT_ROOT / "models" / "full_739k_benign"
T.REPORT_DIR = PROJECT_ROOT / "reports" / "full_739k_benign"

if __name__ == "__main__":
    fresh = "--fresh" in sys.argv
    checkpoint = T.OUTPUT_DIR / "best_model_739k.pt"

    resume_from = None
    if fresh:
        print("[Args] --fresh: 忽略已有 checkpoint，从头训练")
    elif checkpoint.exists():
        resume_from = checkpoint
        print(f"[Args] 自动续训：{resume_from}")
    else:
        print("[Args] 未找到已有 checkpoint，从头训练")

    T.main(resume_from=resume_from)
