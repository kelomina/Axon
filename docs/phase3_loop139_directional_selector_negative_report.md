# Phase 3 Loop139 Direction-aware Selector Report

更新时间：2026-07-05

## 目标

Loop138 显示继续做普通小规则收益有限，但 Loop136 的 pairwise selector 仍有一个低成本模型侧缺口：同一个 accept threshold 同时控制 `baseline=0 -> candidate=1` 和 `baseline=1 -> candidate=0` 两种方向。Loop139 只做一个窄实验：保持训练数据、候选模型、特征集合不变，把 selector 阈值拆成两个方向阈值。

仍遵守身份字段禁令：路径、文件名、后缀、目录、hash、`source_sha256`、`sample_index`、行序只用于对齐和查缓存，不作为模型证据。

## 代码改动

- `scripts/train_loop135_pairwise_selector.py`
  - 新增 `--threshold-mode global|directional`
  - 新增 `apply_selector_directional()`
  - payload 记录 `thresholds_by_direction`
  - `write_predictions()` 去重已有 selector 字段，避免二级 selector 输出重复 CSV 列
- `scripts/evaluate_loop135_pairwise_selector.py`
  - 读取 directional payload 并冻结应用两个方向阈值
  - 支持 `--threshold-0to1` / `--threshold-1to0` 显式覆盖，仅用于审计
- `tests/test_loop135_pairwise_selector.py`
  - 增加方向阈值与 CSV 字段去重单测

单测：`vnev\Scripts\python.exe -m pytest tests\test_loop135_pairwise_selector.py -q` -> `9 passed`。

## 漏斗结果

Val-only 选择结果：

| Model | F1 | Errors | FP | FN |
|---|---:|---:|---:|---:|
| R5 baseline | `0.9907342832` | `186` | `130` | `56` |
| OOF noise candidate | `0.9905796740` | `189` | `126` | `63` |
| Loop136 global selector | `0.9910789933` | `179` | `122` | `57` |
| Loop139 directional selector | `0.9913312077` | `174` | `123` | `51` |

Loop139 Val 选择：

- model：`selector_logreg_balanced_c0.1`
- `threshold_0to1 = 0.55`
- `threshold_1to0 = 0.79`
- accepted candidate rows：`16`
- Val errors：相对 R5 `-12`，相对 Loop136 `-5`

因为 Val 明显优于 Loop136，进入 Test-10k 冻结确认。

Test-10k：

| Model | F1 | Errors | FP | FN | Decision |
|---|---:|---:|---:|---:|---|
| R5 baseline | `0.9915983197` | `84` | `56` | `28` | fallback |
| Loop136 global selector | `0.9916958479` | `83` | `54` | `29` | current strict best gate |
| Loop139 directional selector | `0.9914974492` | `85` | `56` | `29` | reject |

Loop139 没有通过 Test-10k：它比 Loop136 多 `2` 个错误，也比 R5 多 `1` 个错误。因此不进入 full-test。

## Decision

拒绝 Loop139 directional selector，不替代 Loop136。

本轮结论是：方向阈值在 Val 上看起来能恢复 FN，但 Test-10k 没有泛化。后续不应继续围绕同一小分歧集反复扫阈值；方向阈值代码可以保留，因为它修复了 selector 表达能力和 CSV 字段重复问题，但当前实验产物不能作为上线候选。

## Artifacts

- Pre-run guard：`reports/phase3_loop139/pre_run_guard_directional_selector_valonly.json`
- Val report：`reports/phase3_loop139/r5_oof_noise_pairwise_selector_directional_valonly/loop135_pairwise_selector_report.json`
- Val predictions：`reports/phase3_loop139/r5_oof_noise_pairwise_selector_directional_valonly/loop135_pairwise_selector_val_predictions.csv`
- Test-10k guard：`reports/phase3_loop139/pre_run_guard_directional_selector_test10k.json`
- Test-10k eval：`reports/phase3_loop139/r5_oof_noise_pairwise_selector_directional_test10k_eval.json`
- Test-10k predictions：`reports/phase3_loop139/r5_oof_noise_pairwise_selector_directional_test10k_predictions.csv`
