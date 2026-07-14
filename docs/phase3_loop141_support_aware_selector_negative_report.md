# Phase 3 Loop141 Support-aware Selector Report

更新时间：2026-07-05

## 目标

Loop138 证明当前剩余错误里存在大量邻域冲突，Loop139/140 又证明普通方向阈值在 Val 上的收益会在 Test-10k 外溢。Loop141 因此尝试一个更有依据的 support-aware selector：在原有 pairwise selector 特征之外加入 Train-only kNN support 数值特征。

安全约束：

- Train selector rows 使用 OOF kNN support，避免 self-neighbor 泄漏。
- Val/Test rows 使用冻结 Train-memory kNN support。
- support 特征只包含数值邻域统计，不使用路径、文件名、后缀、目录、hash、`source_sha256`、`sample_index` 或 row order 作为模型证据。
- support 特征默认关闭，只有显式传入 `--support-stage2-model` 等参数才启用。

## 代码改动

- `scripts/train_loop135_pairwise_selector.py`
  - 新增 support-aware 特征构建：`build_support_feature_blocks()`、`build_eval_support_feature_block()`
  - Train 侧调用 OOF kNN support；Val/Test 侧调用 frozen train-memory kNN support
  - 新增 support memory label-order 校验，避免传错 Stage2 kNN reference
  - 新增 `--support-key-columns`，support 对齐默认只用 `source_sha256`，身份字段仍只用于对齐
- `scripts/evaluate_loop135_pairwise_selector.py`
  - support-aware payload 评估时强制传入 `--support-predictions`
  - 冻结复用 Val 选出的 support schema/top-k/reference
- `tests/test_loop135_pairwise_selector.py`
  - 增加 support feature name 与 feature append 单测

## Val 结果

### Candidate A: Loop136 vs OOF-noise + support

| Model | F1 | Errors | FP | FN |
|---|---:|---:|---:|---:|
| Loop136 baseline | `0.9910789933` | `179` | `122` | `57` |
| OOF noise candidate | `0.9905796740` | `189` | `126` | `63` |
| Loop141 support selector | `0.9913268867` | `174` | `118` | `56` |

Selected model: `selector_extra_trees_leaf2`; thresholds `0to1=0.53`, `1to0=0.65`; accepted Val rows `19`.

Support coverage: Train `20000/20000` kept, Val `20000/20000` kept, Train fallback rows `4`, Val fallback rows `0`.

### Candidate B: R5 vs OOF-noise + support

| Model | F1 | Errors | FP | FN |
|---|---:|---:|---:|---:|
| R5 baseline | `0.9907342832` | `186` | `130` | `56` |
| OOF noise candidate | `0.9905796740` | `189` | `126` | `63` |
| Loop141 R5 support selector | `0.9910367493` | `180` | `131` | `49` |

This variant improves R5 by `6` errors, but it is still weaker than current best Loop136 (`179` Val errors), so it is not a successor candidate.

## Gate Decision

Reject Loop141 before Test-10k.

The support-aware Loop136 candidate improves Val by `5` errors and keeps both FP/FN below Loop136, but this is below the stricter post-Loop139/140 stability gate. After two consecutive Val-good/Test-bad selector attempts, the minimum gate for another selector family is `Val errors <= 169`, `FP <= 122`, `FN <= 57`, and accepted rows at least `20`. Loop141 achieved `174` errors and `19` accepted rows, so it does not enter Test-10k.

## Artifacts

- Pre-run guard A：`reports/phase3_loop141/pre_run_guard_support_selector_valonly_retry2.json`
- Candidate A Val report：`reports/phase3_loop141/loop136_vs_oof_support_directional_valonly/loop135_pairwise_selector_report.json`
- Candidate A Val predictions：`reports/phase3_loop141/loop136_vs_oof_support_directional_valonly/loop135_pairwise_selector_val_predictions.csv`
- Pre-run guard B：`reports/phase3_loop141/pre_run_guard_r5_oof_support_selector_valonly.json`
- Candidate B Val report：`reports/phase3_loop141/r5_vs_oof_support_directional_valonly/loop135_pairwise_selector_report.json`
- Candidate B Val predictions：`reports/phase3_loop141/r5_vs_oof_support_directional_valonly/loop135_pairwise_selector_val_predictions.csv`

## Next

Support-aware kNN features are useful diagnostic evidence but still do not clear the generalization gate. The next model-improvement attempt should stop reusing the same R5/OOF disagreement family and instead add genuinely new evidence, especially certificate/publisher/version-resource or behavior-side features for the high-confidence FP pool.
