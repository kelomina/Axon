# Phase 3 Loop143 Support Calibrator Negative Report

更新时间：2026-07-05

## 目标

Loop141 的 support-aware selector 只作用在少量模型分歧行上，但已经说明 Train-OOF kNN support 有一点真实信号。Loop143 因此测试更宽的方案：把 Loop136 输出作为基础信号，在全量 Train/Val 行上追加 Train-OOF kNN 邻域支持特征，训练一个轻量二阶段校准器。

本轮仍严格只跑 Train/Val，不进入 Test-10k 或 full-test。路径、文件名、目录、后缀、hash、`source_sha256`、`sample_index`、split 和 row order 仍只用于加载、对齐和缓存审计，不作为模型特征。

## Gate

Loop136 当前 Val 门槛：

| Model | F1 | Errors | FP | FN |
|---|---:|---:|---:|---:|
| Loop136 strict best | `0.9910789933` | `179` | `122` | `57` |

由于 Loop139/140/141 都证明小幅 Val 改善容易外溢，本轮新 selector/calibrator 的 Test-10k 进入门槛保持严格：

- Val errors 必须 `<= 169`
- FP 必须 `<= 122`
- FN 必须 `<= 57`

## 候选结果

### A. Loop136 score alias + kNN support

该候选把 Loop136 CSV 中的 `stage2_prob_malicious` 复制为 `prob_malicious`，再训练 tabular PE/stat + kNN support 校准器。

| Item | Value |
|---|---:|
| Train kept | `20000 / 20000` |
| Val kept | `20000 / 20000` |
| Base feature dim | `311` |
| Feature dim with kNN | `338` |
| Best model | `hgb_lr0.04_leaf15_l2_0__noise_knn_soft_conflict_downweight` |
| Val F1 | `0.9901465513` |
| Val errors | `197` |
| Val FP/FN | `95 / 102` |

结论：召回代价过大，直接拒绝。这个结果也暴露出一个实现语义问题：Loop136 是 selector 的硬决策，`stage2_prob_malicious` 不能完全代表它的最终 prediction。

### B. Loop136 hard-decision alias + HGB kNN support

该候选把 Loop136 最终 prediction 映射为硬概率：`prediction=1 -> 0.999`，`prediction=0 -> 0.001`，用于保留 Loop136 的硬决策先验。

| Item | Value |
|---|---:|
| Best model | `hgb_lr0.06_leaf31_l2_0__noise_knn_soft_conflict_downweight` |
| Val F1 | `0.9910807713` |
| Val errors | `179` |
| Val FP/FN | `124 / 55` |

结论：总错误没有超过 Loop136，只是把少量 FP/FN 做了交换，拒绝。

### C. Loop136 hard-decision alias + tree kNN support

为确认不是 HGB 模型族太弱，又补跑 ExtraTrees / RandomForest 候选。

| Item | Value |
|---|---:|
| Best model | `extra_trees_300_leaf1__noise_none` |
| Val F1 | `0.9911795485` |
| Val errors | `177` |
| Val FP/FN | `122 / 55` |

结论：它比 Loop136 Val 少 `2` 个错误，且 FP/FN 不差于 Loop136，但距离 `<=169` 的稳定性门槛很远。按当前漏斗规则拒绝，不进入 Test-10k。

## 噪声提示

hard-decision alias 候选的 `clean_val_at_val_threshold` 不作为决策证据，因为它使用硬概率构造 `prob_malicious`，会把 Loop136 原始错误几乎全部计入 suspected noise，存在循环解释。真实可用指标只能看完整 Val 的 FP/FN/errors。

## Decision

拒绝 Loop143。kNN support 仍有诊断价值，但当前自动校准器没有产生足够强的 Val 改善。Loop136 保持 strict best。

下一步不应继续把 kNN support 做成小幅自动 selector。它更适合进入噪声治理链路：标出高置信、邻域冲突的疑似坏标签/灰区样本，然后通过独立内容或外部证据确认，再执行同原始标签池 fresh redraw，而不是把冲突样本直接当训练补丁。

## Artifacts

- Score alias candidate：`reports/phase3_loop143/loop136_support_calibrator_valonly/stage2_cache_matrix_report.json`
- Hard-decision HGB candidate：`reports/phase3_loop143/loop136_hardprob_support_calibrator_valonly/stage2_cache_matrix_report.json`
- Hard-decision tree candidate：`reports/phase3_loop143/loop136_hardprob_support_tree_calibrator_valonly/stage2_cache_matrix_report.json`
- Pre-run guard：`reports/phase3_loop143/pre_run_guard_loop136_support_calibrator_valonly.json`
- Hard-decision pre-run guard：`reports/phase3_loop143/pre_run_guard_loop136_hardprob_support_calibrator_valonly.json`
