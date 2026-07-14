# Phase 2 Noise And Error Audit

更新时间：2026-07-01

## 审计边界

本轮只使用 train/val 已导出的预测 CSV 和本地 cache，不重新跑神经网络，不使用 test10k 进行候选选择。test10k 仍只允许作为冻结候选的一次性确认集。

## 输入

- Baseline Val 预测：`reports/random_20w_split/random_20w_8192_replaced_val_predictions.csv`
- Calibrator Val 预测：`reports/random_20w_split/phase2_val_calibrator_predictions.csv`
- Calibrator 模型：`models/random_20w_8192/random20w_replaced_logreg_calibrator.pkl`
- Baseline 阈值：`0.50`
- Calibrator 阈值：`0.44`

## Val 错误对比

| 模型 | Val 样本数 | 错误数 | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 20000 | 1386 | 554 | 832 |
| Calibrator | 20000 | 627 | 350 | 277 |

Calibrator 在 Val 上少了 `759` 个错误，是当前最强的已验证候选；但剩余错误仍然远超 `F1 >= 99.9%` 所允许的错误量。

## 噪声与硬样本信号

Baseline Val 噪声审计：

- 高置信 FN 冲突，`label=1` 且 `prob<=0.05`：`63`
- 高置信 FP 冲突，`label=0` 且 `prob>=0.95`：`16`
- 严重 FP 冲突，`label=0` 且 `prob>=0.99`：`2`
- 近阈值错误，距离阈值 `<=0.05`：`203`
- 近阈值错误，距离阈值 `<=0.10`：`193`
- 疑似噪声/硬样本合计：`477`

Calibrator Val 噪声审计：

- 高置信 FN 冲突，`label=1` 且 `prob<=0.05`：`28`
- 严重 FN 冲突，`label=1` 且 `prob<=0.01`：`9`
- 高置信 FP 冲突，`label=0` 且 `prob>=0.95`：`40`
- 严重 FP 冲突，`label=0` 且 `prob>=0.99`：`39`
- 近阈值错误，距离阈值 `<=0.05`：`78`
- 近阈值错误，距离阈值 `<=0.10`：`59`
- 疑似噪声/硬样本合计：`253`

解释：校准器明显减少了总错误，但剩余错误更像“硬冲突”。尤其是接近 1.0 的白样本 FP 和接近 0 的黑样本 FN，不应简单用阈值调整解决，应优先进入人工/自动复核队列。

## 错误集中区域

Calibrator Val 剩余错误：

- FP：全部来自 `data/待加入白名单`，其中 `<none>` 扩展名占 `310/350`，`.exe` 占 `40/350`。
- FN：全部来自 `data/待拉黑`，其中 `.exe` 占 `218/277`，`.dll` 占 `48/277`。
- FN 月份集中度较高，较多出现在 `2025-11`、`2020-11`、`2026-03`、`2021-11`、`2025-10`、`2026-01`、`2025-12` 等目录批次。

## 后续动作

1. Data-Agent：优先复核 Calibrator Val 的 `39` 个严重 FP 和 `9` 个严重 FN，确认是标签噪声、特征异常，还是模型盲区。
2. Error-Agent：把 FP/FN 拆成三类：近阈值可优化、结构性混淆、高置信疑似标签冲突。
3. Model-Agent：候选训练时避免把疑似噪声直接删掉后汇报“clean F1”，只能把 noise-aware 权重作为 train-only 策略，并在完整 Val 上评估。
4. Eval-Agent：继续封存 test10k 候选排行榜，只允许冻结候选做一次 test10k smoke。

## 产物

- Baseline Val 错误分析：`reports/random_20w_split/phase2_val_baseline_error_analysis/prediction_error_summary.json`
- Baseline Val 噪声审计：`reports/random_20w_split/phase2_val_baseline_noise_audit/noise_audit_summary.json`
- Calibrator Val 错误分析：`reports/random_20w_split/phase2_val_calibrator_error_analysis/prediction_error_summary.json`
- Calibrator Val 噪声审计：`reports/random_20w_split/phase2_val_calibrator_noise_audit/noise_audit_summary.json`
- Calibrator Val 疑似噪声队列：`reports/random_20w_split/phase2_val_calibrator_noise_audit/suspected_noise_and_hard_examples.csv`
