# Phase 2 Loop 1 Review Queue

更新时间：2026-07-01

## 背景

Stage2 extended 冻结候选在 Val 上达到 `F1=0.9818199930`，但仍有 `365/20000` 个错误。Test10k 单次确认 `F1=0.9821285141`，错误 `178/10000`，未达到进入 16w full test 的门槛。

本轮回到 Phase 2，目标是先复核最有价值的 Val 错误，不使用 Test10k 继续调参。

## 复核队列

- 脚本：`scripts/build_error_review_queue.py`
- 输入：`reports/random_20w_split/stage2_cache_matrix_replaced_extended_valonly/frozen_val_predictions.csv`
- 阈值：`0.515`
- 输出 CSV：`reports/random_20w_split/stage2_extended_val_error_review_queue.csv`
- 输出 JSON：`reports/random_20w_split/stage2_extended_val_error_review_queue.json`

## 队列统计

- Val 总样本：`20000`
- 错误总数：`365`
- FP：`221`
- FN：`144`
- P0 严重冲突：`36`
- P1 高置信冲突：`57`
- P2 结构性中置信错误：`218`
- P3/P4 近阈值错误：`54`

原因分布：

- `severe_fp_label0_prob_ge_0.99`：`28`
- `severe_fn_label1_prob_le_0.01`：`8`
- `high_fp_label0_prob_ge_0.95`：`38`
- `high_fn_label1_prob_le_0.05`：`19`
- `mid_confidence_structural_error`：`218`
- `near_threshold_error_le_0.05`：`29`
- `near_threshold_error_le_0.10`：`25`

## 早期归因

P0 FN 主要集中在 `data/待拉黑` 的 `.dll/.exe/.sys`，其中 `2026-03`、`2026-02`、`2021-10` 等批次靠前。它们被模型以极低恶意概率判白，可能是：

1. 标签噪声：目录标黑但文件实际不像恶意。
2. 特征盲区：小体积/驱动/DLL 类样本与白样本相似。
3. 时间批次分布偏移：特定月份批次的恶意样本形态不同于训练集。

P0 FP 主要来自 `data/待加入白名单`，大量是无扩展名文件，也有少量 `.exe`。它们被模型以极高恶意概率判黑，可能是：

1. 白名单标签噪声：白名单里混入真实高风险 PE。
2. packed/obfuscated benign：特征上像恶意软件。
3. 白样本来源分布单一，模型没有学到这类高熵/特殊结构白样本。

## 理论可行性提示

仅 Val 中 P0 严重冲突就有 `36/20000 = 0.18%`。如果人工复核确认其中大部分是不可消除的标签噪声，那么 `F1 >= 99.9%` 会缺乏统计支撑，因为目标容错率大约在 `0.1%` 量级。当前还不能直接降级指标，但必须把这 36 个 P0 样本作为“理论上限审计”的第一批证据。

## 下一步

1. 先复核 P0 的 `36` 个样本，记录标签是否可信。
2. 如果 P0 中噪声率高，再扩展到 P1 的 `57` 个样本。
3. 若 P0/P1 大多不是标签噪声，而是模型盲区，则下一轮 Model-Agent 应围绕 hard-example training、family-aware split、或更强的原始字节特征模型展开。
4. 任何数据清洗策略只能在 Train 上应用，Val 必须保持完整评估，不能通过删除 Val 疑似噪声来汇报虚高 F1。
