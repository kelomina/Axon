# Phase 2 Loop 1 Noise Adjudication

更新时间：2026-07-01

## 目的

本报告承接 `docs/phase2_loop1_review_queue.md`，进一步判断 stage2 extended 冻结候选的 Val 高置信错误更像标签噪声、分布混杂，还是纯模型盲区。

本轮仍只使用 Val 和 Train，不使用 Test10k 调参。

## 输入

- 复核队列：`reports/random_20w_split/stage2_extended_val_error_review_queue.csv`
- Stage2 冻结模型：`reports/random_20w_split/stage2_cache_matrix_replaced_extended_valonly/stage2_selected_model.pkl`
- Train 预测：`reports/random_20w_split/random_20w_8192_replaced_train_predictions.csv`
- Val 基础预测：`reports/random_20w_split/random_20w_8192_replaced_val_predictions.csv`
- 邻居审计脚本：`scripts/audit_error_neighbors.py`
- 邻居审计报告：`reports/random_20w_split/stage2_extended_val_p0_p1_neighbor_audit.json`

## 方法

对 P0/P1 高置信错误样本，在 stage2 extended 的同一套特征空间里查找 Top-10 train 近邻。近邻特征来自：

- baseline 概率特征
- stat features
- PE features
- lightweight features
- byte summary features

判定规则：

- 如果 Top-10 近邻里 `>=80%` 是与当前标签相反的类别，记为 `neighbors_support_model_prediction`。
- 如果 Top-10 近邻里 `>=80%` 与当前标签一致，记为 `neighbors_support_dataset_label`。
- 其它情况记为 `neighbors_mixed`。

这不是最终标签裁决，只是复核优先级排序。

## 结果

审计范围：

- Val 错误总数：`365`
- P0/P1 复核队列：`93`
- 实际匹配审计行：`94`，存在重复 key/重复行现象，按 key 汇总后覆盖 P0/P1 队列。
- Top-K：`10`

近邻支持分布：

- `neighbors_support_model_prediction`：`61`
- `neighbors_mixed`：`27`
- `neighbors_support_dataset_label`：`6`

解释：

大多数 P0/P1 高置信错误在 train 特征空间里更接近“模型预测的那一类”，而不是数据集标注的那一类。换句话说，模型并不是凭空错；很多样本从当前特征证据看确实像相反类别。

## 风险解读

这对 `F1 >= 99.9%` 是一个强风险信号。P0/P1 中 `61` 个样本已经表现为“近邻支持模型预测”，如果人工/多源复核确认其中一批确实标错或语义混杂，那么当前数据集的可达上限可能低于 99.9%。

但不能直接把它们从 Val 删除，也不能把 clean Val F1 当主指标。原因：

1. 近邻只说明特征相似，不等于真实标签。
2. Val 是选择与诊断集，删除 Val 疑似噪声会制造虚高指标。
3. 目标要求最终 16w test，必须保留完整评估口径。

## Model-Agent 策略排序

只读 Model-Agent 给出的下一轮优先级：

1. Val 错样本标签/cache 审计优先。
2. 扩大 cache-first 训练覆盖，再做 stage2 复验。
3. 二阶段特征升级，补充更强 byte/local 统计。
4. 保守动态行为特征只做 FP triage，不进主线。
5. 重新做 GA feature mask，但目标改成 balanced F1/低 FP。

不建议优先重复 SWA/EMA、byte noise、near-threshold weighting、旧 hard-example replay、旧 gated/residual fusion，因为已有报告显示负面或不稳定。

## 下一步决策

进入下一轮 Phase 2/3 前，建议按以下顺序执行：

1. 人工或多源复核 `neighbors_support_model_prediction` 中的 P0 样本，优先确认严重 FP/FN 是否真标签噪声。
2. 若确认噪声比例高，形成“不可达 99.9%”的统计证据，并提出科学降级阈值。
3. 若确认多数不是噪声，而是模型盲区，则开新模型实验分支，优先做 cache-first 扩大训练覆盖和 hard-example aware stage2。
4. 下一轮候选仍必须先跑完整 Val，只有明显超过 `0.9818199930` 才允许冻结后进入 Test10k。

## 关键结论

当前最强模型的错误已经不是简单阈值问题。高置信错误呈现明显的标签/特征冲突信号，继续追 99.9% 前，必须先把 P0/P1 的真实标签可信度搞清楚。
