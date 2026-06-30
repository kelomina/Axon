# Top20 Manual Review Package

更新时间：2026-07-01

## 目的

这是第一轮人工/多源复核包，用于判断当前最强 stage2 extended 模型的高价值 Val 错误究竟是标签噪声、数据整理异常，还是模型盲区。

本包只来自 Val，不使用 Test10k 调参。

## 产物

- 脚本：`scripts/build_top_error_review_package.py`
- 输入队列：`reports/random_20w_split/stage2_extended_val_error_review_queue.csv`
- 输出 CSV：`reports/random_20w_split/stage2_extended_val_top20_manual_review.csv`
- 输出 JSON：`reports/random_20w_split/stage2_extended_val_top20_manual_review.json`

## 选择规则

Top20 采用分层选择，而不是只取最高置信错误：

1. 全部严重 FN：`severe_fn_label1_prob_le_0.01`，共 `8` 条。
2. 严重 FP：`severe_fp_label0_prob_ge_0.99` 中 top `8`，且至少包含 `2` 条 `.exe`。
3. 路径异常 FN：`2020-08/2022-08-*` 中概率最低 `2` 条。
4. 家族路径 FN：`黑文件1/samples/samples` 中最接近阈值 `0.515` 的 `2` 条。

## 复核字段

CSV 中预留了三列：

- `manual_label_verdict`：建议填写 `label_correct`、`label_wrong`、`uncertain`、`out_of_scope`。
- `manual_verdict_note`：填写依据，例如签名、来源、沙箱、VT/内部威胁情报、业务白名单来源等。
- `recommended_action`：建议填写 `keep_label`、`relabel_train_only`、`quarantine_source_group`、`model_blindspot`、`needs_more_evidence`。

## 判定规则

如果 Top20 中多数为 `label_wrong` 或 `out_of_scope`，则说明数据噪声已经足以影响 `F1 >= 99.9%` 的理论上限，应启动科学降级讨论。

如果多数为 `label_correct` 且集中在 family/path 类型，则说明主要问题是模型盲区，下一轮模型实验应转向新特征或更强模型，而不是继续 HGB 权重微调。

## 注意

不要删除 Val 样本来汇报 clean F1。Val 必须保持完整，复核结论只能用于：

1. 判断目标可行性。
2. 设计 train-only 的噪声鲁棒策略。
3. 指导下一轮模型假设。
