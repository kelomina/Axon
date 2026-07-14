# Phase 2 Loop127 Error / Noise Audit Report

日期：2026-07-04

## 结论

本轮 Phase 2 的目标是给下一轮 Train/Val-only 候选方案提供噪声与错误归因线索，不把 final full-test 错误用于训练、调参或特征选择。`source_path`、`cache_path`、`source_sha256` 只用于定位和缓存对齐，不作为模型证据。

本轮修正了 `scripts/analyze_phase2_error_intrinsics.py` 的一个缓存 schema 误判：当前 fixed-v2 训练缓存通常不保存 `orig_length` / `orig_len` / `original_length` 字段。缺少该字段只能说明原始长度不可用，不能说明特征损坏。修正后，`orig_len_missing_or_zero` 只会在缓存明确存在原始长度字段且值不大于 0 时触发。

## 输入与输出

- 输入预测：`reports/phase1_loop127/probability_calibrator_full_test_predictions.csv`
- 阈值：`0.44`
- 概率列：`calibrated_prob_malicious`
- 修正后 JSON：`reports/phase2_loop127/calibrated_full_test_error_intrinsics_corrected.json`
- 修正后 review queue：`reports/phase2_loop127/calibrated_full_test_review_queue_corrected.csv`
- 资源守卫：`reports/logs/guard_phase2_error_intrinsics_rerun_allow_npz_both.json`

## 修正前后差异

修正前，`feature_anomaly_review=3502`，主要原因是所有错误样本都被打上 `orig_len_missing_or_zero`。抽查缓存文件确认当前缓存实际字段为：`byte_sequence`、`pe_features`、`stat_features`、`lightweight_features`、`label`、`source_sha256`，没有 `orig_length`。因此旧的 `feature_anomaly_review` 是 schema 缺字段导致的假异常。

修正后，错误总数不变，但队列语义恢复：

| 队列 | 数量 | 解释 |
|---|---:|---|
| `label_noise_extreme_fn` | 60 | FN 且概率 `<= 0.01`，优先人工复核漏报/标签问题 |
| `label_noise_extreme_fp` | 333 | FP 且概率 `>= 0.99`，优先人工复核误伤/白样本污染 |
| `label_noise_high_fn` | 224 | FN 且 `0.01 < p <= 0.05` |
| `label_noise_high_fp` | 359 | FP 且 `0.95 <= p < 0.99` |
| `calibration_near_threshold` | 222 | 距离阈值 `<= 0.02`，优先用于校准/阈值分析，不直接判标签噪声 |
| `calibration_broad_near_threshold` | 343 | 距离阈值 `<= 0.05` 的扩展近阈值样本 |
| `model_behavior_review` | 3502 | 非近阈值、非高置信噪声队列、且未发现缓存结构异常 |

## 错误结构

全量 test 仍只作为最终审计证据，不作为下一轮模型选择输入。

- 总行数：`160000`
- 错误数：`5043`
- FP：`2923`
- FN：`2120`
- 特征读取失败：`0`

置信度桶：

| bucket | 数量 |
|---|---:|
| `fp_high_conf_ge_0.90` | 943 |
| `fp_mid_conf_0.75_0.90` | 586 |
| `fp_near_threshold_0.44_0.75` | 1394 |
| `fn_high_conf_lt_0.10` | 503 |
| `fn_mid_conf_0.10_0.30` | 859 |
| `fn_near_threshold_0.30_0.44` | 758 |

## 对下一轮模型改进的含义

1. 单纯调阈值不够。近阈值样本只有 `565` 个，而高置信错误至少 `1446` 个，必须增强内容特征或二阶段模型表达能力。
2. full-test 错误不能用于训练或选择候选。下一轮候选必须只用 train 拟合、val 选择；只有 Val 明显超过当前 calibrator，才允许进 Test-10k。
3. 噪声问题真实存在，但本轮不能把缺 `orig_len` 当坏特征。人工/自动复核应优先看极端 FN/FP，高置信冲突比 schema 缺字段更可信。
4. 坏特征或不合格文件的处理原则仍是重抽整批目标集合，而不是用坏样本补齐。任何清洗后 split 仍必须严格保持 `20000/20000/160000` 且总数 `200000`。

## 验证

- `vnev\Scripts\python.exe -m py_compile scripts\analyze_phase2_error_intrinsics.py`
- `vnev\Scripts\python.exe -m pytest tests\test_analyze_phase2_error_intrinsics.py -vv`
- 结果：`2 passed`

