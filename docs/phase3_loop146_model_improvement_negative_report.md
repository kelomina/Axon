# Phase 3 Loop146 Model-improvement Report

更新时间：2026-07-06

## 目标

本轮按用户纠偏，把工作重心从“内存泄漏专项”拉回模型改进。资源 guard 仍作为重脚本运行前的安全闸，但不再把它当主任务。

当前 strict best 仍是 Loop136：

| Split | F1 | Errors | FP | FN |
|---|---:|---:|---:|---:|
| Val | `0.9910789933` | `179` | `122` | `57` |
| Test-10k | `0.9916958479` | `83` | `54` | `29` |
| Full-test | `0.9903723842` | `1544` | `958` | `586` |

本轮候选继续遵守身份字段禁令：`source_path`、文件名、目录、后缀、hash、`source_sha256`、`sample_index`、split、row order 只用于对齐、cache lookup 和审计，不作为模型证据。

## 候选 A：R11 Filter + Fixed-v2/String Union

Loop144 的三路 union 在 Val 到 `167`，但 Test-10k 退化到 `86`。本轮先测试更保守的双路 union：只合并 `r11_filtered` 与 `fixedv2_string`，不接入 `support_tree`。

| Split | Candidate | F1 | Errors | FP | FN | Decision |
|---|---|---:|---:|---:|---:|---|
| Val | Loop136 baseline | `0.9910790` | `179` | `122` | `57` | baseline |
| Val | R11 filtered + fixedv2/string | `0.9915807` | `169` | `121` | `48` | enter Test-10k |
| Test-10k | Loop136 baseline | `0.9916958` | `83` | `54` | `29` | baseline |
| Test-10k | R11 filtered + fixedv2/string | `0.9916017` | `84` | `58` | `26` | reject |

Val 上净减少 `10` 个错误，但 Test-10k 新增 `4` 个 FP、只减少 `3` 个 FN，净错误 `+1`。因此拒绝，不进入 full-test。

## 候选 B：Loop136-aware All-row Calibrator

新增脚本：

- `scripts/train_loop146_loop136_allrow_calibrator.py`
- `tests/test_loop146_loop136_allrow_calibrator.py`

该候选在全量 Train/Val 行上使用 Loop136 final score、R5 baseline score、OOF candidate score、selector score/accept flag、分歧方向，以及 content PE/string 数值特征训练轻量校准器。它不使用路径、hash、sample id 等身份字段作为特征。

Val 结果：

| Model | Val F1 | Errors | FP | FN | Decision |
|---|---:|---:|---:|---:|---|
| Loop136 baseline | `0.9910790` | `179` | `122` | `57` | baseline |
| ExtraTrees all-row calibrator | `0.9911302` | `178` | `123` | `55` | reject |
| Logistic calibrator | `0.9911293` | `178` | `122` | `56` | reject |

最佳只减少 `1` 个错误，且 ExtraTrees 版本 FP 超过 Loop136。未达到 `errors <= 169, FP <= 122, FN <= 57` 的 Val gate，因此不进入 Test-10k。

## 候选 C：kNN Prediction-support Gate

新增脚本：

- `scripts/evaluate_loop146_knn_prediction_support_gate.py`
- `tests/test_loop146_knn_prediction_support_gate.py`

该候选把 Loop138 的邻域想法改成全量 Val 规则：默认保留 Loop136，只有 Train-memory kNN 强烈支持相反类别时才翻转。kNN 数值特征来自 Stage-2 train-memory reference，身份字段只用于读取 cache。

Val 结果：

| Candidate | Val F1 | Errors | FP | FN | Changed rows | Decision |
|---|---:|---:|---:|---:|---:|---|
| Loop136 baseline | `0.9910790` | `179` | `122` | `57` | - | baseline |
| kNN support gate | `0.9911293` | `178` | `122` | `56` | `1` | reject |

该候选只找到 `1` 个安全翻转，远低于 Val gate，拒绝进入 Test-10k。

## 额外验证

已运行单测：

```powershell
vnev\Scripts\python.exe -m pytest tests\test_loop146_knn_prediction_support_gate.py tests\test_loop146_loop136_allrow_calibrator.py -q
```

结果：`5 passed`。

已生成的主要 artifact：

- `reports/phase3_loop146/loop146_r11_fixed_union_val_eval.json`
- `reports/phase3_loop146/loop146_r11_fixed_union_test10k_eval.json`
- `reports/phase3_loop146/loop136_allrow_calibrator_valonly/loop146_allrow_calibrator_report.json`
- `reports/phase3_loop146/knn_support_gate_val_eval.json`

## Decision

Loop146 全部拒绝，Loop136 仍是当前 strict best。

本轮再次确认：能在 Val 上恢复 FN 的规则到了 Test-10k 仍会外溢 FP；而更保守的 all-row calibrator / kNN support gate 又只能改善 `1` 个 Val 错误。后续模型侧继续堆同类小 selector 的收益已经很低。

下一步优先级：

1. 继续只在 Train/Val 上稳定 32768-byte 长上下文候选，但要使用 `batch_size=4`、FP32、禁用 AMP，并把 Val 低于 Loop136 的结果及时归档为负结果。
2. 做 Train/Val 版盲化噪声治理包，独立确认 `label_wrong / feature_broken / out_of_scope` 后 fresh same-label redraw。
3. 如果不能引入动态行为、信誉、外部签名信任链等正交证据，应正式把 99.9% 作为远期目标，而不是下一轮工程 gate。
