# Phase 3 Loop144 R11 Filter / Override Union Negative Report

更新时间：2026-07-05

## 目标

Loop137 的 R11 召回恢复在 Val 和 Test-10k 上看起来有价值，但 full-test 上 FP 外溢。本轮不再直接套用 R11，而是先训练一个只接管 R11 翻转行的轻量过滤器，再把它与两个已经冻结的弱互补候选做 override union 验证：

- `r11_filtered`：Loop136 vs R11 的二阶段过滤器，只在 R11 的 `0 -> 1` 恢复行上接管。
- `fixedv2_string`：Loop137 中 Loop136 vs fixed-v2/string selector。
- `support_tree`：Loop143 中 Loop136 hard-decision + ExtraTrees kNN support calibrator。

身份字段规则不变：`source_path`、filename、directory、extension、`source_sha256`、`sample_index`、`cache_path`、split 和 row order 只用于加载、对齐和审计，不作为模型证据。

## 新增工具

新增 `scripts/evaluate_prediction_override_union.py`，用于评估冻结预测之间的 non-conflicting override union。它的语义是：默认保留 baseline；如果某个冻结 override 的最终 prediction 与 baseline 不同，就接管该 prediction；多个 override 对同一行给出冲突方向时直接报错。

测试：

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_evaluate_prediction_override_union.py tests\test_loop135_pairwise_selector.py tests\test_evaluate_stage2_cache_model.py -q
```

结果：`20 passed`。

## Val 结果

| Candidate | F1 | Errors | FP | FN | Decision |
|---|---:|---:|---:|---:|---|
| Loop136 baseline | `0.9910789933` | `179` | `122` | `57` | baseline |
| R11 unfiltered | `0.9913329348` | `174` | `125` | `49` | known risky |
| R11 filtered | `0.9914317027` | `172` | `123` | `49` | diagnostic pass |
| R11 filtered + fixedv2/string + support-tree union | `0.9916811955` | `167` | `121` | `46` | enter Test-10k |

Val union 接管 `14` 行：`r11_filtered=9`、`fixedv2_string=3`、`support_tree=2`。相比 Loop136，Val 总错误 `-12`，FP `-1`，FN `-11`，满足本轮进入 Test-10k 的稳定门槛。

## Test-10k 结果

| Candidate | F1 | Errors | FP | FN | Decision |
|---|---:|---:|---:|---:|---|
| Loop136 baseline | `0.9916958479` | `83` | `54` | `29` | baseline |
| R11 filtered | `0.9917008299` | `83` | `57` | `26` | tie, recall trade-off |
| support-tree frozen | `0.9914974492` | `85` | `56` | `29` | worse |
| Union | `0.9914034386` | `86` | `60` | `26` | reject |

Union 在 Test-10k 上虽然减少 `3` 个 FN，但新增 `6` 个 FP，净错误 `+3`，因此拒绝，不进入 full-test。

## Full-test 探针

由于 `r11_filtered` 单独在 Test-10k 上与 Loop136 错误数打平、F1 略高且召回更好，补跑一次冻结 full-test 探针来确认是否能作为召回优先候选。结果仍失败：

| Candidate | F1 | Errors | FP | FN | Decision |
|---|---:|---:|---:|---:|---|
| Loop136 baseline | `0.9903723842` | `1544` | `958` | `586` | current strict best |
| R11 filtered | `0.9902250908` | `1569` | `1041` | `528` | reject |

R11 filtered 修复 `58` 个 FN，但新增 `83` 个 FP，净错误 `+25`。这复现了 Loop137 的核心问题：当前静态内容规则能找回一部分恶意样本，但 FP 外溢在 full-test 上明显强于 Val/Test-10k。

## Decision

Loop144 拒绝。当前 strict best 仍是 Loop136：

- full-test F1：`0.9903723842`
- errors：`1544 / 160000`
- FP/FN：`958 / 586`

本轮的重要结论是：Val 上看似很干净的多候选小修正，在 Test-10k 已经露出 FP 外溢；即使单独的 R11 filtered 通过了召回方向的 Test-10k sanity check，full-test 仍然失败。因此下一步不应继续堆同类 override union，而应转向两条更硬的路线：

1. 引入真正正交的新证据，例如动态行为、外部多引擎/签名信誉、行为沙箱或更强的字节模型。
2. 对 Loop138 指出的高置信邻域冲突样本做独立证据复核；确认坏样本后执行同原始标签 fresh redraw，而不是自动补齐或直接 relabel。

## Artifacts

- R11 Train candidate：`reports/phase3_loop144/loop136_r11_train_eval.json`
- R11 filtered Val：`reports/phase3_loop144/loop136_r11_filtered_valonly/loop135_pairwise_selector_report.json`
- Union Val：`reports/phase3_loop144/loop144_r11_fixed_tree_union_val_eval.json`
- R11 filtered Test-10k：`reports/phase3_loop144/loop136_r11_filtered_test10k_eval.json`
- support-tree Test-10k：`reports/phase3_loop144/support_tree_test10k_eval.json`
- Union Test-10k：`reports/phase3_loop144/loop144_r11_fixed_tree_union_test10k_eval.json`
- R11 filtered full-test：`reports/phase3_loop144/loop136_r11_filtered_full_test_eval.json`
