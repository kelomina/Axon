# Phase 3 Loop66: Val-Only Blindspot Content Audit

日期：2026-07-03

## 目标

Loop66 是 Loop57 之后的 Val-only 内容盲区审计。它只读取 Loop57 的冻结 Val 预测表，以及已经存在的 content PE v1 和 overlay boundary sidecar cache，目的是回答两个问题：

1. Loop57 当前 Val 错误到底主要来自 gate 新增误伤，还是 base 和 gate 都看错的持久盲区？
2. FP、FN、修复样本、误伤样本在内容侧 PE/overlay 特征上有什么可复用的差异？

本轮不训练模型、不扫阈值、不触碰 Test-10k 或 16 万 full-test、不改标签、不改 split、不重建 cache。

## 身份字段规则

`source_path`、`source_sha256`、`cache_path`、`sample_index` 和 `split` 只用于读取 Val 行、对齐 sidecar cache、写审计报告。它们不是模型证据，不进入任何特征差异解释。内容证据只来自：

- `CONTENT_PE_V1_FEATURE_NAMES`
- `OVERLAY_BOUNDARY_FEATURE_NAMES`

两组 feature names 都通过 `identity_feature_guard` 检查。

## 实现

新增：

- `scripts/analyze_loop66_val_blindspots.py`
- `tests/test_analyze_loop66_val_blindspots.py`

真实命令：

```powershell
.\vnev\Scripts\python.exe scripts\analyze_loop66_val_blindspots.py `
  --loop57-val-predictions reports\random_20w_split\loop57_fn_overlay_gate_valonly\loop57_fn_overlay_gate_val_predictions.csv `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --overlay-boundary-cache-dir reports\random_20w_split\loop55_overlay_boundary_cache_train_val `
  --output-json reports\random_20w_split\loop66_val_blindspot_content_audit.json `
  --output-csv reports\random_20w_split\loop66_val_blindspot_feature_deltas.csv `
  --top-k 20
```

## 结果

Loop57 Val final groups:

| Group | Rows |
| --- | ---: |
| TP | `9945` |
| TN | `9908` |
| FP | `92` |
| FN | `55` |

Base-to-final exchange:

| Group | Rows |
| --- | ---: |
| both correct | `19833` |
| base error, final repaired | `20` |
| base correct, final harmed | `5` |
| both error | `142` |

关键结论：Loop57 当前 Val 的 `147` 个错误里，只有 `5` 个是 FN gate 新增误伤；`142` 个是 base 和 Loop57 都错的持久错误。因此下一步不能只继续打磨 overlay FP guard。FP guard 有价值，但上限很窄；主战场已经转向持久 FP/FN 的内容盲区与噪声复核。

## 内容差异

FP vs TN 的主要内容差异：

- FP 的 `content_avg_imports_per_dll` 更高：`22.92` vs `9.38`。
- FP 的 import/API/IAT 规模更高，但 security/overlay/cert-like 信号反而低于 TN：例如 `content_dir_security_log_size` 为 `2.16` vs `5.62`，`content_overlay_log_size` 为 `3.25` vs `5.67`。
- overlay boundary 也显示 FP 更常有 payload-like 片段：`overlay_boundary_payload_log_size` 为 `1.79` vs `0.20`，但 security 覆盖 overlay 的证据更弱。

FN vs TP 的主要内容差异：

- FN 明显更偏 signed/overlay/export/exception/basereloc 结构：`content_dir_security_log_size` 为 `5.38` vs `0.81`，`content_overlay_log_size` 为 `6.87` vs `2.52`。
- FN 的 `content_dir_exception_log_size`、`content_dir_basereloc_log_size`、`content_dir_export_log_size`、IAT/import 规模也更高。
- 这说明剩余 FN 不是简单阈值问题，而是复杂 PE 结构样本仍被 base 和 gate 同时低估。

Repair vs harm 的主要内容差异：

- 被 Loop57 修复的 `20` 行比新增误伤的 `5` 行具有更强 overlay/security/basereloc/export 结构：`content_overlay_log_size` 为 `8.04` vs `4.43`，`content_dir_security_log_size` 为 `5.06` vs `1.88`，`content_dir_basereloc_log_size` 为 `5.24` vs `1.90`。
- 这支持 Loop57 的方向：security/overlay boundary 是真实的 FN 修复信号。
- 但新增误伤太少，不能靠这 `5` 行直接训练一个可靠的大模型。下一步更适合做“持久错误分层”和“小而明确的 Val-only 特征假设”，或者继续人工/外部证据复核。

## 决策

Loop66 不产生新候选，不进入 Test-10k。它给下一轮两个约束：

1. 不要继续把主要算力花在同一类 FN gate FP guard 上，除非候选能同时处理持久错误。
2. 下一轮模型假设应围绕 signed/overlay/export/exception/basereloc/import-shape 的更明确分层，先在 Val 上证明能减少 `both_error`，再考虑 Test-10k。

## Artifacts

- Summary: `reports/random_20w_split/loop66_val_blindspot_content_audit.json`
- Feature deltas: `reports/random_20w_split/loop66_val_blindspot_feature_deltas.csv`

Generated reports are not committed.

## 验证

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_analyze_loop66_val_blindspots.py tests\test_identity_feature_guard.py tests\test_analyze_loop61_exchange.py -q
.\vnev\Scripts\python.exe -m py_compile scripts\analyze_loop66_val_blindspots.py scripts\analyze_loop61_exchange.py scripts\identity_feature_guard.py
```

Latest local result: `5 passed`.
