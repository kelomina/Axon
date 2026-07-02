# Phase 3 Loop72: Full-Error Review Wave Plan

日期：2026-07-03

## 目标

Loop72 把 Loop71 的结论落成可执行复核计划。Loop71 已证明：当前 best Loop57 full-test 仍有 `1868` 个错误，要达到 `F1 >= 0.999`，best-case 至少要修复 `1708` 个错误。Loop65 的 `62` 行小批量复核远远不够，因此 Loop72 面向全部 `1868` 个 current-best 错误分波。

这不是模型候选。Loop72 不训练、不扫阈值、不改 split、不自动改标、不从 full-test 错误反推规则。它只生成给人工或外部证据系统使用的复核批次。

## 身份字段规则

`filename`、`path`、`extension`、`directory`、`source_sha256`、`cache_path`、`sample_index`、`split` 和行顺序只用于加载、对齐、缓存审计、重复内容复核和人工复核定位。它们不是模型证据，不能驱动阈值、自动改标、特征工程或生产推理。

本轮还修复了一个行级治理问题：`apply_manual_review_verdicts.py` 过去会按 `source_sha256` 折叠同内容重复行，导致 Loop72 的 `1868` 行空 verdict no-op 检查只识别 `1866` 行。现在人工复核 plan 阶段优先用 `sample_index` 做行级对齐，能保留同 SHA 的不同 split 行。这里的 `sample_index` 仍然是审计/复核身份字段，不是训练特征。

Loop72 CSV 中的 `loop57_*` 和 `loop28_*` 分数只用于复核排序和上下文说明，不是人工 verdict 的充分证据，也不能回流为模型特征、阈值规则或生产策略。

## 实现

新增：

- `scripts/build_loop72_review_wave_plan.py`
- `tests/test_build_loop72_review_wave_plan.py`

更新：

- `scripts/apply_manual_review_verdicts.py`
- `tests/test_apply_manual_review_verdicts.py`

真实命令：

```powershell
.\vnev\Scripts\python.exe scripts\build_loop72_review_wave_plan.py `
  --queue-csv reports\random_20w_split\loop63_persistent_error_review_queue.csv `
  --target-gap-json reports\random_20w_split\loop71_target_gap_noise_roi.json `
  --health-audit-csv reports\random_20w_split\loop63_A_persistent_conflict_content_audit.csv `
  --duplicate-details-csv reports\random_20w_split\loop64_manifest_sha_duplicate_details.csv `
  --output-csv reports\random_20w_split\loop72_full_error_review_wave_plan.csv `
  --output-json reports\random_20w_split\loop72_full_error_review_wave_plan_summary.json `
  --wave-size 200
```

No-op 安全验证：

```powershell
.\vnev\Scripts\python.exe scripts\apply_manual_review_verdicts.py `
  --review-csv reports\random_20w_split\loop72_full_error_review_wave_plan.csv `
  --split-csv reports\random_20w_split\loop27_corrected_split.csv `
  --output-csv reports\random_20w_split\loop72_empty_verdict_adjustment_plan.csv `
  --output-json reports\random_20w_split\loop72_empty_verdict_adjustment_plan.json
```

## 结果

Full review-wave plan:

| Metric | Value |
| --- | ---: |
| Rows | `1868` |
| Wave size | `200` |
| Wave count | `10` |
| First wave reaching target if all actionable | `9` |
| FP/FN | `1195 / 673` |
| Duplicate manifest groups | `2` |
| Duplicate manifest rows | `4` |
| Manual fields blank | `true` |

Category counts:

| Category | Rows |
| --- | ---: |
| `b_duplicate_content_group` | `4` |
| `c_high_conflict_persistent_error` | `639` |
| `d_loop57_new_error` | `108` |
| `e_persistent_fn` | `446` |
| `f_persistent_fp` | `671` |

Wave ROI:

| Wave | Rows | Cumulative rows | Cumulative F1 if all confirmed/fixed | Target reached |
| ---: | ---: | ---: | ---: | --- |
| `1` | `200` | `200` | `0.9896086420` | false |
| `2` | `200` | `400` | `0.9908548361` | false |
| `3` | `200` | `600` | `0.9920935801` | false |
| `4` | `200` | `800` | `0.9933453798` | false |
| `5` | `200` | `1000` | `0.9945860310` | false |
| `6` | `200` | `1200` | `0.9958378506` | false |
| `7` | `200` | `1400` | `0.9970835307` | false |
| `8` | `200` | `1600` | `0.9983278009` | false |
| `9` | `200` | `1800` | `0.9995751805` | true |
| `10` | `68` | `1868` | `1.0000000000` | true |

No-op 安全验证：

| Metric | Value |
| --- | ---: |
| Split rows | `200000` |
| Split counts | `20000 / 20000 / 160000` |
| Review rows | `1868` |
| Planned rows | `0` |
| Ignored rows | `1868` |
| Missing split rows | `0` |
| Duplicate review rows | `0` |
| Training policy rows | `0` |
| Replacement required | `0` |

这个 no-op 结果是期望行为：Loop72 生成的 CSV 在人工或外部证据 verdict 为空时不能产生任何训练、改标或 replacement 动作。

## 决策

1. Loop72 不是可进入 Test-10k 的模型方案。它是噪声/证据治理基础设施。
2. 若继续冲 `F1 >= 99.9%`，复核规模必须接近全量 current-best 错误。按当前分波顺序，best-case 到第 `9` 波才理论覆盖目标缺口。
3. 若人工或外部证据确认 `label_wrong`、`feature_broken` 或 `out_of_scope`，仍必须按 fresh same-original-label redraw 生成新 split。坏样本不能补齐自己，20w 总量不能少。
4. 若多数复核结果是 `label_correct` 或 `model_blindspot`，说明目标差距更多来自模型盲区而不是可自动清洗噪声，下一步应引入真正独立检测视角。

## Artifacts

- Review wave plan: `reports/random_20w_split/loop72_full_error_review_wave_plan.csv`
- Summary: `reports/random_20w_split/loop72_full_error_review_wave_plan_summary.json`
- Empty verdict no-op plan: `reports/random_20w_split/loop72_empty_verdict_adjustment_plan.json`

Generated reports are not committed.

## 验证

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_apply_manual_review_verdicts.py tests\test_build_loop72_review_wave_plan.py -q
.\vnev\Scripts\python.exe -m pytest tests\test_audit_loop68_residual_oof_readiness.py tests\test_materialize_loop69_nested_oof_override.py tests\test_train_loop70_nested_oof_meta.py tests\test_identity_feature_guard.py tests\test_audit_loop71_target_gap_noise_roi.py tests\test_build_loop72_review_wave_plan.py tests\test_apply_manual_review_verdicts.py -q
.\vnev\Scripts\python.exe -m py_compile scripts\build_loop72_review_wave_plan.py scripts\apply_manual_review_verdicts.py
```
