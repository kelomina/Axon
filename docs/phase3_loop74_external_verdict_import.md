# Phase 3 Loop74: Strict External Verdict Import

日期：2026-07-03

## 目标

Loop74 补上 Loop72 review wave plan 和 split adjustment plan 之间的严格入口。它只做外部/人工 verdict 导入校验，不训练、不扫阈值、不改 split、不自动改标，也不把 full-test 错误列表反推成模型规则。

这个入口解决两个风险：

- 外部填回来的 verdict/action 如果格式不严，会把噪声清理变成不可审计的黑箱。
- Loop72 是 full-test current-best error 复核，默认不能直接产生 train/val policy；确认 `label_wrong`、`feature_broken` 或 `out_of_scope` 都必须进入 fresh same-original-label redraw，而不是用坏行补齐或直接把 test verdict 喂给训练。

## 实现

新增：

- `scripts/import_loop72_external_verdicts.py`
- `tests/test_import_loop72_external_verdicts.py`

严格规则：

- 输入默认必须是 Loop72 的 `1868` 行 current-best full-test error plan。
- `sample_index` 必须能和 frozen 20w split 行级对齐；它只是复核/对齐字段，不是模型证据。
- split 必须保持 `200000 = 20000 train + 20000 val + 160000 test`。
- `label_wrong` 必须提供 `corrected_label`，但在 Loop72 口径下仍触发 replacement/redraw，不进入 training policy relabel。
- actionable verdict 必须写 `manual_verdict_note`，用于审计外部证据摘要；路径、文件名、hash、split、模型分数不能作为 verdict 证据。
- `feature_broken/out_of_scope` 必须 fresh same-original-label redraw。
- test verdict 默认只能作为 held-out 噪声审计和目标可行性证据，`training_policy_rows` 必须为 `0`。

## 命令

```powershell
.\vnev\Scripts\python.exe scripts\import_loop72_external_verdicts.py `
  --review-csv reports\random_20w_split\loop72_full_error_review_wave_plan.csv `
  --split-csv reports\random_20w_split\loop27_corrected_split.csv `
  --target-gap-json reports\random_20w_split\loop71_target_gap_noise_roi.json `
  --output-csv reports\random_20w_split\loop74_empty_external_verdict_import.csv `
  --output-json reports\random_20w_split\loop74_empty_external_verdict_import.json `
  --plan-csv reports\random_20w_split\loop74_empty_external_adjustment_plan.csv `
  --plan-json reports\random_20w_split\loop74_empty_external_adjustment_plan.json
```

## 真实 no-op 结果

使用 Loop72 空 verdict 表运行：

| Metric | Value |
| --- | ---: |
| Import ready | `true` |
| Review rows | `1868` |
| Expected rows | `1868` |
| Sample-index matches | `1868` |
| Duplicate review rows | `0` |
| Missing split rows | `0` |
| Split rows | `200000` |
| Split counts | `20000 / 20000 / 160000` |
| Manual content rows | `0` |
| No-op rows | `1868` |
| Invalid rows | `0` |
| Training policy rows | `0` |
| Decision | `ready_noop_no_actionable_verdicts` |

Current target evidence remains unchanged:

- Current best Loop57 full-test F1: `0.9883629658239992`
- Current errors: `1868`
- FP/FN: `1195 / 673`
- Minimum fixed errors for `F1 >= 0.999`: `1708`
- Empty verdict target coverage: `0 / 1708`

## 决策

Loop74 是噪声治理入口，不是模型候选。它让后续外部复核结果可以被机器严格验收，并明确区分：

- no-op/uncertain rows
- confirmed model blindspots
- confirmed bad rows requiring fresh redraw
- target-gap feasibility
- post-redraw must rerun Val-first funnel

若未来 filled verdict 导入后 `confirmed_bad_rows >= 1708`，也只能说明数据治理口径上可能覆盖 99.9% 缺口；仍必须 fresh redraw、补 cache、跑 Val/Test-10k/full-test 漏斗，不能直接宣称模型达标。

## 验证

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\import_loop72_external_verdicts.py scripts\apply_manual_review_verdicts.py scripts\build_loop72_review_wave_plan.py
.\vnev\Scripts\python.exe -m pytest tests\test_import_loop72_external_verdicts.py tests\test_apply_manual_review_verdicts.py tests\test_build_loop72_review_wave_plan.py -q
```

结果：`22 passed`。

Generated reports are not committed.
