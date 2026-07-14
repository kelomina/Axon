# Phase 3 Loop157 External Annotation Package Report

更新时间：2026-07-08

## 目标

Loop156 已经把当前 strict best Loop151 的全部 `162` 个 Val 错误导出成盲化复核包。Loop157 继续把这份包整理成可交给独立人工或外部证据系统的安全标注入口：一个 reviewer context CSV、一个只含表头的 annotation template，以及一份 reviewer guide。

Loop157 不读取 private map、不 unblind、不训练、不评估、不采样 replacement、不改 split/cache。外部 reviewer 只能看到 `review_focus_id`、当前标签、错误方向、review lane 和内容派生数值字段。路径、文件名、目录、后缀、hash、`source_sha256`、`sample_index`、split、row order、模型分数、概率、prediction、threshold、neighbor label 和 similarity 全部不导出，也不能作为 verdict 证据。

## 产物

新增脚本和测试：

- `scripts/export_loop157_current_best_val_external_annotation_package.py`
- `tests/test_export_loop157_current_best_val_external_annotation_package.py`

真实输出：

- `reports/phase3_loop157/loop157_loop151_val_all_errors_external_package_summary.json`
- `reports/phase3_loop157/loop157_loop151_val_all_errors_external_context.csv`
- `reports/phase3_loop157/loop157_loop151_val_all_errors_annotation_template.csv`
- `reports/phase3_loop157/loop157_loop151_val_all_errors_reviewer_guide.json`

## 当前真实状态

| Item | Value |
|---|---:|
| Rows | `162` |
| Context fields | `33` |
| Label 0 / 1 | `105 / 57` |
| FP / FN | `105 / 57` |
| Missing required columns | `0` |
| Forbidden input columns | `0` |
| Context header violations | `0` |
| Context value violations | `0` |
| Missing / duplicate review IDs | `0 / 0` |
| Decision | `ready_for_external_content_annotation` |

Annotation template 当前是 header-only，字段固定为：

```text
review_focus_id, manual_label_verdict, manual_verdict_note, recommended_action
```

## 决策

Loop157 已准备好交给独立内容/外部证据系统标注，但它不产生自动 verdict。后续返回文件必须只包含允许的四列；若混入路径、hash、文件名、`source_sha256`、`sample_index`、模型分数、概率、prediction 或 threshold，必须被 preflight 阻断。

如果外部 verdict 确认某些行 `label_wrong`、`feature_broken` 或 `out_of_scope`，仍然只能走 `exclude_and_replace`，并从 locked manifest 的同原始标签池 fresh redraw。坏样本不能直接改标，不能自己补齐名额，最终仍必须严格保持 `200000 = 20000/20000/160000`。

## 验证

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_export_loop157_current_best_val_external_annotation_package.py -q

.\vnev\Scripts\python.exe scripts\export_loop157_current_best_val_external_annotation_package.py --review-csv reports\phase3_loop156\loop156_loop151_val_all_errors_blinded.csv --context-csv reports\phase3_loop157\loop157_loop151_val_all_errors_external_context.csv --annotation-template-csv reports\phase3_loop157\loop157_loop151_val_all_errors_annotation_template.csv --reviewer-guide-json reports\phase3_loop157\loop157_loop151_val_all_errors_reviewer_guide.json --output-json reports\phase3_loop157\loop157_loop151_val_all_errors_external_package_summary.json --expected-rows 162
```

结果：pytest `4 passed`；真实导出 `decision=ready_for_external_content_annotation`、`blockers=[]`。
