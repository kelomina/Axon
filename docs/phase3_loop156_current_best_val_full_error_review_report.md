# Phase 3 Loop156 Current-Best Val Full-Error Review Report

更新时间：2026-07-08

## 目标

Loop153 已经把当前 strict best Loop151 的 Val 高冲突错误压成 `73` 行优先复核包，但它只覆盖 `73 / 162` 个当前 Val 错误。Loop156 补齐另一半治理面：把 Loop151 当前仍错的全部 `162` 个 Val 样本导出为盲化复核包，让独立内容/外部证据可以覆盖完整 Val 错误面。

Loop156 不训练、不评估、不改阈值、不采样 replacement、不改 split/cache。它只导出 reviewer-facing CSV 和 private map。路径、文件名、目录、后缀、hash、`source_sha256`、`sample_index`、split、row order、模型分数、概率、prediction、neighbor label 和 similarity 只允许用于物流、对齐和审计，不允许作为 verdict、模型、阈值、GA mask、replacement sampling 或生产推理证据。

## 产物

新增脚本和测试：

- `scripts/build_loop156_current_best_val_full_error_review.py`
- `tests/test_build_loop156_current_best_val_full_error_review.py`

真实输出：

- `reports/phase3_loop156/loop156_loop151_val_all_errors_summary.json`
- `reports/phase3_loop156/loop156_loop151_val_all_errors_blinded.csv`
- `reports/phase3_loop156/loop156_loop151_val_all_errors_private_map.csv`
- `reports/phase3_loop156/loop156_loop151_val_all_errors_preflight.json`
- `reports/phase3_loop156/loop156_loop151_val_all_errors_redraw_readiness_summary.json`

## 当前真实状态

| Item | Value |
|---|---:|
| Review rows | `162` |
| FP / FN | `105 / 57` |
| `neighbors_support_model_prediction` | `73` |
| `neighbors_support_dataset_label` | `24` |
| `neighbors_mixed` | `65` |
| Critical / High / Medium / Standard | `4 / 89 / 29 / 40` |
| Annotated rows | `0` |
| Replacement required | `0` |
| Decision | `await_external_verdicts` |

Review lane：

| Lane | Rows |
|---|---:|
| `benign_trust_or_label_quality_review` | `94` |
| `malware_blindspot_or_label_quality_review` | `45` |
| `content_evidence_review` | `23` |

Preflight no-op 结果：`rows=162`、`annotated_rows=0`、`replacement_required_rows=0`、`blockers=[]`、`ready_for_private_mapping=false`。Readiness 继续阻断：fresh redraw、Train/Val、Test-10k 和 full-test 全部 `false`。

## 决策

Loop156 是噪声治理扩面，不是模型候选。它让 reviewer 可以覆盖当前 best 的全 Val 错误面；但在没有独立 verdict 前，不允许自动 redraw、训练、Test-10k 或 full-test。

若后续独立内容/外部证据确认某些行 `label_wrong`、`feature_broken` 或 `out_of_scope`，这些行只能走 `exclude_and_replace`，并从 locked manifest 的同原始标签池 fresh redraw。坏样本不能直接改标，不能自己补齐名额，最终仍必须严格保持 `200000 = 20000/20000/160000`。

## 验证

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_build_loop156_current_best_val_full_error_review.py -q
```

结果：`1 passed`。

真实 no-op 复验：

```powershell
.\vnev\Scripts\python.exe scripts\preflight_loop126_review_annotations.py --annotations-csv reports\phase3_loop156\loop156_loop151_val_all_errors_blinded.csv --output-csv reports\phase3_loop156\loop156_loop151_val_all_errors_preflight.csv --output-json reports\phase3_loop156\loop156_loop151_val_all_errors_preflight.json --expected-rows 162

.\vnev\Scripts\python.exe scripts\run_loop152_loop150_val_focus_redraw_readiness.py --preflight-csv reports\phase3_loop156\loop156_loop151_val_all_errors_preflight.csv --preflight-json reports\phase3_loop156\loop156_loop151_val_all_errors_preflight.json --private-map-csv reports\phase3_loop156\loop156_loop151_val_all_errors_private_map.csv --split-csv reports\random_20w_split\loop127_full_duplicate_corrected_split.csv --manifest-json data\.cache\manifest_38672ba0.json --output-dir reports\phase3_loop156 --output-json reports\phase3_loop156\loop156_loop151_val_all_errors_redraw_readiness_summary.json --output-md reports\phase3_loop156\loop156_loop151_val_all_errors_redraw_readiness_summary.md
```

第一条命令返回非零是预期 no-op 状态，因为 `ready_for_private_mapping=false` 且当前还没有任何独立标注。
