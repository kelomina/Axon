# Phase 3 Loop163 R11 Rescue Support Audit

更新时间：2026-07-08

## 目标

Loop159/160 都围绕 R11 recall rescue 做文章，但它们的真实分歧样本非常少。Loop163 的目标是先回答一个更基础的问题：Val 上到底有没有足够支撑继续训练 selector 或写新规则？

这份审计只读，不训练、不调阈值、不进入 Test-10k/full-test 选择。公开 CSV 只含合成 `loop163_focus_id`、方向、标签、候选结果和分桶分数；真实路径、hash、`source_sha256`、`sample_index` 只在 private map 中保留，用于对齐和审计。

## 产物

新增脚本和测试：

- `scripts/build_loop163_r11_rescue_support_audit.py`
- `tests/test_build_loop163_r11_rescue_support_audit.py`

真实输出：

- `reports/phase3_loop163/loop163_r11_rescue_support_audit.json`
- `reports/phase3_loop163/loop163_r11_rescue_support_audit.md`
- `reports/phase3_loop163/loop163_r11_rescue_support_public.csv`
- `reports/phase3_loop163/loop163_r11_rescue_support_private_map.csv`

## 支撑度审计

阈值要求：

- Val disagreement rows 至少 `30`。
- Val candidate-fixes rows 至少 `10`。
- Val candidate-breaks rows 最多 `0`。

真实结果：

| Split | Rows | Disagreements | Fixes | Breaks | Direction |
|---|---:|---:|---:|---:|---|
| Val | `20000` | `9` | `8` | `1` | `0_to_1=9` |
| Test-10k | `10000` | `6` | `3` | `3` | `0_to_1=6` |
| Full-test posthoc | `160000` | `141` | `58` | `83` | `0_to_1=141` |

Val 支撑失败项：

- `val_disagreement_support_below_minimum`
- `val_fix_support_below_minimum`
- `val_break_rows_exceed_limit`

## 决策

Loop163 判定为 `reject_low_support_no_selector_training`。R11-only rescue 在 Val 上只有 `9` 个分歧，其中已经有 `1` 个会破坏原本正确样本；这不够支撑继续训练 selector，也不够支撑继续写概率阈值规则。

后续停止 probability-only / R11-only selector 搜索。若要继续救 FN，必须先有新的 Val-side 内容证据或外部证据，并且仍要经过 Loop161 promotion guard。

## 验证

```powershell
.\vnev\Scripts\python.exe scripts\build_loop163_r11_rescue_support_audit.py --val-base-csv reports\phase3_loop151\loop151_trusted_signer_guard_val_predictions.csv --val-candidate-csv reports\phase3_loop159\loop159_r11_only_trusted_signer_val_predictions.csv --test10k-base-csv reports\phase3_loop151\loop151_trusted_signer_guard_test10k_predictions.csv --test10k-candidate-csv reports\phase3_loop159\loop159_r11_only_trusted_signer_test10k_predictions.csv --full-base-csv reports\phase3_loop151\loop151_trusted_signer_guard_full_predictions.csv --full-candidate-csv reports\phase3_loop151\loop151_trusted_signer_guard_on_r11_filtered_full_predictions.csv --output-json reports\phase3_loop163\loop163_r11_rescue_support_audit.json --output-public-csv reports\phase3_loop163\loop163_r11_rescue_support_public.csv --output-private-map-csv reports\phase3_loop163\loop163_r11_rescue_support_private_map.csv --output-md reports\phase3_loop163\loop163_r11_rescue_support_audit.md --min-val-disagreements-for-selector 30 --min-val-fix-rows-for-selector 10 --max-val-break-rows-for-selector 0

.\vnev\Scripts\python.exe -m pytest tests\test_build_loop163_r11_rescue_support_audit.py -q
```

结果：Loop163 `decision=reject_low_support_no_selector_training`；单测 `1 passed`。
