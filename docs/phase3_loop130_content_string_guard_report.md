# Phase 3 Loop130 Content String Guard 全量验证报告

更新时间：2026-07-05

## 结论

**更新说明：本报告的“当前最佳”口径已被后续 Loop136 取代。** Loop130 R5 仍是重要 fallback 与对照基线，但当前 full-test strict best 是 `docs/phase3_loop136_recall_pairwise_selector_full_test_report.md` 中的 Loop136，F1 `0.9903723842`、errors `1544 / 160000`、FP/FN `958 / 586`。

`R5_r4_plus_vendor_strings` 是当前严格漏斗下的新最佳候选。它不是重新训练主模型，而是在 Loop129 R4 的 content PE FP guard 后，再对剩余的模型分歧行加入一个极窄的 content-string FP guard。规则仍然只允许 `prediction: 1 -> 0`，不允许 `0 -> 1`。

新 full-test F1 是 `0.9902567651`，错误 `1563 / 160000`，FP/FN 为 `991 / 572`。相比 Loop129 R4 的 `1571` 错误，净减少 `8`；相比 with-logreg OOF primary 的 `1637` 错误，净减少 `74`；相比旧 Local HGB content-cross 的 `1837` 错误，累计减少 `274`。

这仍未达到 F1 >= 99.9%。当前错误仍是千级，距离 160 个以内错误预算还很远。R5 证明 content string 侧信息有轻微信号，但收益很小，下一阶段不应继续无约束堆小规则，而应转向更系统的 FN recovery、噪声审计和正交模型候选。

## 规则定义

R5 规则：

```text
possible_guard =
  with_logreg_prediction == 1
  and (no_logreg_prediction == 0 or old_content_cross_prediction == 0)

R4 =
  possible_guard
  and with_logreg_prob <= 0.65
  and v2_resource_data_entry_count_log >= 2.0
  and v2_resource_type_icon_count_log >= 1.5
  and content_dir_resource_size_ratio >= 0.001

R5 =
  R4
  or (
    possible_guard
    and not R4
    and string_benign_vendor_count_log >= 3.0
  )
```

命中后只执行 `prediction: 1 -> 0`。

`string_benign_vendor_count_log` 来自二进制内容中的 vendor 关键词计数，不来自文件名、目录名、后缀或路径。`source_path`、`cache_path`、`source_sha256`、文件名、后缀、目录、`sample_index`、`split`、row order 只用于对齐、cache lookup 和审计，不作为模型特征。

## 漏斗结果

| 阶段 | 模型/规则 | F1 | Errors | FP | FN | Flips | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| Val | R4 | 0.9905378486 | 190 | 135 | 55 | 20 | 被 R5 替代 |
| Val | R5 | 0.9907342832 | 186 | 130 | 56 | 26 | 通过 Val |
| Test-10k | R4 | 0.9915025492 | 85 | 59 | 26 | 11 | 被 R5 小幅替代 |
| Test-10k | R5 | 0.9915983197 | 84 | 56 | 28 | 16 | 通过 Test-10k |
| Full-test | R4 | 0.9902110451 | 1571 | 1029 | 542 | 142 | 被 R5 替代 |
| Full-test | R5 | 0.9902567651 | 1563 | 991 | 572 | 210 | 当前最佳 |

R5 full-test 具体变化，相比 R4：

- Extra flips over R4：68
- FP：`1029 -> 991`，减少 `38`
- FN：`542 -> 572`，增加 `30`
- 净错误：`1571 -> 1563`，减少 `8`

R5 是一个很窄但有代价的 FP guard：它继续减少误报，但进一步增加漏报。业务上不能只看 F1，需要关注 FN 增量。

## String Cache 证据

Train+Val string cache：

- `reports/phase3_loop128/content_string_train_val_materialization.json`
- rows：`40000`
- failed：`0`
- zero_features：`0`

Test-10k string cache：

- `reports/phase3_loop128/content_string_test10k_materialization.json`
- 后续校验：`reports/phase3_loop128/content_string_test10k_validation.json`
- rows：`10000`
- missing/bad/nonfinite：`0/0/0`
- zero_features：`0`

Full-test string cache：

- 构建报告：`reports/phase3_loop128/content_string_full_test_materialization.json`
- 构建时有 1 个瞬时失败，单样本重跑成功补齐
- 最终校验：`reports/phase3_loop128/content_string_full_test_validation.json`
- rows：`160000`
- missing/bad/nonfinite：`0/0/0`
- zero_features：`0`

这说明 full-test 评估没有缺样，也没有把坏 sidecar 当作有效输入。

## 对照规则

### String OOF 负结果

在资源门通过后，尝试了 `fixed-v2 all + content PE v1 + content_string + logreg base` 的 OOF stacker Val-only 候选：

- 输出目录：`reports/phase3_loop131/oof_fixed_v2_all_string_with_logreg_valonly`
- Val 选择：`meta_logreg_l2_c1`
- Val F1：`0.9899312132`
- Val errors：`202`
- FP/FN：`132 / 70`

它弱于当前 R5 的 Val `186` errors，因此没有资格进入 Test-10k。随后只读检查它与 R5 的错误互补性：R5 错而 string OOF 对的样本有 `26` 个，但 string OOF 错而 R5 对的样本有 `42` 个；简单阈值 override 最多只能打平 R5，不能提升。因此该路线暂记为负结果。

### 固定规则对照

Loop130 full-test 已验证对照：

| 规则 | Full-test F1 | Errors | FP | FN | 结论 |
|---|---:|---:|---:|---:|---|
| R6 version-resource strings | 0.9900622202 | 1594 | 996 | 598 | 弱于 R4/R5 |
| R7 resource entry >= log(40) | 0.9902269979 | 1568 | 1005 | 563 | 弱于 R5 |

Loop130 Test-10k 否决对照：

| 规则 | Val Errors | Test-10k Errors | 结论 |
|---|---:|---:|---|
| R8 dialog protector | 185 | 85 | Val 优于 R5 的 186，但 Test-10k 弱于 R5 的 84，拒绝，不跑 full-test |

Loop132 FN recovery 负结果：

| 规则 | Val Errors | Test-10k Errors | Full-test Errors | 结论 |
|---|---:|---:|---:|---|
| R11 file/version/virtual-raw FN recovery | 180 | 84 | 1594 | Val 明显优于 R5，Test-10k 总错误打平但 full-test 明显退化，拒绝 |
| overlay/last-section/image-base FN probe | 180 | 86 | 未跑 | Val 明显优于 R5，但 Test-10k 退化，拒绝 |

R11 只允许 `0 -> 1`，在 Val 修复 8 个 FN、误伤 2 个 FP；但 full-test 变成修复 62 个 FN、误伤 93 个 FP，净错误从 R5 的 `1563` 退到 `1594`。这说明当前 FN recovery 小规则外推不稳，后续不能只靠 Val 小支持规则推进。

overlay/last-section/image-base 探针规则为：只在 R5 final negative 中恢复 `content_overlay_log_size >= 16.5 OR v2_last_section_entropy >= 0.98 OR content_image_base_log >= 0.5`。它在 Val 修复 6 个 FN 且不增加 FP，但 Test-10k 只增加 2 个 FP、没有修复 FN，因此不进入 full-test。

因此当前只采用 R5。

## 当前最佳 artifacts

- Val 报告：`reports/phase3_loop128/loop130_content_string_guard_val_eval.json`
- Val 预测：`reports/phase3_loop128/loop130_content_string_guard_val_predictions.csv`
- Test-10k 报告：`reports/phase3_loop128/loop130_content_string_guard_r5_test10k_eval.json`
- Test-10k 预测：`reports/phase3_loop128/loop130_content_string_guard_r5_test10k_predictions.csv`
- Full-test 报告：`reports/phase3_loop128/loop130_content_string_guard_r5_full_test_eval.json`
- Full-test 预测：`reports/phase3_loop128/loop130_content_string_guard_r5_full_test_predictions.csv`
- R8 负结果：`reports/phase3_loop128/loop130_content_string_guard_r8_val_eval.json`，`reports/phase3_loop128/loop130_content_string_guard_r8_test10k_eval.json`
- Loop132 FN recovery 负结果：`reports/phase3_loop132/loop132_fn_recovery_val_eval.json`，`reports/phase3_loop132/loop132_fn_recovery_r11_test10k_eval.json`，`reports/phase3_loop132/loop132_fn_recovery_r11_full_test_eval.json`
- 规则脚本：`scripts/evaluate_loop130_content_string_guard_rules.py`

## 复现命令

Val 规则选择：

```powershell
vnev\Scripts\python.exe scripts\evaluate_loop130_content_string_guard_rules.py `
  --primary-predictions reports\phase3_loop128\oof_fixed_v2_all_with_logreg_train_oof\stage2_oof_stacker_val_predictions.csv `
  --conservative-predictions reports\phase3_loop128\oof_fixed_v2_all_no_logreg_train_oof\stage2_oof_stacker_val_predictions.csv `
  --old-predictions reports\phase3_loop127\phase1_content_cross_hgb_local_valonly\loop43_content_cross_val_predictions.csv `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --content-pe-v2-cache-dir reports\phase3_loop127\content_pe_v2_fixed_cache_train_val `
  --content-string-cache-dir reports\phase3_loop128\content_string_cache_train_val `
  --output-json reports\phase3_loop128\loop130_content_string_guard_val_eval.json `
  --output-predictions-csv reports\phase3_loop128\loop130_content_string_guard_val_predictions.csv
```

Test-10k 冻结评估：

```powershell
vnev\Scripts\python.exe scripts\evaluate_loop130_content_string_guard_rules.py `
  --primary-predictions reports\phase3_loop127\oof_fixed_v2_all_with_logreg_test10k_predictions.csv `
  --conservative-predictions reports\phase3_loop127\oof_fixed_v2_all_test10k_predictions.csv `
  --old-predictions reports\phase3_loop127\phase1_content_cross_hgb_local_test10k_predictions.csv `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --content-pe-v2-cache-dir reports\phase3_loop127\content_pe_v2_fixed_cache_train_val `
  --content-string-cache-dir reports\phase3_loop128\content_string_cache_test10k `
  --select-rule R5_r4_plus_vendor_strings `
  --output-json reports\phase3_loop128\loop130_content_string_guard_r5_test10k_eval.json `
  --output-predictions-csv reports\phase3_loop128\loop130_content_string_guard_r5_test10k_predictions.csv
```

Full-test 冻结评估：

```powershell
vnev\Scripts\python.exe scripts\evaluate_loop130_content_string_guard_rules.py `
  --primary-predictions reports\phase3_loop127\oof_fixed_v2_all_with_logreg_full_test_predictions.csv `
  --conservative-predictions reports\phase3_loop127\oof_fixed_v2_all_full_test_predictions.csv `
  --old-predictions reports\phase3_loop127\phase1_content_cross_hgb_local_full_test_predictions.csv `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --content-pe-v2-cache-dir reports\phase3_loop127\content_pe_v2_fixed_cache_train_val `
  --content-string-cache-dir reports\phase3_loop128\content_string_cache_full_test `
  --select-rule R5_r4_plus_vendor_strings `
  --output-json reports\phase3_loop128\loop130_content_string_guard_r5_full_test_eval.json `
  --output-predictions-csv reports\phase3_loop128\loop130_content_string_guard_r5_full_test_predictions.csv
```

## 下一步建议

1. 把 R5 作为当前 strict best，但不要继续围绕同一类 tiny FP guard 过度手调。
   R5 的 Val/Test/full 都为正，但 full-test 只净减少 8 个错误，说明这个方向已经接近边际收益。

2. 优先做 FN recovery，但必须严格用 Val 选择、Test-10k 确认。
   R5 相比 R4 额外增加 30 个 FN；下一轮应重点分析 R4/R5 harmful flips，以及当前 non-guard high-confidence FN。

3. 等系统内存恢复后，再跑 fixed-v2 + string 的 OOF stacker 候选。
   现在 string cache 已覆盖 train/val/test/full，真正的模型级收益仍需 Train/Val OOF 验证。

4. 继续做噪声审计。
   当前 best 仍有 `1563` 个 full-test 错误。若高置信错误中存在明显标签噪声或不可分样本，F1 >= 99.9% 目标需要科学降级或引入更强动态/行为特征。
