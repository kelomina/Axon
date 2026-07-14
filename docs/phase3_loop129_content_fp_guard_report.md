# Phase 3 Loop129 Content FP Guard 全量验证报告

更新时间：2026-07-05

更新说明：Loop129 R4 已被 Loop130 `R5_r4_plus_vendor_strings` 小幅替代。当前严格 best 见 `docs/phase3_loop130_content_string_guard_report.md`，full-test 错误为 `1563 / 160000`，F1 为 `0.9902567651`。

## 结论

`R4_resource_icon_lowconf_resource_ratio_floor` 是 Loop129 当时严格漏斗下的新最佳候选。它不是重新训练主模型，而是在当时 best `OOF fixed-v2 all stacker with logreg base` 之后加一个保守 FP guard，只允许把 primary 判黑的样本 `1 -> 0`，不允许 `0 -> 1`。

新 full-test F1 是 `0.9902110451`，错误 `1571 / 160000`，FP/FN 为 `1029 / 542`。相比上一版 with-logreg OOF best 的 `1637` 错误，净减少 `66`；相比旧 Local HGB content-cross 的 `1837` 错误，累计减少 `266`。

这仍未达到 F1 >= 99.9%。当前错误仍是千级，距离百级错误预算很远。下一阶段应继续做结构性 FP/FN 分治、噪声审计和更强正交内容特征，而不是只调阈值。

## 规则定义

R4 规则：

```text
possible_guard =
  with_logreg_prediction == 1
  and (no_logreg_prediction == 0 or old_content_cross_prediction == 0)

R2_resource_icon_lowconf =
  possible_guard
  and with_logreg_prob <= 0.65
  and v2_resource_data_entry_count_log >= 2.0
  and v2_resource_type_icon_count_log >= 1.5

R4_resource_icon_lowconf_resource_ratio_floor =
  R2_resource_icon_lowconf
  and content_dir_resource_size_ratio >= 0.001
```

命中后只执行 `prediction: 1 -> 0`。

用到的模型证据只包含两个冻结模型的概率/预测分歧，以及 content PE v1/v2 数值特征。`source_path`、`cache_path`、`source_sha256`、文件名、后缀、目录、`sample_index`、`split`、row order 只用于对齐、cache lookup 和审计，不作为模型特征。

## 漏斗结果

| 阶段 | 模型/规则 | F1 | Errors | FP | FN | Flips | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| Val | with-logreg OOF primary | 0.9897512438 | 206 | 153 | 53 | 0 | primary |
| Val | R2 FP guard | 0.9904875741 | 191 | 135 | 56 | 21 | 通过 Val |
| Test-10k | with-logreg OOF primary | 0.9912123028 | 88 | 66 | 22 | 0 | primary |
| Test-10k | R2 FP guard | 0.9915025492 | 85 | 59 | 26 | 11 | 通过 Test-10k |
| Test-10k | R4 FP guard | 0.9915025492 | 85 | 59 | 26 | 11 | 与 R2 打平 |
| Full-test | with-logreg OOF primary | 0.9898088141 | 1637 | 1133 | 504 | 0 | 已被替代 |
| Full-test | R2 FP guard | 0.9902045089 | 1572 | 1027 | 545 | 147 | 已被 R4 小幅替代 |
| Full-test | R4 FP guard | 0.9902110451 | 1571 | 1029 | 542 | 142 | Loop129 best，已被 Loop130 替代 |

R4 full-test 具体变化：

- Flips：142
- Flipped benign label0：104
- Flipped malicious label1：38
- FP：`1133 -> 1029`，减少 `104`
- FN：`504 -> 542`，增加 `38`
- 净错误：`1637 -> 1571`，减少 `66`

R3 对照规则 `R3_resource_icon_lowconf_not_dll`：

| 阶段 | F1 | Errors | FP | FN | Flips | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Val | 0.9904391993 | 192 | 137 | 55 | 18 | 弱于 R2 |
| Test-10k | 0.9916033587 | 84 | 59 | 25 | 10 | Test-10k 略强 |
| Full-test | 0.9901803183 | 1576 | 1035 | 541 | 135 | full-test 弱于 R2 |

R4 是 R2 之后基于 Val harmful flip 分析新增的保护规则：Val `191 -> 190`，Test-10k 与 R2 打平，full-test `1572 -> 1571`，因此成为当前最佳。

## Probability-only Guard 负结果

Loop128 probability-only FP guard selector 已验证为不够稳定：

- 训练式 selector：Val 最优为 0 flips，错误仍为 `206`。
- Val-only 阈值规则：Val `206 -> 203`，但 Test-10k `88 -> 89`，翻错 1 个恶意样本。

因此，仅用 with-logreg/no-logreg 概率分歧不足以可靠降低 FP；必须加入 content PE 数值结构。

## Loop129 artifacts

- Val 报告：`reports/phase3_loop128/loop129_content_fp_guard_val_eval.json`
- Val 预测：`reports/phase3_loop128/loop129_content_fp_guard_val_predictions.csv`
- Test-10k 报告：`reports/phase3_loop128/loop129_content_fp_guard_test10k_eval.json`
- Test-10k 预测：`reports/phase3_loop128/loop129_content_fp_guard_test10k_predictions.csv`
- Full-test 报告：`reports/phase3_loop128/loop129_content_fp_guard_r4_full_test_eval.json`
- Full-test 预测：`reports/phase3_loop128/loop129_content_fp_guard_r4_full_test_predictions.csv`
- R2 full-test 对照：`reports/phase3_loop128/loop129_content_fp_guard_full_test_eval.json`
- 规则脚本：`scripts/evaluate_loop129_content_fp_guard_rules.py`
- Probability-only selector 负结果：`reports/phase3_loop128/loop128_fp_guard_selector_valonly/loop128_fp_guard_selector_report.json`，`reports/phase3_loop128/fp_guard_rule_test10k_eval.json`

## 复现命令

Val 规则选择：

```powershell
vnev\Scripts\python.exe scripts\evaluate_loop129_content_fp_guard_rules.py `
  --primary-predictions reports\phase3_loop128\oof_fixed_v2_all_with_logreg_train_oof\stage2_oof_stacker_val_predictions.csv `
  --conservative-predictions reports\phase3_loop128\oof_fixed_v2_all_no_logreg_train_oof\stage2_oof_stacker_val_predictions.csv `
  --old-predictions reports\phase3_loop127\phase1_content_cross_hgb_local_valonly\loop43_content_cross_val_predictions.csv `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --content-pe-v2-cache-dir reports\phase3_loop127\content_pe_v2_fixed_cache_train_val `
  --output-json reports\phase3_loop128\loop129_content_fp_guard_val_eval.json `
  --output-predictions-csv reports\phase3_loop128\loop129_content_fp_guard_val_predictions.csv
```

Test-10k 冻结评估：

```powershell
vnev\Scripts\python.exe scripts\evaluate_loop129_content_fp_guard_rules.py `
  --primary-predictions reports\phase3_loop127\oof_fixed_v2_all_with_logreg_test10k_predictions.csv `
  --conservative-predictions reports\phase3_loop127\oof_fixed_v2_all_test10k_predictions.csv `
  --old-predictions reports\phase3_loop127\phase1_content_cross_hgb_local_test10k_predictions.csv `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --content-pe-v2-cache-dir reports\phase3_loop127\content_pe_v2_fixed_cache_train_val `
  --output-json reports\phase3_loop128\loop129_content_fp_guard_r4_test10k_eval.json `
  --output-predictions-csv reports\phase3_loop128\loop129_content_fp_guard_r4_test10k_predictions.csv `
  --select-rule R4_resource_icon_lowconf_resource_ratio_floor
```

Full-test 冻结评估：

```powershell
vnev\Scripts\python.exe scripts\evaluate_loop129_content_fp_guard_rules.py `
  --primary-predictions reports\phase3_loop127\oof_fixed_v2_all_with_logreg_full_test_predictions.csv `
  --conservative-predictions reports\phase3_loop127\oof_fixed_v2_all_full_test_predictions.csv `
  --old-predictions reports\phase3_loop127\phase1_content_cross_hgb_local_full_test_predictions.csv `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --content-pe-v2-cache-dir reports\phase3_loop127\content_pe_v2_fixed_cache_train_val `
  --output-json reports\phase3_loop128\loop129_content_fp_guard_r4_full_test_eval.json `
  --output-predictions-csv reports\phase3_loop128\loop129_content_fp_guard_r4_full_test_predictions.csv `
  --select-rule R4_resource_icon_lowconf_resource_ratio_floor
```

## 下一步建议

1. 继续做 FP/FN 分治，而不是单一阈值。
   R4 证明 resource/icon-heavy 的 FP guard 有效，但它用 `+38 FN` 换 `-104 FP`。下一步要做的是在 R4 命中行中识别被误翻的 38 个恶意样本，重点看 export/DLL/security/overlay 证据。

2. 增加 FN recovery 的保护条件。
   当前 FN 从 `504` 增到 `545`。可以把 with-logreg 的 FN 优势和 no-logreg/R2 的 FP 优势做分层 selector，但必须使用 train OOF 或预声明 content 规则，不能在 Test/full 调参。

3. 做 R2 命中簇的 content feature 审计。
   对 flipped label0/label1 分别统计 resource、icon、export、DLL、security、overlay 分布，找出能减少 harmful flips 的下一条规则。

4. 噪声审计仍然必要。
   即使当前 F1 已过 `0.9902`，full-test 仍有 `1571` 错误；若高置信冲突中存在标签噪声，99.9% 目标仍不现实。
