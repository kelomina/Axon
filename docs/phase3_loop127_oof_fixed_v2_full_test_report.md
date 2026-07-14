# Phase 3 Loop127 OOF fixed-v2 全量验证报告

更新时间：2026-07-04

> 更新说明：本文记录的是上一版 OOF full-test best。当前严格漏斗下的新最佳已更新为 Loop129 content FP guard R4，full-test F1 `0.9902110451`、错误 `1571 / 160000`。最新结论见 `docs/phase3_loop129_content_fp_guard_report.md`。

## 结论

`oof_fixed_v2_all_valonly_with_logreg` 曾是严格漏斗下的新最佳候选。它在 Train/Val 上选择模型和阈值，在 Test-10k 上冻结确认，并最终完成 16w full-test。当前已被 Loop129 content FP guard R4 替代。

新 full-test F1 是 `0.9898088141`，错误 `1637 / 160000`，相比上一版 `phase1_content_cross_hgb_local` 的 `0.9885577252` / `1837` 错误，净减少 `200` 个错误。

它仍没有达到最终目标 F1 >= 99.9%。按 16w test 估算，99.9% 级别需要总错误接近百级，而当前仍是千级错误。因此下一轮重点应继续放在高置信错误、噪声审计、FP guard、FN overlay/security gate，而不是继续做小范围 HGB 参数微调。

## 数据与协议

- 总数据：200,000 文件。
- 划分：train 20,000，val 20,000，test 160,000。
- 评估协议：Val 选择模型和阈值；Test-10k 只做冻结确认；只有 Test-10k 优于当前 best 才跑 16w full-test。
- 身份特征约束：`source_path`、`cache_path`、`source_sha256`、文件名、后缀、目录、`sample_index`、`split`、row order 均不得作为模型证据，只允许用于加载、对齐、审计。
- 本候选使用 fixed-v2 修复缓存：`reports/phase3_loop127/content_pe_v2_fixed_cache_train_val`。

## fixed-v2 cache 状态

Train/Val/Test-10k fixed-v2 sidecar 已在前序步骤完成重建和验证。本轮为 full-test 继续补齐同一修复缓存：

- 构建报告：`reports/phase3_loop127/content_pe_v2_fixed_full_test_materialization.json`
- full-test 输入行：160,000
- unique rows：160,000
- exists：159,031
- created：969
- failed：0
- zero_features：1

唯一 zero_features 样本是 `sample_index=166255` / SHA `f2b4aaada89e69174520d0d3988149435ecaca7de454a379fe1dcb0974a34bf4`。复查结果：源文件 SHA 匹配，是有效 `MZ/PE`，但 `pefile` 解析到 sections=0、imports/resources/exports 均不存在；当前 v2 特征只覆盖 import/export/resource/section 结构，所以该全零向量是合法“无 v2 内容信号”，不是缓存损坏。

## 候选路径

本轮先验证了 fixed-v2 子集的直接 Stage-2 候选：

| 候选 | Val threshold | Val F1 | Val errors | Test-10k F1 | Test-10k errors | 结论 |
|---|---:|---:|---:|---:|---:|---|
| fixed-v2 section direct | 0.38 | 0.9882470120 | 236 | - | - | Val 未过 gate |
| fixed-v2 imports direct | 0.585 | 0.9887033890 | 226 | 0.9890704903 | 109 | Test-10k 未过 gate |
| fixed-v2 resource+export direct | 0.435 | 0.9888323861 | 224 | 0.9894989499 | 105 | Test-10k 未过 gate |
| OOF fixed-v2 all stacker, no logreg base | 0.415 | 0.9896300728 | 208 | 0.9907981596 | 92 | 进入 full-test |
| OOF fixed-v2 all stacker, with logreg base | 0.31 | 0.9897512438 | 206 | 0.9912123028 | 88 | 当时最佳 |

直接 Stage-2 子集在 Val 上能降低错误，但 Test-10k 泛化不足。OOF stacker 利用 train 内 out-of-fold 预测给 meta 层训练，减少二阶段过拟合，最终通过 Test-10k gate。

## 新旧 best 对比

| 模型 | 阈值 | Full-test F1 | Errors | FP | FN | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 旧 best：Local HGB content-cross | 0.40 | 0.9885577252 | 1837 | 1191 | 646 | 已被替代 |
| OOF fixed-v2 all stacker, no logreg base | 0.415 | 0.9897809673 | 1639 | 1013 | 626 | FP 更低 |
| 旧 best：OOF fixed-v2 all stacker, with logreg base | 0.31 | 0.9898088141 | 1637 | 1133 | 504 | 已被 Loop129 R4 替代 |

净变化：

- 总错误：`1837 -> 1637`，减少 `200`
- FP：`1191 -> 1133`，减少 `58`
- FN：`646 -> 504`，减少 `142`
- AUC：`0.9992421952`
- records kept：`160000 / 160000`
- skipped missing cache：`0`

差异审计：`reports/phase3_loop127/oof_fixed_v2_all_with_logreg_vs_content_cross_full_delta.json`

- 修复旧错误：493
- 新增回归：293
- 修复 FP：258
- 修复 FN：235
- 新增 FP 回归：200
- 新增 FN 回归：93
- 高置信 FP：`519 -> 474`
- 高置信 FN：`290 -> 291`

相对 no-logreg OOF，对照审计 `reports/phase3_loop127/oof_fixed_v2_all_with_logreg_vs_no_logreg_full_delta.json` 显示：with-logreg 总错误再少 `2`，FN 少 `122`，但 FP 多 `120`。这说明当前 best 是偏召回的版本，下一轮 FP guard 的优先级更高。

## 旧 Best 模型

- 模型：`reports/phase3_loop127/oof_fixed_v2_all_valonly_with_logreg/stage2_oof_stacker_selected_model.pkl`
- Val 报告：`reports/phase3_loop127/oof_fixed_v2_all_valonly_with_logreg/stage2_oof_stacker_report.json`
- Test-10k 报告：`reports/phase3_loop127/oof_fixed_v2_all_with_logreg_test10k_eval.json`
- Test-10k 预测：`reports/phase3_loop127/oof_fixed_v2_all_with_logreg_test10k_predictions.csv`
- Full-test 报告：`reports/phase3_loop127/oof_fixed_v2_all_with_logreg_full_test_eval.json`
- Full-test 预测：`reports/phase3_loop127/oof_fixed_v2_all_with_logreg_full_test_predictions.csv`
- 新旧差异审计：`reports/phase3_loop127/oof_fixed_v2_all_with_logreg_vs_content_cross_full_delta.json`
- OOF 版本差异审计：`reports/phase3_loop127/oof_fixed_v2_all_with_logreg_vs_no_logreg_full_delta.json`

## 复现命令

Val 训练：

```powershell
vnev\Scripts\python.exe scripts\train_stage2_oof_stacker.py `
  --checkpoint models\random_20w_8192\best_model.pt `
  --train-predictions reports\phase3_loop127\phase1_base_train_predictions.csv `
  --val-predictions reports\phase1_loop127\baseline_val_predictions.csv `
  --output-dir reports\phase3_loop127\oof_fixed_v2_all_valonly_with_logreg `
  --feature-set extended `
  --content-pe-features `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --content-pe-v2-features `
  --content-pe-v2-cache-dir reports\phase3_loop127\content_pe_v2_fixed_cache_train_val `
  --content-pe-v2-groups all `
  --drop-base-prob-features `
  --base-model-candidates hgb_lr0.04_leaf15_l2_0,hgb_lr0.06_leaf31_l2_0,hgb_lr0.08_leaf31_l2_1e-3,extra_trees_300_leaf1,logreg_l2_c1 `
  --noise-modes none `
  --thresholds 0.30:0.70:0.005 `
  --folds 5 `
  --seed 128
```

Test-10k 冻结评估：

```powershell
vnev\Scripts\python.exe scripts\evaluate_stage2_oof_stacker.py `
  --model reports\phase3_loop127\oof_fixed_v2_all_valonly_with_logreg\stage2_oof_stacker_selected_model.pkl `
  --predictions reports\phase1_loop127\baseline_full_test_predictions.csv `
  --output-json reports\phase3_loop127\oof_fixed_v2_all_with_logreg_test10k_eval.json `
  --output-predictions-csv reports\phase3_loop127\oof_fixed_v2_all_with_logreg_test10k_predictions.csv `
  --max-rows 10000 `
  --threshold 0.31
```

Full-test 冻结评估：

```powershell
vnev\Scripts\python.exe scripts\evaluate_stage2_oof_stacker.py `
  --model reports\phase3_loop127\oof_fixed_v2_all_valonly_with_logreg\stage2_oof_stacker_selected_model.pkl `
  --predictions reports\phase1_loop127\baseline_full_test_predictions.csv `
  --output-json reports\phase3_loop127\oof_fixed_v2_all_with_logreg_full_test_eval.json `
  --output-predictions-csv reports\phase3_loop127\oof_fixed_v2_all_with_logreg_full_test_predictions.csv `
  --threshold 0.31
```

## 下一步建议

1. 优先做 FP guard。
   新 best 为了降低 FN，把 FP 从 no-logreg OOF 的 `1013` 增到 `1133`。下一轮应训练一个只允许 `1 -> 0` 的窄门 guard，目标是恢复 no-logreg 的低 FP 优势，同时保持 with-logreg 的 FN 收益。

2. 再做 FN overlay/security gate v2。
   新模型 FN 只从 `646` 降到 `626`，高置信 FN 还从 `290` 到 `291`。FN 不是阈值能解决，应围绕 PE header 安全标志、section/overlay/security/cert 结构做保守 `0 -> 1` gate，并用 FP guard 限制副作用。

3. 做 content-structure cluster 审计。
   继续按数值结构聚类，而不是按文件名或目录聚类。重点看 PE header flags、section flag pattern、entropy/import/packer/stat 摘要，对 Val/Test-10k/full-test 的 FP/FN 簇做一致性检查。

4. 暂缓 GA feature mask 作为主线。
   GA mask 在旧 20k sweep 有 trade-off，但在当前 Loop127 严格 Val 没有超过 unmasked baseline，不应进入 Test-10k/full-test 主线。

5. OOF stacker 的 sklearn Pipeline sample_weight 兼容性已修复。
   初次 OOF 运行包含 `logreg_l2_c1` 时失败，因为脚本将 `sample_weight` 直接传给 `Pipeline.fit`。当前已改为对 Pipeline 使用最后一步的 `stepname__sample_weight`，并通过 `tests/test_stage2_oof_stacker.py` 覆盖验证。

## 可行性判断

本轮 full-test 净减少 `198` 个错误，证明 fixed-v2 修复缓存和 OOF stacker 是有效方向。但距离 F1 >= 99.9% 仍很远。若目标维持 99.9%，下一阶段必须同时解决：

- 高置信 FP/FN 的结构性盲区；
- 疑似标签噪声和冲突样本；
- 现有模型之间 disagreement 的回归控制；
- fixed-v2 之外的正交内容特征，例如 overlay/security/cert/string 的严格身份审计版本。

如果连续两轮 full-test 仍停留在千级错误，应正式提出指标降级或数据治理需求。
