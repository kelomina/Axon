# Phase 3 Loop127 Content-Cross 全量验证报告

更新时间：2026-07-04

> 更新说明：本文记录的是旧版 full-test best。当前严格漏斗下的新最佳已更新为 Loop129 content FP guard R4，full-test F1 `0.9902110451`、错误 `1571 / 160000`。最新结论见 `docs/phase3_loop129_content_fp_guard_report.md`。

## 结论

本轮最佳候选是 `hgb_l43_lr0.06_leaf23_iter320_l2_0__noise_none`，它在严格 20w 划分上完成了 Train/Val、Test-10k、16w full-test 漏斗验证。

它显著优于 Phase 1 probability calibrator，但没有达到最终目标 F1 >= 99.9%。当前 full-test F1 是 `0.9885577252`，总错误 `1837 / 160000`。要接近 99.9%，总错误需要从千级降到约百级，因此下一轮不能继续只做小幅阈值或 HGB 参数微调，必须转向错误归因、标签噪声、特征盲区和正交特征。

## 数据与协议

- 总数据：200,000 文件。
- 划分：train 20,000，val 20,000，test 160,000。
- 评估协议：Val 选择模型和阈值；Test-10k 只做泛化确认；只有通过 Test-10k 后才跑 full-test。
- 身份特征约束：`source_path`、`cache_path`、`source_sha256`、文件名、后缀、目录、`sample_index`、`split`、row order 均不得作为模型证据，只允许用于加载、对齐、审计。
- full-test 评估脚本已改为 content sidecar cache-only，缺失 sidecar 不再在评估阶段隐式生成。

## 关键修复

1. 修复 `scripts/evaluate_stage2_cache_model.py`：
   - 支持 Loop43 content-cross payload 的冻结特征重建。
   - 默认 cache-only 读取 content PE v1/v2 sidecar。
   - 缺失或坏 sidecar 立即失败，不再评估时现场解析源文件。
   - 增加模型输入维度检查和预测分块。

2. 补齐 full-test content PE v2 sidecar：
   - 输入：`reports/phase1_loop127/baseline_full_test_predictions.csv`
   - 输出目录：`reports/random_20w_split/content_pe_v2_cache`
   - 结果：160,000 rows，创建 149,994，已存在 10,006，失败 0。
   - 独立复验：v1/v2 均无缺失、无坏维度；v2 全部为零向量。

## 指标汇总

| 阶段/模型 | 阈值 | F1 | Errors | FP | FN | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Phase 1 calibrator Val | 0.44 | 0.9688605451 | 625 | - | - | 基准 |
| Loop43 original Val | 0.44 | 0.9881801406 | 237 | 144 | 93 | 通过 Val |
| Loop43 original Test-10k | 0.44 | 0.9890087930 | 110 | 74 | 36 | 通过 Test-10k |
| Loop43 original full-test | 0.44 | 0.9882065475 | 1892 | 1160 | 732 | 未达 99.9 |
| Local HGB Val | 0.40 | 0.9883466135 | 234 | 157 | 77 | 小幅改进 |
| Local HGB Test-10k | 0.40 | 0.9905066453 | 95 | 66 | 29 | 优于 original |
| Local HGB full-test | 0.40 | 0.9885577252 | 1837 | 1191 | 646 | 当时最佳，仍未达 99.9 |

## 旧 Best 模型

- 模型：`reports/phase3_loop127/phase1_content_cross_hgb_local_valonly/loop43_content_cross_selected_model.pkl`
- full-test 报告：`reports/phase3_loop127/phase1_content_cross_hgb_local_full_test_eval.json`
- full-test 预测：`reports/phase3_loop127/phase1_content_cross_hgb_local_full_test_predictions.csv`
- full-test 错误审计：`reports/phase3_loop127/phase1_content_cross_hgb_local_full_error_intrinsics.json`
- full-test review 队列：`reports/phase3_loop127/phase1_content_cross_hgb_local_full_review_queue.csv`

## 错误审计

本报告记录的旧 best full-test 错误：FP `1191`，FN `646`。

置信度分布：

- 高置信 FN `<0.10`：290
- 中置信 FN `0.10-0.30`：240
- 近阈值 FN `0.30-0.40`：116
- 高置信 FP `>=0.90`：519
- 中置信 FP `0.75-0.90`：222
- 宽近阈值 FP `0.40-0.75`：450

审计队列：

- `label_noise_extreme_fn`: 93
- `label_noise_extreme_fp`: 195
- `label_noise_high_fn`: 117
- `label_noise_high_fp`: 203
- `calibration_near_threshold`: 48
- `calibration_broad_near_threshold`: 65
- `model_behavior_review`: 1116

这说明当前主要问题不是简单阈值校准。高置信 FP/FN 数量很大，下一轮应优先排查标签噪声和模型强误判，而不是继续只扫阈值。

## 下一步建议

1. 先处理高置信 FP。
   当前 FP 多于 FN，且高置信 FP `>=0.90` 有 519 个。应抽样审计这些样本的 PE/stat/content 数值特征、相似样本、标签来源可靠性，确认是标签噪声还是良性样本具有高风险结构。

2. 再处理高置信 FN。
   高置信 FN `<0.10` 有 290 个。这类样本是模型明确看错方向，通常不是阈值能解决，需要找缺失信号，例如字节截断、PE 解析失败、壳/overlay/资源结构、统计分布盲区。

3. 不再优先做 HGB 小网格。
   HGB 局部搜索只把 Val errors 从 237 降到 234，full-test 从 1892 降到 1837，收益太小。

4. 新特征必须先过身份特征审计。
   string/cert/vendor/PDB/original filename 这类内容字段可能包含命名或来源偏置，只能先做诊断或分组 ablation，不能直接作为上线证据。

5. 若两轮错误审计后仍停留在千级错误，应提出指标降级或数据治理需求。
   当前 full-test severe conflict 为 510，且高置信错误超过 800。若其中大量属于标签噪声或不可分样本，F1 >= 99.9% 在当前数据合同下不现实。

## 复现入口

核心 full-test 评估命令：

```powershell
vnev\Scripts\python.exe scripts\evaluate_stage2_cache_model.py `
  --model reports\phase3_loop127\phase1_content_cross_hgb_local_valonly\loop43_content_cross_selected_model.pkl `
  --predictions reports\phase1_loop127\baseline_full_test_predictions.csv `
  --threshold 0.4 `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --content-pe-v2-cache-dir reports\random_20w_split\content_pe_v2_cache `
  --eval-chunk-size 10000 `
  --output-json reports\phase3_loop127\phase1_content_cross_hgb_local_full_test_eval.json `
  --output-predictions-csv reports\phase3_loop127\phase1_content_cross_hgb_local_full_test_predictions.csv
```
