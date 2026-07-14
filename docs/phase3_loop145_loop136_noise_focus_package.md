# Phase 3 Loop145 Loop136 Noise Focus Package

更新时间：2026-07-05

## 目标

Loop136 仍是当前 strict best，但 Loop138/Loop144 已经说明：继续堆小 selector 会反复遇到 FP 外溢，尤其是 `0 -> 1` 召回恢复在 Val 上容易被高估。Loop145 因此不直接训练新模型，而是把 Loop136 full-test 中最像“标签噪声 / 灰区 / 静态特征不可分”的错误样本，整理成一个盲化复核包，给后续独立证据复核和 fresh redraw 链路使用。

这一步服务最终模型目标：只有把坏样本和灰区样本用独立证据治理掉，后续训练/验证才不会继续被同一批冲突样本牵着走。

## 输入

- Loop136 full-test 错误邻域审计：`reports/phase3_loop138/loop136_full_errors_neighbor_audit.csv`
- Loop136 full-test 错误内容证据：`reports/phase3_loop138/loop136_error_review_queue.csv`

筛选策略：

- 只选 `support_bucket = neighbors_support_model_prediction`
- `priority <= 90`
- 按 `priority`、`opposite_label_ratio`、`nearest_similarity` 排序
- 输出 Top `300` 行

这些排序字段只用于复核优先级，不是 verdict，也不是模型特征。

## 新增脚本

新增：

- `scripts/build_loop145_loop136_blinded_noise_focus.py`
- `tests/test_build_loop145_loop136_blinded_noise_focus.py`

脚本输出两份文件：

1. 公开盲化复核表：只含 label、error type 和内容/PE/string 数值证据，不含路径、hash、sample index、概率、模型分数、邻居标签或相似度。
2. 私有映射表：仅用于定位样本和后续导入 verdict，不可作为证据输入。

## 结果

输出：

- `reports/phase3_loop145/loop145_loop136_full_noise_focus_blinded.csv`
- `reports/phase3_loop145/loop145_loop136_full_noise_focus_private_map.csv`
- `reports/phase3_loop145/loop145_loop136_full_noise_focus_summary.json`

复核包概况：

| Item | Count |
|---|---:|
| Input neighbor rows | `1544` |
| Selected focus rows | `300` |
| FP / FN | `137 / 163` |
| Critical / High | `25 / 275` |

Review lane：

| Lane | Count |
|---|---:|
| `benign_trust_or_label_quality_review` | `131` |
| `malware_blindspot_or_label_quality_review` | `113` |
| `content_evidence_review` | `56` |

主要内容标签：

| Tag | Count |
|---|---:|
| `benign_vendor_string_present` | `274` |
| `version_resource_present` | `214` |
| `overlay_present` | `181` |
| `security_directory_present` | `157` |
| `high_overlay_entropy` | `156` |
| `resource_rich` | `141` |
| `large_virtual_raw_gap` | `91` |

## 安全边界

公开 CSV 经字段扫描确认无以下公开列：

`source_path`、`cache_path`、`source_sha256`、`sample_index`、filename、directory、extension、hash、model score、probability、threshold、prediction、neighbor labels、similarity。

私有映射表仅允许用于：

- 查找源样本
- 对齐复核结果
- split/cache/manifest 审计
- 后续 Loop87/112/114 导入链路

它不能作为：

- 模型特征
- verdict 证据
- 阈值或融合输入
- feature mask 输入
- replacement sampling 信号

## 决策

Loop145 不替代 Loop136，也不进入 Test-10k/full-test。它是噪声治理入口。

只有当盲化复核表中的样本被独立内容或外部证据确认 `label_wrong`、`feature_broken` 或 `out_of_scope` 后，才允许进入 `exclude_and_replace`。替换必须是同原始 locked label 的 fresh redraw，不能用坏样本自填，也不能减少 20w 总数。

当前 strict best 仍是 Loop136：

- full-test F1：`0.9903723842`
- errors：`1544 / 160000`
- FP/FN：`958 / 586`

## Verification

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_build_loop145_loop136_blinded_noise_focus.py -q
```

结果：`2 passed`。

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop145_loop136_blinded_noise_focus.py `
  --output-json reports\phase3_loop145\pre_run_guard_loop145_blinded_focus.json `
  --allow-risk reader_materialization
```

结果：`guard_ready=true`。
