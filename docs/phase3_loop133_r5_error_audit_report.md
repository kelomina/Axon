# Phase 3 Loop133 R5 错误与噪声审计报告

更新时间：2026-07-05

## 审计边界

本轮只做 post-hoc 错误归因，不把 full-test 结果用于调参、选阈值或训练。`source_path`、`source_sha256`、`sample_index`、`cache_path` 只保留给人工复核和 sidecar 对齐，不作为模型证据。下一轮任何候选仍必须先在 Train/Val 上成立，再进入 Test-10k，最后才允许 full-test。

## 产物

- 汇总 JSON：`reports/phase3_loop133/loop133_r5_error_audit_summary.json`
- 全量错误队列：`reports/phase3_loop133/loop133_r5_error_review_queue.csv`
- R5 flip 审计：`reports/phase3_loop133/loop133_r5_flips_audit.csv`
- 资源守卫：`reports/phase3_loop133/pre_run_guard_r5_error_audit.json`
- 单测：`tests/test_analyze_loop133_r5_error_audit.py`

## 关键结论

Loop130 R5 在 160000 full-test 上仍是当前 strict best：

| 指标 | 数值 |
|---|---:|
| F1 | 0.9902567651 |
| Errors | 1563 |
| FP | 991 |
| FN | 572 |

错误结构显示，当前瓶颈不是单纯阈值问题：

| 审计项 | 数量 |
|---|---:|
| 高置信 FP (`prob >= 0.90`) | 472 |
| 高置信 FN (`prob < 0.10`) | 291 |
| 近阈值错误 (`0.45 <= prob <= 0.55`) | 75 |
| R5 总 flip | 210 |
| R5 修复 FP | 142 |
| R5 造成 FN | 68 |
| R5 相对 R4 额外修复 FP | 38 |
| R5 相对 R4 额外造成 FN | 30 |

这说明 R5 的收益主要来自更激进地压 FP，但付出了明显漏报代价。更重要的是，高置信 FP/FN 合计 763 个，远大于 99.9% F1 所能承受的错误预算。若其中相当一部分来自标签噪声、静态特征不可分或近重复冲突，那么仅靠当前静态特征和规则微调很难达成 99.9%。

## 特征侧观察

R5 flip 相关样本有共同结构：`guard_repaired_fp` 和 `guard_harmful_fn` 的 resource、overlay、vendor-string 均偏高，说明 R5 捕捉到的是“带正规资源/厂商痕迹的软件形态”。这个形态确实能修复一批 FP，但也覆盖了部分恶意样本。

| 切片 | 数量 | `string_benign_vendor_count_log` 均值 | `content_overlay_log_size` 均值 |
|---|---:|---:|---:|
| 全部 FP | 991 | 2.2180 | 3.9053 |
| 全部 FN | 572 | 2.2567 | 6.4625 |
| R5 修复 FP | 142 | 3.4761 | 8.5086 |
| R5 造成 FN | 68 | 3.6755 | 8.7632 |
| R5 额外修复 FP | 38 | 4.6036 | 8.8368 |
| R5 额外造成 FN | 30 | 4.4515 | 7.5177 |

这个结果不支持继续手写单条 vendor/resource 阈值规则，因为好样本和坏样本在这些特征上高度重叠。下一轮更应该做 Train/Val OOF selector 或数值邻域审计，让模型学习“什么时候 vendor/resource 是 benign 证据，什么时候不是”。

## 下一轮建议

1. 先在 Train/Val 构建 R5 flip selector。
   输入只能使用概率、模型分歧、content PE v1/v2、content string 等数值特征；目标是预测某个 `1 -> 0` flip 是否会造成 FN。若 Val 不能稳定减少 harmful flip 且不牺牲太多 repaired FP，则不进入 Test-10k。

2. 做 Train/Val 数值 kNN 噪声审计。
   对 high-confidence FP/FN 找最近邻标签分布，识别“邻域标签冲突”和“静态不可分”样本簇。full-test 的 review queue 只用于确定审计优先级，不用于设规则阈值。

3. 暂停继续小支持 FN rescue 单规则。
   Loop132 和 overlay/last-section/image-base probe 已证明 Val 上好看的单规则容易外推失败。FN recovery 应转向 OOF selector、分层校准或额外行为特征，而不是继续加窄阈值。

4. 对 99.9% 目标给出阶段性风险标记。
   当前 full-test 高置信错误达到 763 个；若 Train/Val 噪声审计确认大量标签冲突，应把短期工程目标调整为先突破 99.2%/99.5%，同时申请更强数据证据，例如动态行为特征、人工复核标签或更干净的重抽样数据。
