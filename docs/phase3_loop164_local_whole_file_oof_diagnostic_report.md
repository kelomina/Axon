# Loop164 本地 Train-only whole-file OOF 诊断

## 结论

Loop164 的本地五折 whole-file OOF 已完整完成，但 standalone 质量不足，当前 lineage 不应继续增加 seed、epoch、阈值搜索或 heldout 访问。固定阈值 `0.5` 下，supported subset F1 为 `0.9620420177`，共 `748` 个错误；它离 `0.9997` 所允许的最多 `5` 个 supported 错误相差至少 `743` 个。

这不是对 whole-file 信息互补性的最终否定。当前没有 decision-aligned Loop151 Train OOF，因此无法回答它能否专门修复冠军错误。保留本次 OOF 的 score、uncertainty 和 missingness，只允许未来在 Loop151 OOF 重建后执行一次预注册的 cross-fitted complementarity gate；若修复精度不足，关闭 Loop164。

## 边界与授权

- 数据仅来自 canonical Train 前 `20,000` 行；没有读取 Val、Test-10k、legacy full-test、sentinel 或 sealed window。
- 五折每折 `4,000` 行、每类 `2,000`；已发现 content component 不跨折，但 3 个过大 LSH bucket 未展开，因此不能称 family/time/source isolation。
- 本地授权明确 `public_key_required=false`。它不是 A2 training authority，不产生晋级或认证资格。
- 模型为 one-seed、one-epoch、FP32、固定阈值 `0.5`；没有 threshold sweep、checkpoint 或模型状态输出。

## 结果

| 指标 | 结果 |
|---|---:|
| denominator | 20,000 |
| supported / missing | 19,540 / 460 |
| coverage | 0.977 |
| supported F1 | 0.9620420177 |
| supported errors | 748 |
| FP / FN | 488 / 260 |
| conservative all-missing-wrong F1 | 0.9400971933 |
| conservative errors | 1,208 |
| posthoc descriptive ROC AUC | 0.9894106709 |
| fold F1 mean / std | 0.9621561595 / 0.0075271694 |
| fold F1 min / max | 0.9496402878 / 0.9709694142 |

固定阈值五折 F1 分别为 `0.958479`、`0.967874`、`0.963818`、`0.970969`、`0.949640`。`448/748` 个错误是高置信错误：`311` 个 benign score `>=0.9` 的 FP，`137` 个 malicious score `<=0.1` 的 FN。这不是只靠轻微阈值校准即可消除的误差形态。

singleton component 的错误率为 `4.4563%`，non-singleton 为 `0.8751%`；`1-8 MiB` 文件错误率最高，为 `6.1115%`。这些是 posthoc 描述，只用于解释失败，不用于选择新规则或阈值。

## 工程完整性

- completed fit / holdout source calls：`78,160 / 19,540`。
- verified SHA passes：`195,400`，与两遍扫描精确一致。
- backward microbatches / optimizer steps：`78,160 / 9,772`，与冻结期望一致。
- raw bytes read：`116,780,343,760`；elapsed `2,662.703s`。
- peak RSS / CUDA allocated / reserved：`1,991,557,120 / 154,283,008 / 174,063,616` bytes。
- OOM / nonfinite：`0 / 0`；run lease 已释放。

## 证据

- OOF report：`reports/roadmap_9997/loop164/local_whole_file_oof_report.json`，SHA-256 `da55531d39b628a2a02ec008451b7ad0455f6876cabd91dcb8c56f7e18c3e07f`。
- predictions：`reports/roadmap_9997/loop164/local_whole_file_oof_predictions.jsonl`，SHA-256 `4f706788d812987714ebd9f717b77f75b10997309dbe7991c083b9928ad3d4df`。
- posthoc analysis：`reports/roadmap_9997/loop164/local_whole_file_oof_analysis.json`，SHA-256 `a6c0098231e9e358278061bb682410f6065cda9cb42aef80c951f100238e3c10`。
- local authorization：`reports/roadmap_9997/loop164/local_whole_file_oof_authorization.json`，SHA-256 `08601a2d7802cb5c9b774c4a322a66091be33d1329b19034706c0c2a1f7d4f2a`。
- resource guard：`reports/roadmap_9997/loop164/local_whole_file_oof_resource_guard.json`，SHA-256 `240a77fb3faa10c77753ce08991ac5264156f7c1eca1beaf79eff054b13dfe1f`。

相关 `py_compile`、Ruff 和 51 项 OOF/model/loader/resource-guard/posthoc/governance 定向测试通过。ProjectAnalysis doctor 仍被其自身 `C:\MCP\ProjectAnalysis\dist\index.js:14704` 的 JavaScript 语法错误阻断，本次结论不依赖该服务。

## 决策

Loop151 继续是唯一 research champion，legacy development full-test F1 仍为 `0.9908541911`；`>=0.9997` 目标未达成。Loop164 从“下一 standalone 主线”降级为“仅待互补性审计的冻结候选”。下一项高价值工作是重建 decision-aligned Loop151 Train OOF，而不是继续训练当前 whole-file 单专家。
