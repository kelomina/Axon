# Phase 3 Loop56: Loop55 Overlay Error Exchange Audit

日期：2026-07-02

## 目标

Loop56 是 Loop55 的只读错误交换审计，不训练模型、不调阈值、不触碰 Test-10k 或 16 万 full-test。目标是解释：为什么 Loop55 的 overlay/security boundary 特征能修复一部分 Loop28 漏报，但总错误数反而变多。

## 身份字段规则

本轮读取了 `sample_index`、`source_sha256`、`source_path` 等字段，但它们只用于对齐 Loop28/Loop55 的 Val 预测行，以及定位 overlay sidecar cache。它们不是模型特征，也不是阈值、融合、GA 掩码或上线推理依据。

这条规则是硬约束：filename、path、extension、directory、`source_sha256`、`cache_path`、`sample_index`、`split` 和行顺序只能用于加载、对齐、覆盖审计、去重、人工复核或一次性人工标签 manifest。实战命名和训练集命名不在同一分布，攻击者也能低成本改名，所以模型证据必须来自文件内容、PE 结构、字节序列和统计特征。

Loop56 输出的 CSV/JSON 中即使包含 `sample_index`、`source_sha256`、`source_path`，也只允许用于复现实验、人工定位和 cache lookup；任何后续 residual gate、阈值、融合、GA 掩码、噪声处理或自动重抽逻辑，都不得读取这些列作为输入、排序键或分桶依据。

若基于 Loop56 继续设计 FN-specific residual gate，建模矩阵必须在 fit/select 前丢弃所有对齐键、cache key、split 和 CSV 行序字段，并再次通过 `identity_feature_guard` 验证 feature names。

## 实现

新增：

- `scripts/analyze_loop55_overlay_exchange.py`
- `tests/test_analyze_loop55_overlay_exchange.py`

脚本输入两份冻结 Val 预测表和 Loop55 overlay sidecar cache：

- Loop28 Val predictions: `reports/random_20w_split/stage2_loop28_content_pe_valonly/stage2_val_predictions.csv`
- Loop55 Val predictions: `reports/random_20w_split/loop55_overlay_boundary_valonly/loop55_overlay_boundary_val_predictions.csv`
- Loop55 overlay cache: `reports/random_20w_split/loop55_overlay_boundary_cache_train_val`

输出：

- `reports/random_20w_split/loop56_loop55_overlay_exchange_audit.json`
- `reports/random_20w_split/loop56_loop55_overlay_exchange_details.csv`

## 结果

完整 Val 行数为 `20000`，Loop28 与 Loop55 的错误交换如下：

| 分组 | 含义 | 行数 |
| --- | --- | ---: |
| `both_correct` | 两者都预测正确 | `19795` |
| `loop28_only_error` | Loop28 错、Loop55 对 | `31` |
| `loop55_only_error` | Loop28 对、Loop55 错 | `43` |
| `both_error` | 两者都预测错误 | `131` |

因此 Loop55 相比 Loop28 净增加 `12` 个 Val 错误，这与 Loop55 Val `174` errors 对 Loop28 `162` errors 完全一致。

按预测迁移拆开看：

| 迁移 | 含义 | 行数 |
| --- | --- | ---: |
| `0->1|label=1` | 修复 FN | `20` |
| `1->0|label=0` | 修复 FP | `11` |
| `0->1|label=0` | 新增 FP | `35` |
| `1->0|label=1` | 新增 FN | `8` |

Loop55 的真实收益是把一部分恶意样本从白判回黑，修复 `20` 个 FN；但代价是新增 `35` 个 FP。它不是纯增强，而是更偏向“更敏感”的边界信号。

## 特征归因

`loop28_only_error` 代表 Loop55 修复的样本，`loop55_only_error` 代表 Loop55 伤害的样本。修复组相对伤害组更明显的 overlay boundary 差异包括：

| 特征 | 修复组均值 | 伤害组均值 | 差值 |
| --- | ---: | ---: | ---: |
| `overlay_boundary_payload_log_size` | `2.3910` | `0.7152` | `+1.6758` |
| `overlay_boundary_overlay_log_size` | `5.1128` | `3.4791` | `+1.6336` |
| `overlay_boundary_gap_last_section_to_security_log` | `1.7240` | `0.3343` | `+1.3897` |
| `overlay_boundary_security_log_size` | `3.8925` | `3.2188` | `+0.6737` |
| `overlay_boundary_payload_entropy_minus_last_section` | `-0.2869` | `-0.5671` | `+0.2803` |

解释：Loop55 确实抓到了“更大的 overlay/payload、安全目录边界更复杂”的恶意残差信号，这能修复一部分 Loop28 FN。但同类结构在部分正常文件中也存在，尤其是签名、安装器、打包器或复杂正常 PE，因此直接把这组特征拼进 Stage-2 会制造更多 FP。

## 决策

拒绝进入 Test-10k。

Loop56 没有产生新候选模型，只给 Loop55 的失败原因做归因。结论是 overlay/security boundary 信号真实存在，但 standalone 或直接拼接不安全。后续如果复用，只能作为：

- 极保守的 FN-specific residual gate 辅助特征；
- 残差分层分析特征；
- 人工复核队列的解释信号。

不能把它作为默认模型特征直接上线，也不能用路径、命名或扩展名去补偿它的 FP 风险。

## 验证

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\analyze_loop55_overlay_exchange.py
.\vnev\Scripts\python.exe -m pytest tests\test_analyze_loop55_overlay_exchange.py tests\test_loop55_overlay_boundary.py tests\test_identity_feature_guard.py -q
```

结果：`9 passed`。
