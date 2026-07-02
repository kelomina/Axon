# Phase 3 Loop58: Loop57 Full-Test Exchange Audit

日期：2026-07-03

## 目标

Loop58 是 Loop57 之后的只读 full-test 错误交换审计，不训练、不调阈值、不产生新候选模型。目标是解释 Loop57 的 `-81` full-test error 改进来自哪里，以及新增 FP 是否有可被下一轮 Val-only guard 学到的内容结构。

本轮使用 full-test 只做归因，不从 full-test 选择阈值、规则或模型参数。下一轮若要过滤新增 FP，必须重新回到 Train/Val 漏斗验证。

## 身份字段规则

`sample_index`、`source_sha256`、`source_path` 只用于对齐 Loop28/Loop57 预测表和定位 overlay sidecar cache。它们不能作为模型、阈值、gate、GA、噪声判定、自动改标或重抽依据。

## 实现

新增：

- `scripts/analyze_loop57_full_exchange.py`
- `tests/test_analyze_loop57_full_exchange.py`

输入：

- Loop28 full-test predictions: `reports/random_20w_split/stage2_loop28_content_pe_frozen_full_test_predictions.csv`
- Loop57 full-test predictions: `reports/random_20w_split/loop57_fn_overlay_gate_frozen_full_test_predictions.csv`
- Overlay cache: `reports/random_20w_split/loop57_overlay_boundary_cache_full_test`

输出：

- `reports/random_20w_split/loop58_loop57_full_exchange_audit.json`
- `reports/random_20w_split/loop58_loop57_full_exchange_details.csv`

## 结果

完整 full-test 行数为 `160000`。

| 分组 | 含义 | 行数 |
| --- | --- | ---: |
| `both_correct` | Loop28 和 Loop57 都正确 | `157943` |
| `loop28_only_error` | Loop28 错、Loop57 对 | `189` |
| `loop57_only_error` | Loop28 对、Loop57 错 | `108` |
| `both_error` | 两者都错 | `1760` |

迁移全部来自 FN gate 的 `0 -> 1` 覆盖：

| 迁移 | 含义 | 行数 |
| --- | --- | ---: |
| `0->1|label=1` | 修复 FN | `189` |
| `0->1|label=0` | 新增 FP | `108` |
| `0->0|label=1` | 仍漏报 | `673` |
| `1->1|label=0` | 仍误报 | `1087` |

`override_counts_by_group` 显示：Loop57 修复的 `189` 行和新增 FP 的 `108` 行全部来自 gate 覆盖；其它行没有隐藏变化。

## 内容归因

修复 FN 与新增 FP 的 overlay boundary 均值差异：

| Feature | 修复 FN 均值 | 新增 FP 均值 | 差值 |
| --- | ---: | ---: | ---: |
| `overlay_boundary_payload_log_size` | `5.2375` | `6.0661` | `-0.8286` |
| `overlay_boundary_payload_after_cert_log_size` | `0.7082` | `0.0000` | `+0.7082` |
| `overlay_boundary_security_log_size` | `3.5709` | `2.9001` | `+0.6708` |
| `overlay_boundary_gap_last_section_to_security_log` | `1.2844` | `1.6482` | `-0.3638` |
| `overlay_boundary_gap_last_section_to_overlay_log` | `0.7275` | `0.3698` | `+0.3577` |
| `overlay_boundary_overlay_log_size` | `7.2929` | `7.5561` | `-0.2633` |
| `overlay_boundary_payload_high_entropy` | `0.2698` | `0.4722` | `-0.2024` |
| `overlay_boundary_payload_entropy` | `0.3861` | `0.5159` | `-0.1298` |
| `overlay_boundary_payload_after_security` | `0.1164` | `0.0000` | `+0.1164` |

初步解释：

- Loop57 的新增 FP 更像“更大、更高熵的 overlay/payload”，但缺少 `payload_after_cert` 或 `payload_after_security` 证据。
- Loop57 修复 FN 的样本更常出现 security/cert 后 payload 或更明显的 security boundary 结构。
- gate score 本身区分度不够：修复 FN 的 gate mean 为 `0.9467`，新增 FP 为 `0.9348`，两者高度重叠，不能只靠提高 gate 阈值解决。

## 决策

Loop58 不产生新候选，不改变 Loop57 的当前 best full-test reference。

下一轮可以尝试 Val-only FP guard：在 Loop57 的 `0 -> 1` 覆盖候选中，加入内容侧“高熵大 payload 且无 after-cert/after-security 证据”的惩罚或二级 gate。该 guard 必须只用 Train/Val 选择参数；full-test 审计只能提供方向，不能提供最终规则阈值。

## 验证

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\analyze_loop57_full_exchange.py
.\vnev\Scripts\python.exe -m pytest tests\test_analyze_loop57_full_exchange.py tests\test_analyze_loop55_overlay_exchange.py tests\test_identity_feature_guard.py -q
```

Result: `5 passed`.

