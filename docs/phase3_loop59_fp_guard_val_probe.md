# Phase 3 Loop59: FP Guard Val Probe

日期：2026-07-03

## 目标

Loop59 针对 Loop58 的 full-test 归因做一次 Val-only 快速探针：尝试在 Loop57 的 `0 -> 1` 覆盖之后增加二级 FP guard，过滤“高熵大 payload 且缺少 after-cert/after-security 证据”的可疑误伤。

本轮不训练新模型、不触碰 Test-10k、不触碰 full-test。full-test 归因只提供方向，所有候选 guard 都只在 Val 上验证。

## 输入

- Val predictions:
  `reports/random_20w_split/loop57_fn_overlay_gate_valonly/loop57_fn_overlay_gate_val_predictions.csv`
- Overlay cache:
  `reports/random_20w_split/loop55_overlay_boundary_cache_train_val`

身份字段只用于读取 cache；guard 条件只使用 content-derived overlay boundary features。

## Val Reference

| Candidate | Errors | FP/FN | F1 | Overrides |
| --- | ---: | ---: | ---: | ---: |
| Loop28 locked base | `162` | `87 / 75` | `0.9919048571` | `0` |
| Loop57 FN gate | `147` | `92 / 55` | `0.9926635724` | `25` |

Loop57 Val 覆盖中，修复恶意 `20` 行，新增 FP `5` 行。

## 探针结果

测试的 guard 包括：

- 单特征上下界：payload size、overlay size、payload entropy、security size、last-section gap、after-cert size、after-security flag；
- 组合规则：大 payload + 高 entropy + no after-cert + no after-security；
- 保留规则：after-cert / after-security / security size 证据满足其一。

最佳 Val-only 规则：

| Guard | Errors | FP/FN | F1 | Rejected overrides |
| --- | ---: | ---: | ---: | ---: |
| `overlay_boundary_gap_last_section_to_overlay_log <= 0` | `146` | `91 / 55` | `0.9927131164` | `1` |

它只比 Loop57 少 `1` 个 Val error，仅拒绝 `1` 个覆盖，且没有形成足够宽的 margin。其它规则要么不改变 Loop57，要么损伤 FN 修复。

## 决策

Reject for Test-10k.

原因：

- 改进太薄：`147 -> 146` 只少 `1` 个 Val error；
- 没达到浅 guard/gate 候选进入 Test-10k 的 `<=152` 之外的实质 margin；
- Loop37 已证明这种极薄 Val/Test-10k 改善可能在 full-test 反转；
- 当前 guard 只是手工规则探针，还没有形成稳健的 OOF 二级模型。

下一步不应继续手动拧 overlay 阈值。更有价值的方向是训练一个专门区分 Loop57 修复 FN vs 新增 FP 的 OOF 二级模型，并加入更多非 overlay 的内容信号，例如 import/resource/section 正常软件结构。

