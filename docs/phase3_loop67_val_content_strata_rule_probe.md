# Phase 3 Loop67: Val-Only Content Strata Rule Probe

日期：2026-07-03

## 目标

Loop67 接在 Loop66 后面，验证一个非常具体的问题：既然 Loop66 指出剩余 FN 更偏 signed/overlay/export/exception/basereloc/import-shape 复杂 PE，是否能用少量固定内容分层规则直接减少 Loop57 的 Val 错误？

本轮只做 Val-only fixed-rule probe：

- 不训练模型；
- 不做自由组合搜索；
- 不扫 Test-10k 或 full-test；
- 不改标签、不改 split、不改 cache；
- 不使用 filename/path/extension/directory/hash/sample id/split/row order 作为模型证据。

## 实现

新增：

- `scripts/probe_loop67_val_content_strata_rules.py`
- `tests/test_probe_loop67_val_content_strata_rules.py`

规则分两类：

1. `repair`: 只允许 `0 -> 1`，用于 signed/overlay-complex 可疑 FN 修复。
2. `rollback`: 只允许 `1 -> 0`，用于低置信 unsigned/import-heavy/payload-like 可疑 FP 回退。

进入 Test-10k 的最低 Val 门槛设置为：相比 Loop57 至少减少 `10` 个 Val errors。这个门槛是为了避免 Loop37/Loop61 这类 1-3 个样本级别的小改善再次在 Test/full 上反转。

真实命令：

```powershell
.\vnev\Scripts\python.exe scripts\probe_loop67_val_content_strata_rules.py `
  --loop57-val-predictions reports\random_20w_split\loop57_fn_overlay_gate_valonly\loop57_fn_overlay_gate_val_predictions.csv `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --overlay-boundary-cache-dir reports\random_20w_split\loop55_overlay_boundary_cache_train_val `
  --output-json reports\random_20w_split\loop67_val_content_strata_rule_probe.json `
  --output-csv reports\random_20w_split\loop67_val_content_strata_rule_candidates.csv `
  --min-error-reduction-for-test10k 10
```

## 结果

Loop57 baseline Val:

| F1 | Errors | FP/FN |
| ---: | ---: | ---: |
| `0.9926635724` | `147` | `92 / 55` |

最佳规则：

| Rule | Action rows | Beneficial / Harmful | Errors | FP/FN | F1 | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `repair_signed_overlay_complex_c250_g80` | `8` | `5 / 3` | `145` | `95 / 50` | `0.9927662759` | reject |
| `repair_signed_overlay_complex_c515_g80` | `6` | `4 / 2` | `145` | `94 / 51` | `0.9927655541` | reject |

负面规则：

| Rule | Action rows | Beneficial / Harmful | Errors | FP/FN |
| --- | ---: | ---: | ---: | ---: |
| `rollback_lowconf_unsigned_importheavy` | `36` | `10 / 26` | `163` | `82 / 81` |
| `rollback_lowconf_unsigned_payload` | `31` | `5 / 26` | `168` | `87 / 81` |
| `rollback_lowconf_no_security` | `91` | `28 / 63` | `182` | `64 / 118` |

## 决策

拒绝 Test-10k。

原因很直接：最佳规则只把 Val errors 从 `147` 降到 `145`，净改善 `2` 个样本，远小于本轮要求的 `10` 个错误 margin。它本质上仍是 FN/FP 交换：FN 从 `55` 降到 `50`，但 FP 从 `92` 升到 `95`。

更重要的是，所有 rollback FP 规则都显著变差，说明“低置信 unsigned/import-heavy/payload-like 就回退为白”会误伤大量真实恶意样本。这个方向不应继续用手工规则推进。

## 结论

Loop67 把 Loop66 的直觉落成了负面证据：signed/overlay/content strata 确实能找到少量 FN，但手工规则无法形成足够宽的 Val margin。下一步不应继续拧规则阈值，而应转向：

1. 对 `both_error=142` 的持久错误做更细的内容/噪声分层；
2. 若训练候选，必须使用严格 OOF 或明确的 train-only 学习协议；
3. 继续人工/外部证据复核高置信冲突，因为模型侧薄规则已经接近上限。

## Artifacts

- Summary: `reports/random_20w_split/loop67_val_content_strata_rule_probe.json`
- Candidate ranking: `reports/random_20w_split/loop67_val_content_strata_rule_candidates.csv`

Generated reports are not committed.

## 验证

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_probe_loop67_val_content_strata_rules.py tests\test_analyze_loop66_val_blindspots.py tests\test_identity_feature_guard.py -q
.\vnev\Scripts\python.exe -m py_compile scripts\probe_loop67_val_content_strata_rules.py scripts\analyze_loop66_val_blindspots.py scripts\identity_feature_guard.py
```

Latest local result: `5 passed`.
