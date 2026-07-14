# Phase 3 Loop57: FN-Specific Overlay Gate

日期：2026-07-03

## 目标

Loop57 基于 Loop56 的结论，把 overlay/security boundary 信号从“直接拼接”改成“极保守的漏报修复 gate”。它只允许一种覆盖动作：当 Loop28 locked base 判白时，候选视角和 gate 都足够确信，才允许把结果从 `0 -> 1`。它没有任何路径把 base 判黑改成白。

本轮遵守完整漏斗：Train/Val 选模型和 gate，Val 达到进入门槛后只做一次冻结 Test-10k，Test-10k 通过后再做一次 16 万 full-test。所有阈值和模型都来自 Val，不在 Test 上扫阈值。

## 身份字段规则

filename、path、extension、directory、`source_sha256`、`cache_path`、`sample_index`、`split` 和行顺序只用于加载、cache lookup、预测表对齐和审计。建模矩阵在 fit/select 前丢弃所有对齐键、cache key、split 和 CSV 行序字段，并通过 `identity_feature_guard` 验证 feature names。

Loop57 的建模证据只来自：

- Loop28 frozen base probability；
- overlay-aware candidate probability；
- 两者的 score/logit 差值；
- content-derived overlay/security boundary features。

## 实现

新增：

- `scripts/train_loop57_fn_overlay_gate.py`
- `scripts/evaluate_loop57_fn_overlay_gate.py`
- `tests/test_loop57_fn_overlay_gate.py`

训练协议：

- base/candidate train scores 使用 5-fold OOF；
- gate 只在 train OOF 上训练；
- Val 选择 candidate、gate model、candidate threshold 和 gate threshold；
- frozen evaluator 只加载 payload，不重新 fit，不扫阈值。

## Val

Loop28 locked Val reference:

| F1 | Errors | FP/FN |
| ---: | ---: | ---: |
| `0.9919048571` | `162` | `87 / 75` |

Loop57 selected by Val:

| Candidate | Gate | Candidate threshold | Gate threshold | F1 | Errors | FP/FN | Overrides |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `extra_trees_300_leaf1` | `gate_logreg_balanced_c0.25` | `0.515` | `0.88` | `0.9926635724` | `147` | `92 / 55` | `25` |

Val delta vs Loop28: `-15` errors, FP `+5`, FN `-20`。

Val decision: pass. The shallow/gate candidate gate was `<=152` Val errors; Loop57 reached `147`.

## Test-10k

Frozen Test-10k used:

- base predictions: `reports/random_20w_split/stage2_loop28_content_pe_frozen_test10k_predictions.csv`
- test input: `reports/random_20w_split/loop24_dedup_corrected_test10k_base_predictions.csv`
- SHA alignment: `10000/10000`

| Candidate | F1 | Errors | FP/FN | Overrides |
| --- | ---: | ---: | ---: | ---: |
| Loop28 locked base | `0.9888677164` | `111` | `61 / 50` | `0` |
| Loop57 frozen gate | `0.9897877453` | `102` | `65 / 37` | `17` |

Test-10k delta vs Loop28: `-9` errors, FP `+4`, FN `-13`。

Decision: pass. Loop57 improved the locked Test-10k slice and advanced to full-test.

## Full-Test

Frozen full-test used:

- base predictions: `reports/random_20w_split/stage2_loop28_content_pe_frozen_full_test_predictions.csv`
- test input: `reports/random_20w_split/loop24_dedup_corrected_full_test_base_predictions.csv`
- records: `160000/160000`
- SHA alignment: `160000/160000`

Full-test overlay sidecar cache:

| Field | Value |
| --- | ---: |
| unique rows | `159994` |
| processed | `159994` |
| zero features | `203` |
| feature dim | `32` |

`unique_rows=159994` is lower than `160000` because duplicate source hashes share the same sidecar cache. The frozen evaluation still scored all `160000` rows.

| Candidate | F1 | Errors | FP/FN | Overrides |
| --- | ---: | ---: | ---: | ---: |
| Loop28 locked base | `0.9878358558` | `1949` | `1087 / 862` | `0` |
| Loop57 frozen gate | `0.9883629658` | `1868` | `1195 / 673` | `297` |

Full-test delta vs Loop28: `-81` errors, FP `+108`, FN `-189`。

## 决策

Loop57 becomes the current best full-test reference, but it is not the final target. It is a useful low-FN/security-biased improvement: it reduces FN substantially, while increasing FP.

It still remains far from `F1 >= 99.9%`. Current full-test errors are `1868`; the 99.9% target would require roughly hundred-level errors on this 160k balanced test set. The next loop should focus on reducing the new FP introduced by the FN gate while preserving most of the FN repair.

## Artifacts

- Val report:
  `reports/random_20w_split/loop57_fn_overlay_gate_valonly/loop57_fn_overlay_gate_report.json`
- Frozen Test-10k report:
  `reports/random_20w_split/loop57_fn_overlay_gate_frozen_test10k_eval.json`
- Frozen full-test report:
  `reports/random_20w_split/loop57_fn_overlay_gate_frozen_full_test_eval.json`
- Full-test overlay cache report:
  `reports/random_20w_split/loop57_overlay_boundary_cache_full_test_report.json`

Large generated artifacts are intentionally not committed:

- `loop57_fn_overlay_gate_selected_model.pkl`
- `loop57_fn_overlay_gate_*_predictions.csv`
- `loop57_overlay_boundary_cache_*`

## 验证

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\train_loop57_fn_overlay_gate.py scripts\evaluate_loop57_fn_overlay_gate.py
.\vnev\Scripts\python.exe -m pytest tests\test_loop57_fn_overlay_gate.py tests\test_loop42_oof_residual_gate.py tests\test_loop55_overlay_boundary.py tests\test_identity_feature_guard.py -q
```

Latest local result before full-test: `19 passed`.

