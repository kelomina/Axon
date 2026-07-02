# Phase 3 Loop61: Override-Only Classifier

日期：2026-07-03

## 目标

Loop61 针对 Loop57 的新增 FP 问题做更窄的二级判定：只在 locked base 判白、overlay-aware candidate 判黑的 possible override 行上训练一个 allow/block 分类器。它仍然只允许 `0 -> 1`，没有任何路径把 base 判黑改成白。

本轮遵守漏斗：Train OOF 训练 base/candidate/override classifier，Val 选择 candidate、classifier 和 allow threshold；Val 过门槛后只做一次冻结 Test-10k。Test-10k 未超过当前 best，因此不进入 16 万 full-test。

## 身份字段规则

filename、path、extension、directory、`source_sha256`、`cache_path`、`sample_index`、`split` 和行顺序只用于加载、cache lookup、预测表对齐和审计。建模矩阵不编码这些字段，并通过 `identity_feature_guard` 检查 feature names。

Loop61 的 override classifier 只看：

- locked base probability；
- overlay-aware candidate probability；
- 两者的 score/logit 差值；
- content-derived overlay/security boundary features。

## 实现

新增：

- `scripts/train_loop61_override_classifier.py`
- `tests/test_loop61_override_classifier.py`

训练协议：

- base/candidate train scores 使用 5-fold OOF；
- candidate threshold 只由 train OOF 选择；
- override classifier 只在 possible override 行训练，目标为该覆盖是否命中真实 label；
- Val 选择 classifier 和 allow threshold；
- frozen Test-10k 使用 Loop57 evaluator 读取兼容 payload，不重新 fit，不扫阈值。

## Val

Loop57 reference:

| F1 | Errors | FP/FN | Overrides |
| ---: | ---: | ---: | ---: |
| `0.9926635724` | `147` | `92 / 55` | `25` |

Loop61 selected by Val:

| Candidate | Classifier | Candidate threshold | Allow threshold | F1 | Errors | FP/FN | Overrides |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `extra_trees_300_leaf1` | `override_logreg_balanced_c1` | `0.46` | `0.74` | `0.9930139721` | `140` | `90 / 50` | `28` |

Val delta vs Loop57: `-7` errors, FP `-2`, FN `-5`。这通过了进入 Test-10k 的 Val gate。

Train possible override rows were sparse: `160` rows, with `54` beneficial FN repairs and `106` harmful new FP. Val possible override rows were `130`, with `42` beneficial and `88` harmful. This confirms the direction is real but high-variance.

## Test-10k

Frozen Test-10k used the same locked slice as Loop57:

- base predictions: `reports/random_20w_split/stage2_loop28_content_pe_frozen_test10k_predictions.csv`
- test input: `reports/random_20w_split/loop24_dedup_corrected_test10k_base_predictions.csv`
- SHA alignment: `10000/10000`

| Candidate | F1 | Errors | FP/FN | Overrides |
| --- | ---: | ---: | ---: | ---: |
| Loop28 locked base | `0.9888677164` | `111` | `61 / 50` | `0` |
| Loop57 frozen gate | `0.9897877453` | `102` | `65 / 37` | `17` |
| Loop61 override classifier | `0.9897816069` | `102` | `62 / 40` | `11` |

Loop61 vs Loop57 on Test-10k: same total errors, FP `-3`, FN `+3`。It improves the FP profile but gives back the same number of FN repairs, so it does not beat the current best Test-10k reference.

## 决策

Reject for full-test. The Val gain did not translate into fewer Test-10k errors, and running full-test after a Test-10k tie would weaken the funnel protocol. Loop57 remains the current best full-test reference.

The useful lesson is narrower: override-only classification can reduce some new FP, but the possible override training set is tiny and unstable. The next loop should either collect stronger content evidence for these possible override rows or move back to noise/source-label adjudication instead of continuing to tune the same sparse gate.

## Artifacts

- Val report:
  `reports/random_20w_split/loop61_override_classifier_valonly/loop61_override_classifier_report.json`
- Frozen Test-10k report:
  `reports/random_20w_split/loop61_override_classifier_frozen_test10k_eval.json`

Large generated artifacts are intentionally not committed:

- `loop61_override_classifier_selected_model.pkl`
- `loop61_override_classifier_*_predictions.csv`

## 验证

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_loop61_override_classifier.py tests\test_loop57_fn_overlay_gate.py tests\test_loop42_oof_residual_gate.py tests\test_loop55_overlay_boundary.py tests\test_identity_feature_guard.py -q
.\vnev\Scripts\python.exe -m py_compile scripts\train_loop61_override_classifier.py scripts\evaluate_loop57_fn_overlay_gate.py
```

Latest local result: `25 passed`.
