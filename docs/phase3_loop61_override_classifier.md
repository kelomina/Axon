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

## Follow-Up Audits

Loop61 exchange audit:

- New script: `scripts/analyze_loop61_exchange.py`
- Val exchange vs Loop57: Loop61 repaired `13` Loop57 errors and introduced `6` new errors.
- Test-10k exchange vs Loop57: Loop61 repaired `6` Loop57 errors and introduced `6` new errors.

This explains the gate decision: Val had a real but small net advantage, while Test-10k showed no net error reduction.

Loop62 tried the obvious extension, adding full anonymous content matrix features to the override classifier. It was rejected on Val:

| Candidate | Classifier | F1 | Errors | FP/FN | Overrides |
| --- | --- | ---: | ---: | ---: | ---: |
| `extra_trees_300_leaf1` | `override_logreg_balanced_c0.10` | `0.9926096075` | `148` | `87 / 61` | `14` |

Loop62 reduced FP but blocked too many true malicious repairs, so it did not reach Test-10k. This confirms that simply feeding high-dimensional content features into the sparse override-only classifier is not a stable path.

Loop63 then returned to noise/data triage:

- New script: `scripts/build_loop63_persistent_error_review_queue.py`
- Output queue: `reports/random_20w_split/loop63_persistent_error_review_queue.csv`
- Summary: `reports/random_20w_split/loop63_persistent_error_review_queue_summary.json`

Loop63 is read-only full-test triage, not model selection. It found all `1868` current-best Loop57 full-test errors, with `643` rows intersecting the Loop39 high-confidence conflict queue. These rows are the strongest current evidence that the remaining gap is not only a gate-tuning problem.

The `643` A-lane rows were then passed through the Loop50 content/cache health
audit path:

- output: `reports/random_20w_split/loop63_A_persistent_conflict_content_audit_summary.json`
- rows: `643`
- FP/FN: `416 / 227`
- objective cache/source/strict-PE issue rows: `0`
- duplicate SHA group rows: `5`

So the current evidence does not justify automatic replacement. The A-lane rows
remain manual or external-evidence adjudication targets, or model blindspots.
If any are later confirmed `label_wrong`, `feature_broken`, or `out_of_scope`,
the replacement rule remains fresh same-label redraw while preserving the exact
`200000` rows.

Loop64 adds a stricter duplicate-content audit using `manifest.source_sha256`,
because the split CSV does not always expose the true content SHA. Result:

- split rows matched to manifest: `200000/200000`
- manifest SHA duplicate groups: `6`
- duplicate detail rows: `12`
- cross-label duplicate groups: `0`
- cross-split duplicate groups: `0`
- overlap with Loop63 focus queue: `2` groups, `4` rows

This is useful for content-group review but still does not justify automatic
replacement: all duplicate groups are same-label `test` rows, with no
train/val/test leakage and no cross-label contradiction.

Loop65 converts the Loop63 A-lane into a compact manual/external-evidence
review batch:

- New script: `scripts/build_loop65_review_batch.py`
- Output queue: `reports/random_20w_split/loop65_A_lane_review_batch.csv`
- Summary: `reports/random_20w_split/loop65_A_lane_review_batch_summary.json`
- Selected rows: `62`
- Category counts: severe persistent FN `20`, severe persistent FP `20`,
  duplicate content group `2`, corrected-by-other-model `20`
- Error type counts: FN/FP `37 / 25`
- Manual verdict/action fields: blank
- Requested duplicate groups: `2`, matching `4` queue rows; selected duplicate
  group rows: `2`

This batch is not model evidence. `source_path`, `source_sha256`, `cache_path`,
`sample_index`, and `split` are included only so a human or external system can
open the right object and write an auditable verdict.

## Artifacts

- Val report:
  `reports/random_20w_split/loop61_override_classifier_valonly/loop61_override_classifier_report.json`
- Frozen Test-10k report:
  `reports/random_20w_split/loop61_override_classifier_frozen_test10k_eval.json`
- Loop61 Val exchange audit:
  `reports/random_20w_split/loop61_exchange_val_audit.json`
- Loop61 Test-10k exchange audit:
  `reports/random_20w_split/loop61_exchange_test10k_audit.json`
- Loop63 persistent error queue summary:
  `reports/random_20w_split/loop63_persistent_error_review_queue_summary.json`
- Loop63 A-lane content/cache health audit:
  `reports/random_20w_split/loop63_A_persistent_conflict_content_audit_summary.json`
- Loop64 manifest SHA duplicate audit:
  `reports/random_20w_split/loop64_manifest_sha_duplicate_audit.json`
- Loop65 compact review batch summary:
  `reports/random_20w_split/loop65_A_lane_review_batch_summary.json`

Large generated artifacts are intentionally not committed:

- `loop61_override_classifier_selected_model.pkl`
- `loop61_override_classifier_*_predictions.csv`
- `loop61_exchange_*_details.csv`
- `loop63_persistent_error_review_queue.csv`
- `loop65_A_lane_review_batch.csv`

## 验证

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_loop61_override_classifier.py tests\test_loop57_fn_overlay_gate.py tests\test_loop42_oof_residual_gate.py tests\test_loop55_overlay_boundary.py tests\test_identity_feature_guard.py -q
.\vnev\Scripts\python.exe -m py_compile scripts\train_loop61_override_classifier.py scripts\evaluate_loop57_fn_overlay_gate.py
.\vnev\Scripts\python.exe -m pytest tests\test_analyze_loop61_exchange.py tests\test_build_loop63_persistent_error_review_queue.py -q
.\vnev\Scripts\python.exe -m py_compile scripts\analyze_loop61_exchange.py scripts\build_loop63_persistent_error_review_queue.py
.\vnev\Scripts\python.exe -m pytest tests\test_audit_split_manifest_sha_duplicates.py -q
.\vnev\Scripts\python.exe -m py_compile scripts\audit_split_manifest_sha_duplicates.py
.\vnev\Scripts\python.exe -m pytest tests\test_build_loop65_review_batch.py -q
.\vnev\Scripts\python.exe -m py_compile scripts\build_loop65_review_batch.py
```

Latest local results: `25 passed` for Loop61/57/42/55 identity coverage, plus `2 passed` for Loop61 exchange and Loop63 queue coverage. Loop65 local coverage is tracked by `tests/test_build_loop65_review_batch.py`.
