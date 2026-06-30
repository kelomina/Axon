# Stage2 kNN Label-Support Experiment

## Protocol

This experiment follows the locked evaluation funnel:

1. Build candidates from train cache and train predictions only.
2. Select model, noise mode, and threshold on the full 20k validation split.
3. Run one frozen Test-10k confirmation with the validation-selected threshold.
4. Because Test-10k improved over the previous Stage2 result, run one frozen 160k full-test evaluation.

No threshold sweep or candidate selection used Test-10k or full-test metrics.

## Candidate

- Branch: `exp/model-agent-knn-stage2`
- Base checkpoint: `models/random_20w_8192/best_model.pt`
- Train predictions: `reports/random_20w_split/random_20w_8192_replaced_train_predictions.csv`
- Val predictions: `reports/random_20w_split/random_20w_8192_replaced_val_predictions.csv`
- Feature cache manifest: `data/.cache/manifest_38672ba0.json`
- Selected frozen model: `reports/random_20w_split/stage2_knn_extended_valonly/stage2_selected_model.pkl`

The candidate appends train-only kNN label-support features to the existing Stage2 feature matrix. Train rows use 5-fold out-of-fold kNN features, so a train sample never uses its own label as neighbor evidence. Validation, Test-10k, and full-test rows use a frozen memory built from the full 20k train split.

kNN settings:

- Top-k values: `5, 10, 25, 50`
- OOF folds: `5`
- Base feature dimension: `1420`
- Final feature dimension: `1447`

## Validation Selection

Report: `reports/random_20w_split/stage2_knn_extended_valonly/stage2_cache_matrix_report.json`

Selected by Val:

- Model: `hgb_lr0.08_leaf31_l2_1e-3__noise_none`
- Threshold: `0.49`
- Val F1: `0.9833349965`
- Val AUC: `0.9984984350`
- Val errors: `334 / 20000`
- FP/FN: `188 / 146`

Previous best Stage2 Val F1 was `0.9818199930` with `365` errors, so the kNN candidate reduced Val errors by `31`.

## Frozen Test-10k Confirmation

Report: `reports/random_20w_split/stage2_knn_extended_frozen_test10k_eval.json`

- Threshold: `0.49`
- Test-10k F1: `0.9827066157`
- Test-10k AUC: `0.9984840079`
- Test-10k errors: `172 / 10000`
- FP/FN: `82 / 90`

Previous frozen Stage2 Test-10k had `178 / 10000` errors, so this candidate passed the confirmation gate, but only by a small margin.

## Frozen 160k Full-Test Result

Base full-test predictions were exported first:

- Report: `reports/random_20w_split/random_20w_8192_replaced_test_eval.json`
- Predictions: `reports/random_20w_split/random_20w_8192_replaced_test_predictions.csv`
- Rows: `160000 / 160000`
- Missing cache: `0`
- Base model full-test F1 at threshold `0.5`: `0.9283588516`
- Base model errors: `11314 / 160000`

Frozen kNN Stage2 full-test:

- Report: `reports/random_20w_split/stage2_knn_extended_frozen_full_test_eval.json`
- Predictions: `reports/random_20w_split/stage2_knn_extended_frozen_full_test_predictions.csv`
- Rows: `160000 / 160000`
- Missing cache: `0`
- Threshold: `0.49`
- Full-test F1: `0.9828037453`
- Full-test AUC: `0.9983714613`
- Full-test errors: `2753 / 160000`
- FP/FN: `1423 / 1330`

## Noise Signal

Full-test suspected conflict counts from the frozen kNN report:

- Medium conflict: `2279 / 160000` (`1.424375%`)
- Severe conflict: `515 / 160000` (`0.321875%`)
- Severe label-0 conflicts: `107`
- Severe label-1 conflicts: `408`

This is important because the target `F1 >= 99.9%` leaves room for only a very small number of mistakes, while this frozen candidate still makes `2753` full-test errors and the severe-conflict count alone is already large enough to challenge the target.

## Decision

The kNN candidate is a valid improvement and should be kept as the current best frozen Stage2 branch, but it does not achieve the final target. The project should return to Phase 2 with the kNN full-test and Val error files as the next review source.

Recommended next actions:

1. Build a full-test error review queue from `stage2_knn_extended_frozen_full_test_predictions.csv`.
2. Prioritize high-confidence FP/FN and severe conflict samples for label/source adjudication.
3. Compare kNN-fixed versus still-broken errors to separate local-neighborhood misses from true feature/model blind spots.
4. Do not run further full-test experiments until a new Val-selected candidate clearly beats `0.9833349965`.

## Phase 2 Review Queue

The full-test error queue has been generated:

- CSV: `reports/random_20w_split/stage2_knn_full_test_error_review_queue.csv`
- JSON: `reports/random_20w_split/stage2_knn_full_test_error_review_queue.json`
- Total errors: `2753`
- FN/FP: `1330 / 1423`
- Priority-0 severe errors: `696`
- Severe FN (`prob <= 0.05`): `251`
- Severe FP (`prob >= 0.95`): `445`

This is the next Data-Agent/Error-Agent work item. The first review pass should inspect P0 errors before any more model tuning.
