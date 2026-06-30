# Phase 2 Loop 3: Val-Selected Blend Error Queue

## Scope

This loop starts from the current strongest frozen full-test candidate:

- Candidate: Val-selected blend of `stage2_extended:1` and `stage2_knn:2`
- Threshold: `0.55`
- Full-test predictions: `reports/random_20w_split/stage2_blend_val_selected_full_test_predictions.csv`
- Full-test eval: `reports/random_20w_split/stage2_blend_val_selected_full_test_eval.json`

The blend improved full-test F1 from `0.9828037453` to `0.9832884232`, but it still makes `2673` errors and is far below the `F1 >= 99.9%` target.

## Error Queue

Generated outputs:

- CSV: `reports/random_20w_split/stage2_blend_full_test_error_review_queue.csv`
- JSON: `reports/random_20w_split/stage2_blend_full_test_error_review_queue.json`

Queue summary:

- Rows: `160000`
- Errors: `2673`
- FP/FN: `1311 / 1362`
- Priority 0: `570`
- Priority 1: `515`
- Priority 2: `832`
- Priority 3: `756`

Reason counts:

- Severe FN (`prob <= 0.05`): `186`
- Severe FP (`prob >= 0.95`): `384`
- High-confidence FN (`prob <= 0.15`): `227`
- High-confidence FP (`prob >= 0.85`): `288`
- Mid-confidence FN (`prob <= 0.35`): `413`
- Mid-confidence FP (`prob >= 0.65`): `419`
- Near-threshold FN: `536`
- Near-threshold FP: `220`

## Interpretation

The blend reduced full-test errors by `80`, mainly by reducing false positives, but the remaining error count is still much larger than the final target allows. The severe error set alone has `570` rows, so the next useful work is not blind threshold tuning. It is source-label and family-level error attribution.

Next Error-Agent work:

1. Run neighbor audit on P0/P1 blend errors.
2. Compare blend-fixed errors against still-broken kNN errors.
3. Build a new manual review package from model-supported P0/P1 rows.
4. Keep Test-10k/full-test frozen; do not tune on these queues.

