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

## P0/P1 Neighbor Audit

Generated outputs:

- CSV: `reports/random_20w_split/stage2_blend_full_test_p0_p1_neighbor_audit.csv`
- JSON: `reports/random_20w_split/stage2_blend_full_test_p0_p1_neighbor_audit.json`

Audit scope:

- Selected rows: `1085` P0/P1 errors
- Matrix rows kept: `1085`
- Missing cache rows: `0`
- kNN memory: frozen `20000` train rows from the Stage2 train-memory model
- Top-k neighbors: `25`

Support buckets:

- `neighbors_support_model_prediction`: `748`
- `neighbors_mixed`: `284`
- `neighbors_support_dataset_label`: `53`

Model-supported P0/P1 split:

- Total: `748 / 1085`
- FP/FN: `489 / 259`
- P0/P1: `448 / 300`
- Nearest similarity `>= 0.90`: `89`
- Nearest similarity `>= 0.95`: `48`
- Nearest similarity `>= 0.98`: `20`
- Opposite-label neighbor ratio `>= 0.90`: `540`
- Opposite-label neighbor ratio `= 1.00`: `277`

This means most of the highest-priority remaining mistakes are not isolated model guesses. Their nearest train-memory neighborhoods often support the model prediction against the dataset label. That is strong evidence for source-label conflicts, family-boundary ambiguity, or contaminated allow/block lists.

## High-Similarity Conflict Summary

Generated outputs:

- CSV: `reports/random_20w_split/stage2_blend_full_test_p0_p1_high_similarity_conflicts.csv`
- JSON: `reports/random_20w_split/stage2_blend_full_test_p0_p1_neighbor_conflicts.json`

Definition:

`nearest_similarity >= 0.95 and opposite_label_ratio >= 0.8 and support_bucket == neighbors_support_model_prediction`

Result:

- High-similarity opposite-label conflicts: `48`
- Error type split: `48 FP / 0 FN`

Interpretation: these are especially suspicious benign-labeled rows. They look very close to malicious train-memory samples, most of their top-25 neighbors are malicious-labeled, and the blend also predicts them as malicious with high confidence. They should be reviewed manually before any model-side correction is attempted.

## Manual Review Package

Generated outputs:

- CSV: `reports/random_20w_split/stage2_blend_p0_p1_model_supported_manual_review_top260.csv`
- JSON: `reports/random_20w_split/stage2_blend_p0_p1_model_supported_manual_review_top260.json`
- Readiness CSV: `reports/random_20w_split/stage2_blend_p0_p1_model_supported_manual_review_top260_readiness.csv`
- Readiness JSON: `reports/random_20w_split/stage2_blend_p0_p1_model_supported_manual_review_top260_readiness.json`

First-pass high-similarity FP package:

- CSV: `reports/random_20w_split/stage2_blend_high_similarity_fp_manual_review_top48.csv`
- JSON: `reports/random_20w_split/stage2_blend_high_similarity_fp_manual_review_top48.json`
- Readiness CSV: `reports/random_20w_split/stage2_blend_high_similarity_fp_manual_review_top48_readiness.csv`
- Readiness JSON: `reports/random_20w_split/stage2_blend_high_similarity_fp_manual_review_top48_readiness.json`
- Human guide CSV: `reports/random_20w_split/stage2_blend_high_similarity_fp_manual_review_top48_guide.csv`
- Human guide JSON: `reports/random_20w_split/stage2_blend_high_similarity_fp_manual_review_top48_guide.json`
- Human guide MD: `reports/random_20w_split/stage2_blend_high_similarity_fp_manual_review_top48_guide.md`
- Rows: `48`
- Error type: `48 FP`
- Priority split: `38` P0, `10` P1
- Duplicate `source_sha256`: `0`
- Source/cache/PE ready: `48 / 48`
- Top-5 neighbor manifest/source/cache evidence: `240 / 240`
- Review-queue ready: `true`
- Verdict-package ready: `false`, because manual fields are intentionally blank.
- The guide is read-only evidence for adjudication. It does not fill `manual_label_verdict` or `recommended_action`.
- Dry adjustment plan: `reports/random_20w_split/stage2_blend_high_similarity_fp_manual_review_top48_adjustment_plan.json`
- Split placement: `48 / 48` reviewed rows are in the frozen `test` split, label `0`; `training_policy_rows=0`.
- Protocol consequence: these rows are held-out evidence by default. They must not directly tune thresholds, blend weights, or train/val policy.

Selection:

- Source: P0/P1 rows where neighbors support the model prediction
- Selected rows: `260`
- FP/FN: `130 / 130`
- Reasons: `130` severe FP (`prob >= 0.95`), `128` severe FN (`prob <= 0.05`), and `2` high-confidence FN (`prob <= 0.15`)
- Duplicate `source_sha256`: `0`
- Priority split: `258` P0, `2` P1

Readiness audit:

- Source files exist: `260 / 260`
- Source SHA256 matches review row: `260 / 260`
- Manifest matches by source path: `260 / 260`
- Cache files exist and load: `260 / 260`
- NPZ label/SHA/shape checks pass: `260 / 260`
- PE parse succeeds: `260 / 260`
- Top-5 neighbor evidence rows complete: `260 / 260`
- Top-5 neighbor manifest/source/cache evidence: `1300 / 1300`
- Review-queue ready: `true`
- Verdict-package ready: `false`, because `manual_label_verdict` and `recommended_action` are intentionally blank until human/business adjudication.
- Manual field validity: `260 / 260` rows are currently blank but not invalid.

Accepted `manual_label_verdict` values:

- Keep current label: `label_correct`
- Wrong label: `label_wrong`
- Exclude/replace: `out_of_scope`, `feature_broken`
- Defer: `uncertain`

Accepted `recommended_action` values:

- Keep current label: `keep_label`
- Relabel train/val only: `relabel_train_only`
- Replace invalid sample: `replace_sample`
- Quarantine broader source group: `quarantine_source_group`
- Defer: `needs_more_evidence`
- Mark model blind spot without changing data: `model_blindspot`

This package is only an adjudication queue. It must not be used to tune thresholds, blend weights, or model hyperparameters directly. If a reviewed file is confirmed invalid or mislabeled, the corrected split must replace the bad file with a fresh valid candidate of the intended class and then re-audit cache coverage back to exactly `200000` rows.

## Interpretation

The blend reduced full-test errors by `80`, mainly by reducing false positives, but the remaining error count is still much larger than the final target allows. The severe error set alone has `570` rows, so the next useful work is not blind threshold tuning. It is source-label and family-level error attribution.

Updated decision:

1. Prioritize manual/business adjudication of the model-supported P0/P1 rows.
2. Treat the `48` high-similarity FP conflicts as the first review batch.
3. Do not tune on full-test errors. Any corrected-data or model candidate must return to full Val selection first, then Test-10k confirmation, and only then a frozen full-test evaluation.
4. Keep exact dataset cardinality: corrected splits must remain `20000 / 20000 / 160000`, total `200000`, with no missing cache rows.
