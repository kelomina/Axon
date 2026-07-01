# Phase 2 Loop 4: Val-Side Blend Noise Audit

## Scope

This loop moves the current frozen full-test evidence back to an actionable Val-side workflow.

Frozen candidate:

- Blend: `stage2_extended:1 + stage2_knn:2`
- Threshold: `0.55`
- Selection source: Val only
- Full-test status: evidence only, not used for threshold, weight, or model tuning

The full-test result remains far below the target:

- Full-test rows: `160000`
- F1: `0.9832884232`
- Errors: `2673`
- FP/FN: `1311 / 1362`

Because the final test split is frozen, full-test error packages are held-out evidence. Actionable cleaning and adjudication must happen on Train/Val only, then return through the normal funnel: Val selection, Test-10k confirmation, and frozen full-test evaluation.

## Val Blend Reproduction

Generated outputs:

- Eval JSON: `reports/random_20w_split/stage2_blend_val_selected_val_eval.json`
- Predictions CSV: `reports/random_20w_split/stage2_blend_val_selected_val_predictions.csv`

Command:

```powershell
.\vnev\Scripts\python.exe scripts\evaluate_prediction_blend.py `
  --prediction stage2_extended=reports\random_20w_split\stage2_cache_matrix_replaced_extended_valonly\stage2_val_predictions.csv=stage2_prob_malicious=1 `
  --prediction stage2_knn=reports\random_20w_split\stage2_knn_extended_valonly\stage2_val_predictions.csv=stage2_prob_malicious=2 `
  --threshold 0.55 `
  --key-column sample_index `
  --output-json reports\random_20w_split\stage2_blend_val_selected_val_eval.json `
  --output-csv reports\random_20w_split\stage2_blend_val_selected_val_predictions.csv
```

Metrics:

- Rows: `20000`
- F1: `0.9839588226`
- Accuracy: `0.98395`
- Precision: `0.9834182399`
- Recall: `0.9845`
- AUC: `0.99853923`
- FP/FN: `166 / 155`
- Errors: `321`

## Val Error Queue

Generated outputs:

- CSV: `reports/random_20w_split/stage2_blend_val_error_review_queue.csv`
- JSON: `reports/random_20w_split/stage2_blend_val_error_review_queue.json`

Queue summary:

- Total rows: `20000`
- Total errors: `321`
- FP/FN: `166 / 155`
- Priority 0: `69`
- Priority 1: `58`
- Priority 2: `109`
- Priority 3: `85`

Reason counts:

- Severe FN (`prob <= 0.05`): `20`
- Severe FP (`prob >= 0.95`): `49`
- High-confidence FN (`prob <= 0.15`): `22`
- High-confidence FP (`prob >= 0.85`): `36`
- Mid-confidence FN (`prob <= 0.35`): `51`
- Mid-confidence FP (`prob >= 0.65`): `58`
- Near-threshold FN: `62`
- Near-threshold FP: `23`

## P0/P1 Neighbor Audit

Generated outputs:

- CSV: `reports/random_20w_split/stage2_blend_val_p0_p1_neighbor_audit.csv`
- JSON: `reports/random_20w_split/stage2_blend_val_p0_p1_neighbor_audit.json`

Audit scope:

- Selected P0/P1 rows: `127`
- Kept rows: `127`
- Missing cache rows: `0`
- Train-memory rows: `20000`
- Top-k neighbors: `25`

Support buckets:

- `neighbors_support_model_prediction`: `87`
- `neighbors_mixed`: `37`
- `neighbors_support_dataset_label`: `3`

Model-supported split:

- Total: `87 / 127`
- FP/FN: `61 / 26`
- P0/P1: `51 / 36`

Interpretation: most high-priority Val errors are not isolated threshold mistakes. Their frozen train-memory neighbors often support the model prediction against the dataset label. This is actionable evidence for label conflicts, source-list contamination, family-boundary ambiguity, or feature/source problems.

## Manual Review Package

Generated outputs:

- Review CSV: `reports/random_20w_split/stage2_blend_val_p0_p1_model_supported_manual_review_all87.csv`
- Review JSON: `reports/random_20w_split/stage2_blend_val_p0_p1_model_supported_manual_review_all87.json`
- Readiness CSV: `reports/random_20w_split/stage2_blend_val_p0_p1_model_supported_manual_review_all87_readiness.csv`
- Readiness JSON: `reports/random_20w_split/stage2_blend_val_p0_p1_model_supported_manual_review_all87_readiness.json`
- Guide CSV: `reports/random_20w_split/stage2_blend_val_p0_p1_model_supported_manual_review_all87_guide.csv`
- Guide JSON: `reports/random_20w_split/stage2_blend_val_p0_p1_model_supported_manual_review_all87_guide.json`
- Guide MD: `reports/random_20w_split/stage2_blend_val_p0_p1_model_supported_manual_review_all87_guide.md`

Selection:

- Source: Val P0/P1 rows where neighbors support the model prediction
- Selected rows: `87`
- FP/FN: `61 / 26`
- Severe FP: `39`
- Severe FN: `12`
- High-confidence FP: `22`
- High-confidence FN: `14`

Readiness audit:

- Review rows ready: `87 / 87`
- Source files exist: `87 / 87`
- Source SHA256 matches: `87 / 87`
- Cache files exist and load: `87 / 87`
- NPZ label/SHA/shape checks pass: `87 / 87`
- PE parse succeeds: `87 / 87`
- Top-5 neighbor evidence complete: `435 / 435`
- Duplicate source SHA256: `0`
- Verdict package ready: `false`, because manual verdict/action fields are intentionally blank

Guide suspicion levels:

- `critical_label_conflict`: `2`
- `strong_label_conflict`: `7`
- `moderate_label_conflict`: `78`

## Protocol Boundaries

The Val package is actionable because all reviewed rows are from validation-side errors. Human/business adjudication may produce one of these outcomes:

- `label_correct` + `keep_label`: keep as a true model error or blind spot.
- `label_wrong` + `relabel_train_only`: relabel only in train/val correction flow.
- `feature_broken` or `out_of_scope` + `replace_sample`: exclude and redraw a fresh valid sample.
- `uncertain` + `needs_more_evidence`: do not use for training changes yet.

Replacement rule:

- Bad or out-of-scope files must be replaced by newly selected valid candidates.
- The dataset must remain exactly `200000` rows.
- Split sizes must remain exactly `20000 / 20000 / 160000`.
- Excluded samples must never be selected as their own replacements.
- Cache coverage must be re-audited back to `200000 / 200000`.

Full-test artifacts remain held-out evidence only. They can support feasibility and noise-rate reporting, but they must not select thresholds, blend weights, feature masks, replacement policy, or hyperparameters.

## Current Decision

Do not continue blind model-side tuning. The next useful step is to manually adjudicate the 87 Val model-supported P0/P1 rows, then build a corrected Train/Val split if enough rows are confirmed as mislabeled, broken, or out of scope. After any correction, rerun the full Val loop and only promote a candidate to Test-10k if Val improves beyond the frozen baseline.
