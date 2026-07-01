# Phase 2 Loop 5: Val All-Error Neighbor Audit

## Scope

Loop 4 produced an actionable Val P0/P1 manual review package with `87` model-supported high-priority errors. This loop expands the same frozen analysis to every remaining Val error from the selected blend, without changing thresholds, blend weights, model hyperparameters, or any test-set policy.

Frozen candidate:

- Blend: `stage2_extended:1 + stage2_knn:2`
- Threshold: `0.55`
- Val rows: `20000`
- Val F1: `0.9839588226`
- Val errors: `321`
- FP/FN: `166 / 155`

Protocol boundary:

- Full-test errors remain held-out evidence only.
- This loop is Val-side only and can feed human/business adjudication.
- Manual verdicts are intentionally not filled by automation.

## All-Error Neighbor Audit

Generated outputs:

- CSV: `reports/random_20w_split/stage2_blend_val_all_errors_neighbor_audit.csv`
- JSON: `reports/random_20w_split/stage2_blend_val_all_errors_neighbor_audit.json`

Command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_stage2_knn_neighbors.py `
  --stage2-model reports\random_20w_split\stage2_knn_extended_valonly\stage2_selected_model.pkl `
  --train-predictions reports\random_20w_split\random_20w_8192_replaced_train_predictions.csv `
  --eval-base-predictions reports\random_20w_split\random_20w_8192_replaced_val_predictions.csv `
  --review-queue reports\random_20w_split\stage2_blend_val_error_review_queue.csv `
  --max-priority 3 `
  --top-k 25 `
  --batch-size 256 `
  --output-json reports\random_20w_split\stage2_blend_val_all_errors_neighbor_audit.json `
  --output-csv reports\random_20w_split\stage2_blend_val_all_errors_neighbor_audit.csv
```

Audit summary:

- Review rows total: `321`
- Review rows selected: `321`
- Kept rows: `321`
- Missing cache rows: `0`
- Train-memory rows: `20000`
- Top-k neighbors: `25`

Support buckets:

- `neighbors_support_model_prediction`: `160`
- `neighbors_mixed`: `129`
- `neighbors_support_dataset_label`: `32`

Interpretation:

The model-supported pattern is not limited to the most severe P0/P1 rows. Roughly half of all Val errors have train-memory neighborhoods that support the model prediction against the dataset label. This strengthens the noise hypothesis: the current ceiling is likely constrained by label/source conflicts and ambiguous family boundaries, not just by threshold placement.

## Full Model-Supported Review Pool

Generated outputs:

- Review CSV: `reports/random_20w_split/stage2_blend_val_all_model_supported_manual_review_all160.csv`
- Review JSON: `reports/random_20w_split/stage2_blend_val_all_model_supported_manual_review_all160.json`
- Readiness CSV: `reports/random_20w_split/stage2_blend_val_all_model_supported_manual_review_all160_readiness.csv`
- Readiness JSON: `reports/random_20w_split/stage2_blend_val_all_model_supported_manual_review_all160_readiness.json`
- Guide CSV: `reports/random_20w_split/stage2_blend_val_all_model_supported_manual_review_all160_guide.csv`
- Guide JSON: `reports/random_20w_split/stage2_blend_val_all_model_supported_manual_review_all160_guide.json`
- Guide MD: `reports/random_20w_split/stage2_blend_val_all_model_supported_manual_review_all160_guide.md`

Selection:

- Source: all Val errors where `support_bucket == neighbors_support_model_prediction`
- Selected rows: `160`
- FP/FN: `99 / 61`
- P0/P1/P2/P3: `51 / 36 / 47 / 26`

Reason counts:

- Severe FP (`prob >= 0.95`): `39`
- Severe FN (`prob <= 0.05`): `12`
- High-confidence FP (`prob >= 0.85`): `22`
- High-confidence FN (`prob <= 0.15`): `14`
- Mid-confidence FP (`prob >= 0.65`): `29`
- Mid-confidence FN (`prob <= 0.35`): `18`
- Near-threshold FP: `9`
- Near-threshold FN: `17`

## Readiness Audit

Strict readiness passed:

- Review rows ready: `160 / 160`
- Source files exist: `160 / 160`
- Source SHA256 matches: `160 / 160`
- Cache files exist and load: `160 / 160`
- NPZ label/SHA/shape checks pass: `160 / 160`
- PE parse succeeds: `160 / 160`
- Top-5 neighbor evidence complete: `800 / 800`
- Duplicate source SHA256: `0`
- Verdict package ready: `false`, because manual verdict/action fields are intentionally blank

Guide suspicion levels:

- `critical_label_conflict`: `2`
- `strong_label_conflict`: `11`
- `moderate_label_conflict`: `147`

## Queue Policy

Use two nested queues:

1. Priority queue: the Loop 4 P0/P1 package with `87` rows.
2. Extended pool: the Loop 5 all-priority package with `160` rows.

The `87` P0/P1 rows should be manually adjudicated first because they are the highest-confidence errors and provide the fastest signal. The remaining `73` model-supported rows are still actionable, but they should be reviewed after the priority queue or sampled to estimate whether the same noise pattern persists in lower-priority errors.

## Replacement And Relabel Policy

Allowed manual outcomes:

- `label_correct` + `keep_label`
- `label_wrong` + `relabel_train_only`
- `feature_broken` or `out_of_scope` + `replace_sample`
- `uncertain` + `needs_more_evidence`
- `label_correct` + `model_blindspot` when the label is credible but the model family is weak

Hard constraints:

- Bad files must be replaced by freshly selected valid candidates.
- Do not use excluded samples as their own replacements.
- The dataset must remain exactly `200000` rows.
- Split sizes must remain exactly `20000 / 20000 / 160000`.
- Cache coverage must return to exactly `200000 / 200000`.
- Any corrected split must rerun Val before Test-10k.

## Decision

Do not tune thresholds, blend weights, GA feature masks, or calibrators from this evidence. The next step is human/business adjudication of the Val review queues. After enough rows receive valid verdicts, build a corrected Train/Val split, redraw replacements for excluded rows, audit cache coverage, and rerun the frozen Val selection loop.
