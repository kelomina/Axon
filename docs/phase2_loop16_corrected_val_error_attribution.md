# Phase 2 Loop 16: Corrected Val Error Attribution

## Scope

Loop 16 stays inside the corrected Train/Val boundary from Loop 15 and turns
the corrected best Stage-2 Val errors into a manual adjudication workload.

No Test-10k, full-test prediction, Test threshold tuning, feature-mask tuning,
or automatic label correction was used. The goal is evidence triage only: find
which Val errors are most likely to be label noise, feature breakage, source
mixing, or genuine model blind spots.

## Inputs

- Corrected split:
  `reports/random_20w_split/duplicate_source_corrected_split.csv`
- Corrected best blend Val predictions:
  `reports/random_20w_split/stage2_corrected_blend_best_val_predictions.csv`
- Score column: `blend_prob_malicious`
- Corrected best blend threshold: `0.505`
- Stage-2 kNN model:
  `reports/random_20w_split/stage2_corrected_knn_extended_valonly/stage2_selected_model.pkl`
- Corrected Train predictions:
  `reports/random_20w_split/duplicate_source_corrected_train_predictions.csv`
- Corrected Val base predictions:
  `reports/random_20w_split/duplicate_source_corrected_val_predictions.csv`

## Error Review Queue

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_stage2_error_review_queue.py `
  --predictions-csv reports\random_20w_split\stage2_corrected_blend_best_val_predictions.csv `
  --score-column blend_prob_malicious `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_error_review_queue.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_error_review_queue.json `
  --max-examples 40
```

Outputs:

- `reports/random_20w_split/stage2_corrected_best_val_error_review_queue.csv`
- `reports/random_20w_split/stage2_corrected_best_val_error_review_queue.json`

Result:

- Rows total: `20000`
- Errors total: `316`
- FP/FN: `175 / 141`
- Priority counts:
  - P0: `94`
  - P1: `46`
  - P2: `101`
  - P3: `75`
- Reason counts:
  - Severe FP, probability `>= 0.95`: `66`
  - Severe FN, probability `<= 0.05`: `28`
  - High-confidence FP, probability `>= 0.85`: `26`
  - High-confidence FN, probability `<= 0.15`: `20`
  - Mid-confidence FP: `55`
  - Mid-confidence FN: `46`
  - Near-threshold FP/FN: `28 / 47`

Interpretation: the first human review pass should not start from all `316`
rows equally. P0/P1 contains `140` high-value rows, and the `94` P0 rows are
the most direct evidence for either severe label/source noise or stable model
blind spots.

## KNN Neighbor Audit

Command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_stage2_knn_neighbors.py `
  --stage2-model reports\random_20w_split\stage2_corrected_knn_extended_valonly\stage2_selected_model.pkl `
  --train-predictions reports\random_20w_split\duplicate_source_corrected_train_predictions.csv `
  --eval-base-predictions reports\random_20w_split\duplicate_source_corrected_val_predictions.csv `
  --review-queue reports\random_20w_split\stage2_corrected_best_val_error_review_queue.csv `
  --max-priority 3 `
  --top-k 25 `
  --batch-size 256 `
  --output-json reports\random_20w_split\stage2_corrected_best_val_all_errors_neighbor_audit.json `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_all_errors_neighbor_audit.csv
```

Outputs:

- `reports/random_20w_split/stage2_corrected_best_val_all_errors_neighbor_audit.csv`
- `reports/random_20w_split/stage2_corrected_best_val_all_errors_neighbor_audit.json`

Result:

- Review rows selected: `316 / 316`
- Missing-cache skips: `0`
- Memory rows: `20000`
- Top-k neighbors per reviewed error: `25`
- Base feature dimension: `1420`
- Support buckets:
  - `neighbors_support_model_prediction`: `156`
  - `neighbors_mixed`: `132`
  - `neighbors_support_dataset_label`: `28`

Interpretation: `156 / 316` errors have local neighbors that support the model
prediction rather than the dataset label. That is a strong label-noise or
source-mixing signal, but it is not proof by itself. These rows must be
manually adjudicated before any relabel, replacement, quarantine, or model
training policy change.

The `28` rows where neighbors support the dataset label are different: they
are more likely to be genuine model blind spots, feature blind spots, or
calibration failures. They should not be cleaned away as noise without stronger
external evidence.

## Manual Review Packages

### Model-Supported P0/P1

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_stage2_manual_review_package.py `
  --neighbor-csv reports\random_20w_split\stage2_corrected_best_val_all_errors_neighbor_audit.csv `
  --support-bucket neighbors_support_model_prediction `
  --max-priority 1 `
  --fp-count 80 `
  --fn-count 80 `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_model_supported_p0_p1_manual_review.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_model_supported_p0_p1_manual_review.json
```

Result:

- Available rows: `93`
- Selected rows: `93`
- FP/FN: `66 / 27`
- Reasons:
  - Severe FP: `51`
  - Severe FN: `18`
  - High-confidence FP: `15`
  - High-confidence FN: `9`

This is the highest-value human adjudication queue. These rows combine two
signals: the Stage-2 blend is confidently wrong against the current dataset
label, and the kNN neighborhood tends to support the model prediction.

### Mixed P0/P1

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_stage2_manual_review_package.py `
  --neighbor-csv reports\random_20w_split\stage2_corrected_best_val_all_errors_neighbor_audit.csv `
  --support-bucket neighbors_mixed `
  --max-priority 1 `
  --fp-count 40 `
  --fn-count 40 `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_mixed_p0_p1_manual_review.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_mixed_p0_p1_manual_review.json
```

Result:

- Available rows: `44`
- Selected rows: `44`
- FP/FN: `24 / 20`
- Reasons:
  - Severe FP: `15`
  - Severe FN: `9`
  - High-confidence FP: `9`
  - High-confidence FN: `11`

This is the second review queue. These rows are still high priority, but the
neighbor evidence is less one-sided, so the expected outcome is more mixed:
some will be source noise, and some will likely be hard in-distribution model
failures.

## Manual Fields

All manual fields are blank by design:

- `manual_label_verdict`
- `manual_verdict_note`
- `recommended_action`

Allowed `manual_label_verdict` values follow the existing adjudication workflow:

- `label_correct`
- `label_wrong`
- `out_of_scope`
- `feature_broken`
- `uncertain`

Allowed `recommended_action` values:

- `keep_label`
- `relabel_train_only`
- `replace_sample`
- `quarantine_source_group`
- `needs_more_evidence`
- `model_blindspot`

Model evidence must not fill these fields automatically. The human/business
review should use outside evidence such as trusted threat intelligence,
sandboxing, vendor verdicts, source provenance, file integrity, or business
allow-list context.

## Safety Invariants

- The full 20w dataset invariant remains mandatory:
  `200000 = 20000 train + 20000 val + 160000 test`.
- If a reviewed row is judged `feature_broken` or `out_of_scope`, it must be
  excluded and replaced by a fresh valid same-label candidate. It must not be
  kept as its own replacement, and the split must not shrink.
- If `N` rows are excluded, exactly `N` fresh replacement rows are required.
- Replacement candidates must be unused, valid, and cache-ready before any new
  Train/Val experiment.
- Test rows, if ever reviewed, remain held-out evidence by default and must not
  tune thresholds, blend weights, feature masks, or Train/Val policy.
- No automatic label changes are allowed from model-only or neighbor-only
  evidence.

## Adjudication Path

After humans fill one of the new review CSVs, convert it into a non-destructive
adjustment plan before any split rebuild:

```powershell
.\vnev\Scripts\python.exe scripts\apply_manual_review_verdicts.py `
  --review-csv reports\random_20w_split\stage2_corrected_best_val_model_supported_p0_p1_manual_review.csv `
  --split-csv reports\random_20w_split\duplicate_source_corrected_split.csv `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_model_supported_manual_adjustment_plan.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_model_supported_manual_adjustment_plan.json
```

If the plan requires replacements, build a fresh candidate pool using the
reported replacement counts, then build a corrected split and run the strict
cache-readiness gate. This follows the existing workflow in
`docs/manual_review_adjudication_workflow.md`.

## Agent Review

The multi-agent review converged on the same decision:

- Data-Agent confirmed the corrected Train/Val cache and row counts are clean,
  and the Loop 16 artifacts are suitable for noise triage.
- Eval-Agent rejected Test-10k promotion because the corrected best blend only
  improves Val by `5 / 20000` errors over the old frozen Stage-2 baseline,
  below the `0.05` F1 percentage-point improvement threshold.
- Model-Agent agreed that more blind blend or threshold sweeping is unlikely
  to produce a robust gain before noise adjudication.
- Error-Agent prioritized the `316` corrected best Val errors, especially
  P0/P1 high-confidence FP/FN and stable cross-method errors.

## Decision

Do not enter Test-10k from Loop 16.

Loop 16 is a Phase 2 noise and error-attribution step, not a candidate model
promotion. The main outcome is a concrete manual adjudication queue:

1. Review the `93` model-supported P0/P1 rows first.
2. Review the `44` mixed P0/P1 rows second.
3. Treat `neighbors_support_model_prediction` as a strong noise signal, not as
   an automatic relabeling rule.
4. Convert filled manual fields into a non-destructive adjustment plan.
5. If replacements are required, redraw fresh candidates and rerun cache
   readiness before any new Train/Val evaluation.

The scientific implication is also clear: the current corrected best Val F1
of `0.984227` is far below the `99.9%` full-test target, and the remaining
error structure contains enough suspected noise that manual adjudication is
now higher value than additional Val-only micro-tuning.

