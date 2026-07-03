# Axon Malware Classifier ML Improvement Recommendations

## Status

This document records the current strict 200k experiment state after the loop124
full-test confirmation.

The current strict split is:

- Train: 20,000
- Val: 20,000
- Test: 160,000

The active identity rule is hash-first:

- `source_sha256` is used only to align a frozen row with its cached feature file.
- `sample_index` can disambiguate duplicate content rows in frozen split files.
- File name, directory, extension, source path, row order, and hash are not model
  evidence, threshold evidence, GA evidence, or noise verdict evidence.

## Confirmed Results

### Seed43 Baseline

Strict Val best threshold:

- F1: 0.9480857887
- Errors: 1,036
- FP: 496
- FN: 540

Strict Full Test at Val-selected threshold 0.57:

- F1: 0.9481225204
- Errors: 8,277
- FP: 3,913
- FN: 4,364

### Probability Calibrator

The strict calibrator uses only:

- baseline malicious probability
- PE scalar features
- stat scalar features

The calibrator validates every row by `source_sha256` and label before reading
cache features. It does not use path/name/directory/extension metadata.

Strict Val:

- F1: 0.9700996678
- Errors: 603
- FP: 385
- FN: 218
- Delta vs baseline: +0.0220138790 F1, -433 errors

Locked strict Test-10k:

- F1: 0.9723987292
- Errors: 278
- FP: 175
- FN: 103
- Delta vs baseline: +0.0205902931 F1, -203 errors

Strict Full Test:

- F1: 0.9702929778
- Errors: 4,791
- FP: 3,033
- FN: 1,758
- Delta vs baseline: +0.0221704575 F1, -3,486 errors

Decision: useful candidate, but not close to the 99.9% target.

## Feasibility Note

For a balanced 160k test set, F1 >= 99.9% roughly implies only about one to two
hundred total classification errors, depending on the FP/FN mix. The current
best confirmed candidate still has 4,791 errors. This gap is too large for
threshold tuning alone.

Recommended interpretation:

- Probability calibration fixed a real near-threshold problem.
- The remaining error budget is dominated by data noise, feature blind spots,
  and model separability.
- The 99.9% target should remain aspirational until content-level noise and
  feature evidence can account for thousands of remaining errors.

## GA Feature Mask Reverification

Current candidate:

- `config/feature_masks/ga_recall_guard_2000.json`
- kept_total=125, kept_pe=95, kept_stat=30

Strict Val on corrected split:

- GA mask best F1: 0.9475576865
- GA mask errors: 1,050
- Baseline best F1: 0.9480857887
- Baseline errors: 1,036
- Delta: -0.0005281022 F1, +14 errors

Decision: reject. Do not send this GA mask to Test-10k under the current
corrected split.

## Highest-Value Next Work

1. Use the Loop126 strict content-evidence error audit.

   New artifacts:

   - `reports/random_20w_split/loop126_val_calibrated_error_content_evidence.csv`
   - `reports/random_20w_split/loop126_test10k_calibrated_error_content_evidence.csv`
   - `reports/random_20w_split/loop126_val_noise_audit_focus_blinded.csv`
   - `reports/random_20w_split/loop126_test10k_noise_audit_focus_blinded.csv`

   The public focus files are blinded: they exclude source path, cache path,
   source hash, sample index, filename, directory, extension, model probability,
   and threshold fields. Private map files exist only for sample lookup.

2. Separate likely noise from model blind spots.

   High-confidence persistent FP/FN should be reviewed first. A high-confidence
   error is not an automatic relabel; it is only a priority signal for content
   inspection.

   Loop126 Val calibrated errors:

   - Total errors: 603
   - FP: 385
   - FN: 218
   - Persistent errors: 465
   - Broken by calibrator: 138
   - Focus lanes:
     - feature_or_label_quality_review: 262
     - model_blindspot_review: 136
     - calibration_regression_review: 138
     - boundary_model_review: 67

   Loop126 locked Test-10k calibrated errors:

   - Total errors: 278
   - FP: 175
   - FN: 103
   - Persistent errors: 210
   - Broken by calibrator: 68
   - Focus lanes:
     - feature_or_label_quality_review: 131
     - model_blindspot_review: 59
     - calibration_regression_review: 68
     - boundary_model_review: 20

3. Improve features before another full test.

   The current model still misses thousands of samples after calibration. The
   next model work should target content evidence not already captured well by
   PE/stat features, especially overlay/security-boundary and structural
   anomalies.

   Loop126 also corrected the error-evidence extractor to resolve PE features by
   schema name instead of hard-coded legacy indices. This matters because the
   active strict cache is `fixed_v2`: legacy overlay/trailing fields are not
   present in the current 256-dimensional PE input. Those fields must be added as
   real content features or a stage-2 content feature family; they must not be
   inferred from filename/path metadata or guessed from old indices.

4. Keep the full test locked.

   Full test has now been used once to confirm the calibrator. Future iteration
   should return to Train/Val and locked Test-10k. Do not use full-test results
   for repeated tuning.

## Required Guardrails

- Bad feature files must be quarantined and replaced by fresh same-original-label
  samples, not by filling counts with already-failed rows.
- Any redraw must keep the exact required counts; 200,000 means exactly 200,000.
- Any candidate must pass full Val first.
- Test-10k is allowed only after meaningful Val improvement.
- Full test is allowed only after Test-10k confirms the candidate.
- No path/name/directory/extension evidence is allowed in model, GA, threshold,
  noise, or replacement decisions.
