# Phase 1 Loop127 Baseline, Cache Audit, and Calibration Report

Date: 2026-07-04

## Decision Summary

Phase 1 is complete for the current authoritative 20w split. The cache is ready, the strict source-SHA alignment path is verified, baseline and probability calibration were evaluated through the full validation funnel, and the GA feature mask was rejected on Val.

The best confirmed candidate is the probability calibrator:

- Full-test F1: `0.9686386448`
- Full-test errors: `5043 / 160000`
- Full-test FP/FN: `2923 FP`, `2120 FN`
- Baseline full-test F1: `0.9277562179`
- Baseline full-test errors: `11337 / 160000`
- Error reduction vs baseline: `6294`

This is a strong improvement, but it does not support the original `F1 >= 99.9%` target. On an 80k-positive/80k-negative final test set, `F1 >= 99.9%` requires roughly no more than `160` total FP+FN in the best case. The current best candidate has `5043` errors, so it would need to remove about `4883` more errors, or more than `96%` of the remaining error mass.

## Authoritative Data Inputs

Use this split:

- `reports/random_20w_split/loop127_full_duplicate_corrected_split.csv`

Use this cache manifest:

- `data/.cache/manifest_38672ba0.json`

Evidence:

- `reports/random_20w_split/phase1_loop127_strict_metadata_reaudit.json`
  - `audit_ready=true`
  - `rows=200000`
  - `manifest_samples=200145`
  - `row_issue_count=0`
  - `shape_failures=[]`
- `reports/random_20w_split/phase1_loop127_cache_ready_reaudit.json`
  - `cache_ready=true`
  - `total_rows=200000`
  - `covered_rows=200000`
  - `missing_rows=0`
  - `metadata_failure_rows=0`
  - `shape_failures=[]`

The split is balanced:

- Train: `20000` rows, `10000` benign, `10000` malicious
- Val: `20000` rows, `10000` benign, `10000` malicious
- Test: `160000` rows, `80000` benign, `80000` malicious

Identity policy:

- `source_sha256` is used only for cache alignment.
- Path, file name, directory, and extension are not model features and are not malware evidence.

## Resource And Memory Gate

Before each heavy run, `scripts/pre_run_resource_leak_guard.py` was executed. Relevant receipts:

- `reports/logs/phase1_loop127_cache_audit_guard.json`
- `reports/logs/phase1_loop127_strict_val_eval_guard_retry.json`
- `reports/logs/phase1_loop127_probability_calibrator_val_guard.json`
- `reports/logs/phase1_loop127_ga_mask_val_guard.json`
- `reports/logs/phase1_loop127_strict_test10k_eval_guard_after_fix.json`
- `reports/logs/phase1_loop127_probability_calibrator_test10k_guard.json`
- `reports/logs/phase1_loop127_strict_full_test_eval_guard.json`
- `reports/logs/phase1_loop127_probability_calibrator_full_test_guard.json`
- `reports/logs/phase1_loop127_calibrated_full_error_summary_guard.json`

One default guard run blocked when system memory briefly reached `90.81%`. The run was not bypassed; after waiting, memory dropped and the guard passed.

## Evaluated Assets

Baseline checkpoint:

- `models/random_20w_8192/best_model.pt`

Model/config facts:

- `max_byte_length=8192`
- `pe_feature_dim=256`
- `stat_feature_dim=49`
- `pe_schema_version=fixed_v2`

Probability calibrator:

- `models/random_20w_8192/random20w_replaced_logreg_calibrator.pkl`
- Features: baseline probability + stat features + PE features
- `C=0.3`
- `blend_model_weight=1.0`
- selected threshold: `0.44`

GA feature mask:

- `config/feature_masks/ga_recall_guard_2000.json`
- Rejected in this phase because strict Val did not beat the unmasked baseline.

## Validation Results

### Val Baseline

Report:

- `reports/phase1_loop127/baseline_val_strict_eval.json`

At threshold `0.53`:

- F1: `0.9279803861`
- AUC: `0.97578477`
- Errors: `1410`
- FP/FN: `494 FP`, `916 FN`

Best swept threshold:

- Threshold: `0.425`
- F1: `0.9304482226`
- Errors: `1395`

### Val Probability Calibrator

Report:

- `reports/phase1_loop127/probability_calibrator_val_eval.json`

At threshold `0.44`:

- F1: `0.9688605451`
- AUC: `0.99325775`
- Errors: `625`
- FP/FN: `348 FP`, `277 FN`
- Delta vs baseline threshold `0.53`: `+0.0408801589 F1`, `-785 errors`

### Val GA Feature Mask

Report:

- `reports/phase1_loop127/ga_mask_val_strict_eval.json`

At threshold `0.525`:

- F1: `0.9282160625`
- Errors: `1414`

Best swept threshold:

- Threshold: `0.45`
- F1: `0.9300469015`
- Errors: `1402`

Decision:

- Reject GA mask for Test-10k.
- It does not beat the unmasked Val best F1 `0.9304482226`.

## Test-10k Confirmation

`test10k` support in `scripts/evaluate_strict_split_from_cache.py` was fixed during this phase. The script now:

1. Uses explicit `split=test10k` rows if present.
2. Otherwise deterministically falls back to the first `10000` rows from `split=test`.

Tests:

- `vnev\Scripts\python.exe -m pytest tests\test_evaluate_strict_split_from_cache.py -vv`
- Result: `8 passed`

### Test-10k Baseline

Report:

- `reports/phase1_loop127/baseline_test10k_strict_eval.json`

At threshold `0.53`:

- F1: `0.9288711493`
- AUC: `0.9781802036`
- Errors: `695`
- FP/FN: `248 FP`, `447 FN`

Best swept threshold:

- Threshold: `0.45`
- F1: `0.9321012455`
- Errors: `676`

### Test-10k Probability Calibrator

Report:

- `reports/phase1_loop127/probability_calibrator_test10k_eval.json`

At threshold `0.44`:

- F1: `0.9725576290`
- AUC: `0.9941234271`
- Errors: `275`
- FP/FN: `163 FP`, `112 FN`
- Delta vs baseline threshold `0.53`: `+0.0436864797 F1`, `-420 errors`

Decision:

- Calibrator passed Test-10k and was allowed to run full test.

## Full-Test Results

### Baseline Full Test

Report:

- `reports/phase1_loop127/baseline_full_test_strict_eval.json`

At threshold `0.53`:

- F1: `0.9277562179`
- AUC: `0.9766162762`
- Errors: `11337`
- FP/FN: `4132 FP`, `7205 FN`

Best swept threshold:

- Threshold: `0.475`
- F1: `0.9285561390`
- Errors: `11348`

### Probability Calibrator Full Test

Report:

- `reports/phase1_loop127/probability_calibrator_full_test_eval.json`

At threshold `0.44`:

- F1: `0.9686386448`
- AUC: `0.9931262142`
- Errors: `5043`
- FP/FN: `2923 FP`, `2120 FN`
- Delta vs baseline threshold `0.53`: `+0.0408824269 F1`, `-6294 errors`

Error queue:

- `reports/phase1_loop127/calibrated_full_test_errors.csv`

Error summary:

- `reports/phase1_loop127/calibrated_full_test_error_summary.json`

Remaining calibrated full-test errors:

- Total errors: `5043`
- False positives: `2923`
- False negatives: `2120`

Confidence buckets:

- `fp_near_threshold_0.44_0.75`: `1394`
- `fp_mid_conf_0.75_0.90`: `586`
- `fp_high_conf_ge_0.90`: `943`
- `fn_near_threshold_0.30_0.44`: `758`
- `fn_mid_conf_0.10_0.30`: `859`
- `fn_high_conf_lt_0.10`: `503`

## Feasibility Assessment

The original `F1 >= 99.9%` target is not supported by Phase 1 evidence.

Reasoning:

- On the 16w final test set, with roughly 80k positives, `F1 >= 0.999` allows about `160` total errors in the best case.
- The best current candidate has `5043` errors.
- There are `1446` high-confidence errors after calibration:
  - `943` high-confidence FP with calibrated probability `>= 0.90`
  - `503` high-confidence FN with calibrated probability `< 0.10`
- These high-confidence errors alone are about `9x` the approximate total error budget for 99.9 F1.
- If even `0.1%` of the 160k final test labels are noisy, that is already `160` disputed rows, enough to consume the whole 99.9 error budget.

Recommendation:

- Do not use `99.9%` as the next immediate engineering gate.
- Use `97.5%` F1 as the next Phase 2 gate.
- Use `98.5%` F1 as a stretch target after label-noise and high-confidence error audits reduce the error floor.
- Reconsider `99.5%+` only after the hard-error queue is materially reduced and label-noise audits show the disputed/noisy rate is below the corresponding error budget.

## Phase 2 Work Queue

Data-Agent:

- Audit `reports/phase1_loop127/calibrated_full_test_errors.csv`.
- Prioritize high-confidence FP/FN first because thresholding cannot fix them.
- Verify cache/source consistency for suspicious errors using `source_sha256`, not file names or paths as evidence.
- Produce a label-noise review queue with explicit evidence fields.

Error-Agent:

- Split errors into:
  - high-confidence FP
  - high-confidence FN
  - mid-confidence errors
  - near-threshold errors
- For each group, identify whether the likely root is label noise, PE/stat feature anomaly, byte-sequence blind spot, packer/overlay behavior, or model capacity.

Model-Agent:

- Treat the probability calibrator as the current Phase 1 best candidate.
- Do not promote the GA feature mask.
- Candidate families for Phase 2 should target the remaining error buckets:
  - hard-example mining for high-confidence mistakes
  - near-threshold calibration or loss reweighting for near-threshold errors
  - content PE / certificate / string sidecars only if train/val readiness is proven and no path/name identity leakage is introduced

Eval-Agent:

- Keep the strict SHA-only evaluator as the primary evidence path.
- Do not run another full test until a new candidate beats calibrator on Val and Test-10k.
- Preserve all guard receipts for heavy runs.

## Code Changes Made During Phase 1

- `scripts/evaluate_strict_split_from_cache.py`
  - Fixed `--split test10k` fallback so it uses explicit `test10k` rows when present, otherwise first `10000` test rows.
- `tests/test_evaluate_strict_split_from_cache.py`
  - Added fallback coverage.
- `scripts/summarize_strict_val_errors.py`
  - Made error summarization stream CSV rows.
  - Added `--prob-column`.
  - Added `--output-errors-csv`.
- `tests/test_summarize_strict_val_errors.py`
  - Added calibrated probability column and error CSV coverage.

Verification:

- `vnev\Scripts\python.exe -m py_compile scripts\evaluate_strict_split_from_cache.py tests\test_evaluate_strict_split_from_cache.py`
- `vnev\Scripts\python.exe -m pytest tests\test_evaluate_strict_split_from_cache.py -vv`
  - `8 passed`
- `vnev\Scripts\python.exe -m py_compile scripts\summarize_strict_val_errors.py tests\test_summarize_strict_val_errors.py`
- `vnev\Scripts\python.exe -m pytest tests\test_summarize_strict_val_errors.py -vv`
  - `2 passed`

