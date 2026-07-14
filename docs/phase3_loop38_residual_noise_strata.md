# Loop38 Loop28 Residual Noise Strata

Date: 2026-07-02

## Objective

Loop38 did not train a model. It audited the current best Loop28 full-test
residuals to separate likely learnable errors from high-confidence conflict
and near-threshold hard cases.

The audit used only frozen prediction CSVs and existing noise-audit outputs.
`source_sha256` and `source_path` were used only as identity keys for joining
audit tables. They were not used as model features.

## Inputs

- Loop28 full-test predictions:
  `reports/random_20w_split/stage2_loop28_content_pe_frozen_full_test_predictions.csv`
- Loop37 full-test predictions:
  `reports/random_20w_split/loop37_frozen_full_test_blend_loop28_byte_ngram/loop37_blend_full_test_predictions.csv`
- byte n-gram full-test predictions:
  `reports/random_20w_split/byte_ngram_sgd_loop37_loop27_full_test_frozen/byte_ngram_sgd_full_test_predictions.csv`
- Loop26 blend full-test predictions:
  `reports/random_20w_split/stage2_loop26_blend_frozen_full_test_predictions.csv`

## Results

Loop28 remains the current best full-test model:

- Full-test rows: `160000`
- F1: `0.9878358558`
- Errors: `1949`
- FP/FN: `1087/862`

Loop28 noise audit at threshold `0.5`:

- suspected noise or hard examples: `910/1949`
- severe/high confidence conflicts: `649`
- near-threshold errors: `261`

Loop28 residual strata:

| Stratum | Count |
| --- | ---: |
| Loop28 total errors | `1949` |
| FP | `1087` |
| FN | `862` |
| Corrected by Loop37 | `78` |
| Corrected by byte n-gram | `720` |
| Corrected by Loop26 blend | `550` |
| Corrected by at least one compared model | `921` |
| Not corrected by any compared model | `1028` |

Corrected-by-any split:

- corrected FP: `385`
- corrected FN: `536`

Not-corrected split:

- not-corrected FP: `702`
- not-corrected FN: `326`

High-confidence conflicts that no compared model fixed:

- severe FN conflict `<=0.01`: `81`
- high FN conflict `<=0.05`: `80`
- severe FP conflict `>=0.99`: `175`
- high FP conflict `>=0.95`: `165`

## Interpretation

There is still learnable signal: `921` of Loop28's `1949` errors were corrected
by at least one compared model. The strongest opportunity is not broad blending,
because weak models also add many new errors. The useful next step is targeted
residual modeling over the corrected-by-other-model strata, with strict OOF
training and a Val gate.

There is also a real noise or boundary ceiling: `649` Loop28 errors are severe
or high-confidence conflicts, and most of those are not corrected by any
compared model. These rows should not be blindly relabeled, but they should be
treated as a manual adjudication or richer-feature queue. Expecting short-term
`F1 >= 99.9%` without resolving this stratum is not realistic.

## Next Candidate Guidance

The next useful candidate should not be another broad feature dump or shallow
linear blend. It should be one of:

1. A strict OOF residual gate trained to decide when an auxiliary model may
   override Loop28, using only content-derived features and OOF auxiliary
   predictions.
2. A narrow feature extractor aimed at Loop28 FN corrected by byte n-gram or
   Loop26, especially DLL/sys and driver/service-like malicious samples.
3. A manual adjudication package for the not-corrected severe/high confidence
   conflicts, preserving the exact `200000` split by replacement only after a
   verdict.

## Artifacts

- Error overlap report:
  `reports/random_20w_split/loop38_loop28_residual_overlap/full_test_error_overlap.json`
- Error overlap details:
  `reports/random_20w_split/loop38_loop28_residual_overlap/full_test_error_overlap_details.csv`
- Loop28 noise audit:
  `reports/random_20w_split/stage2_loop28_content_pe_full_test_noise_audit/noise_audit_summary.json`
- Residual strata summary:
  `reports/random_20w_split/loop38_loop28_residual_strata/loop28_residual_strata_summary.json`
- Residual strata details:
  `reports/random_20w_split/loop38_loop28_residual_strata/loop28_residual_strata_details.csv`

## Verification

Commands run:

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\summarize_loop28_residual_strata.py
.\vnev\Scripts\python.exe -m pytest tests\test_summarize_loop28_residual_strata.py tests\test_compare_prediction_error_overlap.py tests\test_build_prediction_noise_audit.py
```
