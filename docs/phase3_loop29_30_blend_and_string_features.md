# Loop29/30 Blend And String Feature Follow-Up

Date: 2026-07-01

## Loop29: Loop28 + Loop27 Blend

Loop29 tested whether Loop28 content PE metadata and Loop27 Stage-2 predictions
were complementary.

Val-only selection:

- Inputs:
  - Loop28 content PE: `stage2_loop28_content_pe_valonly/stage2_val_predictions.csv`
  - Loop27 extended: `stage2_loop27_extended_valonly/stage2_val_predictions.csv`
  - Loop27 kNN: `stage2_loop27_knn_extended_valonly/stage2_val_predictions.csv`
- Best Val weights:
  - content PE: `0.5`
  - Loop27 extended: `0.1`
  - Loop27 kNN: `0.4`
- Best Val threshold: `0.49`
- Best Val result: F1 `0.9926621075`, errors `147`, FP/FN `90/57`

Frozen Test-10k confirmation:

- Result: F1 `0.9887685519`, errors `112`, FP/FN `62/50`
- Baseline to beat: Loop28 content-only Test-10k errors `111`

Decision: reject Loop29 for full-test promotion. It improved Val but did not
beat Loop28 on frozen Test-10k. This is mild Val overfitting or insufficient
generalization of the blend weights.

## Loop30: Content String Features

Loop30 added 43 binary string/keyword features extracted from file bytes only.
The features intentionally do not encode filename, extension, directory name,
or path text.

Feature groups include:

- URL/IP/Windows path counters
- script execution keywords
- persistence and registry keywords
- injection and process-manipulation keywords
- crypto/packer/evasion keywords
- vendor/version-resource keywords

Train/Val cache:

- Rows: `40000/40000`
- Feature dim: `43`
- zero_features: `0`

Best Val result:

- Model: `hgb_lr0.06_leaf31_l2_0__noise_none`
- Threshold: `0.51`
- F1: `0.9916537558`
- Errors: `167`
- FP/FN: `88/79`

Decision: reject Loop30 for Test-10k promotion. It is weaker than Loop28
content PE metadata alone, which had Val F1 `0.9919048571` and `162` errors.

## Current Best

The current best promoted model remains Loop28 content PE metadata:

- Val: F1 `0.9919048571`, errors `162`
- Test-10k: F1 `0.9888677164`, errors `111`
- Full-test: F1 `0.9878358558`, errors `1949`

Next useful work should not repeat shallow blends or broad string keyword
features. The more promising direction is targeted residual analysis of Loop28
errors, especially:

- extensionless benign false positives
- malicious `.exe` false negatives
- malicious `.dll` false negatives

Any follow-up features must remain content-derived and Val-first.
