# Stage2 Val-Selected Blend Experiment

## Protocol

This experiment follows the locked evaluation funnel:

1. Analyze candidate blends on the full 20k Val split only.
2. Freeze the selected blend weights and threshold from Val.
3. Run one Test-10k confirmation with the frozen parameters.
4. Do not tune the threshold or weights on Test-10k.

The blend is a weighted average of already frozen Stage2 prediction files:

- `stage2_extended`: weight `1`
- `stage2_knn`: weight `2`
- Normalized weights: approximately `0.3333 / 0.6667`
- Threshold: `0.55`

## Val Selection

Report:

- `reports/random_20w_split/val_stage2_blend_analysis.json`

Val rows:

- `20000`

Current best kNN Stage2 Val baseline:

- F1: `0.9833349965`
- Errors: `334 / 20000`
- FP/FN: `188 / 146`
- Threshold: `0.49`

Selected blend on Val:

- F1: `0.9839588226`
- Errors: `321 / 20000`
- FP/FN: `166 / 155`
- Threshold: `0.55`

This is a Val improvement of `13` fewer errors and about `0.062%` absolute F1, so it clears the `0.05%` stagnation threshold for a Test-10k confirmation.

## Test-10k Confirmation

Report:

- `reports/random_20w_split/stage2_blend_val_selected_test10k_eval.json`
- `reports/random_20w_split/stage2_blend_val_selected_test10k_predictions.csv`

Frozen Test-10k result:

- Rows: `10000`
- Threshold: `0.55`
- F1: `0.9835999598`
- Errors: `163 / 10000`
- FP/FN: `74 / 89`
- AUC: `0.9986185308`

Previous frozen kNN Stage2 Test-10k result:

- F1: `0.9827066157`
- Errors: `172 / 10000`
- FP/FN: `82 / 90`

## Decision

The blend passes the Test-10k confirmation gate. It is eligible for one frozen 160k full-test evaluation with the same weights and threshold.

No full-test threshold sweep or weight tuning is allowed.

