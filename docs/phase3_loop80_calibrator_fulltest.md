# Phase 3 Loop80 Probability Calibrator Full-Test

## Purpose

Loop80 completes the strict full-test confirmation for the probability
calibrator that had already passed Val and Test-10k. This is not a new training
run and not a threshold search. The calibrator was trained on train, selected on
Val, confirmed on Test-10k, and then evaluated once on the full 160k test split.

Identity fields remain audit and alignment metadata only. They are not model,
threshold, GA, gate, or noise-cleaning evidence.

## Entry Evidence

Loop79 allowed this step:

- `reports/random_20w_split/loop79_current_state_gate.json`
- decision: `pass`
- current split cache: `200000/200000`
- current 1% sample integrity: `2000/2000`
- probability calibration status: Val and Test-10k passed

Calibrator train/Val report:

- `reports/random_20w_split/random_20w_8192_replaced_calibrator_train_val.json`
- model: `models/random_20w_8192/random20w_replaced_logreg_calibrator.pkl`
- selected threshold: `0.44`
- train rows kept: `20000/20000`
- val rows kept: `20000/20000`
- Val F1 delta vs 8192 baseline: `+0.03789303556617796`

Test-10k confirmation:

- `reports/random_20w_split/random_20w_8192_replaced_calibrator_test10k_eval.json`
- rows kept: `10000/10000`
- missing cache: `0`
- Test-10k F1 delta vs 8192 baseline: `+0.04255785450909355`
- Test-10k error delta: `-412`

## Full-Test Command

Guard:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\evaluate_probability_calibrator.py `
  --output-json reports\random_20w_split\loop80_calibrator_fulltest_guard.json `
  --allow-risk npz_array_load `
  --allow-risk pickle_model_load
```

Result: `decision=pass`.

Evaluation:

```powershell
.\vnev\Scripts\python.exe scripts\evaluate_probability_calibrator.py `
  --model models\random_20w_8192\random20w_replaced_logreg_calibrator.pkl `
  --predictions reports\random_20w_split\random_20w_8192_replaced_test_predictions.csv `
  --threshold 0.44 `
  --baseline-threshold 0.50 `
  --missing-cache-output reports\random_20w_split\loop80_calibrator_fulltest_missing_cache.csv `
  --output-json reports\random_20w_split\loop80_calibrator_fulltest_eval.json
```

Summary:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop80_calibrator_fulltest_summary.py `
  --output-json reports\random_20w_split\loop80_calibrator_fulltest_summary.json `
  --output-md reports\random_20w_split\loop80_calibrator_fulltest_summary.md
```

## Result

Full-test rows:

- total: `160000`
- kept: `160000`
- skipped missing cache: `0`

8192 baseline on the same prediction CSV:

- F1: `0.9283588516140471`
- AUC: `0.9765983176562499`
- FP/FN: `4620 / 6694`
- errors: `11314`

Loop80 calibrator:

- threshold: `0.44`
- F1: `0.9686442786069652`
- AUC: `0.9931121783593749`
- FP/FN: `2921 / 2121`
- errors: `5042`

Delta vs the 8192 baseline:

- F1: `+0.04028542699291815`
- errors: `-6272`
- FP: `-1699`
- FN: `-4573`

This is a real improvement over the 8192 neural baseline. It confirms that the
train-split calibrator learned useful PE/stat/probability corrections without
using test for training.

## Current-Best Comparison

Loop57 remains the full-test best reference:

- `reports/random_20w_split/loop57_fn_overlay_gate_frozen_full_test_eval.json`
- F1: `0.9883629658239992`
- FP/FN: `1195 / 673`
- errors: `1868`

Loop80 vs Loop57:

- F1 delta: `-0.01971868721703396`
- errors delta: `+3174`
- FP delta: `+1726`
- FN delta: `+1448`

Therefore Loop80 is not the final candidate and must not replace Loop57.

## Target Gap

For a balanced 160k test set, the best-case FP-only allowance for
`F1 >= 0.999` is about `160` errors. Loop80 has `5042` errors, so it would still
need to remove at least `4882` errors even under the most favorable assumption.

This strengthens the earlier Loop71 conclusion: the path to 99.9% is not
available through this single calibrator. The next useful work is not threshold
tuning on test. It is Val-first fusion or new content evidence that can combine
Loop57's strong content/gate behavior with the calibrator's large FN reduction,
without using identity fields and without importing test verdicts into training.

## Validation

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_build_loop80_calibrator_fulltest_summary.py -q
```

Result: `2 passed`.

## Verdict

Loop80 completed the allowed full-test check and produced a useful negative
result:

- Keep Loop57 as current best.
- Do not promote the calibrator as final.
- Use the calibrator only as evidence for future Val-first fusion research.
- Continue noise review and independent content-feature work under the strict
  funnel.
