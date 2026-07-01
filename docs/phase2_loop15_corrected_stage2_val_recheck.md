# Phase 2 Loop 15: Corrected Split Stage-2 Val Recheck

## Scope

Loop 15 reruns the Stage-2 validation-only path on the duplicate-source
corrected split from Loop 12/13. This is the fair follow-up to Loop 14:
instead of comparing the corrected single-checkpoint output to the old Stage-2
blend, it regenerates corrected Train/Val base predictions and retrains the
same Stage-2 families on corrected Train, then evaluates corrected Val only.

No Test-10k, full-test prediction, threshold tuning on Test, feature-mask
tuning, calibrator tuning, or cleanup-policy tuning was used.

## Inputs

- Checkpoint: `models/random_20w_8192/best_model.pt`
- Config: `config/random_20w_8192.toml`
- Manifest: `data/.cache/manifest_38672ba0.json`
- Corrected split: `reports/random_20w_split/duplicate_source_corrected_split.csv`
- Corrected Train predictions:
  `reports/random_20w_split/duplicate_source_corrected_train_predictions.csv`
- Corrected Val predictions:
  `reports/random_20w_split/duplicate_source_corrected_val_predictions.csv`

## Corrected Base Prediction Export

Train export:

```powershell
.\vnev\Scripts\python.exe scripts\evaluate_split_from_cache.py `
  --checkpoint models\random_20w_8192\best_model.pt `
  --config config\random_20w_8192.toml `
  --split-csv reports\random_20w_split\duplicate_source_corrected_split.csv `
  --manifest data\.cache\manifest_38672ba0.json `
  --split train `
  --threshold 0.5 `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda `
  --output-json reports\random_20w_split\duplicate_source_corrected_train_eval.json `
  --output-predictions-csv reports\random_20w_split\duplicate_source_corrected_train_predictions.csv `
  --missing-cache-output reports\random_20w_split\duplicate_source_corrected_train_missing_cache.csv
```

Result:

- Samples: `20000`
- Missing cache samples: `0`
- Accuracy: `0.929200`
- Precision: `0.939754`
- Recall: `0.917200`
- F1: `0.928340`
- AUC: `0.977630`
- FP/FN: `588 / 828`
- Errors: `1416`

The corrected Val base predictions were produced in Loop 14:

- Samples: `20000`
- Missing cache samples: `0`
- F1 at threshold `0.5`: `0.929723`
- FP/FN: `554 / 832`
- Errors: `1386`

## Stage-2 Corrected Val-Only Runs

### Extended

Command:

```powershell
.\vnev\Scripts\python.exe scripts\train_stage2_cache_matrix.py `
  --checkpoint models\random_20w_8192\best_model.pt `
  --train-predictions reports\random_20w_split\duplicate_source_corrected_train_predictions.csv `
  --val-predictions reports\random_20w_split\duplicate_source_corrected_val_predictions.csv `
  --output-dir reports\random_20w_split\stage2_corrected_extended_valonly `
  --thresholds 0.05:0.95:0.005 `
  --feature-set extended `
  --test-val-f1-gate 1.0 `
  --seed 42
```

Report:
`reports/random_20w_split/stage2_corrected_extended_valonly/stage2_cache_matrix_report.json`

Selected by corrected Val:

- Model: `hgb_lr0.08_leaf31_l2_1e-3__noise_none`
- Threshold: `0.575`
- F1: `0.982829`
- AUC: `0.998031`
- FP/FN: `189 / 155`
- Errors: `344`
- Train rows kept: `20000 / 20000`
- Val rows kept: `20000 / 20000`
- `test_predictions`: `null`
- `test_ran`: `false`

### Extended + KNN

Command:

```powershell
.\vnev\Scripts\python.exe scripts\train_stage2_cache_matrix.py `
  --checkpoint models\random_20w_8192\best_model.pt `
  --train-predictions reports\random_20w_split\duplicate_source_corrected_train_predictions.csv `
  --val-predictions reports\random_20w_split\duplicate_source_corrected_val_predictions.csv `
  --output-dir reports\random_20w_split\stage2_corrected_knn_extended_valonly `
  --thresholds 0.05:0.95:0.005 `
  --feature-set extended `
  --knn-features `
  --knn-top-k 5,10,25,50 `
  --knn-folds 5 `
  --knn-batch-size 1024 `
  --test-val-f1-gate 0.98 `
  --seed 42
```

Report:
`reports/random_20w_split/stage2_corrected_knn_extended_valonly/stage2_cache_matrix_report.json`

Selected by corrected Val:

- Model: `hgb_lr0.10_leaf63_l2_1e-3__noise_soft_conflict_downweight`
- Threshold: `0.385`
- F1: `0.983264`
- AUC: `0.998448`
- FP/FN: `206 / 130`
- Errors: `336`
- Train rows kept: `20000 / 20000`
- Val rows kept: `20000 / 20000`
- `test_predictions`: `null`
- `test_ran`: `false`

## Corrected Blend Results

### Frozen Old Blend Recipe

Frozen recipe from the prior best Stage-2 candidate:

- `stage2_extended:1 + stage2_knn:2`
- Threshold: `0.55`

Corrected Val result:

- Report: `reports/random_20w_split/stage2_corrected_blend_frozen_val_eval.json`
- F1: `0.983512`
- AUC: `0.998472`
- FP/FN: `172 / 158`
- Errors: `330`

This is worse than the old split's frozen Stage-2 blend:

- Old frozen Val F1: `0.983959`
- Old frozen FP/FN: `166 / 155`
- Old frozen errors: `321`

Delta, corrected frozen minus old frozen:

- F1: `-0.000447`
- FP: `+6`
- FN: `+3`
- Errors: `+9`

### Corrected Val-Selected Blend

Corrected Val-only blend analysis:
`reports/random_20w_split/stage2_corrected_blend_val_analysis.json`

Best corrected Val blend among the scanned recipes:

- Blend: `stage2_extended:1 + stage2_knn:1`
- Threshold: `0.505`
- Precision: `0.982559`
- Recall: `0.985900`
- F1: `0.984227`
- FP/FN: `175 / 141`
- Errors: `316`

Delta versus the old frozen Stage-2 blend:

- F1: `+0.000268`
- FP: `+9`
- FN: `-14`
- Errors: `-5`

Interpretation: corrected Val can find a small blend/threshold improvement,
mostly by reducing false negatives. The gain is only about `0.027` F1
percentage points and the Val set differs by `6` replacement rows from the old
split. This does not meet a robust "clearly better than baseline" standard.

## Noise Signals

Corrected Stage-2 reports still show non-trivial conflict/noise signals:

- Corrected Train severe conflict count: `56`
- Corrected Val severe conflict count: `81`
- Corrected Val severe conflict ratio: `0.00405`
- KNN conflict medium count on Train: `125`
- KNN conflict strong count on Train: `37`
- KNN exact-opposite count on Train: `9`

The corrected best blend still has `316` Val errors. This is much better than
the single-checkpoint path, but still far away from the `99.9%` full-test target
without additional noise adjudication and model improvement.

## Val Error Overlap Audit

Loop 15 added `scripts/compare_prediction_error_overlap.py` so old and corrected
prediction files can be compared by sample identity instead of by raw
`sample_index`. This matters because the corrected split replaced a few Val
rows; using only `sample_index` would compare different files as if they were
the same sample.

Command:

```powershell
.\vnev\Scripts\python.exe scripts\compare_prediction_error_overlap.py `
  --prediction old_frozen=reports\random_20w_split\stage2_blend_val_selected_val_predictions.csv=blend_prob_malicious=0.55 `
  --prediction corrected_frozen=reports\random_20w_split\stage2_corrected_blend_frozen_val_predictions.csv=blend_prob_malicious=0.55 `
  --prediction corrected_best=reports\random_20w_split\stage2_corrected_blend_best_val_predictions.csv=blend_prob_malicious=0.505 `
  --key-columns source_sha256,source_path `
  --output-json reports\random_20w_split\stage2_corrected_val_error_overlap.json `
  --output-csv reports\random_20w_split\stage2_corrected_val_error_overlap.csv
```

Result:

- Union identities: `20006`
- Common identities across all three predictions: `19994`
- Common rows with any error: `372`
- Common rows all correct: `19622`
- Old frozen common-row errors: `320`
- Corrected frozen common-row errors: `330`
- Corrected best common-row errors: `316`
- Three-way stable errors: `273`

Versus old frozen on common identities:

- Corrected frozen fixed `33` old errors, but introduced `43` new errors.
- Corrected best fixed `42` old errors, but introduced `38` new errors.
- Corrected best net improvement on common identities: `-4` errors.

Interpretation: the corrected best blend is not a broad correction. It trades a
small set of mistakes, with only a four-error net gain on common identities.
The `273` stable errors are now the main Val-only target for noise and feature
blind-spot analysis.

## Corrected Best Error Breakdown

While building the error breakdown, a tooling bug was found and fixed:
`scripts/analyze_prediction_errors.py` did not previously recognize
`blend_prob_malicious`, so blend CSVs were misread as probability `0.0`. The
column is now supported and covered by
`tests/test_analyze_prediction_errors.py`.

Command:

```powershell
.\vnev\Scripts\python.exe scripts\analyze_prediction_errors.py `
  --predictions reports\random_20w_split\stage2_corrected_blend_best_val_predictions.csv `
  --threshold 0.505 `
  --output-dir reports\random_20w_split\stage2_corrected_blend_best_val_error_analysis
```

Corrected result:

- Total predictions: `20000`
- Errors: `316`
- False positives: `175`
- False negatives: `141`
- FP probability average: `0.832742`
- FN probability average: `0.243189`

High-value buckets:

- Extension `<none>`: `154` errors, including `147` FP and `7` FN.
- Extension `.exe`: `130` errors, including `28` FP and `102` FN.
- Extension `.dll`: `31` errors, all FN.
- High-confidence FP `>=0.90`: `80`
- Low-confidence FN `<0.10`: `43`
- Near-threshold FN `0.45-0.51`: `20`

Interpretation: the next useful work is not more blind blend sweeping. The
highest-value Val-only queues are:

1. no-extension benign false positives, especially high-confidence FP;
2. `.exe` and `.dll` malicious false negatives;
3. the `273` stable errors shared by old frozen, corrected frozen, and
   corrected best.

## Decision

Do not promote Loop 15 to Test-10k.

Reasoning:

1. The frozen old blend recipe regresses on corrected Val.
2. The corrected Val-selected `1:1 @ 0.505` blend improves by only `5` errors
   over the old frozen Val baseline.
3. The improvement is below the project stagnation threshold of `0.05` F1
   percentage points.
4. The comparison is not perfectly identical because corrected Val replaced
   `6` rows.
5. Remaining duplicate and source-label conflicts still need manual
   adjudication before treating tiny Val gains as robust.

The corrected Stage-2 result should be kept as a Val-only diagnostic, not a new
candidate for Test-10k or full-test evaluation.

## Agent Review And Tie-Break

The multi-agent review produced one useful disagreement:

- Data-Agent: the corrected Stage-2 path is structurally valid and could be
  frozen for a Test-10k smoke because Train/Val row counts and cache coverage
  are clean.
- Eval-Agent: the corrected best blend improves only `5 / 20000` Val decisions
  over the old frozen baseline, so the gain is not clearly significant.
- Model-Agent: the corrected best blend should not enter Test-10k because the
  apparent improvement is too small and the corrected split changed a few Val
  rows.
- Error-Agent: the remaining `316` Val errors should be explained before
  spending Test-10k budget on a tiny Val-only gain.

Tie-break decision: follow Eval/Model/Error. Structural readiness is necessary,
but not sufficient. The project funnel requires clear Val superiority before
Test-10k, and Loop 15 does not provide that level of evidence.

## Next Actions

1. Prioritize the corrected best blend's `316` Val errors, especially the `141`
   false negatives, for neighbor/source-family adjudication.
2. Start with the `273` stable errors shared across old frozen, corrected
   frozen, and corrected best.
3. Continue manual review of the remaining cross-label duplicate groups before
   drawing strong conclusions from sub-10-error Val changes.
4. Keep full-test frozen.
