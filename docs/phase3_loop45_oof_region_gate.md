# Phase 3 Loop45: OOF Region Byte N-gram Residual Gate

Date: 2026-07-02

## Objective

Loop44 showed that regionized byte n-gram features are more informative than
the earlier prefix-only byte n-gram model, but a direct blend with Loop28 did
not reduce Val errors. Loop45 tests the stricter follow-up: use region n-gram
scores only as a gated residual override, with train scores generated
out-of-fold.

This loop is Val-only. It does not evaluate Test-10k or the 160k full-test.

## Protocol

Inputs:

- checkpoint: `models/random_20w_8192/best_model.pt`
- train predictions: `reports/random_20w_split/loop27_train_predictions.csv`
- val predictions: `reports/random_20w_split/loop27_val_predictions.csv`
- Loop28 Val reference:
  `reports/random_20w_split/stage2_loop28_content_pe_valonly/stage2_val_predictions.csv`
- content PE cache:
  `reports/random_20w_split/content_pe_cache_v1`

Rows:

- train: `20000/20000`, cache misses `0`
- val: `20000/20000`, cache misses `0`

Identity fields remain audit/load-only. Filename, path, extension, directory,
source hash, sample id, split, and row order are not model features.

## Implementation

Loop45 extends the existing Loop42 strict OOF gate instead of introducing a
separate gate implementation:

- `scripts/train_loop42_oof_residual_gate.py`
  - added `--include-region-ngram`
  - added region n-gram OOF score generation via Loop44's region extractor
  - added `--candidate-model-candidates __none__` sentinel for region-only gate
    runs
  - added smoke guard: Val subsets below `20000` rows cannot report
    `eligible_for_test10k`
- `tests/test_loop42_oof_residual_gate.py`
  - verifies region OOF score alignment
  - verifies the `__none__` candidate sentinel

The gate protocol remains Loop42's strict protocol:

1. Stage-2 base train scores are out-of-fold.
2. Region n-gram candidate train scores are out-of-fold.
3. Base and candidate thresholds are selected on train OOF.
4. Gate training targets are computed only from train OOF correction/harm.
5. Val selects gate model and gate threshold.
6. Test-10k is allowed only if full Val reaches the hard error gate.

## Command

```powershell
.\vnev\Scripts\python.exe scripts\train_loop42_oof_residual_gate.py `
  --checkpoint models\random_20w_8192\best_model.pt `
  --train-predictions reports\random_20w_split\loop27_train_predictions.csv `
  --val-predictions reports\random_20w_split\loop27_val_predictions.csv `
  --baseline-val-predictions reports\random_20w_split\stage2_loop28_content_pe_valonly\stage2_val_predictions.csv `
  --output-dir reports\random_20w_split\loop45_oof_region_gate_valonly `
  --content-pe-features `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --candidate-model-candidates __none__ `
  --include-region-ngram `
  --region-ngram-n-features 2097152 `
  --region-ngram-window 1024 `
  --region-ngram-tail-window 1024 `
  --region-ngram-min 2 `
  --region-ngram-max 5 `
  --region-ngram-stride 2 `
  --region-ngram-alpha 3e-6 `
  --region-ngram-epochs 2 `
  --region-ngram-batch-size 256 `
  --region-ngram-include-prefix-features `
  --region-ngram-include-byte-hist `
  --region-ngram-include-cache-features `
  --folds 5 `
  --gate-model-candidates gate_logreg_balanced_c0.25,gate_logreg_balanced_c1,gate_hgb_leaf7 `
  --thresholds 0.20:0.80:0.005 `
  --gate-thresholds 0.50:0.95:0.01 `
  --neutral-weight 0.05 `
  --seed 45
```

## Result

Best Val candidate:

- candidate: `region_byte_ngram_sgd`
- gate: `gate_logreg_balanced_c0.25`
- base train threshold: `0.54`
- candidate train threshold: `0.57`
- gate threshold: `0.94`
- Val F1: `0.9905113863`
- Val errors: `190`
- FP/FN: `107/83`
- overrides: `90`

Train OOF target summary:

- beneficial overrides: `92`
- harmful overrides: `676`
- neutral rows: `19232`
- base OOF errors: `248`
- candidate OOF errors: `832`

Comparison:

| Candidate | Val F1 | Errors | FP/FN | Decision |
| --- | ---: | ---: | ---: | --- |
| Loop28 content PE locked reference | `0.9919048571` | `162` | `87/75` | Current best |
| Loop45 region OOF gate | `0.9905113863` | `190` | `107/83` | Reject |

Loop45 improves the internal OOF gate base by `3` Val errors, but remains `28`
errors worse than the Loop28 locked reference. The candidate has many more
harmful than beneficial train OOF override opportunities, so the gate learns to
override very sparsely.

## Decision

Reject for Test-10k.

The report's `test_gate_decision` is `reject_val_margin_too_small`, with
`test10k_error_gate=152`. No Test-10k or full-test evaluation was run.

This closes the immediate "use region n-gram through OOF residual gate" branch:
the signal is real but not currently clean enough. Future work should focus on
new information sources or higher-quality parsing rather than another shallow
blend of the same region n-gram scores.

## Artifacts

- Report:
  `reports/random_20w_split/loop45_oof_region_gate_valonly/loop42_oof_residual_gate_report.json`
- Val predictions:
  `reports/random_20w_split/loop45_oof_region_gate_valonly/loop42_oof_residual_gate_val_predictions.csv`

Generated model/prediction artifacts are not committed because they are large
experiment outputs.

## Verification

Commands run:

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\train_loop42_oof_residual_gate.py scripts\train_stage2_cache_matrix.py scripts\train_loop44_region_byte_ngram.py
.\vnev\Scripts\python.exe -m pytest tests\test_loop42_oof_residual_gate.py tests\test_loop44_region_byte_ngram.py tests\test_identity_feature_guard.py
```

Result: `14 passed`.
