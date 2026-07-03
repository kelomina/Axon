# Phase 3 Loop82 Same-Manifest Val Complementarity

## Purpose

Loop82 resolves the Loop81 readiness blocker by re-exporting Loop57 and
probability-calibrator Val predictions from the same corrected 20w Val
manifest. This is still Val-only: no training, no threshold search, and no
Test/Test-10k access.

`source_sha256` is used only to align prediction rows for audit. It is not a
model feature, fusion input, threshold rule, label-correction signal, or
production evidence.

## Inputs

The shared corrected Val inputs are:

- base Val predictions:
  `reports/random_20w_split/loop27_val_predictions.csv`
- Loop57 locked base Val predictions:
  `reports/random_20w_split/stage2_loop28_content_pe_valonly/stage2_val_predictions.csv`
- Loop57 frozen payload:
  `reports/random_20w_split/loop57_fn_overlay_gate_valonly/loop57_fn_overlay_gate_selected_model.pkl`
- probability calibrator:
  `models/random_20w_8192/random20w_replaced_logreg_calibrator.pkl`

Input audit before re-export:

- `loop27_val_predictions.csv`: `20000` rows, split `val=20000`, labels
  `0/1=10000/10000`, unique `source_sha256=20000`, missing cache `0`
- `stage2_loop28_content_pe_valonly/stage2_val_predictions.csv`: `20000` rows,
  split `val=20000`, labels `0/1=10000/10000`, unique `source_sha256=20000`,
  missing cache `0`
- SHA set difference between the two inputs: `0 / 0`, intersection `20000`

## Commands

Guard for calibrator export:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\export_probability_calibrator_predictions.py `
  --output-json reports\random_20w_split\loop82_export_calibrator_guard.json `
  --allow-risk npz_array_load `
  --allow-risk pickle_model_load
```

Result: `decision=pass`.

Calibrator export:

```powershell
.\vnev\Scripts\python.exe scripts\export_probability_calibrator_predictions.py `
  --model models\random_20w_8192\random20w_replaced_logreg_calibrator.pkl `
  --predictions reports\random_20w_split\loop27_val_predictions.csv `
  --output-csv reports\random_20w_split\loop82_same_manifest_val\loop82_calibrator_val_predictions.csv `
  --threshold 0.44 `
  --missing-cache-output reports\random_20w_split\loop82_same_manifest_val\loop82_calibrator_val_missing_cache.csv
```

Result: `Rows: 20000`.

Guard for Loop57 frozen export:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\evaluate_loop57_fn_overlay_gate.py `
  --output-json reports\random_20w_split\loop82_export_loop57_guard.json `
  --allow-risk npz_array_load `
  --allow-risk pickle_model_load
```

Result: `decision=pass`.

Loop57 frozen export:

```powershell
.\vnev\Scripts\python.exe scripts\evaluate_loop57_fn_overlay_gate.py `
  --model reports\random_20w_split\loop57_fn_overlay_gate_valonly\loop57_fn_overlay_gate_selected_model.pkl `
  --predictions reports\random_20w_split\loop27_val_predictions.csv `
  --baseline-predictions reports\random_20w_split\stage2_loop28_content_pe_valonly\stage2_val_predictions.csv `
  --overlay-boundary-cache-dir reports\random_20w_split\loop55_overlay_boundary_cache_train_val `
  --output-json reports\random_20w_split\loop82_same_manifest_val\loop82_loop57_val_eval.json `
  --output-predictions-csv reports\random_20w_split\loop82_same_manifest_val\loop82_loop57_val_predictions.csv `
  --baseline-probability-column stage2_prob_malicious `
  --alignment-key-column source_sha256
```

Result:

- records: `20000/20000`
- external base alignment: `sha_checked=20000`
- Loop57 Val F1: `0.9926635723910765`
- Loop57 Val errors: `147`
- Loop57 Val FP/FN: `92 / 55`

Guard for complementarity audit:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\analyze_loop81_val_complementarity.py `
  --output-json reports\random_20w_split\loop82_same_manifest_val\loop82_val_complementarity_guard.json
```

Result: `decision=pass`.

Strict complementarity audit:

```powershell
.\vnev\Scripts\python.exe scripts\analyze_loop81_val_complementarity.py `
  --loop57-predictions reports\random_20w_split\loop82_same_manifest_val\loop82_loop57_val_predictions.csv `
  --calibrator-predictions reports\random_20w_split\loop82_same_manifest_val\loop82_calibrator_val_predictions.csv `
  --join-key source_sha256 `
  --output-json reports\random_20w_split\loop82_same_manifest_val\loop82_val_complementarity.json `
  --output-overlap-csv reports\random_20w_split\loop82_same_manifest_val\loop82_val_complementarity_overlap.csv `
  --strict
```

Result: strict pass.

## Alignment Result

- Loop57 rows: `20000`
- calibrator rows: `20000`
- Loop57 unique keys: `20000`
- calibrator unique keys: `20000`
- common keys: `20000`
- joined rows: `20000`
- missing Loop57 rows: `0`
- missing calibrator rows: `0`
- label mismatches: `0`
- split mismatches: `0`
- ambiguous common keys: `0`
- duplicate keys: `0` on both sides

This clears the Loop81 readiness blocker.

## Complementarity Result

Loop57 on Val:

- F1: `0.9926635723910766`
- errors: `147`
- FP/FN: `92 / 55`

Calibrator on the same Val rows:

- F1: `0.9723470100828591`
- errors: `554`
- FP/FN: `294 / 260`

Oracle choose-correct-if-either diagnostic:

- F1: `0.9954597615127476`
- errors: `91`
- FP/FN: `67 / 24`

Overlap:

- both correct: `19390`
- both wrong: `91`
- Loop57-only-correct: `463`
- calibrator-only-correct: `56`

The calibrator has real complementary signal: it can recover `56` Loop57 Val
errors. But it also breaks `463` rows that Loop57 already gets right, so it
must not replace Loop57 and must not be naively blended.

## Verdict

Loop82 authorizes only the next Val-first step:

- A conservative fusion probe may be attempted on Val only.
- The fusion probe must not use filename, path, extension, directory,
  `sample_index`, `source_sha256`, split, cache path, or row order as features.
- The probe must target the `56` calibrator-only-correct rows while protecting
  the `463` Loop57-only-correct rows.
- No Test-10k or full-test run is allowed until a Val-only fusion probe clearly
  improves Loop57 under the same strict alignment audit.

The route to `F1 >= 99.9%` is still not proven. Even the oracle diagnostic has
`91` Val errors on 20k, which is useful movement but still far from the
hundred-level full-test error budget needed on 160k.
