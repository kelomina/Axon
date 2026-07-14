# Phase 3 Loop84 Content Rescue Separability

## Purpose

Loop84 tests whether non-identity content features can separate two critical
Val groups from Loop82:

- `calibrator_only_correct`: `56` rows where the calibrator fixes Loop57
- `loop57_only_correct`: `463` rows where the calibrator breaks Loop57

This is a Val-only separability diagnostic. It does not train a production
fusion model and does not access Test/Test-10k.

## Identity Policy

`source_sha256` is used only to map Loop82 overlap rows back to cache-backed Val
prediction rows. It is not a model feature.

The probe uses cache-backed PE/stat/lightweight/byte-summary/content-PE
features. The Stage-2 feature names are checked with identity guard. The first
six probability columns are explicitly dropped so this is not another score
delta experiment.

Forbidden as model evidence:

- filename
- path
- extension
- directory
- hash / `source_sha256`
- `sample_index`
- split
- row order

## Commands

Guard:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\analyze_loop84_content_rescue_separability.py `
  --output-json reports\random_20w_split\loop82_same_manifest_val\loop84_content_rescue_guard.json `
  --allow-risk npz_array_load `
  --allow-risk torch_load_checkpoint
```

Result: `decision=pass`.

Tests:

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_analyze_loop84_content_rescue_separability.py tests\test_identity_feature_guard.py -q
```

Result: `6 passed`.

Probe:

```powershell
.\vnev\Scripts\python.exe scripts\analyze_loop84_content_rescue_separability.py `
  --overlap-csv reports\random_20w_split\loop82_same_manifest_val\loop82_val_complementarity_overlap.csv `
  --base-predictions reports\random_20w_split\loop27_val_predictions.csv `
  --checkpoint models\random_20w_8192\best_model.pt `
  --content-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --output-json reports\random_20w_split\loop82_same_manifest_val\loop84_content_rescue_separability.json `
  --folds 5 `
  --seed 8401
```

## Result

Rows:

- Loop82 overlap rows: `20000`
- focus rows: `519`
- regression label `0`: `463`
- rescue label `1`: `56`
- matrix kept rows: `519/519`
- skipped missing cache: `0`
- content matrix shape after dropping probability features: `519 x 1514`

Best selector by F1:

- model: `logreg_balanced_c0.10`
- AUC: `0.6821582844800986`
- precision: `0.24210526315789474`
- recall: `0.4107142857142857`
- F1: `0.304635761589404`
- TP/TN/FP/FN: `23 / 391 / 72 / 33`
- errors on focus task: `105`

Other models:

- `logreg_balanced_c1`: AUC `0.6614085158901574`, recall `0.3392857142857143`
- `extra_trees_100_leaf3`: AUC `0.7681656896019746`, recall `0.125`
- `hgb_leaf3`: AUC `0.6526149336624498`, recall `0.08928571428571429`

The AUC signal is not zero, but it is not operationally sufficient. The only
model with recall above `0.4` has too many false positives against the
Loop57-only-correct group. The model with best AUC captures only `7/56` rescue
rows at threshold `0.5`.

## Verdict

Reject the current content-feature rescue selector path. It does not provide a
reliable enough way to trust the calibrator only when it helps Loop57.

Do not run Test-10k. Do not train a full fusion model from this selector. The
next useful work should shift toward:

- noise review of persistent Val/Test errors,
- stronger external evidence,
- or new content features that are specifically designed for the rescue vs
  regression distinction.

The current evidence does not support further tuning of the same calibrator
fusion route.
