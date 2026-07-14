# Phase 3 Loop43: Content-Cross Feature Probe

Date: 2026-07-02

## Objective

Loop43 tested whether narrow content-derived interaction features can improve
Loop28 residuals better than broad PE v2 expansion. The feature family targets
the error themes found by Loop32/38/39:

- DLL/driver-like malicious false negatives.
- security-directory plus overlay/export/exception/debug patterns.
- section, entrypoint, entropy, and packer-like structure.
- import/API semantic combinations.
- resource/export structure.

This loop is Val-only. It does not evaluate Test-10k or full-test.

## Protocol

Inputs:

- checkpoint: `models/random_20w_8192/best_model.pt`
- train predictions: `reports/random_20w_split/loop27_train_predictions.csv`
- val predictions: `reports/random_20w_split/loop27_val_predictions.csv`
- content PE cache: `reports/random_20w_split/content_pe_cache_v1`
- content PE v2 cache: `reports/random_20w_split/content_pe_v2_cache`

Rows:

- train: `20000/20000`
- val: `20000/20000`
- skipped missing cache: `0`

Identity fields remain audit/load-only. Filename, path, extension, directory,
source hash, sample id, split, and row order are not model features.

## Implementation

New tooling:

- `scripts/train_loop43_content_cross.py`
- `tests/test_loop43_content_cross.py`

The script reuses the existing Stage-2 matrix and appends `66` hand-built
content-cross features derived from PE v1 and PE v2 sidecar arrays. The final
feature dimension is:

- base feature dim: `1520`
- content-cross dim: `66`
- total dim: `1586`

## Fast Val Probe

Command:

```powershell
.\vnev\Scripts\python.exe scripts\train_loop43_content_cross.py `
  --checkpoint models\random_20w_8192\best_model.pt `
  --train-predictions reports\random_20w_split\loop27_train_predictions.csv `
  --val-predictions reports\random_20w_split\loop27_val_predictions.csv `
  --output-dir reports\random_20w_split\loop43_content_cross_valonly `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --content-pe-v2-cache-dir reports\random_20w_split\content_pe_v2_cache `
  --model-candidates hgb_lr0.06_leaf31_l2_0,hgb_lr0.08_leaf31_l2_1e-3 `
  --noise-modes none `
  --thresholds 0.35:0.65:0.005 `
  --seed 42
```

Best fast result:

- model: `hgb_lr0.06_leaf31_l2_0__noise_none`
- Val F1: `0.9912131802`
- Val errors: `176`
- FP/FN: `103/73`

This was worse than Loop28's `162` errors.

## Full Candidate Matrix

Command:

```powershell
.\vnev\Scripts\python.exe scripts\train_loop43_content_cross.py `
  --checkpoint models\random_20w_8192\best_model.pt `
  --train-predictions reports\random_20w_split\loop27_train_predictions.csv `
  --val-predictions reports\random_20w_split\loop27_val_predictions.csv `
  --output-dir reports\random_20w_split\loop43_content_cross_full_valonly `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --content-pe-v2-cache-dir reports\random_20w_split\content_pe_v2_cache `
  --noise-modes none,soft_conflict_downweight,trim_extreme_conflict `
  --thresholds 0.35:0.65:0.005 `
  --seed 42
```

Best full-matrix result:

- model: `hgb_lr0.08_leaf31_l2_1e-3__noise_trim_extreme_conflict`
- Val F1: `0.9914180222`
- Val errors: `172`
- FP/FN: `107/65`
- effective train rows: `19983`

This remains worse than Loop28:

| Candidate | Val F1 | Errors | FP/FN | Decision |
| --- | ---: | ---: | ---: | --- |
| Loop28 content PE | `0.9919048571` | `162` | `87/75` | Current best |
| Loop43 fast | `0.9912131802` | `176` | `103/73` | Reject |
| Loop43 full matrix | `0.9914180222` | `172` | `107/65` | Reject |

## Decision

Reject for Test-10k.

Loop43 did what it was meant to test: it converted the residual-analysis ideas
into explicit content-derived interactions and ran both a fast full-Val probe
and a full candidate/noise matrix. The result is negative. The hand-built
interaction features improve some FN behavior but add too many FP, and the best
model is still `10` Val errors worse than Loop28.

The next content work should not continue this exact hand-multiplied interaction
set. Better candidates are:

- real Authenticode/PKCS#7 parsing and signature coverage checks,
- regionized byte n-gram features around entrypoint, resources, imports,
  sections, and overlay,
- parser-quality improvements that distinguish certificate overlay from payload
  overlay.

## Artifacts

- Fast report:
  `reports/random_20w_split/loop43_content_cross_valonly/loop43_content_cross_report.json`
- Full report:
  `reports/random_20w_split/loop43_content_cross_full_valonly/loop43_content_cross_report.json`
- Full Val predictions:
  `reports/random_20w_split/loop43_content_cross_full_valonly/loop43_content_cross_val_predictions.csv`

Generated model/prediction artifacts are not committed because they are large
experiment outputs.

## Verification

Commands run:

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\train_loop43_content_cross.py
.\vnev\Scripts\python.exe -m pytest tests\test_loop43_content_cross.py tests\test_identity_feature_guard.py
```

Result: `5 passed`.
