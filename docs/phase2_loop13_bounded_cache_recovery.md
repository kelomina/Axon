# Phase 2 Loop 13: Bounded Cache Recovery

## Scope

Loop 12 built a duplicate-source corrected split that replaced `9` Train/Val
duplicate rows with fresh raw PE candidates outside the active 20w split. Cache
readiness was blocked because those `9` new rows had not yet been extracted.

Loop 13 performs bounded cache recovery for only those `9` rows, then reruns the
strict cache and duplicate-source audits.

No model training, threshold tuning, blend-weight tuning, feature-mask tuning,
or full-test feedback was used.

## Inputs

- Corrected split: `reports/random_20w_split/duplicate_source_corrected_split.csv`
- Missing cache CSV: `reports/random_20w_split/duplicate_source_corrected_missing_cache.csv`
- Checkpoint config source: `models/random_20w_8192/best_model.pt`
- Cache directory: `data/.cache`
- Manifest: `data/.cache/manifest_38672ba0.json`

The missing-cache CSV contains `9` rows:

- `train`: `3`
- `val`: `6`
- label `0`: `1`
- label `1`: `8`

## Dry Run

Command:

```powershell
.\vnev\Scripts\python.exe scripts\recover_missing_feature_cache.py `
  --missing-csv reports\random_20w_split\duplicate_source_corrected_missing_cache.csv `
  --checkpoint models\random_20w_8192\best_model.pt `
  --cache-dir data\.cache `
  --workers 1 `
  --backend thread `
  --dry-run `
  --storage-format uncompressed `
  --output-json reports\random_20w_split\duplicate_source_missing_cache_recovery_dry_run.json
```

Result:

- `planned_rows`: `9`
- `status_counts`: `would_extract=9`
- `cache_config_hash`: `38672ba0`
- `manifest_path`: `data/.cache/manifest_38672ba0.json`
- `storage_format`: `uncompressed`

## Recovery

Command:

```powershell
.\vnev\Scripts\python.exe scripts\recover_missing_feature_cache.py `
  --missing-csv reports\random_20w_split\duplicate_source_corrected_missing_cache.csv `
  --checkpoint models\random_20w_8192\best_model.pt `
  --cache-dir data\.cache `
  --workers 1 `
  --backend thread `
  --storage-format uncompressed `
  --progress-interval 1 `
  --output-json reports\random_20w_split\duplicate_source_missing_cache_recovery.json
```

Result:

- `input_rows`: `9`
- `status_counts`: `extracted=9`
- `manifest_added`: `9`
- Failed examples: none

## Cache Readiness After Recovery

Command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_corrected_split_cache_ready.py `
  --split-csv reports\random_20w_split\duplicate_source_corrected_split.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --output-json reports\random_20w_split\duplicate_source_corrected_cache_ready_after_recovery.json `
  --missing-cache-output reports\random_20w_split\duplicate_source_corrected_missing_cache_after_recovery.csv `
  --enforce-label-balance `
  --strict
```

Result:

- `cache_ready`: `true`
- Total rows: `200000`
- Covered rows: `200000`
- Missing rows: `0`
- Coverage ratio: `1.0`
- Split counts: `train=20000`, `val=20000`, `test=160000`
- Label balance preserved:
  - Train: `0=10000`, `1=10000`
  - Val: `0=10000`, `1=10000`
  - Test: `0=80000`, `1=80000`
- Shape failures: none
- Label balance drift: none

## Duplicate-Source Audit After Recovery

Command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_split_duplicate_sources.py `
  --split-csv reports\random_20w_split\duplicate_source_corrected_split.csv `
  --output-json reports\random_20w_split\duplicate_source_corrected_duplicate_sources.json `
  --output-csv reports\random_20w_split\duplicate_source_corrected_duplicate_sources.csv
```

Result:

- Duplicate groups: `14`
- Duplicate extra rows: `14`
- Cross-label groups: `4`
- Cross-split groups: `2`
- Same-path duplicate groups: `0`

Change versus original split:

- Duplicate groups: `23 -> 14`
- Duplicate extra rows: `23 -> 14`
- Cross-split groups: `9 -> 2`
- Cross-label groups: unchanged at `4`

Interpretation: the automatic Train/Val duplicate cleanup removed the safe
same-label duplicate rows. Remaining duplicate identities require manual review
or involve frozen Test boundaries.

## Current Status

The duplicate-source corrected split is now structurally ready for full Val
evaluation:

1. Exactly `200000` rows.
2. Exact `20000 / 20000 / 160000` split shape.
3. Per-split label balance preserved.
4. Replacement integrity passed.
5. Cache coverage is `200000 / 200000`.

It is not a final cleaned dataset because:

- `14` duplicate groups remain.
- `4` remaining groups are cross-label conflicts.
- `2` remaining groups cross split boundaries.
- Manual adjudication is still required for cross-label and frozen-Test cases.

## Decision

Proceed next to full Val evaluation on
`reports/random_20w_split/duplicate_source_corrected_split.csv` only. Do not use
full-test feedback for threshold, blend, mask, calibrator, or hyperparameter
selection.

If Val improves, the candidate can enter the normal funnel. If Val does not
improve, treat the result as evidence that duplicate-source cleanup alone is not
enough and continue Phase 2 noise adjudication.
