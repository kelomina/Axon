# Phase 2 Loop 12: Fresh Replacement Candidate Pool

## Scope

Loop 11 produced a duplicate-source auto cleanup plan with `9` Train/Val rows
that can be safely replaced without mutating frozen Test. Loop 12 verifies that
fresh raw PE candidates exist outside the current 20w split, then materializes a
candidate corrected split and runs the replacement/cache gates.

This loop does not train a model and does not use full-test errors to tune any
threshold, blend weight, feature mask, calibrator, or hyperparameter.

## Candidate Pool

Existing tool validated:

- Script: `scripts/build_replacement_candidate_pool.py`
- Test: `tests/test_build_replacement_candidate_pool.py`

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_replacement_candidate_pool.py `
  --data-dir data `
  --split-csv reports\random_20w_split\random_20w_split.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --output-csv reports\random_20w_split\duplicate_source_replacement_candidates.csv `
  --output-json reports\random_20w_split\duplicate_source_replacement_candidates.json `
  --required-label0 1 `
  --required-label1 8 `
  --max-candidates-per-label 50 `
  --hash-files
```

Result:

- Candidate rows: `100`
- Label counts: `0=50`, `1=50`
- Required replacements: `0=1`, `1=8`
- Replacement shortfall: none
- Enough for required replacements: `true`
- Cache-present candidates: `0`

Interpretation: there are enough fresh raw PE candidates outside the active 20w
split, but they need feature-cache extraction before the corrected split can be
used for training/evaluation.

## Corrected Split Materialization

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_corrected_split_from_plan.py `
  --split-csv reports\random_20w_split\random_20w_split.csv `
  --plan-csv reports\random_20w_split\duplicate_source_auto_cleanup_plan.csv `
  --candidate-csv reports\random_20w_split\duplicate_source_replacement_candidates.csv `
  --output-csv reports\random_20w_split\duplicate_source_corrected_split.csv `
  --output-json reports\random_20w_split\duplicate_source_corrected_split_summary.json `
  --seed 42
```

Result:

- Original rows: `200000`
- Corrected rows: `200000`
- Split counts preserved: `train=20000`, `val=20000`, `test=160000`
- Label balance preserved:
  - Train: `0=10000`, `1=10000`
  - Val: `0=10000`, `1=10000`
  - Test: `0=80000`, `1=80000`
- Plan rows: `9`
- Excluded rows: `9`
- Selected replacements: `9`
- Replacement shortfall: none

## Replacement Integrity

Command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_corrected_split_replacements.py `
  --original-split-csv reports\random_20w_split\random_20w_split.csv `
  --corrected-split-csv reports\random_20w_split\duplicate_source_corrected_split.csv `
  --plan-csv reports\random_20w_split\duplicate_source_auto_cleanup_plan.csv `
  --output-json reports\random_20w_split\duplicate_source_replacement_integrity.json `
  --detail-output-csv reports\random_20w_split\duplicate_source_replacement_integrity_details.csv `
  --enforce-label-balance `
  --strict
```

Result:

- `replacement_integrity_ok`: `true`
- Plan rows: `9`
- Replacement requests: `9`
- Excluded rows found in original: `9`
- Planned excluded rows removed: `9`
- Excluded rows still present: `0`
- Fresh replacement rows: `9`
- Unplanned original rows removed: `0`
- Test replacement requests: `0`
- Original duplicate key rows: `23`
- Corrected duplicate key rows: `14`
- Duplicate key row delta: `-9`

This confirms the corrected split removes the planned Train/Val duplicate rows
and replaces them with fresh split-external candidates.

## Cache Readiness

Command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_corrected_split_cache_ready.py `
  --split-csv reports\random_20w_split\duplicate_source_corrected_split.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --output-json reports\random_20w_split\duplicate_source_corrected_cache_ready.json `
  --missing-cache-output reports\random_20w_split\duplicate_source_corrected_missing_cache.csv `
  --enforce-label-balance `
  --strict
```

Expected blocking result:

- Total rows: `200000`
- Covered rows: `199991`
- Missing rows: `9`
- Missing labels: `0=1`, `1=8`
- Missing splits: `train=3`, `val=6`
- Missing reason: `manifest_missing=9`
- Shape failures: none
- Label balance drift: none
- `cache_ready`: `false`

Interpretation: the split is structurally valid, but the `9` fresh replacement
rows need bounded feature-cache extraction before Val evaluation.

## Code Fixes

Two exact-row matching fixes were needed:

1. `build_corrected_split_from_plan.py`
   - Plans with `sample_index + source_path` now match the exact split row
     first.
   - This prevents duplicate-source cleanup from excluding the canonical row
     with the same sha-like identity.

2. `audit_corrected_split_replacements.py`
   - Replacement audit also uses exact-row matching for planned exclusions.
   - This prevents a kept canonical duplicate from being counted as an excluded
     row still present.

## Test Coverage

Validation commands:

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_build_replacement_candidate_pool.py -q
.\vnev\Scripts\python.exe -m pytest tests\test_audit_corrected_split_replacements.py tests\test_build_corrected_split_from_plan.py -q
```

Results:

- Candidate pool tests: `3 passed`
- Corrected split/replacement tests: `13 passed`

## Decision

The duplicate-source Train/Val auto-cleanup path is now executable up to cache
readiness:

1. Fresh candidates exist outside the current 20w split.
2. Corrected split keeps exactly `200000` rows.
3. Replacement integrity is clean.
4. Cache is intentionally not ready until the `9` fresh rows are extracted.

Next required step: run a bounded cache extraction/recovery for
`reports/random_20w_split/duplicate_source_corrected_missing_cache.csv`, then
rerun cache readiness. Only after `cache_ready=true` can this corrected split be
used for full Val evaluation.
