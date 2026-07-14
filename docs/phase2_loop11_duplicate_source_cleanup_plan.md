# Phase 2 Loop 11: Duplicate Source Cleanup Plan

## Scope

Loop 10 found duplicate source identities in the current 20w split:

- Duplicate groups: `23`
- Duplicate extra rows: `23`
- Cross-label groups: `4`
- Cross-split groups: `9`

Loop 11 turns that read-only audit into a non-destructive cleanup plan. It does
not mutate the split, labels, cache, thresholds, blend weights, feature masks,
calibrators, model hyperparameters, or test-set artifacts.

## Tooling Added

- Script: `scripts/build_duplicate_source_cleanup_plan.py`
- Test: `tests/test_build_duplicate_source_cleanup_plan.py`

The script consumes the duplicate-source detail CSV and emits two artifacts:

1. Auto cleanup plan for safe same-label duplicate rows.
2. Manual review queue for rows that cannot be safely auto-handled.

## Command

```powershell
.\vnev\Scripts\python.exe scripts\build_duplicate_source_cleanup_plan.py `
  --duplicate-csv reports\random_20w_split\random_20w_split_duplicate_sources.csv `
  --output-plan-csv reports\random_20w_split\duplicate_source_auto_cleanup_plan.csv `
  --output-review-csv reports\random_20w_split\duplicate_source_cross_label_review_queue.csv `
  --output-json reports\random_20w_split\duplicate_source_cleanup_plan_summary.json `
  --keep-policy protect_test_then_val
```

Default policy:

- `freeze_test=true`
- keep policy: `protect_test_then_val`

Meaning:

- Same-label duplicate groups that can be fixed by replacing Train/Val rows are
  converted to `exclude_and_replace` plan rows.
- Cross-label duplicate groups go to manual review.
- Same-label duplicate groups that would require mutating frozen Test go to
  review, not to the auto plan.

## Current Result

Input duplicate groups: `23`

Auto cleanup:

- Auto plan rows: `9`
- Planned replacements by split:
  - `train`: `3`
  - `val`: `6`
- Planned replacements by label:
  - label `0`: `1`
  - label `1`: `8`

Manual review:

- Manual review rows: `28`
- Manual-review groups:
  - Cross-label duplicate groups: `4`
  - Same-label groups requiring frozen Test mutation: `10`

Group action counts:

- `auto_replace_duplicates`: `9`
- `manual_review_required`: `4`
- `manual_review_required_frozen_test`: `10`

## Materialization Dry Run

Attempted command:

```powershell
.\vnev\Scripts\python.exe scripts\build_corrected_split_from_plan.py `
  --split-csv reports\random_20w_split\random_20w_split.csv `
  --plan-csv reports\random_20w_split\duplicate_source_auto_cleanup_plan.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --output-csv reports\random_20w_split\duplicate_source_corrected_split.csv `
  --output-json reports\random_20w_split\duplicate_source_corrected_split_summary.json `
  --seed 42
```

Result: blocked as expected.

```text
ValueError: Not enough unused same-label replacement candidates: {"0": 1, "1": 8}
```

Interpretation:

The current manifest only covers the active 20w cache. It cannot provide fresh
replacement candidates outside the original split. This is the correct failure
mode because excluded duplicate rows must not be filled by samples already in
the 20w split.

To materialize the auto cleanup plan, the replacement source must be one of:

- raw data directory scan with enough unused valid PE files, or
- a candidate CSV built from valid files outside the current 20w split, or
- a newly extracted cache manifest that includes fresh replacement candidates.

## Test Coverage

Validation command:

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_build_duplicate_source_cleanup_plan.py -q
```

Result: `5 passed`.

Covered cases:

- Same-label duplicate creates one replacement plan row when Test is not
  mutated.
- Cross-label duplicate goes to review only.
- Keep policy can explicitly prefer Train when Test replacement is allowed.
- Same-label duplicate that would mutate frozen Test goes to review by default.
- Summary JSON is written.

## Decision

Do not use the current 20w cache manifest as a replacement pool for duplicate
cleanup. It has no unused same-label candidates outside the active split.

The next executable data step is to build a fresh candidate pool from raw valid
PE files outside the current split, then rerun:

1. `build_corrected_split_from_plan.py`
2. `audit_corrected_split_replacements.py --strict --enforce-label-balance`
3. `audit_corrected_split_cache_ready.py --strict --enforce-label-balance`
4. full Val evaluation

Manual review is still required for cross-label duplicate identities and for
groups that would require changing frozen Test. Full-test evidence remains held
out and must not be used to tune thresholds, blend weights, feature masks,
calibrators, or model hyperparameters.
