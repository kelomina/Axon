# Phase 2 Loop 9: Corrected Split Replacement Integrity

## Scope

Loop 8 proved that a blank Val adjudication queue does not create any split
mutation. Loop 9 adds a post-build integrity gate for the future point where
manual verdicts are filled and `build_corrected_split_from_plan.py` emits a
corrected split.

This loop does not change labels, splits, thresholds, blend weights, feature
masks, calibrators, model hyperparameters, or test-set artifacts.

## Tooling Added

- Script: `scripts/audit_corrected_split_replacements.py`
- Test: `tests/test_audit_corrected_split_replacements.py`

The script audits three promises that matter for noisy-file cleanup:

1. Excluded or feature-broken rows must disappear from the corrected split.
2. Replacement rows must be fresh rows not present in the original split.
3. Replacement counts must match replacement requests by split and label.

It also checks that Train/Val/Test shape is still exact and, when requested,
that label balance is unchanged.

## Command

```powershell
.\vnev\Scripts\python.exe scripts\audit_corrected_split_replacements.py `
  --original-split-csv reports\random_20w_split\random_20w_split.csv `
  --corrected-split-csv reports\random_20w_split\corrected_manual_review_split.csv `
  --plan-csv reports\random_20w_split\manual_review_adjustment_plan.csv `
  --output-json reports\random_20w_split\corrected_manual_review_replacement_integrity.json `
  --detail-output-csv reports\random_20w_split\corrected_manual_review_replacement_integrity_details.csv `
  --enforce-label-balance `
  --strict
```

## Current Dry-Run Result

Current input plan is empty, so this is a baseline gate check rather than a data
cleanup action.

Result:

- `replacement_integrity_ok`: `true`
- Original rows: `200000`
- Corrected rows: `200000`
- Split counts: `train=20000`, `val=20000`, `test=160000`
- Label balance:
  - Train: `0=10000`, `1=10000`
  - Val: `0=10000`, `1=10000`
  - Test: `0=80000`, `1=80000`
- Plan rows: `0`
- Replacement requests: `0`
- Relabel requests: `0`
- Fresh replacement rows: `0`
- Unplanned original rows removed: `0`
- Test replacement requests: `0`
- Test relabel requests: `0`
- Integrity failures: none

Historical duplicate key note:

- Original duplicate source-key rows: `23`
- Corrected duplicate source-key rows: `23`
- Duplicate source-key delta: `0`

Interpretation: this gate does not introduce or hide replacements. It also
surfaces an existing duplicate-key data-quality issue for later noise audit, but
does not treat unchanged historical duplicates as a replacement failure.

## Test Coverage

Validation command:

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_audit_corrected_split_replacements.py -q
```

Result: `7 passed`.

Covered cases:

- Empty plan preserves integrity.
- Fresh same split/label replacement is accepted.
- Self-replacement is rejected.
- Unplanned original-row removal is rejected.
- Relabel requests must be reflected in the corrected split.
- Historical duplicate source keys are reported without blocking when unchanged.
- Newly introduced duplicate source keys are rejected.

## Placement In The Cleanup Pipeline

After human/business verdicts are filled for the Val queue, the safe sequence is:

1. Run `audit_manual_review_package_readiness.py --strict`.
2. Run `apply_manual_review_verdicts.py` to produce a non-destructive plan.
3. Run `build_corrected_split_from_plan.py` to build the corrected split.
4. Run `audit_corrected_split_replacements.py --strict --enforce-label-balance`.
5. Run `audit_corrected_split_cache_ready.py --strict --enforce-label-balance`.
6. If cache rows are missing, recover or extract fresh cache rows and re-run the
   cache audit.
7. Only then run full Val evaluation.

## Decision

This gate should be mandatory for any future corrected split. It directly
addresses the replacement rule: bad or feature-broken files are not used to fill
their own slots, and a corrected split must still contain exactly `200000`
samples with the required `20000 / 20000 / 160000` shape before any model-side
validation resumes.

Full-test evidence remains held out and must not be used to tune thresholds,
blend weights, feature masks, calibrators, replacement policy, or manual label
policy.
