# Phase 2 Loop 22: Corrected Split Plan Guard

## Scope

Loop 22 adds a pre-build safety gate to the corrected split builder.

This loop does not train, relabel, replace samples, edit the original split,
edit the cache, tune thresholds, tune blend weights, tune feature masks, run
Test-10k, or touch the full-test split. It only prevents unresolved or held-out
manual plan rows from being accepted by the split-generation step.

Conflict note: the project guidance prefers explanation and confirmation before
new code. The active Phase 2 objective is already in an authorized hardening
loop, and this is a narrow safety fix to prevent incomplete human decisions
from entering training data. I proceeded and documented the decision here.

## Issue

After Loop 21, `apply_manual_review_verdicts.py` correctly emits
`needs_manual_target_label` when a reviewer marks `label_wrong` but does not
provide an explicit corrected label.

The next script, `scripts/build_corrected_split_from_plan.py`, previously did
not reject such unresolved rows up front. In many cases it would silently keep
the original row and continue. That is safer than applying a bad relabel, but
still too permissive for this workflow: a half-complete manual plan should not
produce a corrected split artifact that looks training-ready.

The same principle applies to held-out test verdicts and inconsistent
replacement flags. These must be rejected before a corrected split is written,
not merely discovered by later audits.

## Fix

Changed:

- `scripts/build_corrected_split_from_plan.py`

Added `validate_plan_rows()` before any split mutation. The builder now rejects:

- `needs_manual_target_label`
- `held_out_test_verdict_only`
- any plan row whose `split` is `test`
- unsupported or blank `plan_action`
- `relabel` rows not marked usable for training policy
- `relabel` rows without `planned_label` equal to `0` or `1`
- `exclude_and_replace` rows without `replacement_required=true`
- replacement rows without `replacement_label` equal to `0` or `1`
- `replacement_required=true` on non-replacement actions

This moves the guard from "post-build audit might catch it" to "unsafe plan
cannot emit a corrected split."

## Regression Tests

Changed:

- `tests/test_build_corrected_split_from_plan.py`

Added coverage:

- unresolved relabel target plans are rejected before split generation
- test-split plan rows are rejected before split generation
- `replacement_required=true` on a non-replacement action is rejected

Command:

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_build_corrected_split_from_plan.py -q
```

Result:

- `8 passed`

Broader related command:

```powershell
.\vnev\Scripts\python.exe -m pytest `
  tests\test_build_corrected_split_from_plan.py `
  tests\test_apply_manual_review_verdicts.py `
  tests\test_audit_corrected_split_replacements.py `
  tests\test_audit_corrected_split_cache_ready.py `
  tests\test_audit_manual_review_package_readiness.py -q
```

Result:

- `41 passed`

## No-Op Corrected Split Reverification

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_corrected_split_from_plan.py `
  --split-csv reports\random_20w_split\duplicate_source_corrected_split.csv `
  --plan-csv reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_empty_manual_adjustment_plan_loop21.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_empty_corrected_split_loop22.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_empty_corrected_split_loop22.json
```

Result:

- Original rows: `200000`
- Corrected rows: `200000`
- Split counts: `train=20000`, `val=20000`, `test=160000`
- Label counts: `0=100000`, `1=100000`
- Per-split label balance:
  - train: `0=10000`, `1=10000`
  - val: `0=10000`, `1=10000`
  - test: `0=80000`, `1=80000`
- Plan rows: `0`
- Excluded rows: `0`
- Relabeled rows: `0`
- Selected replacements: `0`
- Replacement shortfall: none

Interpretation: the empty plan still produces a no-op corrected split with the
exact 20w invariant preserved.

## Cache Readiness Reverification

Command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_corrected_split_cache_ready.py `
  --split-csv reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_empty_corrected_split_loop22.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --missing-cache-output reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_empty_corrected_split_loop22_missing_cache.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_empty_corrected_split_loop22_cache_ready.json `
  --strict `
  --enforce-label-balance
```

Result:

- Total rows: `200000`
- Covered rows: `200000`
- Missing rows: `0`
- Coverage ratio: `1.0`
- Manifest match counts: `source_path=200000`
- Shape failures: none
- Label balance drift: none
- `cache_ready=true`

## Safety Decision

Do not enter Test-10k from Loop 22.

Reasoning:

1. Loop 22 changes only corrected split plan validation.
2. It produces no candidate model, threshold, blend, feature mask, actionable
   manual verdict set, or Val metric gain.
3. The combined queue still has blank manual fields.
4. Incomplete relabels and held-out test verdicts are now blocked before split
   generation.
5. Any later excluded row must still be replaced by exactly one fresh unused
   same-label valid candidate.

The 20w invariant remains unchanged:

```text
200000 = 20000 train + 20000 val + 160000 test
```

## Next Procedure

1. Human reviewers complete the combined manual review queue.
2. Rerun readiness with `--strict`; require exit code `0`.
3. Build a manual adjustment plan.
4. If the plan contains unresolved rows, return them to human review; do not
   build a corrected split.
5. If the plan is clean, build the corrected split, then run replacement
   integrity and strict cache readiness audits before any Train/Val rerun.
