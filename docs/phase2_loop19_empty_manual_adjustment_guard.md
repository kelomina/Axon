# Phase 2 Loop 19: Empty Manual Adjustment Guard

## Scope

Loop 19 verifies that the combined P0/P1 manual review queue can be passed to
`scripts/apply_manual_review_verdicts.py` safely while all human verdict/action
fields are still blank.

This loop does not train, relabel, replace samples, edit the split, edit the
cache, tune thresholds, tune blend weights, tune feature masks, run Test-10k,
or touch the full-test split. It only hardens the non-destructive adjustment
planning tool before any human-filled verdict package is trusted.

## Issue Found

The first no-op dry run produced the expected destructive-action counts:

- `planned_rows=0`
- `replacement_required=0`
- `training_policy_rows=0`
- `review_rows_in_test_split=0`

However, its review label summary was inconsistent with an independent match
of the same combined queue against the corrected 20w split:

- Tool summary before fix: `val:0=89`, `val:1=48`
- Independent match: `val:0=90`, `val:1=47`

Root cause: the script used one mixed lookup map for both real
`source_sha256` keys and SHA-like aliases derived from path filenames. If one
row's filename stem looked like a 64-character SHA and another row used that
same value as its real `source_sha256`, the path-derived alias could select the
wrong split row.

## Fix

Script changed:

- `scripts/apply_manual_review_verdicts.py`

The split index is now separated by match type:

- `by_sha`: real split `source_sha256`
- `by_path`: normalized split `source_path`
- `by_path_stem_sha`: SHA-like filename stem fallback, only for split rows
  without an explicit `source_sha256`

Lookup order is now:

1. Review row explicit `source_sha256` -> split explicit `source_sha256`
2. Review row normalized `source_path` -> split normalized `source_path`
3. Review SHA or review path-stem SHA -> split path-stem SHA fallback, only
   when the split row has no explicit SHA

This preserves the old fallback where a review SHA can match a split path
filename when the split CSV lacks `source_sha256`, but prevents path aliases
from shadowing real SHA identity.

## Regression Tests

Test file changed:

- `tests/test_apply_manual_review_verdicts.py`

Added coverage:

- Explicit review SHA wins over a colliding path-stem SHA alias.
- Path-stem SHA fallback is ignored when the split row has an explicit real
  SHA.
- Existing behavior still works when review SHA must match a split path
  filename because the split row has no SHA column/value.

Commands:

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_apply_manual_review_verdicts.py -q
```

Result:

- `9 passed`

```powershell
.\vnev\Scripts\python.exe -m pytest `
  tests\test_apply_manual_review_verdicts.py `
  tests\test_build_combined_manual_review_queue.py `
  tests\test_audit_manual_review_package_readiness.py -q
```

Result:

- `18 passed`

## No-Op Plan Reverification

Command:

```powershell
.\vnev\Scripts\python.exe scripts\apply_manual_review_verdicts.py `
  --review-csv reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_manual_review.csv `
  --split-csv reports\random_20w_split\duplicate_source_corrected_split.csv `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_empty_manual_adjustment_plan.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_empty_manual_adjustment_plan.json
```

Result:

- Split rows: `200000`
- Split counts: `train=20000`, `val=20000`, `test=160000`
- Label balance:
  - train: `0=10000`, `1=10000`
  - val: `0=10000`, `1=10000`
  - test: `0=80000`, `1=80000`
- Review rows: `137`
- Planned rows: `0`
- Ignored rows: `137`
- Missing split rows: `0`
- Duplicate review rows: `0`
- Review split counts: `val=137`
- Review label split counts: `val:0=90`, `val:1=47`
- Review rows in test split: `0`
- Replacement required: `0`
- Training policy rows: `0`

Interpretation: empty manual verdict/action fields remain a strict no-op. The
corrected label summary now matches the combined queue's FP/FN distribution and
the independent split match.

## Safety Decision

Do not enter Test-10k from Loop 19.

Reasoning:

1. Loop 19 fixed review-plan identity matching and verified no-op safety.
2. It produced no candidate model, threshold, blend, feature mask, relabel
   decision, replacement set, or Val metric gain.
3. The combined queue still has blank manual fields by design.
4. Neighbor evidence and model confidence remain review signals only; they are
   not labels.
5. Any later `feature_broken`, `corrupt`, `invalid_pe`, or `out_of_scope`
   verdict must trigger fresh same-label replacement. Bad rows must not fill
   their own slots, and the 20w invariant must stay exact.

The 20w invariant remains unchanged:

```text
200000 = 20000 train + 20000 val + 160000 test
```

## Next Procedure

1. Human reviewers fill the combined queue's `manual_label_verdict`,
   `manual_verdict_note`, and `recommended_action` fields.
2. Rerun readiness with strict checks before any plan is accepted.
3. Convert filled verdicts into a non-destructive adjustment plan.
4. For every excluded row, redraw exactly one fresh unused same-label valid
   candidate, then rebuild cache/readiness.
5. Only a corrected Train/Val rerun with clear Val improvement can reopen the
   Test-10k gate.
