# Phase 2 Loop 21: Explicit Relabel Target Gate

## Scope

Loop 21 tightens the manual adjustment planner so that relabeling never invents
a target label.

This loop does not train, relabel, replace samples, edit the split, edit the
cache, tune thresholds, tune blend weights, tune feature masks, run Test-10k,
or touch the full-test split. It only prevents the planning tool from turning a
human "label is wrong" signal into an automatic binary flip.

## Issue

`scripts/apply_manual_review_verdicts.py` previously inferred a relabel target
as follows:

1. If `corrected_label`, `new_label`, `target_label`, or `manual_label` was
   present, use it.
2. Otherwise, if the current label was `0` or `1`, automatically use
   `1 - current_label`.

That second step is unsafe for the Phase 2 noise workflow. A reviewer saying
`label_wrong` is not the same as explicitly saying the corrected class is the
opposite label. In this project, neighbor evidence and model confidence are
signals only; they are not labels.

## Fix

Changed:

- `scripts/apply_manual_review_verdicts.py`
- `docs/manual_review_adjudication_workflow.md`

New behavior:

- Relabel actions require an explicit corrected target label in one of:
  - `corrected_label`
  - `new_label`
  - `target_label`
  - `manual_label`
- If a row has `label_wrong` / `relabel_train_only` but no target label, the
  plan row becomes:
  - `plan_action=needs_manual_target_label`
  - `planned_label=original_label`
  - `usable_for_training_policy=false`
- The planner no longer auto-flips `0` to `1` or `1` to `0`.

This matches the user's constraint: do not invent labels.

## Regression Tests

Changed:

- `tests/test_apply_manual_review_verdicts.py`

Added or updated coverage:

- A relabel verdict without a target label requires manual target completion.
- A relabel verdict with explicit `corrected_label=1` produces an actionable
  relabel.
- The SHA/path fallback matching test now includes an explicit corrected label,
  so it does not depend on auto-flip behavior.

Command:

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_apply_manual_review_verdicts.py -q
```

Result:

- `10 passed`

Broader related test command:

```powershell
.\vnev\Scripts\python.exe -m pytest `
  tests\test_apply_manual_review_verdicts.py `
  tests\test_audit_manual_review_package_readiness.py `
  tests\test_build_combined_manual_review_queue.py `
  tests\test_build_corrected_split_from_plan.py `
  tests\test_audit_corrected_split_replacements.py -q
```

Result:

- `36 passed`

## No-Op Plan Reverification

Command:

```powershell
.\vnev\Scripts\python.exe scripts\apply_manual_review_verdicts.py `
  --review-csv reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_manual_review.csv `
  --split-csv reports\random_20w_split\duplicate_source_corrected_split.csv `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_empty_manual_adjustment_plan_loop21.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_empty_manual_adjustment_plan_loop21.json
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

Interpretation: the combined queue is still a no-op while manual fields are
blank. The stricter relabel target rule only affects future filled verdicts.

## Safety Decision

Do not enter Test-10k from Loop 21.

Reasoning:

1. Loop 21 changes only manual-adjustment planning semantics.
2. It produces no candidate model, threshold, blend, feature mask, replacement
   set, or Val metric gain.
3. The combined queue still has blank manual fields.
4. No labels were invented or auto-applied.
5. Any later `label_wrong` row must include a clear target label before it can
   affect training policy.

The 20w invariant remains unchanged:

```text
200000 = 20000 train + 20000 val + 160000 test
```

## Next Procedure

1. Human reviewers fill `manual_label_verdict`, `manual_verdict_note`, and
   `recommended_action`.
2. For every `label_wrong` row, also fill one explicit target label column:
   `corrected_label`, `new_label`, `target_label`, or `manual_label`.
3. Rerun readiness with `--strict`.
4. Convert filled verdicts into a non-destructive adjustment plan.
5. Rows that still produce `needs_manual_target_label` must go back to human
   review before any corrected Train/Val evaluation.
