# Phase 2 Loop 20: Strict Manual Readiness Gate

## Scope

Loop 20 tightens the command-line gate for manual-review readiness.

This loop does not train, relabel, replace samples, edit the split, edit the
cache, tune thresholds, tune blend weights, tune feature masks, run Test-10k,
or touch the full-test split. It only fixes the meaning of
`audit_manual_review_package_readiness.py --strict` so that later human verdicts
cannot be accidentally skipped.

Conflict note: the project guidance prefers explanation and confirmation before
new code. The active optimization objective requires continued validated
progress, and this change is a narrow safety fix inside the already authorized
Phase 2 review pipeline. I proceeded because it prevents an unsafe automated
transition from review evidence to adjustment planning.

## Issue

Before this loop, the readiness audit reported two separate states:

- `manual_review_ready`: source/cache/PE/top-5 evidence is complete, so a human
  can inspect the row.
- `verdict_package_ready`: evidence is complete and human
  `manual_label_verdict` / `recommended_action` fields are non-empty, valid,
  and mutually consistent.

However, the CLI `--strict` exit condition only checked
`manual_review_ready`. That was too weak for Phase 2. A package with blank
human verdicts could be technically review-ready but still not safe to apply.

## Fix

Changed:

- `scripts/audit_manual_review_package_readiness.py`

New behavior:

- Non-strict mode remains report-only and can exit `0` while reporting blocking
  issues.
- `--strict` now exits non-zero unless `verdict_package_ready=true`.
- The help text now says strict requires both complete evidence and complete,
  consistent manual verdict/action fields.

This makes the CLI match the documented Phase 2 gate:

```text
review_queue_ready=true  -> humans may review
verdict_package_ready=true -> downstream adjustment planning may proceed
```

## Regression Tests

Changed:

- `tests/test_audit_manual_review_package_readiness.py`

Added CLI exit-code coverage:

- `--strict` fails when the review queue is evidence-ready but manual verdicts
  are blank.
- `--strict` passes when the verdict package is fully ready.
- non-strict mode reports blank verdicts without failing.
- `--strict` fails when verdict/action fields are individually valid but
  inconsistent, such as `feature_broken + relabel_train_only`.

Commands:

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_audit_manual_review_package_readiness.py -q
```

Result:

- `10 passed`

```powershell
.\vnev\Scripts\python.exe -m pytest `
  tests\test_audit_manual_review_package_readiness.py `
  tests\test_apply_manual_review_verdicts.py `
  tests\test_build_combined_manual_review_queue.py `
  tests\test_build_manual_review_adjudication_guide.py `
  tests\test_summarize_manual_review_sources.py -q
```

Result:

- `24 passed`

## Combined Queue Reverification

Strict command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_manual_review_package_readiness.py `
  --review-csv reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_manual_review.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_readiness_strict_check.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_readiness_strict_check.json `
  --strict
```

Result:

- Exit code: `2`
- Total rows: `137`
- Ready rows: `137`
- Not-ready rows: `0`
- `review_queue_ready=true`
- `manual_review_ready=true`
- `verdict_package_ready=false`
- Blank `manual_label_verdict`: `137`
- Blank `recommended_action`: `137`
- Blocking issues:
  - `manual_verdict_empty`
  - `recommended_action_empty`

Non-strict command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_manual_review_package_readiness.py `
  --review-csv reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_manual_review.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_readiness_non_strict_check.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_readiness_non_strict_check.json
```

Result:

- Exit code: `0`
- The JSON reports the same blocking issues.

Interpretation: the combined queue is still technically ready for human review,
but it is now correctly blocked at the strict CLI gate until humans fill
verdict/action fields.

## Agent Review

Eval-Agent performed a read-only gate review and confirmed:

- Checking only `manual_review_ready` violates the Phase 2 manual-review gate.
- `manual_review_ready=true` means "safe to show to humans", not "safe to
  execute".
- `--strict` should require `verdict_package_ready=true`.
- CLI exit-code tests are required because function-level summary tests alone
  do not protect downstream shell pipelines.

## Safety Decision

Do not enter Test-10k from Loop 20.

Reasoning:

1. Loop 20 changes only a manual-review readiness gate.
2. It produces no candidate model, threshold, blend, feature mask, relabel
   decision, replacement set, or Val metric gain.
3. The combined queue still has blank manual fields.
4. Neighbor evidence and model confidence remain review signals only; they are
   not labels.
5. Any later exclusion must still be followed by exactly one fresh same-label
   valid replacement per excluded row. Bad files must not be reused to fill
   their own slots.

The 20w invariant remains unchanged:

```text
200000 = 20000 train + 20000 val + 160000 test
```

## Next Procedure

1. Human reviewers fill the combined queue's `manual_label_verdict`,
   `manual_verdict_note`, and `recommended_action` fields.
2. Rerun readiness with `--strict`; it must exit `0` before any plan is
   accepted.
3. Convert filled verdicts into a non-destructive adjustment plan.
4. For every excluded row, redraw exactly one fresh unused same-label valid
   candidate.
5. Run replacement integrity and strict cache readiness audits before any new
   Train/Val evaluation.
