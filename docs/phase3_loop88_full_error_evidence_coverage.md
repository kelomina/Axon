# Phase 3 Loop88 Full-Error Evidence Coverage

## Purpose

Loop88 measures whether the current evidence-review pipeline covers enough of
the full current-best error set to matter for the `F1 >= 0.999` target.

It is read-only. It does not train, tune thresholds, relabel, mutate the split,
mutate cache, generate model features, or authorize Test/Test-10k.

## Inputs

- full current-best error queue:
  `reports/random_20w_split/loop63_persistent_error_review_queue.csv`
- target-gap audit:
  `reports/random_20w_split/loop71_target_gap_noise_roi.json`
- full wave plan:
  `reports/random_20w_split/loop72_full_error_review_wave_plan.csv`
- first evidence package:
  `reports/random_20w_split/loop86_review_evidence_package_summary.json`
- first verdict gate:
  `reports/random_20w_split/loop87_review_evidence_verdict_import.json`
- A-lane content/cache health:
  `reports/random_20w_split/loop63_A_persistent_conflict_content_audit_summary.json`
- duplicate audit:
  `reports/random_20w_split/loop64_manifest_sha_duplicate_audit.json`

## Identity Policy

`source_path`, `cache_path`, `source_sha256`, `sample_index`, split, review
rank, and model score columns are allowed only for loading, alignment, priority,
cache audit, duplicate review, and manual review indexing. They are not model
evidence, verdict evidence, threshold inputs, or replacement sampling keys.

## Commands

Tests:

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_build_loop88_full_error_evidence_coverage.py -q
```

Result: `2 passed`.

Guard:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop88_full_error_evidence_coverage.py `
  --output-json reports\random_20w_split\loop88_full_error_evidence_coverage_guard.json
```

Result: `decision=pass`, static findings `0`.

Coverage report:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop88_full_error_evidence_coverage.py `
  --queue-csv reports\random_20w_split\loop63_persistent_error_review_queue.csv `
  --target-gap-json reports\random_20w_split\loop71_target_gap_noise_roi.json `
  --loop72-wave-csv reports\random_20w_split\loop72_full_error_review_wave_plan.csv `
  --loop86-summary-json reports\random_20w_split\loop86_review_evidence_package_summary.json `
  --loop87-import-json reports\random_20w_split\loop87_review_evidence_verdict_import.json `
  --loop63-health-summary-json reports\random_20w_split\loop63_A_persistent_conflict_content_audit_summary.json `
  --loop64-duplicate-summary-json reports\random_20w_split\loop64_manifest_sha_duplicate_audit.json `
  --output-json reports\random_20w_split\loop88_full_error_evidence_coverage.json
```

Regression:

```powershell
.\vnev\Scripts\python.exe -m pytest `
  tests\test_build_loop88_full_error_evidence_coverage.py `
  tests\test_import_loop87_review_evidence_verdicts.py `
  tests\test_build_loop86_review_evidence_package.py `
  tests\test_identity_feature_guard.py -q
```

Result: `14 passed`.

## Result

Current best:

- F1: `0.9883629658239992`
- full-test errors: `1868`
- FP/FN: `1195 / 673`

Target gap:

- target F1: `0.999`
- minimum fixed errors under best-case math: `1708`
- required reduction ratio: `91.43468950749465%`

Full queue and wave plan:

- queue rows: `1868`
- unique queue keys: `1868`
- Loop72 wave rows: `1868`
- Loop72 covers queue keys: `true`
- wave count: `10`
- first wave covering target-gap row count: `9`

First evidence package coverage:

- Loop86 rows: `62`
- source exists: `62`
- cache exists: `62`
- source SHA mismatch: `0`
- PE parse status: `ok=62`
- coverage of full queue: `62/1868 = 3.3190578158458245%`
- coverage of target-gap minimum: `62/1708 = 3.629976580796253%`
- remaining queue rows without Loop86-style evidence package: `1806`
- remaining rows to cover best-case target gap: `1646`

Verdict gate status:

- Loop87 rows: `62`
- import ready: `true`
- decision: `ready_noop_no_actionable_verdicts`
- blank verdict rows: `62`
- actionable rows: `0`
- replacement required rows: `0`
- training policy rows: `0`

Existing audits:

- A-lane health rows: `643`
- A-lane objective issue rows: `0`
- Loop64 duplicate groups: `6`
- cross-label duplicate groups: `0`
- cross-split duplicate groups: `0`
- focus duplicate detail rows: `4`

## Decision

The first evidence package is technically clean, but too small to move the
target. It covers only `3.32%` of current-best errors and only `3.63%` of the
best-case target-gap fixes.

Next allowed step:

Expand Loop86-style evidence packaging from the first `62` rows toward the full
Loop72 wave plan. Empty verdicts remain no-op. Confirmed bad rows still require
strict Loop87-style ingress and fresh redraw from the locked-manifest
original-label pool.

## Verdict

Loop88 reinforces that the bottleneck is now evidence coverage. The current
state does not justify training or Test-10k access. To keep pushing toward
`F1 >= 0.999`, the review pipeline must scale from one compact batch to at least
the first `1708` current-best errors under best-case math, or it must prove that
most errors are `label_correct/model_blindspot`, in which case the remaining
gap is model capability rather than removable data noise.
