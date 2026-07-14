# Phase 3 Loop94 Wave10 Full-Queue Evidence Coverage

## Purpose

Loop94 expands the Loop86-style review evidence package to the final Loop72
Wave10 batch and summarizes Waves1-10 together. This completes evidence-package
coverage for the entire current-best full-error queue.

This remains a read-only evidence workflow. It does not train, tune thresholds,
relabel, mutate the split, mutate cache, or authorize Test/Test-10k.

## Inputs

- Loop72 wave plan:
  `reports/random_20w_split/loop72_full_error_review_wave_plan.csv`
- Wave10 review input:
  `reports/random_20w_split/loop94_wave10_review_input.csv`
- Wave10 evidence package:
  `reports/random_20w_split/loop94_wave10_review_evidence_package.csv`
- Wave10 verdict import:
  `reports/random_20w_split/loop94_wave10_review_evidence_verdict_import.json`
- Waves1-10 multi-wave summary:
  `reports/random_20w_split/loop94_multiwave_evidence_summary.json`

The Wave10 input was produced by filtering `review_wave_id == 10` from the
Loop72 wave plan. This is a report-row selection only; it does not change raw
files, cache, labels, or split membership.

## Identity Policy

Filenames, paths, extensions, directories, `source_sha256`, `cache_path`,
`sample_index`, split, row order, review rank, and model score columns are
loading, alignment, cache-audit, duplicate-review, and manual-index fields only.
They are not model evidence, verdict evidence, replacement sampling keys, or
threshold/fusion inputs.

Manual verdict notes must cite PE/content facts or independent external
evidence. Notes that only cite identity fields or model scores are blocked by
the Loop87 verdict gate.

## Commands

Build Wave10 input:

```powershell
$rows = Import-Csv reports\random_20w_split\loop72_full_error_review_wave_plan.csv |
  Where-Object { $_.review_wave_id -eq '10' }
if ($rows.Count -ne 68) { throw "unexpected_wave10_rows=$($rows.Count)" }
$rows | Export-Csv reports\random_20w_split\loop94_wave10_review_input.csv -NoTypeInformation -Encoding UTF8
```

Result: `wave10_rows=68`.

Guard for Wave10 evidence package:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop86_review_evidence_package.py `
  --output-json reports\random_20w_split\loop94_wave10_evidence_guard.json
```

Result: `decision=pass`, static findings `0`.

Build Wave10 evidence package:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop86_review_evidence_package.py `
  --review-csv reports\random_20w_split\loop94_wave10_review_input.csv `
  --output-csv reports\random_20w_split\loop94_wave10_review_evidence_package.csv `
  --output-json reports\random_20w_split\loop94_wave10_review_evidence_package_summary.json `
  --max-entropy-bytes 67108864
```

Guard for Wave10 verdict gate:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\import_loop87_review_evidence_verdicts.py `
  --output-json reports\random_20w_split\loop94_wave10_verdict_guard.json
```

Result: `decision=pass`, static findings `0`.

Validate Wave10 blank verdict import:

```powershell
.\vnev\Scripts\python.exe scripts\import_loop87_review_evidence_verdicts.py `
  --evidence-csv reports\random_20w_split\loop94_wave10_review_evidence_package.csv `
  --output-csv reports\random_20w_split\loop94_wave10_review_evidence_verdict_import.csv `
  --output-json reports\random_20w_split\loop94_wave10_review_evidence_verdict_import.json `
  --expected-rows 68
```

Guard for Waves1-10 summary:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop90_multiwave_evidence_summary.py `
  --output-json reports\random_20w_split\loop94_multiwave_summary_guard.json
```

Build Waves1-10 summary:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop90_multiwave_evidence_summary.py `
  --loop72-summary-json reports\random_20w_split\loop72_full_error_review_wave_plan_summary.json `
  --loop88-coverage-json reports\random_20w_split\loop88_full_error_evidence_coverage.json `
  --wave 1=reports\random_20w_split\loop89_wave1_review_evidence_package_summary.json,reports\random_20w_split\loop89_wave1_review_evidence_verdict_import.json `
  --wave 2=reports\random_20w_split\loop90_wave2_review_evidence_package_summary.json,reports\random_20w_split\loop90_wave2_review_evidence_verdict_import.json `
  --wave 3=reports\random_20w_split\loop91_wave3_review_evidence_package_summary.json,reports\random_20w_split\loop91_wave3_review_evidence_verdict_import.json `
  --wave 4=reports\random_20w_split\loop92_wave4_review_evidence_package_summary.json,reports\random_20w_split\loop92_wave4_review_evidence_verdict_import.json `
  --wave 5=reports\random_20w_split\loop93_wave5_review_evidence_package_summary.json,reports\random_20w_split\loop93_wave5_review_evidence_verdict_import.json `
  --wave 6=reports\random_20w_split\loop93_wave6_review_evidence_package_summary.json,reports\random_20w_split\loop93_wave6_review_evidence_verdict_import.json `
  --wave 7=reports\random_20w_split\loop93_wave7_review_evidence_package_summary.json,reports\random_20w_split\loop93_wave7_review_evidence_verdict_import.json `
  --wave 8=reports\random_20w_split\loop93_wave8_review_evidence_package_summary.json,reports\random_20w_split\loop93_wave8_review_evidence_verdict_import.json `
  --wave 9=reports\random_20w_split\loop93_wave9_review_evidence_package_summary.json,reports\random_20w_split\loop93_wave9_review_evidence_verdict_import.json `
  --wave 10=reports\random_20w_split\loop94_wave10_review_evidence_package_summary.json,reports\random_20w_split\loop94_wave10_review_evidence_verdict_import.json `
  --output-json reports\random_20w_split\loop94_multiwave_evidence_summary.json
```

## Wave10 Result

Wave10 evidence package:

- rows: `68`
- category counts:
  - Loop57 new error: `68`
- error type counts: FP `68`
- source exists: `68`
- cache exists: `68`
- source SHA mismatch: `0`
- PE parse status: `ok=68`

Wave10 review tags:

- `has_resource_directory`: `63`
- `overlay_present`: `54`
- `high_overlay_entropy`: `48`
- `overlay_after_security_present`: `37`
- `high_file_entropy`: `28`
- `has_security_directory`: `21`
- `many_sections`: `11`

Wave10 verdict gate:

- decision: `ready_noop_no_actionable_verdicts`
- blank verdict rows: `68`
- actionable rows: `0`
- replacement required rows: `0`
- training policy rows: `0`

## Full-Queue Coverage

Combined Wave1-Wave10 coverage:

- covered waves: `1, 2, 3, 4, 5, 6, 7, 8, 9, 10`
- combined rows: `1868`
- full current-best error queue: `1868`
- best-case target-gap fixes needed: `1708`
- coverage of queue: `1868/1868 = 100%`
- coverage of target gap: `1868/1708 = 109.36768149882905%`
- remaining queue rows without evidence package: `0`
- remaining target-gap rows without evidence package: `0`
- combined error type counts: FN `673`, FP `1195`
- combined category counts:
  - duplicate content group: `4`
  - high-conflict persistent error: `639`
  - Loop57 new error: `108`
  - persistent FN: `446`
  - persistent FP: `671`
- combined blank verdict rows: `1868`
- actionable rows: `0`
- replacement required rows: `0`
- training policy rows: `0`

## Decision

Loop94 completes evidence-package coverage for the whole current-best full-error
queue. The project now has review material for all `1868` errors behind the
current best full-test F1.

This is coverage, not correction. With blank verdicts, all `1868` rows remain
no-op. The workflow still does not authorize training, replacement, Test-10k, or
full-test evaluation.

Next allowed step:

- import independent manual/external verdicts for Waves1-10 through Loop87.

Confirmed `label_wrong`, `feature_broken`, or `out_of_scope` rows only create a
non-destructive quarantine plus fresh redraw request. The bad row must not
self-fill its slot. Replacement must be a fresh valid unused sample from the
same original-label pool, preserving the exact `200000` rows and split/class
balance.
