# Phase 3 Loop89 Wave1 Evidence Expansion

## Purpose

Loop89 expands the Loop86-style review evidence package from the initial `62`
rows to the full Loop72 Wave1 batch of `200` current-best errors.

This is still a read-only evidence workflow. It does not train, tune thresholds,
relabel, mutate the split, mutate cache, or authorize Test/Test-10k.

## Inputs

- Loop72 wave plan:
  `reports/random_20w_split/loop72_full_error_review_wave_plan.csv`
- Wave1 review input:
  `reports/random_20w_split/loop89_wave1_review_input.csv`
- Wave1 evidence package:
  `reports/random_20w_split/loop89_wave1_review_evidence_package.csv`
- Wave1 verdict import:
  `reports/random_20w_split/loop89_wave1_review_evidence_verdict_import.json`

The Wave1 input was produced by filtering `review_wave_id == 1` from the
Loop72 wave plan. This is a report-row selection only; it does not change raw
files, cache, labels, or split membership.

## Identity Policy

Filenames, paths, extensions, directories, `source_sha256`, `cache_path`,
`sample_index`, split, review rank, and model score columns are alignment and
review context only. They are not model evidence, verdict evidence, replacement
sampling keys, or threshold/fusion inputs.

## Commands

Build Wave1 input:

```powershell
$rows = Import-Csv reports\random_20w_split\loop72_full_error_review_wave_plan.csv |
  Where-Object { $_.review_wave_id -eq '1' }
$rows | Export-Csv reports\random_20w_split\loop89_wave1_review_input.csv -NoTypeInformation -Encoding UTF8
```

Result: `wave1_rows=200`.

Guard for evidence package:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop86_review_evidence_package.py `
  --output-json reports\random_20w_split\loop89_wave1_evidence_guard.json
```

Result: `decision=pass`, static findings `0`.

Build Wave1 evidence package:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop86_review_evidence_package.py `
  --review-csv reports\random_20w_split\loop89_wave1_review_input.csv `
  --output-csv reports\random_20w_split\loop89_wave1_review_evidence_package.csv `
  --output-json reports\random_20w_split\loop89_wave1_review_evidence_package_summary.json `
  --max-entropy-bytes 67108864
```

Guard for verdict gate:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\import_loop87_review_evidence_verdicts.py `
  --output-json reports\random_20w_split\loop89_wave1_verdict_guard.json
```

Result: `decision=pass`, static findings `0`.

Validate Wave1 blank verdict import:

```powershell
.\vnev\Scripts\python.exe scripts\import_loop87_review_evidence_verdicts.py `
  --evidence-csv reports\random_20w_split\loop89_wave1_review_evidence_package.csv `
  --output-csv reports\random_20w_split\loop89_wave1_review_evidence_verdict_import.csv `
  --output-json reports\random_20w_split\loop89_wave1_review_evidence_verdict_import.json `
  --expected-rows 200
```

Build Loop89 summary:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop89_wave1_evidence_summary.py `
  --loop72-summary-json reports\random_20w_split\loop72_full_error_review_wave_plan_summary.json `
  --loop88-coverage-json reports\random_20w_split\loop88_full_error_evidence_coverage.json `
  --wave1-evidence-json reports\random_20w_split\loop89_wave1_review_evidence_package_summary.json `
  --wave1-verdict-json reports\random_20w_split\loop89_wave1_review_evidence_verdict_import.json `
  --output-json reports\random_20w_split\loop89_wave1_evidence_summary.json
```

Regression:

```powershell
.\vnev\Scripts\python.exe -m pytest `
  tests\test_build_loop89_wave1_evidence_summary.py `
  tests\test_build_loop88_full_error_evidence_coverage.py `
  tests\test_import_loop87_review_evidence_verdicts.py `
  tests\test_build_loop86_review_evidence_package.py `
  tests\test_identity_feature_guard.py -q
```

Result: `16 passed`.

## Result

Wave1 evidence package:

- rows: `200`
- category counts:
  - duplicate content group: `4`
  - high-conflict persistent error: `196`
- error type counts: FN `98`, FP `102`
- source exists: `200`
- cache exists: `200`
- source SHA mismatch: `0`
- PE parse status: `ok=200`

Review tags:

- `has_resource_directory`: `185`
- `overlay_present`: `98`
- `high_overlay_entropy`: `77`
- `has_security_directory`: `64`
- `many_sections`: `43`
- `overlay_after_security_present`: `39`
- `high_file_entropy`: `32`
- `no_import_directory`: `5`

Wave1 verdict gate:

- decision: `ready_noop_no_actionable_verdicts`
- blank verdict rows: `200`
- actionable rows: `0`
- replacement required rows: `0`
- training policy rows: `0`

Coverage after Wave1:

- full current-best error queue: `1868`
- best-case target-gap fixes needed: `1708`
- Wave1 evidence coverage of queue: `200/1868 = 10.706638115631692%`
- Wave1 evidence coverage of target gap: `200/1708 = 11.7096018735363%`
- remaining queue rows without evidence package: `1668`
- remaining target-gap rows without evidence package: `1508`

## Decision

Wave1 expansion proves the evidence package workflow can scale beyond the first
compact batch while keeping resource and identity-safety gates intact.

It still does not authorize training or Test-10k. With blank verdicts, all `200`
rows remain no-op. The next allowed step is either:

- continue packaging Loop72 Wave2; or
- import independent manual/external verdicts for Wave1 through Loop87.

Confirmed bad rows must still go through fresh redraw from the locked-manifest
original-label pool.
