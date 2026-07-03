# Phase 3 Loop91 Wave3 Evidence Expansion

## Purpose

Loop91 expands the Loop86-style review evidence package to Loop72 Wave3 and
summarizes Waves1-3 together.

This remains a read-only evidence workflow. It does not train, tune thresholds,
relabel, mutate the split, mutate cache, or authorize Test/Test-10k.

## Inputs

- Loop72 wave plan:
  `reports/random_20w_split/loop72_full_error_review_wave_plan.csv`
- Wave3 review input:
  `reports/random_20w_split/loop91_wave3_review_input.csv`
- Wave3 evidence package:
  `reports/random_20w_split/loop91_wave3_review_evidence_package.csv`
- Wave3 verdict import:
  `reports/random_20w_split/loop91_wave3_review_evidence_verdict_import.json`
- Waves1-3 multi-wave summary:
  `reports/random_20w_split/loop91_multiwave_evidence_summary.json`

The Wave3 input was produced by filtering `review_wave_id == 3` from the
Loop72 wave plan. This is a report-row selection only; it does not change raw
files, cache, labels, or split membership.

## Identity Policy

Filenames, paths, extensions, directories, `source_sha256`, `cache_path`,
`sample_index`, split, row order, review rank, and model score columns are
loading, alignment, cache-audit, duplicate-review, and manual-index fields only.
They are not model evidence, verdict evidence, replacement sampling keys, or
threshold/fusion inputs.

Wave3 carries these fields so a reviewer can find the sample and audit cache
integrity. A manual verdict note still must cite PE/content facts or independent
external evidence. Notes that only cite identity fields or model scores are
blocked by the Loop87 verdict gate.

## Commands

Build Wave3 input:

```powershell
$rows = Import-Csv reports\random_20w_split\loop72_full_error_review_wave_plan.csv |
  Where-Object { $_.review_wave_id -eq '3' }
if ($rows.Count -ne 200) { throw "unexpected_wave3_rows=$($rows.Count)" }
$rows | Export-Csv reports\random_20w_split\loop91_wave3_review_input.csv -NoTypeInformation -Encoding UTF8
```

Result: `wave3_rows=200`.

Guard for Wave3 evidence package:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop86_review_evidence_package.py `
  --output-json reports\random_20w_split\loop91_wave3_evidence_guard.json
```

Result: `decision=pass`, static findings `0`.

Build Wave3 evidence package:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop86_review_evidence_package.py `
  --review-csv reports\random_20w_split\loop91_wave3_review_input.csv `
  --output-csv reports\random_20w_split\loop91_wave3_review_evidence_package.csv `
  --output-json reports\random_20w_split\loop91_wave3_review_evidence_package_summary.json `
  --max-entropy-bytes 67108864
```

Guard for Wave3 verdict gate:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\import_loop87_review_evidence_verdicts.py `
  --output-json reports\random_20w_split\loop91_wave3_verdict_guard.json
```

Result: `decision=pass`, static findings `0`.

Validate Wave3 blank verdict import:

```powershell
.\vnev\Scripts\python.exe scripts\import_loop87_review_evidence_verdicts.py `
  --evidence-csv reports\random_20w_split\loop91_wave3_review_evidence_package.csv `
  --output-csv reports\random_20w_split\loop91_wave3_review_evidence_verdict_import.csv `
  --output-json reports\random_20w_split\loop91_wave3_review_evidence_verdict_import.json `
  --expected-rows 200
```

Guard for Waves1-3 summary:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop90_multiwave_evidence_summary.py `
  --output-json reports\random_20w_split\loop91_multiwave_summary_guard.json
```

Build Waves1-3 summary:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop90_multiwave_evidence_summary.py `
  --loop72-summary-json reports\random_20w_split\loop72_full_error_review_wave_plan_summary.json `
  --loop88-coverage-json reports\random_20w_split\loop88_full_error_evidence_coverage.json `
  --wave 1=reports\random_20w_split\loop89_wave1_review_evidence_package_summary.json,reports\random_20w_split\loop89_wave1_review_evidence_verdict_import.json `
  --wave 2=reports\random_20w_split\loop90_wave2_review_evidence_package_summary.json,reports\random_20w_split\loop90_wave2_review_evidence_verdict_import.json `
  --wave 3=reports\random_20w_split\loop91_wave3_review_evidence_package_summary.json,reports\random_20w_split\loop91_wave3_review_evidence_verdict_import.json `
  --output-json reports\random_20w_split\loop91_multiwave_evidence_summary.json
```

## Result

Wave3 evidence package:

- rows: `200`
- category counts:
  - high-conflict persistent error: `200`
- error type counts: FN `27`, FP `173`
- source exists: `200`
- cache exists: `200`
- source SHA mismatch: `0`
- PE parse status: `ok=200`

Wave3 review tags:

- `has_resource_directory`: `183`
- `overlay_present`: `70`
- `high_overlay_entropy`: `61`
- `has_security_directory`: `54`
- `many_sections`: `46`
- `high_file_entropy`: `27`
- `overlay_after_security_present`: `18`
- `no_import_directory`: `1`

Wave3 verdict gate:

- decision: `ready_noop_no_actionable_verdicts`
- blank verdict rows: `200`
- actionable rows: `0`
- replacement required rows: `0`
- training policy rows: `0`

Combined Wave1 + Wave2 + Wave3 coverage:

- covered waves: `1, 2, 3`
- combined rows: `600`
- full current-best error queue: `1868`
- best-case target-gap fixes needed: `1708`
- coverage of queue: `600/1868 = 32.119914346895073%`
- coverage of target gap: `600/1708 = 35.1288056206089%`
- remaining queue rows without evidence package: `1268`
- remaining target-gap rows without evidence package: `1108`
- combined error type counts: FN `227`, FP `373`
- combined blank verdict rows: `600`
- actionable rows: `0`
- replacement required rows: `0`
- training policy rows: `0`

## Decision

Loop91 extends evidence coverage to the first three Loop72 waves and keeps row
counts aligned with the locked full-error queue.

It still does not authorize training, replacement, Test-10k, or full-test
evaluation. With blank verdicts, all `600` rows remain no-op. The next allowed
step is either:

- continue packaging Loop72 Wave4; or
- import independent manual/external verdicts for Waves1-3 through Loop87.

Confirmed `label_wrong`, `feature_broken`, or `out_of_scope` rows only create a
non-destructive quarantine plus fresh redraw request. The bad row must not
self-fill its slot. Replacement must be a fresh valid unused sample from the
same original-label pool, preserving the exact `200000` rows and split/class
balance.
