# Phase 3 Loop92 Wave4 Evidence Expansion

## Purpose

Loop92 expands the Loop86-style review evidence package to Loop72 Wave4 and
summarizes Waves1-4 together.

This remains a read-only evidence workflow. It does not train, tune thresholds,
relabel, mutate the split, mutate cache, or authorize Test/Test-10k.

## Inputs

- Loop72 wave plan:
  `reports/random_20w_split/loop72_full_error_review_wave_plan.csv`
- Wave4 review input:
  `reports/random_20w_split/loop92_wave4_review_input.csv`
- Wave4 evidence package:
  `reports/random_20w_split/loop92_wave4_review_evidence_package.csv`
- Wave4 verdict import:
  `reports/random_20w_split/loop92_wave4_review_evidence_verdict_import.json`
- Waves1-4 multi-wave summary:
  `reports/random_20w_split/loop92_multiwave_evidence_summary.json`

The Wave4 input was produced by filtering `review_wave_id == 4` from the
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

Build Wave4 input:

```powershell
$rows = Import-Csv reports\random_20w_split\loop72_full_error_review_wave_plan.csv |
  Where-Object { $_.review_wave_id -eq '4' }
if ($rows.Count -ne 200) { throw "unexpected_wave4_rows=$($rows.Count)" }
$rows | Export-Csv reports\random_20w_split\loop92_wave4_review_input.csv -NoTypeInformation -Encoding UTF8
```

Result: `wave4_rows=200`.

Guard for Wave4 evidence package:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop86_review_evidence_package.py `
  --output-json reports\random_20w_split\loop92_wave4_evidence_guard.json
```

Result: `decision=pass`, static findings `0`.

Build Wave4 evidence package:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop86_review_evidence_package.py `
  --review-csv reports\random_20w_split\loop92_wave4_review_input.csv `
  --output-csv reports\random_20w_split\loop92_wave4_review_evidence_package.csv `
  --output-json reports\random_20w_split\loop92_wave4_review_evidence_package_summary.json `
  --max-entropy-bytes 67108864
```

Guard for Wave4 verdict gate:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\import_loop87_review_evidence_verdicts.py `
  --output-json reports\random_20w_split\loop92_wave4_verdict_guard.json
```

Result: `decision=pass`, static findings `0`.

Validate Wave4 blank verdict import:

```powershell
.\vnev\Scripts\python.exe scripts\import_loop87_review_evidence_verdicts.py `
  --evidence-csv reports\random_20w_split\loop92_wave4_review_evidence_package.csv `
  --output-csv reports\random_20w_split\loop92_wave4_review_evidence_verdict_import.csv `
  --output-json reports\random_20w_split\loop92_wave4_review_evidence_verdict_import.json `
  --expected-rows 200
```

Guard for Waves1-4 summary:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop90_multiwave_evidence_summary.py `
  --output-json reports\random_20w_split\loop92_multiwave_summary_guard.json
```

Build Waves1-4 summary:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop90_multiwave_evidence_summary.py `
  --loop72-summary-json reports\random_20w_split\loop72_full_error_review_wave_plan_summary.json `
  --loop88-coverage-json reports\random_20w_split\loop88_full_error_evidence_coverage.json `
  --wave 1=reports\random_20w_split\loop89_wave1_review_evidence_package_summary.json,reports\random_20w_split\loop89_wave1_review_evidence_verdict_import.json `
  --wave 2=reports\random_20w_split\loop90_wave2_review_evidence_package_summary.json,reports\random_20w_split\loop90_wave2_review_evidence_verdict_import.json `
  --wave 3=reports\random_20w_split\loop91_wave3_review_evidence_package_summary.json,reports\random_20w_split\loop91_wave3_review_evidence_verdict_import.json `
  --wave 4=reports\random_20w_split\loop92_wave4_review_evidence_package_summary.json,reports\random_20w_split\loop92_wave4_review_evidence_verdict_import.json `
  --output-json reports\random_20w_split\loop92_multiwave_evidence_summary.json
```

## Result

Wave4 evidence package:

- rows: `200`
- category counts:
  - high-conflict persistent error: `43`
  - persistent FN: `157`
- error type counts: FN `157`, FP `43`
- source exists: `200`
- cache exists: `200`
- source SHA mismatch: `0`
- PE parse status: `ok=200`

Wave4 review tags:

- `has_resource_directory`: `158`
- `overlay_present`: `101`
- `has_security_directory`: `87`
- `high_overlay_entropy`: `83`
- `high_file_entropy`: `45`
- `many_sections`: `31`
- `overlay_after_security_present`: `19`
- `no_import_directory`: `15`

Wave4 verdict gate:

- decision: `ready_noop_no_actionable_verdicts`
- blank verdict rows: `200`
- actionable rows: `0`
- replacement required rows: `0`
- training policy rows: `0`

Combined Wave1 + Wave2 + Wave3 + Wave4 coverage:

- covered waves: `1, 2, 3, 4`
- combined rows: `800`
- full current-best error queue: `1868`
- best-case target-gap fixes needed: `1708`
- coverage of queue: `800/1868 = 42.82655246252677%`
- coverage of target gap: `800/1708 = 46.8384074941452%`
- remaining queue rows without evidence package: `1068`
- remaining target-gap rows without evidence package: `908`
- combined error type counts: FN `384`, FP `416`
- combined blank verdict rows: `800`
- actionable rows: `0`
- replacement required rows: `0`
- training policy rows: `0`

## Decision

Loop92 extends evidence coverage to the first four Loop72 waves and keeps row
counts aligned with the locked full-error queue.

It still does not authorize training, replacement, Test-10k, or full-test
evaluation. With blank verdicts, all `800` rows remain no-op. The next allowed
step is either:

- continue packaging Loop72 Wave5; or
- import independent manual/external verdicts for Waves1-4 through Loop87.

Confirmed `label_wrong`, `feature_broken`, or `out_of_scope` rows only create a
non-destructive quarantine plus fresh redraw request. The bad row must not
self-fill its slot. Replacement must be a fresh valid unused sample from the
same original-label pool, preserving the exact `200000` rows and split/class
balance.
