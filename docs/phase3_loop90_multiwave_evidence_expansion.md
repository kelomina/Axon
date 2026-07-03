# Phase 3 Loop90 Multi-Wave Evidence Expansion

## Purpose

Loop90 generalizes the Loop89 Wave1 summary into a multi-wave evidence coverage
summary, then adds Loop72 Wave2 to the current review package accounting.

This remains a read-only evidence workflow. It does not train, tune thresholds,
relabel, mutate the split, mutate cache, or authorize Test/Test-10k.

## Inputs

- Loop72 wave plan summary:
  `reports/random_20w_split/loop72_full_error_review_wave_plan_summary.json`
- Loop88 coverage gate:
  `reports/random_20w_split/loop88_full_error_evidence_coverage.json`
- Wave1 evidence package summary:
  `reports/random_20w_split/loop89_wave1_review_evidence_package_summary.json`
- Wave1 verdict import:
  `reports/random_20w_split/loop89_wave1_review_evidence_verdict_import.json`
- Wave2 review input:
  `reports/random_20w_split/loop90_wave2_review_input.csv`
- Wave2 evidence package summary:
  `reports/random_20w_split/loop90_wave2_review_evidence_package_summary.json`
- Wave2 verdict import:
  `reports/random_20w_split/loop90_wave2_review_evidence_verdict_import.json`
- Multi-wave summary:
  `reports/random_20w_split/loop90_multiwave_evidence_summary.json`

The Wave2 input was produced by filtering `review_wave_id == 2` from the
Loop72 wave plan. This is a report-row selection only; it does not change raw
files, cache, labels, or split membership.

## Identity Policy

Filenames, paths, extensions, directories, `source_sha256`, `cache_path`,
`sample_index`, split, row order, review rank, and model score columns are
loading, alignment, cache-audit, duplicate-review, and manual-index fields only.
They are not model evidence, verdict evidence, replacement sampling keys, or
threshold/fusion inputs.

This distinction matters operationally: real deployment filenames can be
renamed freely and will not match training names. Therefore naming can help find
the same sample in a manifest, but it cannot justify a malware verdict.

## Commands

Build Wave2 input:

```powershell
$rows = Import-Csv reports\random_20w_split\loop72_full_error_review_wave_plan.csv |
  Where-Object { $_.review_wave_id -eq '2' }
$rows | Export-Csv reports\random_20w_split\loop90_wave2_review_input.csv -NoTypeInformation -Encoding UTF8
```

Result: `wave2_rows=200`.

Guard for Wave2 evidence package:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop86_review_evidence_package.py `
  --output-json reports\random_20w_split\loop90_wave2_evidence_guard.json
```

Result: `decision=pass`, static findings `0`.

Build Wave2 evidence package:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop86_review_evidence_package.py `
  --review-csv reports\random_20w_split\loop90_wave2_review_input.csv `
  --output-csv reports\random_20w_split\loop90_wave2_review_evidence_package.csv `
  --output-json reports\random_20w_split\loop90_wave2_review_evidence_package_summary.json `
  --max-entropy-bytes 67108864
```

Guard for Wave2 verdict gate:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\import_loop87_review_evidence_verdicts.py `
  --output-json reports\random_20w_split\loop90_wave2_verdict_guard.json
```

Result: `decision=pass`, static findings `0`.

Validate Wave2 blank verdict import:

```powershell
.\vnev\Scripts\python.exe scripts\import_loop87_review_evidence_verdicts.py `
  --evidence-csv reports\random_20w_split\loop90_wave2_review_evidence_package.csv `
  --output-csv reports\random_20w_split\loop90_wave2_review_evidence_verdict_import.csv `
  --output-json reports\random_20w_split\loop90_wave2_review_evidence_verdict_import.json `
  --expected-rows 200
```

Guard for Loop90 summary:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop90_multiwave_evidence_summary.py `
  --output-json reports\random_20w_split\loop90_multiwave_summary_guard.json
```

Build Loop90 summary:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop90_multiwave_evidence_summary.py `
  --loop72-summary-json reports\random_20w_split\loop72_full_error_review_wave_plan_summary.json `
  --loop88-coverage-json reports\random_20w_split\loop88_full_error_evidence_coverage.json `
  --wave 1=reports\random_20w_split\loop89_wave1_review_evidence_package_summary.json,reports\random_20w_split\loop89_wave1_review_evidence_verdict_import.json `
  --wave 2=reports\random_20w_split\loop90_wave2_review_evidence_package_summary.json,reports\random_20w_split\loop90_wave2_review_evidence_verdict_import.json `
  --output-json reports\random_20w_split\loop90_multiwave_evidence_summary.json
```

Regression:

```powershell
.\vnev\Scripts\python.exe -m pytest `
  tests\test_build_loop90_multiwave_evidence_summary.py `
  tests\test_build_loop89_wave1_evidence_summary.py `
  tests\test_build_loop88_full_error_evidence_coverage.py `
  tests\test_import_loop87_review_evidence_verdicts.py `
  tests\test_build_loop86_review_evidence_package.py `
  tests\test_identity_feature_guard.py -q
```

## Result

Wave2 evidence package:

- rows: `200`
- category counts:
  - high-conflict persistent error: `200`
- error type counts: FN `102`, FP `98`
- source exists: `200`
- cache exists: `200`
- source SHA mismatch: `0`
- PE parse status: `ok=200`

Wave2 review tags:

- `has_resource_directory`: `176`
- `overlay_present`: `92`
- `has_security_directory`: `81`
- `high_overlay_entropy`: `81`
- `high_file_entropy`: `36`
- `many_sections`: `31`
- `overlay_after_security_present`: `21`
- `no_import_directory`: `11`

Wave2 verdict gate:

- decision: `ready_noop_no_actionable_verdicts`
- blank verdict rows: `200`
- actionable rows: `0`
- replacement required rows: `0`
- training policy rows: `0`

Combined Wave1 + Wave2 coverage:

- covered waves: `1, 2`
- combined rows: `400`
- full current-best error queue: `1868`
- best-case target-gap fixes needed: `1708`
- coverage of queue: `400/1868 = 21.413276231263384%`
- coverage of target gap: `400/1708 = 23.4192037470726%`
- remaining queue rows without evidence package: `1468`
- remaining target-gap rows without evidence package: `1308`
- combined error type counts: FN `200`, FP `200`
- combined blank verdict rows: `400`
- actionable rows: `0`
- replacement required rows: `0`
- training policy rows: `0`

## Decision

Loop90 proves the evidence workflow now scales across multiple Loop72 waves and
keeps row counts aligned with the locked full-error queue.

It still does not authorize training, replacement, Test-10k, or full-test
evaluation. With blank verdicts, all `400` rows remain no-op. The next allowed
step is either:

- continue packaging Loop72 Wave3; or
- import independent manual/external verdicts for Waves1-2 through Loop87.

Confirmed bad rows must still be quarantined and replaced by fresh valid samples
from the locked-manifest original-label pool. Bad rows do not self-fill counts,
and identity fields must not be used to choose replacements.
