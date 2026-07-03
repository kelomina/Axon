# Phase 3 Loop93 Waves5-9 Evidence Expansion

## Purpose

Loop93 expands the Loop86-style review evidence package from Waves1-4 to
Waves1-9. Loop72 predicted that Wave9 is the first wave where best-case confirmed
fixes could cross the `F1 >= 0.999` target gap.

This remains a read-only evidence workflow. It does not train, tune thresholds,
relabel, mutate the split, mutate cache, or authorize Test/Test-10k.

## Inputs

- Loop72 wave plan:
  `reports/random_20w_split/loop72_full_error_review_wave_plan.csv`
- Wave5-Wave9 review inputs:
  `reports/random_20w_split/loop93_wave{5..9}_review_input.csv`
- Wave5-Wave9 evidence package summaries:
  `reports/random_20w_split/loop93_wave{5..9}_review_evidence_package_summary.json`
- Wave5-Wave9 verdict imports:
  `reports/random_20w_split/loop93_wave{5..9}_review_evidence_verdict_import.json`
- Waves1-9 multi-wave summary:
  `reports/random_20w_split/loop93_multiwave_evidence_summary.json`

Each Wave5-Wave9 input was produced by filtering the corresponding
`review_wave_id` from the Loop72 wave plan. These are report-row selections
only; they do not change raw files, cache, labels, or split membership.

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

Build Wave5-Wave9 inputs:

```powershell
foreach ($wave in 5..9) {
  $rows = Import-Csv reports\random_20w_split\loop72_full_error_review_wave_plan.csv |
    Where-Object { $_.review_wave_id -eq [string]$wave }
  if ($rows.Count -ne 200) { throw "unexpected_wave${wave}_rows=$($rows.Count)" }
  $rows | Export-Csv "reports\random_20w_split\loop93_wave${wave}_review_input.csv" -NoTypeInformation -Encoding UTF8
}
```

Result: each wave has exactly `200` rows.

Guard for Wave5-Wave9 evidence packages:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop86_review_evidence_package.py `
  --output-json reports\random_20w_split\loop93_waves5_9_evidence_guard.json
```

Result: `decision=pass`, static findings `0`.

Build Wave5-Wave9 evidence packages:

```powershell
foreach ($wave in 5..9) {
  .\vnev\Scripts\python.exe scripts\build_loop86_review_evidence_package.py `
    --review-csv "reports\random_20w_split\loop93_wave${wave}_review_input.csv" `
    --output-csv "reports\random_20w_split\loop93_wave${wave}_review_evidence_package.csv" `
    --output-json "reports\random_20w_split\loop93_wave${wave}_review_evidence_package_summary.json" `
    --max-entropy-bytes 67108864
}
```

Guard for Wave5-Wave9 verdict gates:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\import_loop87_review_evidence_verdicts.py `
  --output-json reports\random_20w_split\loop93_waves5_9_verdict_guard.json
```

Result: `decision=pass`, static findings `0`.

Validate Wave5-Wave9 blank verdict imports:

```powershell
foreach ($wave in 5..9) {
  .\vnev\Scripts\python.exe scripts\import_loop87_review_evidence_verdicts.py `
    --evidence-csv "reports\random_20w_split\loop93_wave${wave}_review_evidence_package.csv" `
    --output-csv "reports\random_20w_split\loop93_wave${wave}_review_evidence_verdict_import.csv" `
    --output-json "reports\random_20w_split\loop93_wave${wave}_review_evidence_verdict_import.json" `
    --expected-rows 200
}
```

Guard for Waves1-9 summary:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop90_multiwave_evidence_summary.py `
  --output-json reports\random_20w_split\loop93_multiwave_summary_guard.json
```

Build Waves1-9 summary:

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
  --output-json reports\random_20w_split\loop93_multiwave_evidence_summary.json
```

## Wave Results

Wave5:

- rows: `200`
- error type counts: FN `18`, FP `182`
- category counts: persistent FN `18`, persistent FP `182`
- source/cache: `200/200`
- source SHA mismatch: `0`
- PE parse status: `ok=200`

Wave6:

- rows: `200`
- error type counts: FN `184`, FP `16`
- category counts: persistent FN `184`, persistent FP `16`
- source/cache: `200/200`
- source SHA mismatch: `0`
- PE parse status: `ok=200`

Wave7:

- rows: `200`
- error type counts: FN `87`, FP `113`
- category counts: persistent FN `87`, persistent FP `113`
- source/cache: `200/200`
- source SHA mismatch: `0`
- PE parse status: `ok=200`

Wave8:

- rows: `200`
- error type counts: FP `200`
- category counts: persistent FP `200`
- source/cache: `200/200`
- source SHA mismatch: `0`
- PE parse status: `ok=200`

Wave9:

- rows: `200`
- error type counts: FP `200`
- category counts: Loop57 new error `40`, persistent FP `160`
- source/cache: `200/200`
- source SHA mismatch: `0`
- PE parse status: `ok=200`

All Wave5-Wave9 verdict gates:

- decision: `ready_noop_no_actionable_verdicts`
- blank verdict rows per wave: `200`
- actionable rows: `0`
- replacement required rows: `0`
- training policy rows: `0`

## Combined Coverage

Combined Wave1-Wave9 coverage:

- covered waves: `1, 2, 3, 4, 5, 6, 7, 8, 9`
- combined rows: `1800`
- full current-best error queue: `1868`
- best-case target-gap fixes needed: `1708`
- coverage of queue: `1800/1868 = 96.35974304068522%`
- coverage of target gap: `1800/1708 = 105.3864168618267%`
- remaining queue rows without evidence package: `68`
- remaining target-gap rows without evidence package: `0`
- combined error type counts: FN `673`, FP `1127`
- combined category counts:
  - duplicate content group: `4`
  - high-conflict persistent error: `639`
  - Loop57 new error: `40`
  - persistent FN: `446`
  - persistent FP: `671`
- combined blank verdict rows: `1800`
- actionable rows: `0`
- replacement required rows: `0`
- training policy rows: `0`

## Decision

Loop93 reaches and exceeds the best-case target-gap evidence coverage threshold:
the first nine waves cover `1800` current-best errors against the `1708` minimum
best-case fixes needed for `F1 >= 0.999`.

This is coverage, not correction. With blank verdicts, all `1800` rows remain
no-op. The workflow still does not authorize training, replacement, Test-10k, or
full-test evaluation.

Next allowed step:

- import independent manual/external verdicts for Waves1-9 through Loop87; or
- package Wave10 to cover the remaining `68` current-best errors.

Confirmed `label_wrong`, `feature_broken`, or `out_of_scope` rows only create a
non-destructive quarantine plus fresh redraw request. The bad row must not
self-fill its slot. Replacement must be a fresh valid unused sample from the
same original-label pool, preserving the exact `200000` rows and split/class
balance.
