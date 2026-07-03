# Phase 3 Loop95 Full-Queue Verdict Intake

## Purpose

Loop95 turns the ten Loop72 evidence waves into one strict full-queue intake
CSV for the Loop87 verdict gate. This closes the operational gap between
"evidence packages exist" and "the full `1868`-row queue can be validated by one
manual/external verdict import command".

This is still a read-only workflow. It does not train, tune thresholds, relabel,
sample replacements, mutate cache, mutate the split, or authorize Test/Test-10k.

## Identity Policy

Filenames, paths, extensions, directories, `source_sha256`, `cache_path`,
`sample_index`, split, row order, review rank, and model score columns remain
loading, alignment, cache-audit, duplicate-review, and manual-index fields only.
They are not model evidence, verdict evidence, replacement sampling keys, or
threshold/fusion inputs.

Loop95 uses `sample_index` only to prove that all Loop72 rows are present exactly
once. It does not infer malware status from row identity. `source_sha256`
duplicates are counted for group review, not used as verdict evidence.

## Commands

Guard for Loop95 intake builder:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop95_full_queue_verdict_intake.py `
  --output-json reports\random_20w_split\loop95_intake_guard.json
```

Result: `decision=pass`, static findings `0`.

Build the full-queue intake:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop95_full_queue_verdict_intake.py `
  --loop72-plan-csv reports\random_20w_split\loop72_full_error_review_wave_plan.csv `
  --multiwave-summary-json reports\random_20w_split\loop94_multiwave_evidence_summary.json `
  --wave 1=reports\random_20w_split\loop89_wave1_review_evidence_package.csv `
  --wave 2=reports\random_20w_split\loop90_wave2_review_evidence_package.csv `
  --wave 3=reports\random_20w_split\loop91_wave3_review_evidence_package.csv `
  --wave 4=reports\random_20w_split\loop92_wave4_review_evidence_package.csv `
  --wave 5=reports\random_20w_split\loop93_wave5_review_evidence_package.csv `
  --wave 6=reports\random_20w_split\loop93_wave6_review_evidence_package.csv `
  --wave 7=reports\random_20w_split\loop93_wave7_review_evidence_package.csv `
  --wave 8=reports\random_20w_split\loop93_wave8_review_evidence_package.csv `
  --wave 9=reports\random_20w_split\loop93_wave9_review_evidence_package.csv `
  --wave 10=reports\random_20w_split\loop94_wave10_review_evidence_package.csv `
  --output-csv reports\random_20w_split\loop95_full_queue_review_evidence_intake.csv `
  --output-json reports\random_20w_split\loop95_full_queue_review_evidence_intake.json `
  --expected-rows 1868
```

Guard for the full-queue Loop87 verdict gate:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\import_loop87_review_evidence_verdicts.py `
  --output-json reports\random_20w_split\loop95_full_queue_verdict_guard.json
```

Result: `decision=pass`, static findings `0`.

Validate the combined blank verdict intake:

```powershell
.\vnev\Scripts\python.exe scripts\import_loop87_review_evidence_verdicts.py `
  --evidence-csv reports\random_20w_split\loop95_full_queue_review_evidence_intake.csv `
  --output-csv reports\random_20w_split\loop95_full_queue_review_evidence_verdict_import.csv `
  --output-json reports\random_20w_split\loop95_full_queue_review_evidence_verdict_import.json `
  --expected-rows 1868
```

## Results

Loop95 intake builder:

- rows: `1868`
- expected rows: `1868`
- queue rows: `1868`
- blockers: none
- covered waves: `1-10`
- per-wave Loop72 row mismatches: `0`
- per-wave unexpected sample indices: `0`
- duplicate sample index rows: `0`
- split counts: test `1868`
- label counts: label `0 = 1195`, label `1 = 673`
- error type counts: FP `1195`, FN `673`
- blank verdict rows before Loop87: `1868`
- duplicate source SHA groups: `2` groups, `4` rows

The duplicate source SHA rows are retained as separate rows because
`sample_index` is the row-preserving audit key. They require content-group
review, but they are not automatic replacement or relabel evidence.

Full-queue Loop87 verdict import:

- rows: `1868`
- expected rows: `1868`
- import ready: `true`
- decision: `ready_noop_no_actionable_verdicts`
- duplicate sample index rows: `0`
- duplicate review batch rank rows: `0`
- invalid rows: `0`
- blank verdict rows: `1868`
- actionable rows: `0`
- replacement required rows: `0`
- training policy rows: `0`

## Decision

Loop95 proves the full current-best error queue can enter a single strict
Loop87 verdict gate without row loss, duplicate folding, or identity-field
evidence leakage.

It still does not authorize training, replacement, Test-10k, or full-test
evaluation. The next allowed step remains independent manual/external verdict
annotation on `reports/random_20w_split/loop95_full_queue_review_evidence_intake.csv`,
then re-running Loop87. Any actionable verdict must cite PE/content facts or
independent external evidence; notes that only cite names, paths, hashes,
`sample_index`, split, row order, review rank, model scores, probabilities, or
thresholds are invalid.

Confirmed `label_wrong`, `feature_broken`, or `out_of_scope` rows only create a
non-destructive quarantine plus fresh redraw request from the same original-label
pool. Bad rows never self-fill their slots, and the final dataset must remain
exactly `200000 = 20000/20000/160000`.
