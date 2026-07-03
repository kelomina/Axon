# Phase 3 Loop96 Blinded Review Package

## Purpose

Loop96 reduces review bias before manual/external verdict annotation. Loop95
proved the full `1868`-row error queue can enter a single strict verdict gate;
Loop96 adds a blinded reviewer-facing CSV that removes names, paths, hashes,
row ids, split membership, wave/rank order, and model scores from the table that
humans or external reviewers annotate.

This remains read-only infrastructure. It does not train, tune thresholds,
relabel, sample replacements, mutate cache, mutate the split, or authorize
Test/Test-10k.

## Files

- Blinded reviewer CSV:
  `reports/random_20w_split/loop96_full_queue_blinded_review.csv`
- Private alignment map:
  `reports/random_20w_split/loop96_full_queue_private_map.csv`
- Unblinded Loop87 input from the current blank review CSV:
  `reports/random_20w_split/loop96_full_queue_unblinded_loop87_input.csv`
- Loop87 validation of that unblinded input:
  `reports/random_20w_split/loop96_full_queue_verdict_import.json`

## Identity Policy

The blinded reviewer CSV omits:

- filenames, paths, extensions, and directories
- `source_sha256`, actual source hashes, and hash-match columns
- cache paths
- `sample_index`
- split membership
- row order, wave id, and review rank
- Loop57/Loop39 model probabilities, predictions, gates, scores, and categories

The blinded CSV keeps only a synthetic `blind_review_id`, current label, PE and
content facts, objective issue flags, and manual verdict fields. The private map
restores identity/alignment fields only after annotation so the strict Loop87
verdict gate can validate the result. The private map is not verdict evidence.

## Commands

Guard:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop96_blinded_review_package.py `
  --output-json reports\random_20w_split\loop96_blinded_review_guard.json
```

Result: `decision=pass`, static findings `0`.

Build blinded package:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop96_blinded_review_package.py build `
  --input-csv reports\random_20w_split\loop95_full_queue_review_evidence_intake.csv `
  --blinded-csv reports\random_20w_split\loop96_full_queue_blinded_review.csv `
  --private-map-csv reports\random_20w_split\loop96_full_queue_private_map.csv `
  --output-json reports\random_20w_split\loop96_full_queue_blinded_review.json `
  --expected-rows 1868 `
  --seed 9601
```

Unblind the current blank review CSV:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop96_blinded_review_package.py unblind `
  --annotated-blinded-csv reports\random_20w_split\loop96_full_queue_blinded_review.csv `
  --private-map-csv reports\random_20w_split\loop96_full_queue_private_map.csv `
  --output-csv reports\random_20w_split\loop96_full_queue_unblinded_loop87_input.csv `
  --output-json reports\random_20w_split\loop96_full_queue_unblinded_loop87_input.json `
  --expected-rows 1868
```

Validate with Loop87:

```powershell
.\vnev\Scripts\python.exe scripts\import_loop87_review_evidence_verdicts.py `
  --evidence-csv reports\random_20w_split\loop96_full_queue_unblinded_loop87_input.csv `
  --output-csv reports\random_20w_split\loop96_full_queue_verdict_import.csv `
  --output-json reports\random_20w_split\loop96_full_queue_verdict_import.json `
  --expected-rows 1868
```

## Results

Blinded package build:

- rows: `1868`
- expected rows: `1868`
- blockers: none
- label counts: label `0 = 1195`, label `1 = 673`
- blinded field count: `48`
- private map field count: `71`
- forbidden blinded columns: none
- ready for blinded review: `true`

The blinded CSV header starts with:

```text
blind_review_id,current_label,review_tags,content_evidence_fields,source_exists,source_size_bytes,file_entropy,...
```

It does not include `source_path`, `cache_path`, `source_sha256`,
`sample_index`, split, review rank, wave id, or model score columns.

Unblind of the current blank review CSV:

- rows: `1868`
- expected rows: `1868`
- blockers: none
- verdict counts: blank `1868`
- action counts: blank `1868`
- ready for Loop87 import: `true`

Loop87 validation after unblind:

- rows: `1868`
- expected rows: `1868`
- import ready: `true`
- decision: `ready_noop_no_actionable_verdicts`
- duplicate sample index rows: `0`
- invalid rows: `0`
- blank verdict rows: `1868`
- actionable rows: `0`
- replacement required rows: `0`
- training policy rows: `0`

## Decision

Loop96 makes the full-error review workflow safer against the naming/path/hash
problem. Reviewers can now annotate the blinded CSV using PE/content facts or
external evidence without seeing deployment-unstable identity metadata or model
probabilities.

This is still not a correction. The current blinded CSV has blank verdicts, so
no sample can be replaced, relabeled, trained on, or used to enter Test-10k.
After independent annotation, the only allowed path is:

1. unblind through `scripts/build_loop96_blinded_review_package.py unblind`;
2. validate through `scripts/import_loop87_review_evidence_verdicts.py`;
3. only confirmed `label_wrong`, `feature_broken`, or `out_of_scope` rows may
   create non-destructive quarantine plus fresh same-original-label redraw
   requests.

The final split must remain exactly `200000 = 20000/20000/160000`; bad rows
never self-fill their slots.
