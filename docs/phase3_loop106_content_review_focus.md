# Phase 3 Loop106 Content Review Focus

Date: 2026-07-03

## Purpose

Loop106 reduces the cost of obtaining independent content verdicts. It builds a
review focus batch from the Loop96 blinded CSV using only PE/static-content
fields. It does not train, evaluate, tune thresholds, read the private map,
restore identity fields, mutate split/cache, relabel, or replace samples.

## Input

- `reports/random_20w_split/loop96_full_queue_blinded_review.csv`

The input has `1868` current-best error rows and is already blinded. It omits
filenames, paths, hashes, sample indices, split, row order, wave/rank fields, and
model scores.

## Output

- `reports/random_20w_split/loop106_content_review_focus_top240.csv`
- `reports/random_20w_split/loop106_content_review_focus_top240.json`

Real result:

- rows: `1868`
- selected rows: `240`
- blockers: `[]`
- forbidden input columns: `[]`
- ready for independent content review: `true`
- automatic verdict/relabel/replacement/training/Test-10k: all `false`

Selected bucket counts:

- benign-label content review: `207`
- malicious-label content review: `33`

Top focus reasons include:

- overlay present: `240`
- high overlay entropy: `235`
- benign-label malware-like static shape: `207`
- high file entropy: `184`
- post-security overlay present: `136`
- high post-security overlay entropy: `130`

## Identity Boundary

The focus builder rejects inputs containing identity/model columns such as
`source_path`, `cache_path`, `source_sha256`, `sample_index`, split, row order,
`loop57_*`, `loop39_*`, probability, score, prediction, or threshold fields.

The output scan passed:

- focus rows: `240`
- forbidden header/content-evidence violations: `0`

`blind_review_id` is only a reviewer row handle. It is not model evidence and
does not imply original row order.

## Review Semantics

Loop106 is not a verdict. It only chooses which blinded rows should be reviewed
first. A row becomes actionable only after an independent reviewer or external
engine fills `manual_label_verdict`, `manual_verdict_note`, and
`recommended_action`, and Loop87 accepts the note as content/external evidence.

Confirmed bad rows still follow the existing policy:

- quarantine the bad row
- redraw from the locked manifest same-original-label pool
- no direct relabel
- no self-fill
- keep strict `200000 = 20000/20000/160000`

## Verification

Resource/static guard:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts/pre_run_resource_leak_guard.py --target-script scripts/build_loop106_content_review_focus.py --target-script tests/test_build_loop106_content_review_focus.py --output-json reports/random_20w_split/loop106_content_review_focus_guard.json
```

Result: pass.

Tests:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" -m pytest tests/test_build_loop106_content_review_focus.py tests/test_build_loop96_blinded_review_package.py tests/test_import_loop87_review_evidence_verdicts.py
```

Result: `12 passed`.

Generation:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts/build_loop106_content_review_focus.py --blinded-csv reports/random_20w_split/loop96_full_queue_blinded_review.csv --output-csv reports/random_20w_split/loop106_content_review_focus_top240.csv --output-json reports/random_20w_split/loop106_content_review_focus_top240.json --max-rows 240 --expected-rows 1868
```

Result: report generated with no blockers.
