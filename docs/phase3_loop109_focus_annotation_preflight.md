# Phase 3 Loop109 Focus Annotation Preflight

Date: 2026-07-03

## Purpose

Loop109 adds a blinded preflight gate before Loop107 focus merge and Loop96
unblind. It validates the three manual fields in the Loop106 focus CSV:

- `manual_label_verdict`
- `manual_verdict_note`
- `recommended_action`

The loop is read-only. It does not read the private map, unblind rows, merge
annotations, train models, tune thresholds, load checkpoints, open NPZ arrays,
replace samples, or mutate split/cache.

## Added Tool

- `scripts/preflight_loop106_focus_annotations.py`

The tool checks:

- required columns: `blind_review_id` and the three manual fields
- duplicate or missing `blind_review_id`
- forbidden identity/model columns
- verdict/action legality and pairing
- actionable verdicts require a manual note
- manual notes must cite content or external evidence
- identity fields or model scores alone are blocked before merge/unblind

This uses the same evidence-note vocabulary as Loop87 so the focus preflight and
the downstream unblinded import gate do not drift.

## Real No-op Result

Input:

- `reports/random_20w_split/loop106_content_review_focus_top240.csv`

Outputs:

- `reports/random_20w_split/loop109_focus_annotation_preflight_noop.csv`
- `reports/random_20w_split/loop109_focus_annotation_preflight_noop.json`
- `reports/random_20w_split/loop109_focus_annotation_preflight_guard.json`

Result:

- rows: `240`
- decision: `ready_noop_no_focus_annotations`
- blockers: `[]`
- annotated rows: `0`
- actionable rows: `0`
- invalid rows: `0`
- identity/model term mention rows: `0`
- automatic verdict/relabel/replacement/training/Test-10k allowed: `false`

The focus file is therefore structurally ready to merge as a no-op, but it still
contains no independent verdicts and does not unlock redraw, Train/Val,
Test-10k, or full-test.

## Identity Boundary

Filename, path, directory, extension, hash, `source_sha256`, `sample_index`,
split, row order, review rank, model score, probability, prediction, and
threshold terms are not sufficient evidence in manual fields. A reviewer or
external engine must cite content or external evidence such as PE structure,
entropy, imports/resources, overlay/security-directory facts, parser failure,
multi-engine result, sandbox behavior, signature/publisher facts, or equivalent
content-derived evidence.

## Commands

Resource/static guard:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\pre_run_resource_leak_guard.py --target-script scripts/preflight_loop106_focus_annotations.py --output-json reports/random_20w_split/loop109_focus_annotation_preflight_guard.json
```

Real focus preflight:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\preflight_loop106_focus_annotations.py --focus-annotations-csv reports/random_20w_split/loop106_content_review_focus_top240.csv --output-csv reports/random_20w_split/loop109_focus_annotation_preflight_noop.csv --output-json reports/random_20w_split/loop109_focus_annotation_preflight_noop.json --expected-rows 240
```

Tests:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" -m pytest tests/test_preflight_loop106_focus_annotations.py tests/test_import_loop87_review_evidence_verdicts.py tests/test_merge_loop106_focus_annotations.py
```

Result: `15 passed`.
