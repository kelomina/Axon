# Phase 3 Loop107 Focus Annotation Merge

Date: 2026-07-03

## Purpose

Loop107 connects the Loop106 focus CSV back to the existing Loop96/Loop87 verdict
gate. Reviewers or external engines can annotate only the 240-row focus CSV, and
this loop safely merges those manual fields into the full 1868-row blinded CSV.

The loop does not train, evaluate, tune thresholds, read the private map during
merge, create verdicts, relabel, replace samples, or mutate split/cache.

## Added Tool

- `scripts/merge_loop106_focus_annotations.py`

The tool reads:

- full blinded CSV: `reports/random_20w_split/loop96_full_queue_blinded_review.csv`
- focus annotation CSV: `reports/random_20w_split/loop106_content_review_focus_top240.csv`

It writes:

- merged full blinded CSV: `reports/random_20w_split/loop107_focus_merged_full_blinded_noop.csv`
- merge summary: `reports/random_20w_split/loop107_focus_annotation_merge_noop.json`

Only these fields are merged from the focus CSV:

- `manual_label_verdict`
- `manual_verdict_note`
- `recommended_action`

All content columns in the final full blinded CSV come from the original Loop96
full blinded CSV. Focus rank/score/reason columns are not propagated downstream.

## Real No-op Chain

Current focus annotations are blank, so this is a no-op route proof:

1. Merge focus annotations into full blinded CSV.
2. Unblind with Loop96 private map.
3. Validate with Loop87.

Results:

- merge blockers: `[]`
- full blinded rows: `1868`
- focus rows: `240`
- annotated focus rows: `0`
- merged annotated rows: `0`
- Loop96 unblind blockers: `[]`
- Loop87 decision: `ready_noop_no_actionable_verdicts`
- Loop87 actionable rows: `0`
- replacement required rows: `0`
- training/test10k allowed: `false`

## Identity Boundary

The focus annotation CSV is keyed only by `blind_review_id`. The merge tool
blocks focus input columns containing identity or model evidence, including:

- filename/path/directory/extension
- source/cache path
- hash/source SHA
- sample index
- split/row order/review rank
- `loop57_*`, `loop39_*`, probability, score, prediction, threshold

`blind_review_id` is a blind review handle only. It does not expose source row
identity and is not model evidence.

## Verification

Resource/static guard:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts/pre_run_resource_leak_guard.py --target-script scripts/merge_loop106_focus_annotations.py --target-script tests/test_merge_loop106_focus_annotations.py --output-json reports/random_20w_split/loop107_focus_annotation_merge_guard.json
```

Result: pass.

Tests:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" -m pytest tests/test_merge_loop106_focus_annotations.py tests/test_build_loop106_content_review_focus.py tests/test_build_loop96_blinded_review_package.py tests/test_import_loop87_review_evidence_verdicts.py
```

Result: `15 passed`.

No-op route commands:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts/merge_loop106_focus_annotations.py --full-blinded-csv reports/random_20w_split/loop96_full_queue_blinded_review.csv --focus-annotations-csv reports/random_20w_split/loop106_content_review_focus_top240.csv --output-csv reports/random_20w_split/loop107_focus_merged_full_blinded_noop.csv --output-json reports/random_20w_split/loop107_focus_annotation_merge_noop.json --expected-full-rows 1868 --expected-focus-rows 240

& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts/build_loop96_blinded_review_package.py unblind --annotated-blinded-csv reports/random_20w_split/loop107_focus_merged_full_blinded_noop.csv --private-map-csv reports/random_20w_split/loop96_full_queue_private_map.csv --output-csv reports/random_20w_split/loop107_focus_merged_unblinded_loop87_input_noop.csv --output-json reports/random_20w_split/loop107_focus_merged_unblind_noop.json --expected-rows 1868

& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts/import_loop87_review_evidence_verdicts.py --evidence-csv reports/random_20w_split/loop107_focus_merged_unblinded_loop87_input_noop.csv --output-csv reports/random_20w_split/loop107_focus_merged_verdict_import_noop.csv --output-json reports/random_20w_split/loop107_focus_merged_verdict_import_noop.json --expected-rows 1868
```

Result: route proved as no-op with `actionable_rows=0`.
