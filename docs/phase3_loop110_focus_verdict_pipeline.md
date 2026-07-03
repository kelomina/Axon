# Phase 3 Loop110 Focus Verdict Pipeline

Date: 2026-07-03

## Purpose

Loop110 adds one strict entrypoint for focus annotations after Loop106. It runs
the full review-ingress chain in order:

1. Loop109 focus annotation preflight
2. Loop107 focus annotation merge
3. Loop96 unblind
4. Loop87 strict verdict import

The pipeline is read-only. It does not train, tune thresholds, load checkpoints,
open NPZ arrays, sample replacements, mutate split/cache, or produce automatic
verdicts.

## Added Tool

- `scripts/run_loop110_focus_verdict_pipeline.py`

The tool stops immediately if an earlier stage blocks. This prevents bypassing
the blinded focus preflight and prevents invalid focus annotations from being
merged or unblinded.

## Real No-op Result

Input:

- full blinded CSV: `reports/random_20w_split/loop96_full_queue_blinded_review.csv`
- focus CSV: `reports/random_20w_split/loop106_content_review_focus_top240.csv`
- private map: `reports/random_20w_split/loop96_full_queue_private_map.csv`

Outputs:

- summary: `reports/random_20w_split/loop110_focus_verdict_pipeline_noop_summary.json`
- stage outputs: `reports/random_20w_split/loop110_focus_verdict_pipeline_noop/`
- guard: `reports/random_20w_split/loop110_focus_verdict_pipeline_guard.json`

Result:

- pipeline decision: `ready_noop_no_actionable_verdicts`
- blockers: `[]`
- preflight rows: `240`
- preflight annotated rows: `0`
- merged annotated rows: `0`
- Loop87 rows: `1868`
- Loop87 actionable rows: `0`
- replacement required rows: `0`
- training policy rows: `0`
- redraw preflight allowed: `false`
- Train/Val allowed: `false`
- Test-10k allowed: `false`
- full-test allowed: `false`

This keeps the project moving without violating the current evidence gate: the
pipeline is ready for future independent content/external verdicts, but the
current focus table is still blank and cannot unlock heavy operations.

## Commands

Resource/static guard:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\pre_run_resource_leak_guard.py --target-script scripts/run_loop110_focus_verdict_pipeline.py --output-json reports/random_20w_split/loop110_focus_verdict_pipeline_guard.json
```

Real no-op pipeline:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\run_loop110_focus_verdict_pipeline.py --full-blinded-csv reports/random_20w_split/loop96_full_queue_blinded_review.csv --focus-annotations-csv reports/random_20w_split/loop106_content_review_focus_top240.csv --private-map-csv reports/random_20w_split/loop96_full_queue_private_map.csv --output-dir reports/random_20w_split/loop110_focus_verdict_pipeline_noop --output-json reports/random_20w_split/loop110_focus_verdict_pipeline_noop_summary.json --expected-full-rows 1868 --expected-focus-rows 240
```

Tests:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" -m pytest tests/test_run_loop110_focus_verdict_pipeline.py tests/test_preflight_loop106_focus_annotations.py tests/test_merge_loop106_focus_annotations.py tests/test_build_loop96_blinded_review_package.py tests/test_import_loop87_review_evidence_verdicts.py
```

Result: `21 passed`.
