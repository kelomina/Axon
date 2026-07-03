# Phase 3 Loop103 Corrected Split Candidate SHA Gate

Loop103 hardens the low-level corrected split builder. Loop102 made the candidate pool content-hash based, but a caller could still bypass Loop76 and pass an old candidate CSV with blank `source_sha256` directly into `build_corrected_split_from_plan.py`.

## Changes

- `scripts/build_corrected_split_from_plan.py`
  - requires replacement candidates to carry a valid 64-character SHA-256 by default;
  - filters blank or invalid candidate hashes before replacement selection;
  - records `candidate_load_summary`;
  - adds `--allow-unhashed-candidates-legacy` as an explicit compatibility escape hatch.

The legacy switch is not allowed for the strict 20w full-error redraw workflow. It exists only for old small tests or historical workflows.

## Why

Fresh redraw means the replacement file content must be different from excluded or already-used content. Path-only candidates are not enough because file names and directories can be changed freely. Content SHA is used only for replacement integrity and duplicate detection; it is not malware evidence.

## Current Decision

No new 20w corrected split was generated in this loop. Current Loop101/98 evidence still shows:

- actionable verdict rows: `0`
- replacement required rows: `0`
- `training_allowed_now=false`
- `test10k_allowed_now=false`
- `full_test_allowed_now=false`

So there is no legitimate replacement plan to materialize yet.

## Verification

Resource guard:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_corrected_split_from_plan.py `
  --output-json reports\random_20w_split\loop103_corrected_split_builder_guard.json
```

Tests:

```powershell
.\vnev\Scripts\python.exe -m py_compile `
  scripts\build_corrected_split_from_plan.py `
  scripts\build_replacement_candidate_pool.py `
  scripts\build_loop76_redraw_readiness.py `
  scripts\audit_corrected_split_replacements.py

.\vnev\Scripts\python.exe -m pytest `
  tests\test_build_corrected_split_from_plan.py `
  tests\test_build_replacement_candidate_pool.py `
  tests\test_build_loop76_redraw_readiness.py `
  tests\test_audit_corrected_split_replacements.py `
  tests\test_audit_corrected_split_cache_ready.py `
  tests\test_build_loop79_current_state_gate.py `
  tests\test_build_loop98_route_audit.py `
  tests\test_import_loop87_review_evidence_verdicts.py `
  -q
```

Result: `69 passed`.

## Boundary

`source_sha256` remains audit metadata. It proves replacement identity and duplicate status only. It must not be used as model input, automatic verdict evidence, threshold evidence, feature-mask input, or production inference evidence.
