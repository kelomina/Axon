# Phase 3 Loop101 Metadata-Ready Route Gate

Loop101 wires Loop100 full-cache metadata readiness into the higher-level current-state and route gates. This prevents old cache-ready reports that only prove file existence from authorizing later workflow steps.

## Changes

- `scripts/build_loop79_current_state_gate.py`
  - Default `--current-cache-ready` now points to `reports/random_20w_split/loop100_cache_ready_metadata.json`.
  - The current split gate now requires:
    - `cache_ready=true`
    - `total_rows=200000`
    - `covered_rows=200000`
    - `missing_rows=0`
    - `label_balance_enforced=true`
    - `cache_metadata_validation_enabled=true`
    - `metadata_checked_rows=200000`
    - `metadata_failure_rows=0`
- `scripts/build_loop98_route_audit.py`
  - The fixed-v2 route section now also verifies Loop79 metadata readiness evidence.
  - Missing metadata validation, incomplete metadata coverage, or metadata failures block the route.

## Real Reports

Loop79 rerun:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop79_current_state_gate.py `
  --current-cache-ready reports\random_20w_split\loop100_cache_ready_metadata.json `
  --current-coverage reports\random_20w_split\current_split_cache_coverage_reaudit.json `
  --sample-integrity reports\random_20w_split\loop78_cache_sample_integrity_1pct.json `
  --output-json reports\random_20w_split\loop101_current_state_gate_metadata.json `
  --output-md reports\random_20w_split\loop101_current_state_gate_metadata.md `
  --strict
```

Result: `decision=pass`, blockers `{}`.

Key evidence:

- replacement rows: `130`
- replacement status: `strict_extracted=130`
- self replacements: `0`
- current rows: `200000`
- current covered rows: `200000`
- missing rows: `0`
- metadata validation: `true`
- metadata checked rows: `200000`
- metadata failure rows: `0`
- sampled integrity rows: `2000`
- sampled failed rows: `0`

Loop98 rerun:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop98_route_audit.py `
  --loop79-current-state reports\random_20w_split\loop101_current_state_gate_metadata.json `
  --output-json reports\random_20w_split\loop101_identity_safe_route_audit.json
```

Result: `decision=await_independent_blinded_verdicts`.

The route confirms metadata-ready fixed-v2 cache and redraw state, but still forbids immediate training, Test-10k, and full-test:

- `training_allowed_now=false`
- `test10k_allowed_now=false`
- `full_test_allowed_now=false`
- `ready_for_redraw_preflight=false`

## Identity Boundary

Path-like and identity fields remain loading and audit metadata only:

- filename
- path
- directory
- extension
- hash
- `source_sha256`
- `sample_index`
- split
- row order
- model score

They are still forbidden as model, threshold, fusion, GA mask, automatic verdict, replacement-sampling, relabel, or production inference evidence.

## Verification

Resource guards:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop79_current_state_gate.py `
  --output-json reports\random_20w_split\loop101_loop79_guard.json

.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop98_route_audit.py `
  --output-json reports\random_20w_split\loop101_loop98_guard.json
```

Tests:

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\build_loop79_current_state_gate.py scripts\build_loop98_route_audit.py
.\vnev\Scripts\python.exe -m pytest tests\test_build_loop79_current_state_gate.py tests\test_build_loop98_route_audit.py -q
```

Result: `9 passed`.

## Current Decision

The current corrected 20w cache is acceptable for future Val-first work only after a valid upstream reason exists. No automatic model route is open. The next legitimate route is still independent blinded verdicts for the full-error queue, followed by non-destructive quarantine plus fresh same-original-label redraw if bad rows are confirmed.
