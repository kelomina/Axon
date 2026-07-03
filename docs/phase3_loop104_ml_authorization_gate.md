# Phase 3 Loop104 ML Authorization Gate

Date: 2026-07-03

## Purpose

Loop104 hardens the ML authorization preflight so completed experiment records
cannot be mistaken for permission to train, tune thresholds, run Test-10k, or
run the 160k full test.

The gate is read-only. It does not train, load models, open NPZ arrays, mutate
cache, or scan raw data.

## Inputs

- `reports/random_20w_split/loop101_identity_safe_route_audit.json`
- `reports/random_20w_split/loop101_current_state_gate_metadata.json`
- `reports/random_20w_split/loop100_cache_ready_metadata.json`
- `reports/model_review/final_model_selection/ml_experiment_authorization_plan.json`
- `reports/model_review/final_model_selection/ml_recommendation_status.json`

## Output

- `reports/random_20w_split/loop104_ml_authorization_preflight.json`

Key result:

- `ml_gate_result.passed=false`
- `allowed_operations=[]`
- `train_val_allowed=false`
- `threshold_sweep_allowed=false`
- `test10k_allowed=false`
- `full_test_allowed=false`
- `redraw_preflight_allowed=false`

The cache and split gates pass:

- total rows: `200000`
- split: `20000/20000/160000`
- covered rows: `200000`
- missing rows: `0`
- metadata checked rows: `200000`
- metadata failure rows: `0`

The route gate still blocks heavy operation:

- `decision=await_independent_blinded_verdicts`
- `actionable_rows=0`
- `replacement_required_rows=0`
- `training_policy_rows=0`

## Identity Boundary

Filename, path, extension, directory, hash, `source_sha256`, `sample_index`,
split, row order, and model score are logistics metadata only. They are allowed
for loading, alignment, cache audit, duplicate detection, and manual/external
review indexing.

They are not allowed as model evidence, verdict evidence, feature-mask input,
threshold evidence, fusion evidence, replacement-sampling evidence, relabel
evidence, or production inference evidence.

If original corpus labels were bootstrapped from curated directories, that step
ends at the locked manifest/split label. Fresh redraws use the locked manifest
same-original-label pool, not names or paths.

## Verification

Resource/static guard:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts/pre_run_resource_leak_guard.py --target-script scripts/build_ml_authorization_preflight.py --target-script tests/test_build_ml_authorization_preflight.py --output-json reports/random_20w_split/loop104_ml_authorization_guard_rerun.json
```

Result: pass.

Tests:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" -m pytest tests/test_build_ml_authorization_preflight.py tests/test_build_loop79_current_state_gate.py tests/test_build_loop98_route_audit.py tests/test_audit_corrected_split_cache_ready.py
```

Result: `27 passed`.

Report generation:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts/build_ml_authorization_preflight.py --authorization-plan reports/model_review/final_model_selection/ml_experiment_authorization_plan.json --status-json reports/model_review/final_model_selection/ml_recommendation_status.json --route-audit reports/random_20w_split/loop101_identity_safe_route_audit.json --current-state-gate reports/random_20w_split/loop101_current_state_gate_metadata.json --cache-ready reports/random_20w_split/loop100_cache_ready_metadata.json --output-json reports/random_20w_split/loop104_ml_authorization_preflight.json
```

Result: report generated.
