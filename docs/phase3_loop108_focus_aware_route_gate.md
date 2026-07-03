# Phase 3 Loop108 Focus-Aware Route Gate

Date: 2026-07-03

## Purpose

Loop108 reconnects the Loop107 focus-annotation no-op result to the high-level
route audit and ML authorization preflight. This proves that the 240-row focus
review path is wired into the gate, while still blocking training and Test-10k
until independent content or external verdicts exist.

The loop is read-only. It does not train, tune thresholds, load checkpoints,
open NPZ feature arrays, rebuild cache, replace samples, or mutate split files.
Any JSON field named `evidence` in the generated route/preflight reports means
authorization audit evidence only; it is not model evidence, verdict evidence,
threshold evidence, fusion evidence, relabel evidence, replacement-sampling
evidence, or production inference evidence.

## Inputs

- `reports/random_20w_split/loop101_current_state_gate_metadata.json`
- `reports/random_20w_split/loop100_cache_ready_metadata.json`
- `reports/random_20w_split/loop107_focus_merged_verdict_import_noop.json`
- `reports/random_20w_split/loop80_calibrator_fulltest_summary.json`
- `reports/random_20w_split/loop85_noise_strategy_gate.json`
- `reports/random_20w_split/loop95_full_queue_review_evidence_intake.json`
- `reports/random_20w_split/loop96_full_queue_blinded_review.json`
- `reports/random_20w_split/loop97_speakeasy_triage_decision.json`

## Outputs

- `reports/random_20w_split/loop108_focus_route_guard.json`
- `reports/random_20w_split/loop108_ml_preflight_guard.json`
- `reports/random_20w_split/loop108_focus_aware_route_audit_noop.json`
- `reports/random_20w_split/loop108_focus_aware_ml_authorization_preflight_noop.json`
- `reports/random_20w_split/loop108_focus_aware_route_summary.json`

## Result

The fixed-v2 cache and current split remain healthy:

- total rows: `200000`
- split: `20000/20000/160000`
- covered rows: `200000`
- missing rows: `0`
- metadata checked rows: `200000`
- metadata failure rows: `0`
- replacement rows: `130`
- self replacements: `0`

The focus-aware route remains blocked:

- route decision: `await_independent_blinded_verdicts`
- Loop87 decision: `ready_noop_no_actionable_verdicts`
- actionable rows: `0`
- replacement required rows: `0`
- training policy rows: `0`
- Train/Val allowed: `false`
- threshold sweep allowed: `false`
- Test-10k allowed: `false`
- full-test allowed: `false`
- redraw preflight allowed: `false`

## Identity Boundary

This loop explicitly keeps filename, path, extension, directory, hash,
`source_sha256`, `sample_index`, split, row order, and model score out of model
and verdict evidence. Those fields are allowed only for loading, alignment,
cache audit, duplicate detection, and manual or external review indexing.

This directly addresses the deployment issue: real-world names do not follow
training-corpus names, and an attacker can rename a file at essentially zero
cost. Directory or filename inference can only bootstrap a human-curated label
manifest. After the locked 20w split exists, noisy rows must be resolved by
independent content/external evidence, quarantine, and fresh redraw from the
locked manifest same-original-label pool.

## Commands

Resource/static guards:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\pre_run_resource_leak_guard.py --target-script scripts/build_loop98_route_audit.py --output-json reports/random_20w_split/loop108_focus_route_guard.json

& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\pre_run_resource_leak_guard.py --target-script scripts/build_ml_authorization_preflight.py --output-json reports/random_20w_split/loop108_ml_preflight_guard.json
```

Focus-aware route audit:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\build_loop98_route_audit.py --loop79-current-state reports/random_20w_split/loop101_current_state_gate_metadata.json --loop80-calibrator-fulltest reports/random_20w_split/loop80_calibrator_fulltest_summary.json --loop85-noise-strategy reports/random_20w_split/loop85_noise_strategy_gate.json --loop95-intake reports/random_20w_split/loop95_full_queue_review_evidence_intake.json --loop96-blinded-review reports/random_20w_split/loop96_full_queue_blinded_review.json --loop96-verdict-import reports/random_20w_split/loop107_focus_merged_verdict_import_noop.json --loop97-speakeasy reports/random_20w_split/loop97_speakeasy_triage_decision.json --output-json reports/random_20w_split/loop108_focus_aware_route_audit_noop.json
```

Focus-aware ML authorization preflight:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\build_ml_authorization_preflight.py --authorization-plan reports/model_review/final_model_selection/ml_experiment_authorization_plan.json --status-json reports/model_review/final_model_selection/ml_recommendation_status.json --route-audit reports/random_20w_split/loop108_focus_aware_route_audit_noop.json --current-state-gate reports/random_20w_split/loop101_current_state_gate_metadata.json --cache-ready reports/random_20w_split/loop100_cache_ready_metadata.json --output-json reports/random_20w_split/loop108_focus_aware_ml_authorization_preflight_noop.json
```

## Verification

Resource/static guards: pass.

Tests:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" -m pytest tests/test_build_loop98_route_audit.py tests/test_build_ml_authorization_preflight.py
```

Result: `13 passed`.
