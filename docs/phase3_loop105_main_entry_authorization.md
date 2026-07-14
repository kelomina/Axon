# Phase 3 Loop105 Main Entry Authorization

Date: 2026-07-03

## Purpose

Loop105 adds a lightweight authorized entrypoint for official 20w train/eval
commands. Loop104 already produces the machine-readable preflight decision, but
operators could still copy older package commands that call `scripts/main.py`
directly. This loop makes the approved command path explicit.

## Added Files

- `scripts/ml_authorization_runtime.py`
- `scripts/authorized_main.py`
- `tests/test_ml_authorization_runtime.py`

`scripts/authorized_main.py` parses only the small set of arguments needed to
classify the requested operation before importing `scripts/main.py`. When the
preflight blocks an operation, it exits before loading torch, checkpoints,
models, data, CUDA, or NPZ cache.

## Operation Mapping

- non-fast `train --skip-test-eval`: requires `train_val_allowed`
- non-fast `train` without `--skip-test-eval`: requires `train_val_allowed` and `full_test_allowed`
- `eval --split test` without `--max-eval-samples`: requires `full_test_allowed`
- `eval --split test --max-eval-samples ...`: requires `test10k_allowed`
- any eval threshold sweep or decision-threshold override: requires `threshold_sweep_allowed`
- fast/smoke training does not require the official 20w heavy-operation gate

## Package Command Hardening

The hard-error and hard-family fine-tune package builders now generate official
train/eval commands through:

```powershell
scripts\authorized_main.py --ml-preflight "reports\random_20w_split\loop104_ml_authorization_preflight.json" -- ...
```

This keeps copied README commands from bypassing the preflight. Prediction export
commands are unchanged because they operate on explicit sample CSVs and are not
the official 20w Train/Val/Test gate.

## Current Real-World Behavior

With current Loop104 evidence, full-test is blocked before any checkpoint or data
load:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts/authorized_main.py --ml-preflight reports/random_20w_split/loop104_ml_authorization_preflight.json -- eval --checkpoint missing.pt --data-dir missing-data --split test
```

Result:

```text
[ML Authorization Blocked] ML operation is blocked by authorization preflight. required=['full_test']; full_test: no_actionable_independent_verdicts, route_audit_awaits_independent_blinded_verdicts, route_audit_full_test_allowed_now_false
```

## Verification

Resource/static guard:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts/pre_run_resource_leak_guard.py --target-script scripts/ml_authorization_runtime.py --target-script scripts/authorized_main.py --target-script scripts/build_hard_error_finetune_package.py --target-script scripts/build_hard_family_finetune_package.py --target-script tests/test_ml_authorization_runtime.py --target-script tests/test_build_hard_error_finetune_package.py --output-json reports/random_20w_split/loop105_entry_authorization_guard_rerun.json
```

Result: pass.

Tests:

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" -m pytest tests/test_ml_authorization_runtime.py tests/test_build_ml_authorization_preflight.py tests/test_build_hard_error_finetune_package.py
```

Result: `16 passed`.

Direct-command scan:

```powershell
rg -n "scripts\\main.py (train|eval)|scripts/main.py (train|eval)" scripts/build_hard_error_finetune_package.py scripts/build_hard_family_finetune_package.py tests/test_build_hard_error_finetune_package.py
```

Result: no matches.
