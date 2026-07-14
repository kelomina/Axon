# Phase 3 Loop85 Noise Strategy Gate

## Purpose

Loop85 is a read-only strategy gate after Loop82-84 rejected the current
calibrator-fusion route. It consolidates the current best full-test result,
persistent-error review evidence, duplicate/content health audits, and Val-only
fusion diagnostics into one decision point.

It does not train, tune thresholds, relabel, mutate the split, mutate cache, or
access Test/Test-10k for candidate selection.

## Identity Policy

Filename, path, extension, directory, hash, `source_sha256`, `sample_index`,
split, and row order are forbidden as model evidence.

Allowed uses are limited to loading, alignment, cache audit, duplicate
detection, and manual/external review indexing.

The important boundary is this:

- file or directory names may be a bootstrap source when creating a one-time
  human-curated label manifest, if no independent label list exists;
- after the 20w split/manifest is locked, labels come from that manifest;
- replacement redraws are selected from the locked manifest's original-label
  pool, not by filename, extension, path, directory, hash, split, row order, or
  other identity similarity.

This matches the deployment reality: real-world malware and benign filenames
do not follow the training-corpus naming convention, and attackers can rename
files cheaply.

## Commands

Guard:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop85_noise_strategy_gate.py `
  --output-json reports\random_20w_split\loop85_noise_strategy_guard.json
```

Result: `decision=pass`.

Tests:

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_build_loop85_noise_strategy_gate.py -q
```

Result before report generation: `2 passed`.

Report:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop85_noise_strategy_gate.py `
  --loop57-full-eval reports\random_20w_split\loop57_fn_overlay_gate_frozen_full_test_eval.json `
  --loop63-queue-summary reports\random_20w_split\loop63_persistent_error_review_queue_summary.json `
  --loop63-health-summary reports\random_20w_split\loop63_A_persistent_conflict_content_audit_summary.json `
  --loop64-duplicate-summary reports\random_20w_split\loop64_manifest_sha_duplicate_audit.json `
  --loop65-review-summary reports\random_20w_split\loop65_A_lane_review_batch_summary.json `
  --loop82-complementarity reports\random_20w_split\loop82_same_manifest_val\loop82_val_complementarity.json `
  --loop83-rescue-profile reports\random_20w_split\loop82_same_manifest_val\loop83_calibrator_rescue_profile.json `
  --loop84-content-rescue reports\random_20w_split\loop82_same_manifest_val\loop84_content_rescue_separability.json `
  --output-json reports\random_20w_split\loop85_noise_strategy_gate.json
```

Result: exit code `0`, blockers `[]`.

## Result

Current best remains Loop57 FN overlay gate:

- full-test rows: `160000`
- F1: `0.9883629658239992`
- errors: `1868`
- FP/FN: `1195 / 673`

Target gap:

- approximate `F1 >= 0.999` full-test error budget: `160`
- minimum error reduction still needed: `1708`

Noise and review evidence:

- Loop63 queue covers every Loop57 full-test error: `1868/1868`
- persistent/high-conflict A-lane: `643` rows
- A-lane objective cache/source/strict-PE issue rows: `0`
- manifest duplicate groups: `6`, detail rows `12`
- cross-label duplicate groups: `0`
- cross-split duplicate groups: `0`
- Loop65 compact review batch: `62` rows, manual fields blank

Fusion evidence:

- Loop82 same-manifest Val alignment passed with `20000/20000` unique rows
- calibrator-only-correct: `56`
- Loop57-only-correct: `463`
- Loop83 score-delta rule does not improve Loop57
- Loop84 content selector is not promising enough
- current calibrator-fusion route must not enter Test-10k

## Decisions

- Automatic replacement allowed: `false`
- Automatic relabel allowed: `false`
- Test-10k allowed for current calibrator fusion: `false`
- Next phase: manual/external-evidence noise review before more fusion

Replacement rule:

If `label_wrong`, `feature_broken`, or `out_of_scope` is confirmed, quarantine
the bad row and fresh-redraw one valid sample from the same locked-manifest
original-label pool. Do not use bad samples to fill counts, and do not choose
replacements by filename/path/directory similarity.

## Verdict

Stop the current calibrator-fusion path. The next useful work is not more
score mixing, and not naming-based cleanup. It is evidence-grade noise review:
use Loop65 as the first compact manual/external review batch, then expand toward
the full Loop63 persistent-error queue if the review channel can handle it.

Any confirmed bad row triggers quarantine plus fresh redraw under the strict
`20000 train / 20000 val / 160000 test` protocol.
