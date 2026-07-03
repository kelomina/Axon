# Phase 3 Loop79 Current State Gate

## Purpose

Loop79 turns the current 20w protocol state into a single read-only gate. It
does not train, evaluate, load checkpoints, open NPZ arrays, mutate cache, or
scan raw data. It consolidates strict evidence for four items that were easy to
confuse across older reports:

- the historical fixed-v2 130-row cache failure has been handled by fresh
  same-label redraw, not by filling bad samples;
- the current authoritative split is still exactly `200000 = 20000 train +
  20000 val + 160000 test`;
- current cache coverage and sampled cache integrity are clean;
- probability calibration and GA feature-mask decisions remain separated.

## Evidence

Primary gate:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop79_current_state_gate.py `
  --current-cache-ready reports\random_20w_split\loop79_fresh_cache_ready.json `
  --current-coverage reports\random_20w_split\loop79_fresh_cache_coverage.json `
  --sample-integrity reports\random_20w_split\loop79_cache_sample_integrity_1pct_seed7901.json `
  --output-json reports\random_20w_split\loop79_current_state_gate.json `
  --output-md reports\random_20w_split\loop79_current_state_gate.md `
  --strict
```

Result: `decision=pass`.

Fresh cache readiness:

- `reports/random_20w_split/loop79_fresh_cache_ready.json`
- rows: `200000`
- split counts: train `20000`, val `20000`, test `160000`
- per-split label counts: train `10000/10000`, val `10000/10000`, test `80000/80000`
- coverage: `200000/200000`
- missing: `0`
- `cache_ready=true`
- `label_balance_enforced=true`

Fresh full coverage:

- `reports/random_20w_split/loop79_fresh_cache_coverage.json`
- coverage: `200000/200000`
- missing: `0`

Fresh 1% sampled integrity:

- guard: `reports/random_20w_split/loop79_fresh_sample_integrity_guard.json`
- report: `reports/random_20w_split/loop79_cache_sample_integrity_1pct_seed7901.json`
- detail CSV: `reports/random_20w_split/loop79_cache_sample_integrity_1pct_seed7901_detail.csv`
- seed: `7901`
- sampled rows: `2000`
- sampled split counts: train `200`, val `200`, test `1600`
- sampled labels: `0=1000`, `1=1000`
- declared shapes: byte `8192`, PE `256`, stat `49`, lightweight `256`
- failed rows: `0`
- `audit_ready=true`

## 130-row Fixed-v2 Resolution

The old failure source remains:

- `reports/random_20w_split/random_20w_8192_uncompressed_cache_rebuild_full_current_split.json`
- `reports/random_20w_split/random_20w_8192_uncompressed_cache_coverage_audit.json`

Those reports showed `199870/200000` coverage and `130` strict PE extraction
failures. They are historical failure evidence, not the current training-ready
state.

The current replacement evidence is:

- `reports/random_20w_split/random_20w_8192_replace_130_bad_features.json`
- `reports/random_20w_split/random_20w_8192_replacement_130_strict.csv`
- `reports/random_20w_split/random_20w_8192_uncompressed_cache_coverage_audit_replaced_130.json`

Loop79 verifies:

- replacement rows: `130`
- selection status: `strict_extracted=130`
- self replacements: `0`
- replacement counts match the old failed slots by split and label:
  - `train:0=12`
  - `val:0=19`
  - `test:0=94`
  - `test:1=5`
- manifest after replacement: `200000`
- split after replacement: `200000`
- replacement cache format: `uncompressed`
- missing after replacement: `0`

This satisfies the operational rule: bad feature files triggered fresh
same-label redraw. They were not used to fill their own failed cache rows.

## Probability Calibration

Loop79 preserves the A/B strict conclusion:

- source: `reports/model_review/final_model_selection/ab_strict_reverification_report.json`
- conclusion: `strictly_reverified_useful`
- no test used for training: `true`
- all strict rows kept: `true`

Current replacement-split calibrator evidence:

- train/val: `reports/random_20w_split/random_20w_8192_replaced_calibrator_train_val.json`
- Test-10k: `reports/random_20w_split/random_20w_8192_replaced_calibrator_test10k_eval.json`
- Val delta F1: `+0.03789303556617796`
- Test-10k delta F1: `+0.04255785450909355`
- Test-10k delta errors: `-412`
- skipped missing cache rows: `0`

This keeps the funnel valid: train split fits the calibrator, val selects the
threshold, and Test-10k is confirmation evidence only.

## GA Feature Mask

Loop79 also preserves the stricter GA feature-mask verdict:

- source: `reports/model_review/final_model_selection/ab_strict_reverification_report.json`
- conclusion: `strictly_reverified_high_security_candidate_not_default`
- 20k validation delta errors: `-130`
- hard holdout row-count mismatches: none
- high-value benign FP delta: `+34`

The mask remains a candidate for special high-security/recall-biased use, not a
default replacement for the full feature set. The high-value benign FP increase
is the blocking reason for making it default.

## Identity Evidence Policy

Identity fields are forbidden as model, threshold, GA, or noise-cleaning
evidence:

- filename
- path
- extension
- directory
- hash
- source_sha256
- sample_index
- split
- row order

They remain allowed only for loading, cache alignment, manifest audit, duplicate
detection, and manual review indexing.

## Validation

Commands run after the Loop77 guard:

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_build_loop79_current_state_gate.py -q
.\vnev\Scripts\python.exe -m pytest `
  tests\test_build_ab_strict_reverification_report.py `
  tests\test_audit_split_cache_coverage.py `
  tests\test_audit_corrected_split_cache_ready.py `
  tests\test_audit_loop78_cache_sample_integrity.py -q
```

Results:

- Loop79 tests: `2 passed`
- related cache/A-B/Loop78 tests: `13 passed`

The next allowed action is Val-first model or calibrator work from the current
corrected 20w split. GA mask must remain non-default unless a later Val-first
experiment resolves the high-value benign FP regression.
