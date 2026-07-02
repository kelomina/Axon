# Phase 3 Loop48: Fresh Seed43 Current-Split Val-only Check

Date: 2026-07-02

## Objective

Loop47 showed that the existing `models/` directory does not contain a safe
pool of diverse checkpoints for the current random 20w split. Loop48 therefore
tested one fresh current-split neural checkpoint with seed `43`, using the same
fixed-v2 8192-byte cache protocol, to see whether a simple seed change is worth
promoting into later OOF stacking.

This loop is Val-only. It does not tune on Test-10k, does not run full-test, and
does not use filename, path, extension, directory, hash, sample id, split, or row
order as model evidence.

## Inputs

- split: `reports/random_20w_split/loop27_corrected_split.csv`
- cache manifest: `data/.cache/manifest_38672ba0.json`
- config: `config/random_20w_8192_seed43.toml`
- checkpoint directory: `models/random_20w_8192_seed43`

The seed43 config keeps the same current-split model/data signature:

- `max_byte_length = 8192`
- `pe_feature_dim = 256`
- `stat_feature_dim = 49`
- `pe_schema_version = fixed_v2`
- `pe_fixed_section_slots = 32`
- `seed = 43`

## Split And Cache Audit

The current split/cache coverage re-audit reports:

- total rows: `200000`
- covered rows: `200000`
- missing rows: `0`
- coverage ratio: `1.0`

The split remains the strict `20000 train / 20000 val / 160000 test` protocol.
When a sample is later confirmed as label-wrong, feature-broken, or out of
scope, it must be replaced by a fresh same-label draw through the replacement
preflight. Bad rows are not used to "fill" the quota.

Coverage artifact:

- `reports/random_20w_split/current_split_cache_coverage_reaudit.json`

## Identity Feature Rule

This experiment follows `docs/identity_feature_policy.md`:

- allowed: using path/hash/cache metadata for loading, joining, coverage audit,
  duplicate detection, and manual review
- forbidden: using filename, path, extension, directory, `source_sha256`,
  `sample_index`, split, or row order as training features, blending features,
  threshold shortcuts, relabel evidence, or production inference evidence

The reason is operational, not stylistic: deployment filenames and training
corpus filenames are different distributions, and adversaries can rename files
at essentially zero cost.

## Result

Smoke run:

- 1 epoch smoke Val F1: `0.8158`
- output directory: `models/smoke_random_20w_8192_seed43_e1`

Full seed43 training was stopped early after it became clear that the candidate
was far below the current Loop28 reference. The retained checkpoint is:

- checkpoint: `models/random_20w_8192_seed43/best_model.pt`
- epoch: `17`
- last_epoch: `17`
- best Val F1: `0.9500494559841741`
- checkpoint mtime: `2026-07-02T13:46:30.801599`
- checkpoint size: `10359449` bytes
- final checkpoint: missing

Reference to beat:

- Loop28 content PE metadata Val F1: `0.9919048571`
- Loop28 Val errors: `162`
- Loop28 Test-10k F1: `0.9888677164`
- Loop28 full-test F1: `0.9878358558`

## Decision

Reject seed43 as a promotion candidate.

The candidate is around `4.19` F1 percentage points below Loop28 on Val, so it
does not qualify for Test-10k and should not be used as a Stage-2 base learner.
It is also not worth exporting predictions for stacker training in its current
form, because weak same-architecture seed variation is not the kind of diversity
Loop36/Loop47 require.

## Implication

Do not keep blindly training the same 8192 fixed-v2 neural recipe with only seed
changes. The next valid diverse-checkpoint attempt should change at least one
meaningful information axis, for example:

- byte length or region view
- model architecture or fusion constraint
- stable content PE metadata schema
- independently trained OOF base learners with current-split provenance

Noise work remains a separate first-class path. High-confidence conflicts from
Loop38/Loop39 should be manually adjudicated, and confirmed bad rows must go
through fresh same-label replacement while preserving exact `200000` rows.

## Verification

Already run in this branch:

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\evaluate_split_from_cache.py scripts\train_stage2_cache_matrix.py scripts\analyze_val_prediction_ensemble.py scripts\evaluate_prediction_blend.py scripts\identity_feature_guard.py scripts\train_stage2_oof_stacker.py

.\vnev\Scripts\python.exe -m pytest tests\test_analyze_val_prediction_ensemble.py tests\test_analyze_val_prediction_ensemble_alignment.py tests\test_evaluate_prediction_blend.py -q

.\vnev\Scripts\python.exe -m pytest tests\test_identity_feature_guard.py -q
```

Results recorded from the run:

- prediction/ensemble tests: `5 passed`
- identity feature guard tests: `3 passed`
