# Phase 3 Loop47: Checkpoint Provenance Audit for Diverse Stacking

Date: 2026-07-02

## Objective

Loop41-46 triggered strategy review. The next promising direction in the main
recommendation doc is true diverse OOF stacking, but that is only valid if the
base checkpoints were trained on the current train split and did not see the
current Val/Test rows. Loop47 audits existing checkpoints before any stacking
attempt.

This loop is read-only. It does not train, tune thresholds, run Test-10k, or use
checkpoint predictions as model inputs.

## Protocol

Inputs:

- split: `reports/random_20w_split/loop27_corrected_split.csv`
- checkpoint root: `models/`

Implementation:

- `scripts/audit_checkpoint_provenance.py`
- `tests/test_audit_checkpoint_provenance.py`

The audit checks checkpoint metadata and path provenance against the current
random 20w 8192/fixed-v2 signature:

- `max_byte_length = 8192`
- `pe_feature_dim = 256`
- `stat_feature_dim = 49`
- `pe_schema_version = fixed_v2`
- `pe_fixed_section_slots = 32`

Identity fields remain audit-only. Filename, path, extension, directory,
source hash, sample id, split, and row order are not model features.

## Command

```powershell
.\vnev\Scripts\python.exe scripts\audit_checkpoint_provenance.py `
  --models-dir models `
  --split-csv reports\random_20w_split\loop27_corrected_split.csv `
  --output-json reports\random_20w_split\loop47_checkpoint_provenance_audit\checkpoint_provenance_audit.json
```

## Result

Total checkpoints scanned: `177`

| Status | Count |
| --- | ---: |
| compatible_current_random20w | `1` |
| provenance_mismatch | `2` |
| incompatible | `40` |
| unknown | `134` |

Only one checkpoint is clearly compatible with the current random 20w 8192
split:

- `models/random_20w_8192/best_model.pt`

The compatible checkpoint uses:

- `max_byte_length=8192`
- `pe_feature_dim=256`
- `stat_feature_dim=49`
- `pe_schema_version=fixed_v2`
- `pe_fixed_section_slots=32`

Decision from the audit report:

```text
current repo has no safe diverse current-split checkpoint pool
```

## Decision

Do not build a multi-checkpoint stack from the existing checkpoint directory.

Most available checkpoints come from older experiments, group-isolated subsets,
comparison-cache experiments, hard replay / fine-tuning, incompatible
architectures, or unknown provenance. Using them as base learners for the
current 20w split would risk hidden data leakage or incomparable training
distributions.

The next valid version of this idea requires training fresh current-split
checkpoints, for example:

- same `loop27_corrected_split.csv`
- same fixed-v2 8192 cache/manifest
- different seed or byte length
- train split only for fitting
- Val only for model/threshold/stacker selection
- Test-10k only after the Val gate is passed

## Artifacts

- Audit report:
  `reports/random_20w_split/loop47_checkpoint_provenance_audit/checkpoint_provenance_audit.json`

## Verification

Commands run:

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\audit_checkpoint_provenance.py
.\vnev\Scripts\python.exe -m pytest tests\test_audit_checkpoint_provenance.py
```

Result: `1 passed`.
