# Identity Feature Policy

## Rule

Filename, path, extension, directory name, `source_sha256`, `cache_path`,
`sample_index`, `split`, and row order are not model features.

They may only be used for:

- loading files or cache rows
- joining predictions back to source samples
- cache coverage audits
- duplicate or content-group detection
- manual review queues
- building a one-time split/label manifest from human-curated roots

They must not be used for:

- training features
- model blending features
- threshold tuning shortcuts
- relabel evidence
- production inference decisions

## Why

Real deployment filenames and paths are usually unrelated to training-corpus
names, and attackers can rename files at essentially zero cost. A model that
learns external naming patterns will look strong in a local corpus and then
fail in the field.

Dataset roots such as `待加入白名单` and `待拉黑` are only human labeling buckets.
After the split CSV or manifest has a label, the model evidence must come from
file content: bytes, PE structure, statistics, or other content-derived
features.

## Current Guard

`scripts/identity_feature_guard.py` enforces this at feature-name level. It is
currently wired into:

- `scripts/train_stage2_cache_matrix.py`
- `scripts/train_stage2_oof_stacker.py`

If a future Stage-2 experiment introduces a feature named like `source_path`,
`file_extension_*`, `sample_index`, `split_*`, `filename_*`, or similar external
identity metadata, the training script raises before writing the selected model.
