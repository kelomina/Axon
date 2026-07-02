# Identity Feature Policy

## Rule

Filename, path, extension, directory name, `source_sha256`, `cache_path`,
`sample_index`, `split`, and row order are not model features. Alias fields
such as `manifest_source_sha256`, `file_hash`, `filename_hash`, `row_number`,
`record_index`, and `dir_name` follow the same rule.

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

## Label Manifests

`label_inference=filename` or `label_inference=directory` is only a bootstrap
mechanism for turning a human-curated corpus into an explicit label manifest.
It is not a modeling signal and should not be used once an official split CSV
or cache manifest exists.

For the 20w protocol, the frozen split/manifest label is the label source.
Names, paths, extensions, directory buckets, hashes, `sample_index`, split
membership, and row order remain identity fields only. If those labels are
suspect, the fix is manual or external-evidence adjudication followed by a
fresh same-original-label redraw, not training on naming patterns.

## Current Guard

`scripts/identity_feature_guard.py` enforces this at feature-name level. It is
currently wired into:

- `scripts/train_stage2_cache_matrix.py`
- `scripts/train_stage2_oof_stacker.py`
- Loop57/Loop61-style residual gate feature construction
- `scripts/audit_loop68_residual_oof_readiness.py`
- `scripts/materialize_loop69_nested_oof_override.py`
- `scripts/train_loop70_nested_oof_meta.py`

If a future Stage-2 experiment introduces a feature named like `source_path`,
`file_extension_*`, `sample_index`, `split_*`, `filename_*`,
`manifest_source_sha256`, `file_hash`, `row_number`, `record_index`, `dir_name`,
or similar external identity metadata, the training script raises before writing
the selected model.

Loop68 adds a separate readiness check for stacked residual experiments. It does
not train anything; it verifies that a candidate has row-level train final OOF
predictions before another residual layer is allowed. Identity fields in those
prediction CSVs remain alignment-only columns, not model inputs.

Loop69 materializes train-only nested OOF predictions so a later residual layer
can train without seeing in-fold upstream predictions. Its identity columns are
for row alignment and cache audit only. Loop70 trains a meta layer from Loop69
OOF score fields and validates on Val; it explicitly excludes fold id, path,
hash, filename, directory, extension, split, `correct`, and row-order fields
from the model matrix.

Loop71 is a read-only target-gap and review-ROI audit. It may read full-test
errors to quantify feasibility and prioritize manual or external-evidence
review, but it does not train, tune thresholds, mutate splits, auto-relabel, or
turn full-test error identity fields into model rules.
