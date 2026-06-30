# Stage2 kNN Conflict-Filter Experiment

## Protocol

This experiment tests whether train-only out-of-fold kNN conflict evidence can improve the current Stage2 model by downweighting or trimming suspected noisy train rows.

The evaluation follows the locked funnel:

1. Build features from train cache and train predictions.
2. Generate kNN conflict evidence only from train OOF neighborhoods.
3. Select model, noise mode, and threshold on the full 20k validation split.
4. Do not run Test-10k unless the selected validation result beats the current best validation F1.

No validation or test label is used to define the kNN conflict rule.

## Candidate

- Branch: `exp/model-agent-knn-conflict-filter`
- Base checkpoint: `models/random_20w_8192/best_model.pt`
- Train predictions: `reports/random_20w_split/random_20w_8192_replaced_train_predictions.csv`
- Val predictions: `reports/random_20w_split/random_20w_8192_replaced_val_predictions.csv`
- Output report: `reports/random_20w_split/stage2_knn_conflict_filter_valonly/stage2_cache_matrix_report.json`

The script now supports these additional train-only noise modes:

- `knn_soft_conflict_downweight`
- `knn_trim_strong_conflict`
- `knn_trim_exact_opposite`

The default script behavior remains unchanged unless these modes are explicitly requested.

## Conflict Rule

Rule version: `train_oof_knn_conflict_v2`

The conflict rule uses OOF kNN features from the training split:

- Reference neighborhood: `knn25`
- Auxiliary neighborhood: `knn10`
- Similarity guard: `knn_top1_similarity`

For malicious-labeled rows, the opposite ratio is the benign neighbor ratio. For benign-labeled rows, the opposite ratio is the malicious neighbor ratio.

Observed train conflict counts:

- Medium conflict: `132 / 20000` (`0.66%`)
- Strong conflict: `41 / 20000` (`0.205%`)
- Exact-opposite conflict: `9 / 20000` (`0.045%`)
- Strong label-0 / label-1: `23 / 18`
- Exact-opposite label-0 / label-1: `9 / 0`

## Validation Result

Current best validation baseline:

- Model: `hgb_lr0.08_leaf31_l2_1e-3__noise_none`
- Threshold: `0.49`
- Val F1: `0.9833349965`
- Val errors: `334 / 20000`
- FP/FN: `188 / 146`

Best result in this experiment:

- Model: `hgb_lr0.08_leaf31_l2_1e-3__noise_none`
- Threshold: `0.49`
- Val F1: `0.9833349965`
- Val errors: `334 / 20000`

Top conflict-filter candidates:

- `hgb_lr0.04_leaf15_l2_0__noise_knn_trim_strong_conflict`: F1 `0.9833175639`, errors `335`
- `hgb_lr0.06_leaf31_l2_0__noise_knn_trim_exact_opposite`: F1 `0.9832692404`, errors `335`
- `hgb_lr0.06_leaf31_l2_0__noise_knn_soft_conflict_downweight`: F1 `0.9832675691`, errors `335`
- `hgb_lr0.08_leaf31_l2_1e-3__noise_knn_soft_conflict_downweight`: F1 `0.9832491625`, errors `335`

## Decision

The kNN conflict-filter implementation is valid and reportable, but it does not beat the current validation best. Under the locked funnel, this branch does not qualify for Test-10k.

This is a useful negative result: automatic train-row trimming/downweighting based on strict OOF neighborhood conflict is not enough to move the validation ceiling. The next phase should prioritize manual/business adjudication of the high-similarity label conflicts already identified in the Phase 2 review package, rather than more automatic deletion.

