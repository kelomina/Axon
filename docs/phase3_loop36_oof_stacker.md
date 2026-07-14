# Loop36 Strict OOF Stage-2 Stacker

Date: 2026-07-02

## Objective

Loop36 tested whether a strict out-of-fold Stage-2 stacker can improve on the
Loop28 content PE model.

The experiment was designed to avoid training-time prediction leakage:

- Train split was split into 5 stratified folds.
- Each train row's base-learner score came only from a base learner trained on
  the other folds.
- The meta model was trained only on these OOF base scores.
- Val scores came from base learners trained on the full train split.
- Val selected the meta model and threshold.
- No Test-10k was used because Val did not beat Loop28.

Strict mode also used `--drop-base-prob-features`, which removes the six
exported base-probability features from the Stage-2 feature matrix before
training base learners. This avoids using non-OOF train-side model probabilities
as base-learner input.

No filename, path, extension, directory, source hash, sample id, split, label,
prediction, or correctness field was used as a model feature.

## Code

New files:

- `scripts/train_stage2_oof_stacker.py`
- `scripts/evaluate_stage2_oof_stacker.py`
- `tests/test_stage2_oof_stacker.py`

The frozen evaluator is present for future candidates, but Loop36 did not enter
Test-10k.

## Configuration

Base features:

- Extended Stage-2 cache matrix.
- Loop28 content PE v1 sidecar.
- Dropped first 6 exported probability-derived features.

Base learners:

- `hgb_lr0.04_leaf15_l2_0`
- `hgb_lr0.06_leaf31_l2_0`
- `hgb_lr0.08_leaf31_l2_1e-3`

Meta candidates:

- logistic regression C=0.1
- logistic regression C=1.0
- balanced logistic regression C=0.1
- small HGB leaf7
- small HGB leaf15

## Result

Best Val meta model:

- Model: `meta_logreg_l2_c1`
- Threshold: `0.47`
- Val F1: `0.9917594766`
- Val errors: `165`
- FP/FN: `94/71`

Comparison:

| Candidate | Val F1 | Val errors | FP/FN | Decision |
| --- | ---: | ---: | ---: | --- |
| Loop28 content PE | `0.9919048571` | `162` | `87/75` | Current best |
| Loop36 strict OOF stacker | `0.9917594766` | `165` | `94/71` | Reject |

Loop36 missed the Val gate by 3 errors and therefore did not enter Test-10k.

## Interpretation

OOF stacking remains a valid protocol, but this specific implementation is not
enough. The base learners were too similar: all three were HGB variants over the
same content PE v1 matrix. The next stacking attempt needs genuinely diverse
base predictions, such as multiple neural checkpoints/seeds, different byte
lengths, or independent feature families. Re-stacking near-identical HGB models
is not a useful next loop.

## Verification

Commands run:

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\train_stage2_oof_stacker.py scripts\evaluate_stage2_oof_stacker.py
.\vnev\Scripts\python.exe -m pytest tests\test_stage2_oof_stacker.py tests\test_stage2_content_pe_features.py
```
