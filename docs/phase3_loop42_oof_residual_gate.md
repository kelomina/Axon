# Phase 3 Loop42: Strict OOF Residual Gate

Date: 2026-07-02

## Objective

Loop38 showed that some Loop28 errors are corrected by other model views, but
Loop37 and Loop41 also showed that shallow linear blending is too unstable.
Loop42 tests a controlled override gate: keep Loop28-style Stage-2 as the base,
and allow an auxiliary candidate to override only when a gate predicts that the
override is likely to reduce errors.

This loop is Val-only. It does not evaluate Test-10k or full-test.

## Protocol

The experiment uses the full current train/val lane:

- train rows: `20000`
- val rows: `20000`
- cache misses: `0`
- checkpoint: `models/random_20w_8192/best_model.pt`
- train predictions: `reports/random_20w_split/loop27_train_predictions.csv`
- val predictions: `reports/random_20w_split/loop27_val_predictions.csv`

Base and candidate train scores are generated out-of-fold. The gate trains only
on train OOF signals and chooses gate model/threshold on Val. It does not use
Test-10k or full-test.

Identity fields remain audit/alignment only. Filename, path, extension,
directory, source hash, sample id, split, and row order are not model features.
The Val baseline alignment audit checked `20000/20000` SHA values.

## Implementation

New tooling:

- `scripts/train_loop42_oof_residual_gate.py`
- `tests/test_loop42_oof_residual_gate.py`

The gate feature set contains score-derived features from OOF/frozen model
scores. In this first run, `--gate-content-features` was not enabled, so the
gate did not consume the full content matrix. Stage-2 base/candidate learners
used content PE v1 and dropped the first six non-OOF base probability features.

Command:

```powershell
.\vnev\Scripts\python.exe scripts\train_loop42_oof_residual_gate.py `
  --checkpoint models\random_20w_8192\best_model.pt `
  --train-predictions reports\random_20w_split\loop27_train_predictions.csv `
  --val-predictions reports\random_20w_split\loop27_val_predictions.csv `
  --output-dir reports\random_20w_split\loop42_oof_residual_gate_valonly `
  --content-pe-features `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --drop-base-prob-features `
  --base-model-candidate hgb_lr0.06_leaf31_l2_0 `
  --candidate-model-candidates hgb_lr0.08_leaf31_l2_1e-3,extra_trees_300_leaf1 `
  --include-byte-ngram `
  --byte-ngram-n-features 2097152 `
  --byte-ngram-prefix-len 4096 `
  --byte-ngram-min 2 `
  --byte-ngram-max 5 `
  --byte-ngram-stride 2 `
  --byte-ngram-alpha 3e-6 `
  --byte-ngram-epochs 3 `
  --byte-ngram-batch-size 256 `
  --byte-ngram-include-byte-hist `
  --byte-ngram-include-cache-features `
  --neutral-weight 0.05 `
  --thresholds 0.35:0.65:0.005 `
  --gate-thresholds 0.50:0.95:0.01 `
  --folds 5 `
  --seed 42 `
  --baseline-val-predictions reports\random_20w_split\stage2_loop28_content_pe_valonly\stage2_val_predictions.csv `
  --baseline-probability-column stage2_prob_malicious `
  --alignment-key-column sample_index
```

## Results

Loop28 locked Val baseline:

- F1: `0.9919048571`
- errors: `162`
- FP/FN: `87/75`

Loop42 internal OOF base at its train-selected threshold:

- threshold: `0.47`
- Val F1: `0.9910134798`
- Val errors: `180`
- FP/FN: `105/75`

Best residual gate:

- candidate: `extra_trees_300_leaf1`
- gate: `gate_logreg_balanced_c0.25`
- base train threshold: `0.47`
- candidate train threshold: `0.495`
- gate threshold: `0.89`
- Val F1: `0.9920143741`
- Val errors: `160`
- FP/FN: `98/62`
- overrides: `74/20000`

Candidate comparison:

| Candidate | Best gate | Val F1 | Errors | FP/FN | Overrides |
| --- | --- | ---: | ---: | ---: | ---: |
| `extra_trees_300_leaf1` | `gate_logreg_balanced_c0.25` | `0.9920143741` | `160` | `98/62` | `74` |
| `byte_ngram_sgd` | `gate_logreg_balanced_c0.25` | `0.9912140575` | `176` | `104/72` | `137` |
| `hgb_lr0.08_leaf31_l2_1e-3` | `gate_hgb_leaf7` | `0.9910629587` | `179` | `104/75` | `2` |

## Decision

Reject for Test-10k.

Loop42 confirms that controlled residual overrides are a real signal: the best
gate reduces the internal base from `180` Val errors to `160`. However, the
official comparison is Loop28 locked Val at `162` errors, so the net gain is
only `2` errors. After Loop37 and Loop41, this margin is too thin.

The Loop42 Test-10k entry gate was set to `<=152` Val errors for shallow
override/blend-like candidates. The selected gate reached `160`, so
`test_gate_decision=reject_val_margin_too_small`.

## Artifacts

- Report:
  `reports/random_20w_split/loop42_oof_residual_gate_valonly/loop42_oof_residual_gate_report.json`
- Val predictions:
  `reports/random_20w_split/loop42_oof_residual_gate_valonly/loop42_oof_residual_gate_val_predictions.csv`
- Selected model payload:
  `reports/random_20w_split/loop42_oof_residual_gate_valonly/loop42_oof_residual_gate_selected_model.pkl`

Generated model/prediction artifacts are not committed because they are large
experiment outputs.

## Verification

Commands run:

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\train_loop42_oof_residual_gate.py
.\vnev\Scripts\python.exe -m pytest tests\test_loop42_oof_residual_gate.py tests\test_identity_feature_guard.py
```

Result: `7 passed`.
