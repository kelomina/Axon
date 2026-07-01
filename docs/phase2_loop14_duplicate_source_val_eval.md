# Phase 2 Loop 14: Duplicate-Source Corrected Val Evaluation

## Scope

Loop 14 evaluates the duplicate-source corrected split from Loop 12/13 on the
validation split only. This is a same-checkpoint, same-threshold comparison
against the original validation split to determine whether the automatic
Train/Val duplicate cleanup changes the current single-model baseline.

No model training, threshold tuning, blend-weight tuning, feature-mask tuning,
calibrator tuning, or full-test feedback was used.

## Inputs

- Checkpoint: `models/random_20w_8192/best_model.pt`
- Config: `config/random_20w_8192.toml`
- Manifest: `data/.cache/manifest_38672ba0.json`
- Original split: `reports/random_20w_split/random_20w_split.csv`
- Corrected split: `reports/random_20w_split/duplicate_source_corrected_split.csv`
- Original Val eval: `reports/random_20w_split/original_single_checkpoint_val_eval.json`
- Corrected Val eval: `reports/random_20w_split/duplicate_source_corrected_val_eval.json`

## Data-Agent Audit

The corrected split remains structurally valid:

- Total rows: `200000`
- Split counts: `train=20000`, `val=20000`, `test=160000`
- Train labels: `0=10000`, `1=10000`
- Val labels: `0=10000`, `1=10000`
- Test labels: `0=80000`, `1=80000`
- Cache coverage: `200000 / 200000`
- Missing cache rows: `0`
- Cache ready: `true`

Remaining duplicate-source risks after automatic cleanup:

- Duplicate groups: `14`
- Duplicate extra rows: `14`
- Cross-label groups: `4`
- Cross-split groups: `2`
- Same-path duplicate groups: `0`

Interpretation: automatic replacement removed the safe Train/Val duplicate
rows while preserving exact dataset size and label balance. Remaining duplicate
identities are not safe for automatic action because they involve cross-label
conflicts or frozen Test boundaries.

## Eval-Agent Fair Comparison

Original Val, single checkpoint, threshold `0.5`:

- Samples: `20000`
- Accuracy: `0.930650`
- Precision: `0.942828`
- Recall: `0.916900`
- F1: `0.929683`
- AUC: `0.975704`
- FP: `556`
- FN: `831`
- Errors: `1387`

Corrected Val, same checkpoint, threshold `0.5`:

- Samples: `20000`
- Accuracy: `0.930700`
- Precision: `0.943016`
- Recall: `0.916800`
- F1: `0.929723`
- AUC: `0.975755`
- FP: `554`
- FN: `832`
- Errors: `1386`

Delta, corrected minus original:

- Accuracy: `+0.000050`
- Precision: `+0.000188`
- Recall: `-0.000100`
- F1: `+0.000040`
- AUC: `+0.000051`
- FP: `-2`
- FN: `+1`
- Errors: `-1`

Interpretation: the corrected split is essentially metric-neutral for the
current single checkpoint. The improvement is about `0.004` F1 percentage
points, far below the project funnel threshold for a new model candidate.

Important comparison note: this is a **controlled split修正前后对比**, not a
perfectly identical Val sample set. The cache and checkpoint are the same, but
the corrected split replaced `6` Val rows, so the sample set is mostly the same
yet not mathematically identical.

## Model-Agent Decision

This result does not justify entering Test-10k as a model-improvement
candidate. It should be kept as a data hygiene improvement only:

1. It preserves the exact 20w dataset shape.
2. It removes safe duplicate-source contamination from Train/Val.
3. It does not materially improve the current single-checkpoint Val result.

The prior Stage 2 blend result is not directly comparable to this Loop 14
single-checkpoint evaluation. The frozen best known blend was:

- Blend: `stage2_extended:1 + stage2_knn:2`
- Threshold: `0.55`
- Val F1: `0.9839588226475439`
- Test-10k: passed
- Full-test F1: `0.9832884231848902`

That is a different inference pipeline and threshold. A fair Stage 2 comparison
on the corrected split requires regenerating the same Stage 2 prediction
features from `duplicate_source_corrected_split.csv`, then rerunning the same
blend/threshold validation logic without using full-test feedback.

For decision-making, the safer conclusion is:

- single-model Loop 14: not a Test-10k candidate;
- Stage 2 blend: re-evaluate on the corrected split before any new promotion;
- full-test: continue to remain frozen until Val-only evidence is clearly
  stronger than the current baseline.

## Error-Agent Noise Assessment

The corrected Val run still has `1386` validation errors:

- False positives: `554`
- False negatives: `832`

Given that the safe duplicate-source cleanup changed only one net error on Val,
duplicate cleanup alone is not a sufficient path toward `99.9%` F1. The
remaining error mass is more likely driven by one or more of:

- unresolved cross-label source conflicts;
- source-family or near-duplicate label noise;
- feature-extraction artifacts;
- long-tail families not represented well by the current 20k training split;
- model capacity or calibration limits in the single-checkpoint path.

The `4` remaining cross-label duplicate groups must go to manual adjudication.
The `2` cross-split duplicate groups touch frozen Test boundaries and should not
be used for tuning or cleanup policy decisions without an explicit frozen-set
governance decision.

## Decision

Loop 14 is accepted as a data hygiene checkpoint, not as a model candidate.

Do not promote the corrected single-checkpoint result to Test-10k. Continue
Phase 2 with noise adjudication and source-aware error analysis on Train/Val.

Recommended next actions:

1. Regenerate Stage 2 blend predictions on the corrected split if a fair
   corrected-vs-original blend comparison is needed.
2. Build a manual review queue for the `4` cross-label duplicate groups.
3. Keep frozen Test out of threshold, mask, calibrator, and cleanup-policy
   selection.
4. Prioritize Val false negatives because the corrected single model still has
   more FN than FP (`832` vs `554`).
