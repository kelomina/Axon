# Phase 2 Loop 8: Adjudication Apply Dry Run

## Scope

Loop 7 created a source-aware Val adjudication queue for `160` model-supported errors.
This loop checks whether that queue can safely pass through the existing evidence
readiness and manual-verdict application tools.

No labels, splits, thresholds, blend weights, feature masks, calibrators, model
hyperparameters, or test-set artifacts were changed.

## Inputs

- Queue CSV: `reports/random_20w_split/stage2_blend_val_all_model_supported_adjudication_queue.csv`
- Cache manifest: `data/.cache/manifest_38672ba0.json`
- Frozen split CSV: `reports/random_20w_split/random_20w_split.csv`

## Readiness Audit

Command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_manual_review_package_readiness.py `
  --review-csv reports\random_20w_split\stage2_blend_val_all_model_supported_adjudication_queue.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --output-csv reports\random_20w_split\stage2_blend_val_all_model_supported_adjudication_queue_readiness.csv `
  --output-json reports\random_20w_split\stage2_blend_val_all_model_supported_adjudication_queue_readiness.json `
  --strict
```

Result:

- Total rows: `160`
- Ready rows: `160`
- Not ready rows: `0`
- `review_queue_ready`: `true`
- `manual_review_ready`: `true`
- `verdict_package_ready`: `false`
- Blank `manual_label_verdict`: `160`
- Blank `recommended_action`: `160`
- Invalid manual verdict/action values: `0`
- Inconsistent manual field pairs: `0`
- Duplicate `source_sha256`: `0`

Evidence checks:

- Manifest matched by `source_path`: `160 / 160`
- Source exists: `160 / 160`
- Source sha256 OK: `160 / 160`
- Cache exists: `160 / 160`
- NPZ loaded: `160 / 160`
- NPZ label OK: `160 / 160`
- NPZ source sha256 OK: `160 / 160`
- NPZ shape OK: `160 / 160`
- PE parse OK: `160 / 160`
- Top-5 neighbor evidence OK: `160 / 160`
- Neighbor manifest/cache/source checks: `800 / 800`

Blocking issues are intentional:

- `manual_verdict_empty`
- `recommended_action_empty`

Interpretation: the queue is evidence-ready, but it is not a verdict package yet
because the human/business decision fields are deliberately blank.

## Dry Apply

Command:

```powershell
.\vnev\Scripts\python.exe scripts\apply_manual_review_verdicts.py `
  --review-csv reports\random_20w_split\stage2_blend_val_all_model_supported_adjudication_queue.csv `
  --split-csv reports\random_20w_split\random_20w_split.csv `
  --output-csv reports\random_20w_split\stage2_blend_val_all_model_supported_adjudication_queue_adjustment_plan.csv `
  --output-json reports\random_20w_split\stage2_blend_val_all_model_supported_adjudication_queue_adjustment_plan.json
```

Result:

- Split rows: `200000`
- Split counts: `train=20000`, `val=20000`, `test=160000`
- Split label counts:
  - Train: `0=10000`, `1=10000`
  - Val: `0=10000`, `1=10000`
  - Test: `0=80000`, `1=80000`
- Review rows: `160`
- Review split: `val=160`
- Review labels: `val:0=99`, `val:1=61`
- Planned rows: `0`
- Ignored rows: `160`
- Unknown verdict rows: `0`
- Missing split rows: `0`
- Duplicate review rows: `0`
- Test review rows: `0`
- Replacement required: `0`
- Training policy rows: `0`

Interpretation: blank manual verdict/action fields are treated as no manual
decision. They create no relabel action, no replacement action, no quarantine
action, and no training policy row.

## Safety Checks Already Covered By Tests

Existing tests cover the critical behavior required by this loop:

- Blank manual verdicts create no actions.
- Feature-broken rows require `exclude_and_replace`, not self-fill.
- Exclude verdicts take priority over conflicting relabel actions.
- Test-split verdicts are withheld from training policy by default.
- Blank test-split reviews are counted but do not create actions.
- Review sha256 can match split rows through source-path filename sha.
- Readiness distinguishes evidence-ready queues from verdict-ready packages.
- Readiness rejects invalid or inconsistent manual field pairs.

Relevant tests:

- `tests/test_apply_manual_review_verdicts.py`
- `tests/test_audit_manual_review_package_readiness.py`

## Decision

The source-aware adjudication queue is safe to hand to human/business review.
It is not allowed to mutate the dataset until manual fields are filled and the
readiness audit reports a verdict-ready package.

Next allowed step:

1. Fill manual decisions for the Val queue only.
2. Re-run readiness in strict mode.
3. Apply verdicts into a non-destructive adjustment plan.
4. Build a corrected Train/Val split using fresh replacement candidates for any
   excluded or feature-broken rows.
5. Re-audit exact split size, label balance, cache coverage, and Val metrics
   before any Test-10k confirmation.

Hard boundary: full-test evidence remains held out and must not be used to tune
thresholds, blend weights, feature masks, calibrators, replacement policy, or
manual label policy.
