# Phase 2 Loop 17: Manual Review Readiness

## Scope

Loop 17 prepares the Loop 16 manual adjudication packages for human review.
It does not fill verdict fields, relabel samples, replace files, tune
thresholds, tune blend weights, tune feature masks, run Test-10k, or touch the
full test split.

The purpose is to verify that the two P0/P1 review packages are technically
ready for a human reviewer: source files exist, hashes match, feature caches
load, NPZ metadata is coherent, rows are PE files, and top-5 neighbor evidence
is present.

## Inputs

- Manifest: `data/.cache/manifest_38672ba0.json`
- Model-supported review package:
  `reports/random_20w_split/stage2_corrected_best_val_model_supported_p0_p1_manual_review.csv`
- Mixed-neighbor review package:
  `reports/random_20w_split/stage2_corrected_best_val_mixed_p0_p1_manual_review.csv`

Both packages are derived from corrected Val errors only. No Test-10k or
full-test row was used.

## Readiness Audit

### Model-Supported P0/P1

Command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_manual_review_package_readiness.py `
  --review-csv reports\random_20w_split\stage2_corrected_best_val_model_supported_p0_p1_manual_review.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_model_supported_p0_p1_readiness.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_model_supported_p0_p1_readiness.json
```

Result:

- Rows: `93`
- Review queue ready: `true`
- Manual review ready: `true`
- Verdict package ready: `false`
- Blocking issues: `manual_verdict_empty`, `recommended_action_empty`
- FP/FN: `66 / 27`
- Priority P0/P1: `69 / 24`
- Source files exist: `93 / 93`
- SHA-256 OK: `93 / 93`
- Cache exists and NPZ loads: `93 / 93`
- NPZ label/source SHA/shape OK: `93 / 93`
- PE rows: `93 / 93`
- Top-5 neighbor evidence OK: `93 / 93`
- Duplicate source SHA-256 count: `0`

Interpretation: this package is ready for human review, but it is not ready
for any automated action because every `manual_label_verdict` and
`recommended_action` cell is still blank.

### Mixed P0/P1

Command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_manual_review_package_readiness.py `
  --review-csv reports\random_20w_split\stage2_corrected_best_val_mixed_p0_p1_manual_review.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_mixed_p0_p1_readiness.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_mixed_p0_p1_readiness.json
```

Result:

- Rows: `44`
- Review queue ready: `true`
- Manual review ready: `true`
- Verdict package ready: `false`
- Blocking issues: `manual_verdict_empty`, `recommended_action_empty`
- FP/FN: `24 / 20`
- Priority P0/P1: `24 / 20`
- Source files exist: `44 / 44`
- SHA-256 OK: `44 / 44`
- Cache exists and NPZ loads: `44 / 44`
- NPZ label/source SHA/shape OK: `44 / 44`
- PE rows: `44 / 44`
- Top-5 neighbor evidence OK: `44 / 44`
- Duplicate source SHA-256 count: `0`

Interpretation: this package is also ready for human review and also blocked
from action until humans fill verdict/action fields.

## Adjudication Guides

### Model-Supported P0/P1

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_manual_review_adjudication_guide.py `
  --readiness-csv reports\random_20w_split\stage2_corrected_best_val_model_supported_p0_p1_readiness.csv `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_model_supported_p0_p1_adjudication_guide.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_model_supported_p0_p1_adjudication_guide.json `
  --output-md reports\random_20w_split\stage2_corrected_best_val_model_supported_p0_p1_adjudication_guide.md `
  --markdown-rows 25
```

Result:

- Rows: `93`
- Manual-review ready rows: `93`
- Suspicion levels:
  - `critical_label_conflict`: `3`
  - `strong_label_conflict`: `6`
  - `moderate_label_conflict`: `84`

The guide is read-only evidence. It intentionally does not include
`manual_label_verdict` or `recommended_action`; those fields must be filled in
the original manual review CSV.

### Mixed P0/P1

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_manual_review_adjudication_guide.py `
  --readiness-csv reports\random_20w_split\stage2_corrected_best_val_mixed_p0_p1_readiness.csv `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_mixed_p0_p1_adjudication_guide.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_mixed_p0_p1_adjudication_guide.json `
  --output-md reports\random_20w_split\stage2_corrected_best_val_mixed_p0_p1_adjudication_guide.md `
  --markdown-rows 25
```

Result:

- Rows: `44`
- Manual-review ready rows: `44`
- Suspicion levels:
  - `moderate_label_conflict`: `27`
  - `review_required`: `17`

## Source Summary

### Model-Supported P0/P1

Command:

```powershell
.\vnev\Scripts\python.exe scripts\summarize_manual_review_sources.py `
  --review-csv reports\random_20w_split\stage2_corrected_best_val_model_supported_p0_p1_manual_review.csv `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_model_supported_p0_p1_source_summary.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_model_supported_p0_p1_source_summary.json `
  --prefix-depth 3 `
  --example-limit 8
```

Result:

- Rows: `93`
- FP/FN: `66 / 27`
- P0/P1: `69 / 24`
- Data directories:
  - `待加入白名单`: `66`
  - `待拉黑`: `27`
- High-similarity conflicts, nearest similarity `>= 0.90`: `9`
- Critical conflicts, nearest similarity `>= 0.95`: `4`
- Largest source prefix: `待加入白名单/<flat>` with `66` rows, all FP
- For that largest prefix:
  - Average model-supported malicious probability: `0.967480`
  - Average opposite-label neighbor ratio: `0.949697`
  - Maximum opposite-label neighbor ratio: `1.000000`

Interpretation: model-supported high-priority FP errors are heavily
concentrated in the flat white-list directory. This is a strong review signal:
the reviewer should check whether these rows are true business-approved benign
software, mislabeled riskware/malware, source-ingestion mistakes, or a model
blind spot around benign files that resemble known malicious families.

### Mixed P0/P1

Command:

```powershell
.\vnev\Scripts\python.exe scripts\summarize_manual_review_sources.py `
  --review-csv reports\random_20w_split\stage2_corrected_best_val_mixed_p0_p1_manual_review.csv `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_mixed_p0_p1_source_summary.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_mixed_p0_p1_source_summary.json `
  --prefix-depth 3 `
  --example-limit 8
```

Result:

- Rows: `44`
- FP/FN: `24 / 20`
- P0/P1: `24 / 20`
- Data directories:
  - `待加入白名单`: `24`
  - `待拉黑`: `20`
- High-similarity conflicts, nearest similarity `>= 0.90`: `0`
- Critical conflicts, nearest similarity `>= 0.95`: `0`
- Largest source prefix: `待加入白名单/<flat>` with `24` rows, all FP
- For that largest prefix:
  - Average model-supported malicious probability: `0.952286`
  - Average opposite-label neighbor ratio: `0.536667`
  - Maximum opposite-label neighbor ratio: `0.760000`

Interpretation: the mixed package is less one-sided than the model-supported
package. It is still worth reviewing, but the expected outcome should be more
diverse: some rows may be source noise, while others may be genuine model
blind spots or ambiguous borderline samples.

## Safety Decision

Do not enter Test-10k from Loop 17.

Reasoning:

1. Loop 17 added no candidate model or threshold improvement.
2. The current outputs are human-review scaffolding, not validated model gains.
3. Both packages are blocked from action until manual fields are filled.
4. The full-test split remains frozen and unused.
5. Any later `feature_broken` or `out_of_scope` verdict must trigger fresh
   same-label replacement, not dataset shrinkage and not reuse of the bad file.

The 20w invariant remains:

```text
200000 = 20000 train + 20000 val + 160000 test
```

## Agent Review

Data-Agent reviewed the new readiness, source summary, and adjudication guide
artifacts. The conclusion was:

- Both review packages are technically complete for human review.
- Model-supported package: `93 / 93` rows ready, with `465 / 465` top-5
  neighbor evidence entries present.
- Mixed package: `44 / 44` rows ready, with `220 / 220` top-5 neighbor
  evidence entries present.
- Both packages still have blank `manual_label_verdict` and
  `recommended_action` fields, so neither can be automatically applied.
- The main data risk is concentrated in `待加入白名单/<flat>`: `66 / 93`
  model-supported rows and `24 / 44` mixed rows come from that source prefix,
  all as FP rows.
- No violation of the 20w no-shrink invariant was found.

Eval-Agent reviewed the funnel compliance. The conclusion was:

- Loop 17 remains inside corrected Train/Val evidence.
- No new Test-10k or full-test artifact was used for selection or tuning.
- Readiness, source summary, and adjudication guides are review aids only.
- No evidence was found that these artifacts tuned thresholds, blend weights,
  or feature masks.
- Test-10k should stay blocked until filled manual verdicts produce a
  non-destructive plan, any required fresh replacements pass cache readiness,
  and a new Val-only evaluation shows a clear improvement.

## Next Procedure

1. Human reviewers fill `manual_label_verdict`, `manual_verdict_note`, and
   `recommended_action` in the original review CSVs.
2. Rerun readiness with `--strict`. It should only pass after verdict/action
   fields are complete and internally consistent.
3. Convert filled verdicts into a non-destructive adjustment plan with
   `scripts/apply_manual_review_verdicts.py`.
4. If any rows require replacement, redraw fresh unused candidates with the
   same intended label. The number of replacements must exactly match the
   number of excluded rows.
5. Build the corrected split, then run strict cache readiness before any new
   Train/Val evaluation.
6. Only after a corrected Train/Val rerun produces a clear Val improvement
   above the funnel threshold should a frozen candidate be considered for
   Test-10k.
