# Phase 2 Loop 7: Source-Aware Adjudication Queue

## Scope

Loop 6 showed that the Val model-supported errors split into two different modes:

- White-list FP errors are source-concentrated.
- Malicious FN errors are date/batch distributed.

This loop turns that evidence into a source-aware manual adjudication queue. It does not fill verdicts and does not mutate labels, splits, thresholds, blend weights, feature masks, or test-set artifacts.

## Tooling

Added:

- Script: `scripts/build_source_aware_adjudication_queue.py`
- Test: `tests/test_build_source_aware_adjudication_queue.py`

The script takes the ready manual-review CSV and adds:

- `review_priority_rank`
- `review_lane`
- `source_group_key`
- `source_group_size`
- `source_group_error_type_counts`
- `source_group_priority_counts`
- `suspicion_level`
- `review_question_focus`
- allowed verdict/action fields
- replacement rule

Manual fields remain blank:

- `manual_label_verdict`
- `manual_verdict_note`
- `recommended_action`

## Generated Outputs

- CSV: `reports/random_20w_split/stage2_blend_val_all_model_supported_adjudication_queue.csv`
- JSON: `reports/random_20w_split/stage2_blend_val_all_model_supported_adjudication_queue.json`

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_source_aware_adjudication_queue.py `
  --review-csv reports\random_20w_split\stage2_blend_val_all_model_supported_manual_review_all160.csv `
  --output-csv reports\random_20w_split\stage2_blend_val_all_model_supported_adjudication_queue.csv `
  --output-json reports\random_20w_split\stage2_blend_val_all_model_supported_adjudication_queue.json
```

Validation:

```powershell
.\vnev\Scripts\python.exe -m pytest `
  tests\test_build_source_aware_adjudication_queue.py `
  tests\test_summarize_manual_review_sources.py `
  -q
```

Result: `2 passed`.

## Queue Summary

Rows: `160`

Lane counts:

- `A_whitelist_critical_fp`: `2`
- `B_whitelist_high_similarity_fp`: `10`
- `C_whitelist_remaining_fp`: `87`
- `D_malicious_batch_fn`: `61`

Suspicion levels:

- `critical_label_conflict`: `2`
- `strong_label_conflict`: `11`
- `moderate_label_conflict`: `147`

Manual fields:

- Blank `manual_label_verdict`: `160 / 160`
- Blank `recommended_action`: `160 / 160`

Source groups:

- `待加入白名单/<flat>`: `99`
- largest malicious source group: `待拉黑/2020-11/2020-11-07`, `5`
- next largest malicious groups: `待拉黑/2020-11/2020-11-17`, `3`; `待拉黑/2026-03/2026-03-01`, `3`

## Lane Definitions

`A_whitelist_critical_fp`

- Benign-labeled file
- Model predicts malicious
- White-list flat source
- `nearest_similarity >= 0.95`
- `opposite_label_ratio >= 0.80`
- high prediction confidence

`B_whitelist_high_similarity_fp`

- Benign-labeled file
- Model predicts malicious
- White-list flat source
- strong high-similarity conflict, but below critical rule

`C_whitelist_remaining_fp`

- Remaining white-list FP rows
- Still source-concentrated and model-supported
- Review as source-policy risk, not as isolated one-off mistakes

`D_malicious_batch_fn`

- Malicious-labeled file
- Model predicts benign
- Review by source/date batch
- Sorts larger source groups first, then priority and confidence

## Review Policy

The queue is an ordering aid, not a verdict engine.

Allowed `manual_label_verdict`:

- `label_correct`
- `label_wrong`
- `out_of_scope`
- `feature_broken`
- `uncertain`

Allowed `recommended_action`:

- `keep_label`
- `relabel_train_only`
- `replace_sample`
- `quarantine_source_group`
- `needs_more_evidence`
- `model_blindspot`

Decision rules:

- `label_wrong` pairs with `relabel_train_only`.
- `feature_broken` or `out_of_scope` pairs with `replace_sample` or `quarantine_source_group`.
- `label_correct` pairs with `keep_label` or `model_blindspot`.
- `uncertain` pairs with `needs_more_evidence` or no correction.

## Hard Constraints

- Do not use this queue to tune thresholds, blend weights, GA feature masks, calibrators, or model hyperparameters.
- Do not use full-test errors to decide labels or source policy.
- Bad files must be replaced by freshly selected valid candidates.
- Excluded samples must never self-replace.
- Total dataset must remain exactly `200000`.
- Split sizes must remain exactly `20000 / 20000 / 160000`.
- Cache coverage must re-audit to exactly `200000 / 200000`.
- Any corrected split must rerun full Val before Test-10k.

## Decision

Use this adjudication queue as the next human/business review artifact. Review the first two lanes (`A` and `B`) before touching lower-priority rows. If those rows confirm white-list contamination, move from individual row review to source-policy correction, then rebuild the corrected Train/Val split with fresh replacements and strict cache re-audit.
