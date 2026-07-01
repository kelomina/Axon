# Phase 2 Loop 6: Val Source-Level Noise Audit

## Scope

Loop 5 produced a complete Val model-supported review pool:

- Rows: `160`
- FP/FN: `99 / 61`
- Priority 0/1/2/3: `51 / 36 / 47 / 26`
- Strict evidence readiness: `160 / 160`

This loop adds source-level grouping so manual review can move from isolated rows to source and batch hypotheses. It is still read-only: no labels, splits, thresholds, blend weights, feature masks, or test-set artifacts are changed.

## Tooling

Added a read-only summary utility:

- Script: `scripts/summarize_manual_review_sources.py`
- Test: `tests/test_summarize_manual_review_sources.py`

The script groups manual-review rows by:

- `source_prefix`
- `data_dir`
- `month`
- `date_dir`
- `parent_dir`
- `extension`
- `error_type`
- `priority`
- `label`

It also counts high-similarity conflicts:

- High similarity: `nearest_similarity >= 0.90 and opposite_label_ratio >= 0.80`
- Critical: `nearest_similarity >= 0.95 and opposite_label_ratio >= 0.80`

## Generated Outputs

- CSV: `reports/random_20w_split/stage2_blend_val_all_model_supported_source_summary.csv`
- JSON: `reports/random_20w_split/stage2_blend_val_all_model_supported_source_summary.json`

Command:

```powershell
.\vnev\Scripts\python.exe scripts\summarize_manual_review_sources.py `
  --review-csv reports\random_20w_split\stage2_blend_val_all_model_supported_manual_review_all160.csv `
  --output-csv reports\random_20w_split\stage2_blend_val_all_model_supported_source_summary.csv `
  --output-json reports\random_20w_split\stage2_blend_val_all_model_supported_source_summary.json `
  --prefix-depth 3 `
  --example-limit 3
```

## Source-Level Findings

Overall:

- Rows: `160`
- FP/FN: `99 / 61`
- High-similarity conflicts: `13`
- Critical conflicts: `7`

By data directory:

- `待加入白名单`: `99` rows, all FP
- `待拉黑`: `61` rows, all FN

White-list side:

- Source prefix: `待加入白名单/<flat>`
- Count: `99`
- FP/FN: `99 / 0`
- Priority 0/1/2/3: `39 / 22 / 29 / 9`
- Average malicious probability: `0.8655965864`
- Average nearest similarity: `0.7445066022`
- Average opposite-label neighbor ratio: `0.9470707071`
- High-similarity conflicts: `12`
- Critical conflicts: `7`

Malicious side:

- Source directory: `待拉黑`
- Count: `61`
- FP/FN: `0 / 61`
- Priority 0/1/2/3: `12 / 14 / 18 / 17`
- Average malicious probability: `0.2259009909`
- Average nearest similarity: `0.6809839228`
- Average opposite-label neighbor ratio: `0.8898360656`
- High-similarity conflicts: `1`
- Critical conflicts: `0`

Largest malicious date/month clusters:

- `2020-11`: `11` FN
- `2026-03`: `8` FN
- `2025-11`: `6` FN
- `2025-12`: `6` FN
- `2025-10`: `4` FN
- `2020-10`: `3` FN
- `2021-06`: `3` FN

Largest malicious source-prefix clusters:

- `待拉黑/2020-11/2020-11-07`: `5` FN
- `待拉黑/2020-11/2020-11-17`: `3` FN
- `待拉黑/2026-03/2026-03-01`: `3` FN
- `待拉黑/2020-11/2020-11-10`: `2` FN
- `待拉黑/2021-06/2021-06-03`: `2` FN
- `待拉黑/2025-03/2025-03-21`: `2` FN

## Interpretation

The two error modes look different:

1. White-list FP errors are source-concentrated.

   All `99` model-supported FP rows come from the flat white-list directory. The average opposite-label neighbor ratio is very high (`0.9471`), and `7` rows meet the critical high-similarity conflict rule. This is consistent with white-list contamination, risky benign-like labels, copied lineage from malicious families, or a source policy that is too broad for this task.

2. Malicious FN errors are batch-distributed.

   The `61` model-supported FN rows are spread across many date folders. The largest single date bucket has only `5` rows. This looks less like one bad source folder and more like long-tail family coverage, weak representation of some batches, or ambiguous malicious samples whose nearest memory often resembles benign rows.

## Review Strategy

Recommended manual review order:

1. Review the `7` critical white-list FP conflicts.
2. Review the remaining `12` high-similarity white-list FP conflicts.
3. Review the full `99` white-list FP set by source policy, not only by individual file.
4. Review malicious FN clusters by date batch, starting with:
   - `2020-11`
   - `2026-03`
   - `2025-11`
   - `2025-12`
5. For malicious FN batches, sample both severe and near-threshold cases to separate true model blind spots from label/source ambiguity.

## Policy

No automatic relabeling is allowed from this report.

Allowed next actions after human/business adjudication:

- If `label_wrong`: apply `relabel_train_only` only through the Train/Val correction flow.
- If `feature_broken` or `out_of_scope`: replace with a fresh valid candidate.
- If `label_correct`: mark as model blind spot or keep label.
- If uncertain: hold out of correction until more evidence exists.

Hard constraints:

- Total dataset remains exactly `200000`.
- Split sizes remain exactly `20000 / 20000 / 160000`.
- Excluded files must never self-replace.
- Replacement candidates must be newly selected valid files.
- Cache coverage must re-audit to `200000 / 200000`.
- Full-test evidence remains frozen and must not tune thresholds, weights, masks, or correction policy.

## Decision

Do not proceed to more model-side tuning yet. The next highest-value action is source-aware adjudication of the white-list FP pool and batch-aware adjudication of the malicious FN pool. The observed pattern supports treating white-list contamination as the first cleanup target and long-tail malicious batches as the second target.
