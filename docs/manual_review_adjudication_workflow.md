# Manual Review Adjudication Workflow

## Purpose

The manual review CSV is the human/business evidence layer. It should not directly rewrite the split or cache. The adjudication script converts filled review rows into a non-destructive plan that can be inspected before any new train/val experiment.

Script:

- `scripts/apply_manual_review_verdicts.py`

Primary inputs:

- Review CSV: `reports/random_20w_split/stage2_knn_model_supported_p0_p1_manual_review.csv`
- Split CSV: `reports/random_20w_split/random_20w_split.csv`

## Accepted Manual Fields

Use `manual_label_verdict` for the human decision:

- `label_correct`: current label is correct
- `label_wrong`: current label is wrong; the reviewer must also provide an explicit corrected label in one of `corrected_label`, `new_label`, `target_label`, or `manual_label`
- `out_of_scope`: sample should not count as an in-scope malware/benign sample
- `feature_broken`: sample features or source file are not valid for this experiment
- `uncertain`: not enough evidence

Use `recommended_action` for the operational recommendation:

- `keep_label`
- `relabel_train_only`
- `replace_sample`
- `quarantine_source_group`
- `needs_more_evidence`
- `model_blindspot`

The verdict and action should describe the same operational class. For example, `label_wrong` pairs with `relabel_train_only`, while `feature_broken` and `out_of_scope` pair with `replace_sample` or `quarantine_source_group`. If a row conflicts, such as `feature_broken + relabel_train_only`, the safer exclude/replace interpretation wins because a broken sample must be replaced rather than relabeled.

Important: `label_wrong` does not automatically flip `0` to `1` or `1` to `0`.
The adjustment plan only creates a relabel action when the reviewer provides a
clear corrected label. Without that explicit target, the row becomes
`needs_manual_target_label` and is not eligible for training policy.

## Run

```powershell
.\vnev\Scripts\python.exe scripts\apply_manual_review_verdicts.py `
  --review-csv reports\random_20w_split\stage2_knn_model_supported_p0_p1_manual_review.csv `
  --split-csv reports\random_20w_split\random_20w_split.csv `
  --output-csv reports\random_20w_split\manual_review_adjustment_plan.csv `
  --output-json reports\random_20w_split\manual_review_adjustment_plan.json
```

## Safety Rules

- The script does not edit the original split, raw data, or feature cache.
- Filename, path, extension, directory, source hash, sample id, split, and row
  order are audit/join fields only. They can help locate the reviewed file, but
  they must not become model features or relabel evidence.
- When the same content or `source_sha256` appears in multiple split rows,
  `sample_index` preserves row-level review identity. This prevents a duplicate
  content group from accidentally collapsing into one adjustment row. It is
  still an audit/join field only, not a model feature.
- SHA-like filename fallback is allowed only as a row-matching fallback when an
  explicit `source_sha256` is missing. It is not naming evidence and must not
  influence labels, thresholds, or model features.
- Empty or uncertain review rows produce no action.
- Test-split verdicts are held out of training policy by default.
- `label_wrong` rows without an explicit corrected label produce
  `needs_manual_target_label`; they are not auto-flipped and are not eligible
  for training policy.
- `out_of_scope` and `feature_broken` rows become `exclude_and_replace`.
- Excluded rows require fresh replacement sampling from valid unused candidates with the same intended label. They are not used to fill their own slots.
- The corrected split builder also rejects candidate rows that match an excluded sample, so a manually edited candidate CSV cannot accidentally put the bad file back into the 20w split.

## Interpretation

The JSON summary reports:

- `planned_rows`: rows with an actionable manual decision
- `review_split_counts` and `review_rows_in_test_split`: where the reviewed rows live in the frozen 20w split
- `replacement_required`: rows that must be excluded and replaced
- `replacement_counts_by_original_label`: how many same-label replacement candidates are needed
- `training_policy_rows`: rows that are eligible to affect train/val policy

If `review_rows_in_test_split > 0`, treat those verdicts as held-out evidence by default. They can support noise adjudication and target-feasibility analysis, but they must not directly tune thresholds, blend weights, or train/val policy unless a separate business decision explicitly rebuilds the dataset and restarts Val-first selection.

If `replacement_required > 0`, the next data step is to regenerate a corrected split from a candidate pool that has enough unused valid PE files. Do not reduce the 20w total and do not claim the bad rows as replacements.

## Build a Corrected Split

If replacements are required, first build an unused raw-PE candidate pool. The pool must contain valid fresh files, not the excluded files being replaced:

```powershell
.\vnev\Scripts\python.exe scripts\build_replacement_candidate_pool.py `
  --data-dir data `
  --split-csv reports\random_20w_split\random_20w_split.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --required-label0 0 `
  --required-label1 0 `
  --output-csv reports\random_20w_split\replacement_candidate_pool.csv `
  --output-json reports\random_20w_split\replacement_candidate_pool_summary.json
```

Set `--required-label0` and `--required-label1` to the counts reported by `replacement_counts_by_original_label` in the adjustment plan. If `replacement_shortfall` is not empty, stop and collect more valid samples before rebuilding the corrected split.

After reviewing the candidate pool, build a corrected split:

```powershell
.\vnev\Scripts\python.exe scripts\build_corrected_split_from_plan.py `
  --split-csv reports\random_20w_split\random_20w_split.csv `
  --plan-csv reports\random_20w_split\manual_review_adjustment_plan.csv `
  --candidate-csv reports\random_20w_split\replacement_candidate_pool.csv `
  --output-csv reports\random_20w_split\corrected_manual_review_split.csv `
  --output-json reports\random_20w_split\corrected_manual_review_split_summary.json
```

Use `--data-dir data` when replacements are required, because the existing fixed cache manifest only covers the current 20w selected rows. If you provide `--manifest-json data\.cache\manifest_38672ba0.json`, it can validate no-op plans but cannot supply fresh replacements beyond the already selected split.

The corrected split builder refuses to emit a short split. If a reviewed row is excluded and no unused same-label replacement exists, it stops with an error instead of writing a 199999-row dataset.

## Cache Readiness Gate

Before training from a corrected split, run the strict cache readiness audit:

```powershell
.\vnev\Scripts\python.exe scripts\audit_corrected_split_cache_ready.py `
  --split-csv reports\random_20w_split\corrected_manual_review_split.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --missing-cache-output reports\random_20w_split\corrected_manual_review_missing_cache.csv `
  --output-json reports\random_20w_split\corrected_manual_review_cache_ready.json `
  --strict
```

This gate checks both requirements:

- the corrected split is still exactly `200000 = 20000 train + 20000 val + 160000 test`
- every row has a cache entry and the cache file exists

By default, label-count drift after manual relabeling is reported in `label_balance_drift` but does not block cache readiness. Add `--enforce-label-balance` if the next experiment must preserve the original `10000/10000` train/val and `80000/80000` test class balance exactly.

If `cache_ready=false`, do not train. First extract or recover cache for rows listed in `corrected_manual_review_missing_cache.csv`, then rerun the audit.

Build a bounded recovery plan from the missing-cache CSV:

```powershell
.\vnev\Scripts\python.exe scripts\build_corrected_split_cache_recovery_plan.py `
  --missing-csv reports\random_20w_split\corrected_manual_review_missing_cache.csv `
  --checkpoint models\random_20w_8192\best_model.pt `
  --cache-dir data\.cache `
  --recovery-output-json reports\random_20w_split\corrected_manual_review_cache_recovery_run.json `
  --post-recovery-audit-command ".\vnev\Scripts\python.exe scripts\audit_corrected_split_cache_ready.py --split-csv reports\random_20w_split\corrected_manual_review_split.csv --manifest-json data\.cache\manifest_38672ba0.json --missing-cache-output reports\random_20w_split\corrected_manual_review_missing_cache.csv --output-json reports\random_20w_split\corrected_manual_review_cache_ready.json --strict" `
  --output-json reports\random_20w_split\corrected_manual_review_cache_recovery_plan.json `
  --output-md reports\random_20w_split\corrected_manual_review_cache_recovery_plan.md
```

The generated plan includes a dry-run command and a recovery command. Run the dry-run first. The default storage format is `uncompressed`, matching the current preference for newly extracted replacement caches.
