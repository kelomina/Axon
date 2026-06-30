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
- `label_wrong`: current label is wrong and should be flipped unless a corrected label column says otherwise
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
- Empty or uncertain review rows produce no action.
- Test-split verdicts are held out of training policy by default.
- `out_of_scope` and `feature_broken` rows become `exclude_and_replace`.
- Excluded rows require fresh replacement sampling from valid unused candidates with the same intended label. They are not used to fill their own slots.

## Interpretation

The JSON summary reports:

- `planned_rows`: rows with an actionable manual decision
- `replacement_required`: rows that must be excluded and replaced
- `replacement_counts_by_original_label`: how many same-label replacement candidates are needed
- `training_policy_rows`: rows that are eligible to affect train/val policy

If `replacement_required > 0`, the next data step is to regenerate a corrected split from a candidate pool that has enough unused valid PE files. Do not reduce the 20w total and do not claim the bad rows as replacements.

## Build a Corrected Split

If replacements are required, first build an unused raw-PE candidate pool:

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
