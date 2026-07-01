# Phase 2 Loop 23: Automatic Noise Replacement

## Scope

Loop 23 converts the previously manual-only review queue into a conservative
automatic replacement pass for high-confidence Train/Val noise candidates.

This loop does not relabel samples, tune thresholds, tune blend weights, tune
feature masks, run Test-10k, or touch the full-test split for decision making.
It only removes selected high-confidence noisy Train/Val rows from the active
benchmark split and redraws the same number of fresh unused valid files with
the same intended label.

Conflict note: the project guidance prefers explanation and confirmation
before new code. The user explicitly authorized autonomous execution after the
manual review flow stalled. I kept the safer interpretation: no fabricated
labels and no model-driven relabeling; automatic action is limited to
same-label exclude-and-replace.

## Policy

The automatic policy is intentionally narrow:

- support bucket: `neighbors_support_model_prediction`
- priority: `<= 0`
- model confidence against the current label: `>= 0.95`
- opposite-label neighbor ratio: `>= 0.80`
- nearest similarity: `>= 0.0`
- allowed splits: `train`, `val`
- forbidden split: `test`
- action: `exclude_and_replace`
- replacement rule: one fresh unused valid same-label file per excluded row
- relabeling: disabled

In plain terms, this treats these rows as unsuitable measuring cups rather than
rewriting their truth. If a row is too suspicious to keep in Train/Val, it is
removed and replaced by a new same-label candidate. The dataset is never
shrunk and the old sample is not reused to "fill" the split.

## New Guardrail Script

Added:

- `scripts/build_automatic_noise_replacement_plan.py`

Added tests:

- `tests/test_build_automatic_noise_replacement_plan.py`

The script reads a review queue plus the active split and emits a manual-plan
compatible CSV/JSON where all selected rows are:

- `plan_action=exclude_and_replace`
- `replacement_required=true`
- `replacement_label` equal to the original split label
- `usable_for_training_policy=false`
- `manual_label_verdict=automatic_high_confidence_noise_candidate`

It skips rows that fail the evidence policy, rows missing from the split, label
mismatches, and held-out rows.

## Automatic Plan Result

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_automatic_noise_replacement_plan.py `
  --review-csv reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_manual_review.csv `
  --split-csv reports\random_20w_split\duplicate_source_corrected_split.csv `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_replacement_plan_loop23.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_auto_noise_replacement_plan_loop23.json `
  --max-priority 0 `
  --min-confidence 0.95 `
  --min-opposite-ratio 0.80 `
  --min-nearest-similarity 0.0 `
  --support-bucket neighbors_support_model_prediction
```

Result:

- Review rows: `137`
- Planned rows: `69`
- Skipped rows: `68`
- Planned split counts: `val=69`
- Planned label counts: `0=51`, `1=18`
- Replacement required: `69`
- Training policy rows: `0`
- Test actions: disabled
- Relabeling: disabled

## Fresh Replacement Pool

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_replacement_candidate_pool.py `
  --data-dir data `
  --split-csv reports\random_20w_split\duplicate_source_corrected_split.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --required-label0 51 `
  --required-label1 18 `
  --hash-files `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_replacement_candidates_loop23.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_auto_noise_replacement_candidates_loop23.json
```

Result:

- Candidate rows: `164819`
- Available label `0`: `44958`
- Available label `1`: `119861`
- Required label `0`: `51`
- Required label `1`: `18`
- Replacement shortfall: none
- Enough candidates: `true`

## Corrected Split Result

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_corrected_split_from_plan.py `
  --split-csv reports\random_20w_split\duplicate_source_corrected_split.csv `
  --plan-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_replacement_plan_loop23.csv `
  --candidate-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_replacement_candidates_loop23.csv `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_corrected_split_loop23.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_auto_noise_corrected_split_loop23.json `
  --seed 42
```

Result:

- Original rows: `200000`
- Corrected rows: `200000`
- Split counts: `train=20000`, `val=20000`, `test=160000`
- Label counts: `0=100000`, `1=100000`
- Train labels: `0=10000`, `1=10000`
- Val labels: `0=10000`, `1=10000`
- Test labels: `0=80000`, `1=80000`
- Excluded rows: `69`
- Relabeled rows: `0`
- Selected replacements: `69`
- Replacement shortfall: none

The corrected split preserves the exact 20w invariant:

```text
200000 = 20000 train + 20000 val + 160000 test
```

## Replacement Integrity

Command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_corrected_split_replacements.py `
  --original-split-csv reports\random_20w_split\duplicate_source_corrected_split.csv `
  --corrected-split-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_corrected_split_loop23.csv `
  --plan-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_replacement_plan_loop23.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_auto_noise_replacement_integrity_loop23.json `
  --detail-output-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_replacement_integrity_loop23.csv `
  --strict `
  --enforce-label-balance
```

Result:

- Replacement integrity: `true`
- Planned excluded rows removed: `69`
- Excluded rows still present after correction: `0`
- Unplanned original rows removed: `0`
- Fresh replacement rows: `69`
- Request counts: `val:0=51`, `val:1=18`
- Fresh counts: `val:0=51`, `val:1=18`
- Test replacement requests: `0`
- Test relabel requests: `0`
- Integrity failures: none

## Cache Recovery

Before recovery, the new split had `69` missing cache rows, all caused by fresh
replacement files not yet present in the manifest.

Dry-run command:

```powershell
.\vnev\Scripts\python.exe scripts\recover_missing_feature_cache.py `
  --missing-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_missing_cache_loop23.csv `
  --checkpoint models\random_20w_8192\best_model.pt `
  --cache-dir data\.cache `
  --workers 4 `
  --backend process `
  --storage-format uncompressed `
  --dry-run `
  --output-json reports\random_20w_split\stage2_corrected_best_val_auto_noise_cache_recovery_dry_run_loop23.json
```

Run command:

```powershell
.\vnev\Scripts\python.exe scripts\recover_missing_feature_cache.py `
  --missing-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_missing_cache_loop23.csv `
  --checkpoint models\random_20w_8192\best_model.pt `
  --cache-dir data\.cache `
  --workers 4 `
  --backend process `
  --storage-format uncompressed `
  --progress-interval 10 `
  --output-json reports\random_20w_split\stage2_corrected_best_val_auto_noise_cache_recovery_run_loop23.json
```

Result:

- Input rows: `69`
- Status counts: `extracted=69`
- Manifest added: `69`
- Failed examples: none
- Storage format: `uncompressed`

## Cache Readiness

Command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_corrected_split_cache_ready.py `
  --split-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_corrected_split_loop23.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --missing-cache-output reports\random_20w_split\stage2_corrected_best_val_auto_noise_missing_cache_after_recovery_loop23.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_auto_noise_cache_ready_after_recovery_loop23.json `
  --strict `
  --enforce-label-balance
```

Result:

- Total rows: `200000`
- Covered rows: `200000`
- Missing rows: `0`
- Coverage ratio: `1.0`
- Manifest match counts: `source_path=200000`
- Shape failures: none
- Label balance drift: none
- Cache ready: `true`

## Regression Tests

Command:

```powershell
.\vnev\Scripts\python.exe -m pytest `
  tests\test_build_automatic_noise_replacement_plan.py `
  tests\test_build_corrected_split_from_plan.py `
  tests\test_audit_corrected_split_replacements.py `
  tests\test_audit_corrected_split_cache_ready.py `
  tests\test_recover_missing_feature_cache.py -q
```

Result:

- `28 passed`

## Safety Decision

Do not enter Test-10k from Loop 23 yet.

Reasoning:

1. Loop 23 creates a cleaner Train/Val split, not a proven better model.
2. The automatic action is deliberately not a relabeling policy.
3. The corrected split and cache now pass strict invariants.
4. The next gate is an apples-to-apples Val evaluation using the same
   checkpoint and the same candidate threshold protocol.
5. Test-10k is allowed only if Val clearly improves under the fixed funnel.

## Next Procedure

1. Evaluate the old corrected split and Loop 23 corrected split on Val with the
   same checkpoint.
2. Compare threshold-sweep results on Val only.
3. If Val improves clearly, train or refit the next candidate on the corrected
   Train/Val protocol.
4. Only after a candidate beats the current Val gate should Test-10k be used.
5. Full-test remains final confirmation only, never a tuning surface.
