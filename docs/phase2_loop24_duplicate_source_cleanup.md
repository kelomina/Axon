# Phase 2 Loop 24: Duplicate Source Cleanup

## Scope

Loop 24 removes source-identity duplicates that remained after Loop 23.

This loop is data hygiene, not model tuning. It does not use model errors,
thresholds, Test-10k scores, or full-test performance to decide what to
replace. Decisions are based only on duplicate source identity inferred from
SHA-like file names and split metadata.

Default safeguards remain in place: test split plan rows are rejected unless a
command explicitly opts into data-hygiene replacements. Even with that override,
only `exclude_and_replace` is allowed; relabeling test rows remains forbidden.

## Issue

Loop 23 passed replacement integrity and cache readiness, but a SHA-only
duplicate audit still found duplicate source identities:

- Duplicate groups: `14`
- Duplicate extra rows: `14`
- Cross-label groups: `4`
- Cross-split groups: `2`
- Same-path duplicate groups: `0`

These rows were not explicit `source_sha256` duplicates in the split CSV. They
were inferred from SHA-like 64-hex file names. The risk was still real:

- cross-split duplicates can leak the same file identity across Train/Val/Test
- cross-label duplicates mean the same identity appears as both benign and
  malicious
- same-label duplicates in Test reduce independence of the final benchmark

## Code Changes

Changed:

- `scripts/build_duplicate_source_cleanup_plan.py`
- `scripts/build_corrected_split_from_plan.py`
- `scripts/audit_corrected_split_replacements.py`

Changed tests:

- `tests/test_build_duplicate_source_cleanup_plan.py`
- `tests/test_build_corrected_split_from_plan.py`
- `tests/test_audit_corrected_split_replacements.py`

New behavior:

- duplicate cleanup supports `--cross-label-policy replace_all`
- corrected split builder supports explicit `--allow-test-replacements`
- replacement audit supports explicit `--allow-test-replacements`
- default behavior still rejects test replacements
- test relabeling remains blocked

## Duplicate Audit Before Cleanup

Command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_split_duplicate_sources.py `
  --split-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_corrected_split_loop23.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_auto_noise_sha_duplicate_audit_loop23.json `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_sha_duplicate_audit_loop23.csv `
  --sha-only
```

Result:

- Rows: `200000`
- Split counts: `train=20000`, `val=20000`, `test=160000`
- Label balance: preserved in every split
- Duplicate groups: `14`
- Duplicate extra rows: `14`
- Cross-label groups: `4`
- Cross-split groups: `2`

## Cleanup Plan

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_duplicate_source_cleanup_plan.py `
  --duplicate-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_sha_duplicate_audit_loop23.csv `
  --output-plan-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_duplicate_cleanup_plan_loop24.csv `
  --output-review-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_duplicate_cleanup_review_loop24.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_auto_noise_duplicate_cleanup_plan_loop24.json `
  --allow-test-replacements `
  --cross-label-policy replace_all
```

Result:

- Duplicate groups: `14`
- Detail rows: `28`
- Auto plan rows: `18`
- Manual review rows: `0`
- Group actions:
  - `auto_replace_cross_label_all=4`
  - `auto_replace_duplicates=10`
- Planned split counts:
  - `test=14`
  - `train=1`
  - `val=3`
- Planned label counts:
  - `0=7`
  - `1=11`

Policy:

- Same-label duplicate group: keep one canonical row and redraw the extra row.
- Cross-label duplicate group: redraw all conflicting rows with fresh same-label
  files.
- No relabeling is performed.

## Fresh Candidate Pool

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_replacement_candidate_pool.py `
  --data-dir data `
  --split-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_corrected_split_loop23.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --required-label0 7 `
  --required-label1 11 `
  --hash-files `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_duplicate_cleanup_candidates_loop24.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_auto_noise_duplicate_cleanup_candidates_loop24.json
```

Result:

- Candidate rows: `164819`
- Available label `0`: `44958`
- Available label `1`: `119861`
- Required label `0`: `7`
- Required label `1`: `11`
- Replacement shortfall: none
- Enough candidates: `true`

## Corrected Split

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_corrected_split_from_plan.py `
  --split-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_corrected_split_loop23.csv `
  --plan-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_duplicate_cleanup_plan_loop24.csv `
  --candidate-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_duplicate_cleanup_candidates_loop24.csv `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_dedup_corrected_split_loop24.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_auto_noise_dedup_corrected_split_loop24.json `
  --seed 43 `
  --allow-test-replacements
```

Result:

- Original rows: `200000`
- Corrected rows: `200000`
- Split counts: `train=20000`, `val=20000`, `test=160000`
- Label counts: `0=100000`, `1=100000`
- Train labels: `0=10000`, `1=10000`
- Val labels: `0=10000`, `1=10000`
- Test labels: `0=80000`, `1=80000`
- Excluded rows: `18`
- Relabeled rows: `0`
- Selected replacements: `18`
- Replacement shortfall: none

## Replacement Integrity

Command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_corrected_split_replacements.py `
  --original-split-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_corrected_split_loop23.csv `
  --corrected-split-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_dedup_corrected_split_loop24.csv `
  --plan-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_duplicate_cleanup_plan_loop24.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_auto_noise_dedup_replacement_integrity_loop24.json `
  --detail-output-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_dedup_replacement_integrity_loop24.csv `
  --strict `
  --enforce-label-balance `
  --allow-test-replacements
```

Result:

- Replacement integrity: `true`
- Original duplicate key rows: `14`
- Corrected duplicate key rows: `0`
- Duplicate key row delta: `-14`
- Planned excluded rows removed: `18`
- Excluded rows still present after correction: `0`
- Unplanned original rows removed: `0`
- Fresh replacement rows: `18`
- Test replacement requests: `14`
- Test relabel requests: `0`
- Integrity failures: none

## Duplicate Audit After Cleanup

Command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_split_duplicate_sources.py `
  --split-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_dedup_corrected_split_loop24.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_auto_noise_dedup_duplicate_audit_loop24.json `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_dedup_duplicate_audit_loop24.csv `
  --sha-only `
  --strict
```

Result:

- Duplicate groups: `0`
- Duplicate extra rows: `0`
- Cross-label groups: `0`
- Cross-split groups: `0`
- Same-path duplicate groups: `0`

## Cache Recovery

Initial cache readiness after Loop 24 found:

- Covered rows: `199982`
- Missing rows: `18`
- Missing labels: `0=7`, `1=11`
- Missing splits: `test=14`, `train=1`, `val=3`
- Missing reason: `manifest_missing=18`

Dry-run:

```powershell
.\vnev\Scripts\python.exe scripts\recover_missing_feature_cache.py `
  --missing-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_dedup_missing_cache_loop24.csv `
  --checkpoint models\random_20w_8192\best_model.pt `
  --cache-dir data\.cache `
  --workers 4 `
  --backend process `
  --storage-format uncompressed `
  --dry-run `
  --output-json reports\random_20w_split\stage2_corrected_best_val_auto_noise_dedup_cache_recovery_dry_run_loop24.json
```

Result:

- Planned rows: `18`
- Would extract: `18`

Run:

```powershell
.\vnev\Scripts\python.exe scripts\recover_missing_feature_cache.py `
  --missing-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_dedup_missing_cache_loop24.csv `
  --checkpoint models\random_20w_8192\best_model.pt `
  --cache-dir data\.cache `
  --workers 4 `
  --backend process `
  --storage-format uncompressed `
  --progress-interval 5 `
  --output-json reports\random_20w_split\stage2_corrected_best_val_auto_noise_dedup_cache_recovery_run_loop24.json
```

Result:

- Input rows: `18`
- Extracted: `18`
- Manifest added: `18`
- Failed examples: none
- Storage format: `uncompressed`

## Final Cache Readiness

Command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_corrected_split_cache_ready.py `
  --split-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_dedup_corrected_split_loop24.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --missing-cache-output reports\random_20w_split\stage2_corrected_best_val_auto_noise_dedup_missing_cache_after_recovery_loop24.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_auto_noise_dedup_cache_ready_after_recovery_loop24.json `
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
  tests\test_recover_missing_feature_cache.py `
  tests\test_audit_split_duplicate_sources.py `
  tests\test_build_duplicate_source_cleanup_plan.py -q
```

Result:

- `42 passed`

## Safety Decision

Loop 24 still does not authorize Test-10k model evaluation.

Reasoning:

1. The loop fixed dataset identity leakage and label-conflict duplicates.
2. It did not create a new model, threshold, blend, or feature mask candidate.
3. Full-test performance was not inspected to make the cleanup decision.
4. The next gate remains Val-only evaluation using a fixed checkpoint and fixed
   threshold protocol.

The clean active split for the next Val gate is:

```text
reports\random_20w_split\stage2_corrected_best_val_auto_noise_dedup_corrected_split_loop24.csv
```
