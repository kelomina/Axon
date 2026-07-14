# Phase 2 Loop 10: Split Duplicate Source Audit

## Scope

Loop 9 surfaced `23` historical duplicate source-key rows in the current 20w
split. Loop 10 separates that finding from replacement integrity and audits it
as a data-noise and leakage risk.

This loop is read-only. It does not change labels, splits, thresholds, blend
weights, feature masks, calibrators, model hyperparameters, or test-set
artifacts.

## Tooling Added

- Script: `scripts/audit_split_duplicate_sources.py`
- Test: `tests/test_audit_split_duplicate_sources.py`

The script groups split rows by normalized `source_path` and sha-like identity
keys inferred from `source_sha256` or sha-like filenames. It reports:

- duplicate groups
- duplicate extra rows
- cross-label duplicate groups
- cross-split duplicate groups
- same-path duplicate groups
- detail rows for manual/data-policy review

## Command

```powershell
.\vnev\Scripts\python.exe scripts\audit_split_duplicate_sources.py `
  --split-csv reports\random_20w_split\random_20w_split.csv `
  --output-json reports\random_20w_split\random_20w_split_duplicate_sources.json `
  --output-csv reports\random_20w_split\random_20w_split_duplicate_sources.csv
```

## Result

The 20w split shape is still exact:

- Rows: `200000`
- Split counts: `train=20000`, `val=20000`, `test=160000`
- Label balance:
  - Train: `0=10000`, `1=10000`
  - Val: `0=10000`, `1=10000`
  - Test: `0=80000`, `1=80000`

Duplicate source identity findings:

- Duplicate groups: `23`
- Duplicate extra rows: `23`
- Maximum group size: `2`
- Same-path duplicate groups: `0`
- Cross-label duplicate groups: `4`
- Cross-split duplicate groups: `9`

Label patterns:

- `0`: `4`
- `1`: `15`
- `0|1`: `4`

Split patterns:

- `test`: `11`
- `train`: `1`
- `val`: `2`
- `test|train`: `3`
- `test|val`: `6`

Cross-split patterns:

- `test|train`: `3`
- `test|val`: `6`

## Interpretation

This is not a row-count failure. The split still has exactly `200000` samples.
The problem is identity duplication:

1. `9` duplicate groups cross split boundaries.

   These are potential validation/test leakage risks because the same source
   identity can appear in both model-selection evidence and held-out evidence.

2. `4` duplicate groups cross labels.

   These are direct label-noise conflicts: the same source identity appears as
   both benign and malicious.

3. `0` groups are same-path duplicates.

   The issue is not repeated CSV rows. It is duplicate source identity across
   different directories or differently cased filenames.

Examples in the detail CSV show the pattern:

- Same malicious SHA appearing under different malicious family/date folders.
- Same SHA appearing once under `待加入白名单` and once under `待拉黑`.
- Some duplicate identities cross `test|train` or `test|val`.

## Policy

Do not fix this by deleting rows and accepting a short split.

Any corrected split must:

- remove duplicate source identities from the active 20w split,
- replace removed rows with fresh valid candidates,
- preserve exactly `200000` rows,
- preserve `20000 / 20000 / 160000`,
- preserve per-split label balance unless a documented label-policy change is
  explicitly approved,
- re-run replacement integrity audit,
- re-run cache readiness audit,
- re-run full Val before any Test-10k confirmation.

Full-test duplicate findings are data-quality evidence only. They must not be
used to tune thresholds, blend weights, feature masks, calibrators, or model
hyperparameters.

## Test Coverage

Validation command:

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_audit_split_duplicate_sources.py -q
```

Result: `5 passed`.

Covered cases:

- Clean split reports no duplicates.
- Same-path duplicates are detected.
- Cross-split and cross-label conflicts are detected.
- Sha-like path stems are detected as duplicate source identities.
- Detail CSV rows are written for review.

## Decision

Duplicate-source cleanup should become the next data-side correction target
after the Val manual adjudication queue. It is a measurable noise/leakage issue
and must be handled through fresh resampling rather than self-filling or row
deletion.
