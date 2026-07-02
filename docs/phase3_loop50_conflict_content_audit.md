# Phase 3 Loop50: Conflict Content-Health Audit

Date: 2026-07-02

## Objective

Loop38/39 showed that Loop28 still has many high-confidence conflicts, but the
manual verdict columns are blank. Loop50 builds a read-only content/cache health
audit for those rows before any replacement attempt.

This loop does not train, tune thresholds, run Test-10k, alter labels, or alter
the split.

## Protocol

Inputs:

- active split: `reports/random_20w_split/loop27_corrected_split.csv`
- cache manifest: `data/.cache/manifest_38672ba0.json`
- review queue:
  `reports/random_20w_split/loop39_loop28_conflict_adjudication/loop28_conflict_adjudication_queue.csv`

New script:

- `scripts/build_loop50_conflict_content_audit.py`

Checks:

- row is still present in the active split
- cache row exists in the active manifest
- cache NPZ has expected shapes and dtypes
- cache label/source SHA match the queue row
- source file exists and SHA matches
- strict PE parsing succeeds
- duplicate SHA groups are surfaced for group-level review

Identity rule:

- filename, path, extension, directory, hash, sample id, split, and row order are
  used only for joining, loading, and audit
- none of them is used as model evidence or relabel evidence

## Result

Full Loop39 queue audit:

- rows: `649`
- lanes:
  - `A_unfixed_severe_conflict`: `256`
  - `B_corrected_severe_conflict`: `42`
  - `C_unfixed_high_conflict`: `245`
  - `D_corrected_high_conflict`: `106`
- error types: FP `416`, FN `233`
- manual verdict blank count: `649`
- recommended action blank count: `649`
- objective content/cache issue rows: `0`
- duplicate SHA group rows: `5`

The audit found no direct objective hygiene issue that justifies automatic
replacement. Cache/source SHA alignment, NPZ tensor health, active split
presence, and strict PE parsing all passed for the queue rows. The only surfaced
issue is duplicate SHA grouping, which must be reviewed by content group rather
than row-by-row.

## Decision

Do not replace samples automatically.

Loop50 strengthens the earlier noise conclusion: severe conflicts are real and
important, but current evidence does not prove the rows are broken. They remain
manual/adjudication candidates or model blindspots until independent evidence
marks them as `label_wrong`, `feature_broken`, or `out_of_scope`.

If a row is later confirmed bad, the rule remains:

```text
remove that bad row and draw one fresh same-label valid candidate; preserve the
exact 200000-row split and rerun replacement/cache readiness audits.
```

## Artifacts

- Summary:
  `reports/random_20w_split/loop50_conflict_content_audit/loop50_conflict_content_audit_summary.json`
- CSV:
  `reports/random_20w_split/loop50_conflict_content_audit/loop50_conflict_content_audit.csv`
- Smoke:
  `reports/random_20w_split/loop50_conflict_content_audit/smoke_a5.json`

## Verification

Commands run:

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\build_loop50_conflict_content_audit.py

.\vnev\Scripts\python.exe -m pytest tests\test_build_loop50_conflict_content_audit.py -q

.\vnev\Scripts\python.exe scripts\build_loop50_conflict_content_audit.py --lane A_unfixed_severe_conflict --limit 5 --output-csv reports\random_20w_split\loop50_conflict_content_audit\smoke_a5.csv --output-json reports\random_20w_split\loop50_conflict_content_audit\smoke_a5.json

.\vnev\Scripts\python.exe scripts\build_loop50_conflict_content_audit.py --output-csv reports\random_20w_split\loop50_conflict_content_audit\loop50_conflict_content_audit.csv --output-json reports\random_20w_split\loop50_conflict_content_audit\loop50_conflict_content_audit_summary.json
```

Results:

- test: `1 passed`
- smoke: `5` rows, objective issue rows `0`
- full audit: `649` rows, objective issue rows `0`, duplicate SHA group rows `5`
