# Phase 3 Loop86 Review Evidence Package

## Purpose

Loop86 turns the Loop65 compact review batch into a read-only content-evidence
package for manual or external adjudication.

It does not train, tune thresholds, relabel, mutate the split, mutate cache, or
authorize Test/Test-10k. It exists to make the next noise-review step auditable:
reviewers can inspect content facts without treating filenames, paths, hashes,
row ids, model scores, or review ranks as label evidence.

## Inputs

- review batch:
  `reports/random_20w_split/loop65_A_lane_review_batch.csv`
- rows: `62`
- categories:
  - severe persistent FN: `20`
  - severe persistent FP: `20`
  - duplicate content group entry: `2`
  - corrected by another model: `20`

Loop65 manual fields were blank before Loop86 and remain blank after Loop86.

## Evidence Boundaries

Content evidence fields include:

- file size and file SHA match status
- sampled byte entropy
- MZ/PE parse status
- PE machine, optional-header magic, subsystem, characteristics
- section count and section names
- import/export/resource/security directory presence and sizes
- overlay size and overlay entropy
- overlay-after-security size and entropy

Alignment and priority fields are not evidence:

- `source_path`
- `cache_path`
- `source_sha256`
- `sample_index`
- `split`
- Loop57 probabilities
- Loop57 error type
- review rank/category/lane
- whether another model corrected the row

Those fields are allowed only for loading, cache alignment, duplicate grouping,
manual review indexing, and explaining why a row was selected for review. They
must not be used as training features, fusion features, threshold shortcuts,
automatic relabel evidence, production inference evidence, or replacement
sampling keys.

## Commands

Initial unit test:

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_build_loop86_review_evidence_package.py -q
```

Result: `2 passed`.

First guard attempt:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop86_review_evidence_package.py `
  --output-json reports\random_20w_split\loop86_review_evidence_guard.json
```

Result: blocked by static `while True` risk in chunked file reading.

Fix: replaced the loop with `iter(lambda: handle.read(chunk_size), b"")`.

Re-test and guard:

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_build_loop86_review_evidence_package.py -q
```

Result: `2 passed`.

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop86_review_evidence_package.py `
  --output-json reports\random_20w_split\loop86_review_evidence_guard.json
```

Result: `decision=pass`, static findings `0`.

Evidence package:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop86_review_evidence_package.py `
  --review-csv reports\random_20w_split\loop65_A_lane_review_batch.csv `
  --output-csv reports\random_20w_split\loop86_review_evidence_package.csv `
  --output-json reports\random_20w_split\loop86_review_evidence_package_summary.json `
  --max-entropy-bytes 67108864
```

## Result

Loop86 generated a complete `62` row evidence package.

Health checks:

- source exists: `62/62`
- cache exists: `62/62`
- source SHA mismatch: `0`
- PE parse status: `ok=62`
- manual fields blank: `true`

Review tags:

- `has_resource_directory`: `58`
- `has_security_directory`: `29`
- `overlay_present`: `40`
- `high_overlay_entropy`: `34`
- `overlay_after_security_present`: `14`
- `high_file_entropy`: `10`
- `many_sections`: `9`
- `no_import_directory`: `1`

These are content facts for review. They are not label verdicts. For example,
overlay presence, high entropy, or a security directory can support investigation,
but none of them automatically proves maliciousness or benignness.

## Decisions

- automatic relabel allowed: `false`
- automatic replacement allowed: `false`
- training allowed from this package: `false`
- Test-10k allowed from this package: `false`

Only an independent manual or external verdict can trigger quarantine plus fresh
redraw from the locked-manifest original-label pool. This package alone does not
change labels or counts.

## Verdict

Loop86 confirms that the first `62` high-priority review rows are technically
accessible and parseable. There is no source/cache/SHA/PE-parse failure that
would justify automatic replacement.

The next useful step is external adjudication or deeper content review of these
rows, using Loop86 as the evidence sheet. Any confirmed `label_wrong`,
`feature_broken`, or `out_of_scope` row must be quarantined and replaced with a
fresh valid sample from the locked manifest's same original-label pool.
