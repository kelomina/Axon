# Phase 3 Loop102 Content-Hash Candidate Pool

Loop102 hardens the fresh-redraw candidate pool. Replacement candidates must be de-duplicated by real file content hash, not by filename, path, directory, extension, or SHA-like names.

## Why

The strict 20w protocol requires bad files or bad features to be quarantined and replaced with fresh same-original-label samples. A candidate is not fresh if the same file content is already present in the locked split or manifest under another name.

Filename-derived SHA values are not enough for this. Real deployment names and corpus names can differ, and a file can be renamed without changing content.

## Changes

- `scripts/build_replacement_candidate_pool.py`
  - hashes candidate files by content by default;
  - records `source_sha256_origin`;
  - supports `--no-hash-files` only as an explicit compatibility escape hatch;
  - uses manifest `source_sha256` values as already-used content identities;
  - de-duplicates candidates globally, not just inside a label bucket;
  - defaults to bounded scans when replacement counts are provided.
- `scripts/build_loop76_redraw_readiness.py`
  - blocks candidate pools where strict content hashing was not required;
  - blocks candidate pools containing unhashed rows.

`source_sha256` remains audit metadata only. It is used for duplicate detection, cache alignment, and replacement integrity. It is not malware evidence and must not be used as a model, threshold, fusion, verdict, or production inference feature.

## Real Candidate Pool

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_replacement_candidate_pool.py `
  --data-dir data `
  --split-csv reports\random_20w_split\loop27_corrected_split.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --required-label0 125 `
  --required-label1 5 `
  --output-csv reports\random_20w_split\loop102_replacement_candidate_pool_content_hash.csv `
  --output-json reports\random_20w_split\loop102_replacement_candidate_pool_content_hash.json
```

Result:

- rows: `305`
- label counts: `0=250`, `1=55`
- required replacements: `0=125`, `1=5`
- replacement shortfall: `{}`
- enough for required replacements: `true`
- source SHA origins: `content_hash=305`
- unhashed candidates: `0`
- cache-present candidates: `0`
- bounded scan limits: label `0=250`, label `1=55`
- runtime: about `16.9s`

The earlier unbounded full candidate scan was stopped after a 30-minute timeout. The bounded scan keeps strict content hashing while avoiding long-running opaque work.

## Verification

Resource guard:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_replacement_candidate_pool.py `
  --output-json reports\random_20w_split\loop102_candidate_pool_guard.json
```

Tests:

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\build_replacement_candidate_pool.py scripts\build_loop76_redraw_readiness.py
.\vnev\Scripts\python.exe -m pytest tests\test_build_replacement_candidate_pool.py tests\test_build_loop76_redraw_readiness.py -q
```

Result: `21 passed`.

## Current Decision

The replacement candidate pool is sufficient for a `125/5` same-original-label redraw if future independent verdicts confirm bad rows. It does not authorize replacement, training, Test-10k, or full-test by itself.
