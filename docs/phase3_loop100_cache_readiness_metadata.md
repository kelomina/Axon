# Phase 3 Loop100 Cache Readiness Metadata

Loop100 closes a cache governance gap in the corrected 20w split pipeline. The previous readiness audit proved that every split row had a cache file, but it did not prove that the NPZ content still matched the locked split and manifest.

## What Changed

- `scripts/audit_corrected_split_cache_ready.py` now validates cache metadata by default.
- The audit checks required NPZ fields, label consistency, source SHA consistency, and manifest-declared array shapes.
- Shape checks read `.npy` headers from the NPZ zip members instead of loading full feature arrays.
- `--no-validate-cache-metadata` exists only as an explicit compatibility escape hatch.
- `scripts/build_loop76_redraw_readiness.py` now blocks cache readiness reports that did not enable metadata validation.
- Cache metadata failures are routed to quarantine plus same-original-label redraw, not missing-cache recovery.

## Real 20w Result

Command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_corrected_split_cache_ready.py `
  --split-csv reports\random_20w_split\loop27_corrected_split.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --missing-cache-output reports\random_20w_split\loop100_cache_ready_missing_cache.csv `
  --metadata-issue-output reports\random_20w_split\loop100_cache_ready_metadata_issues.csv `
  --output-json reports\random_20w_split\loop100_cache_ready_metadata.json `
  --enforce-label-balance `
  --strict
```

Output summary:

- total rows: `200000`
- split counts: train `20000`, val `20000`, test `160000`
- label balance: train `10000/10000`, val `10000/10000`, test `80000/80000`
- covered rows: `200000`
- missing rows: `0`
- metadata checked rows: `200000`
- metadata failure rows: `0`
- declared shapes: byte `8192`, PE `256`, stat `49`, lightweight `256`
- decision: `cache_ready=true`

The generated missing-cache and metadata-issue CSV files contain only headers.

## Identity Boundary

`source_path` is still allowed for loading and manifest/cache alignment. It is not evidence that a file is malicious or benign.

The readiness gate blocks on content-stable metadata:

- locked split label
- manifest label
- NPZ label
- manifest source SHA
- NPZ source SHA
- NPZ field presence
- NPZ array shape

Filename, directory, extension, path text, hash text, sample index, split name, row order, and model scores remain forbidden as model, threshold, fusion, GA mask, automatic verdict, or production inference evidence.

## Verification

Resource guard:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\audit_corrected_split_cache_ready.py `
  --output-json reports\random_20w_split\loop100_cache_ready_guard_rerun.json `
  --allow-risk npz_array_load
```

Tests:

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\audit_corrected_split_cache_ready.py scripts\build_loop76_redraw_readiness.py
.\vnev\Scripts\python.exe -m pytest tests\test_audit_corrected_split_cache_ready.py tests\test_build_loop76_redraw_readiness.py tests\test_build_corrected_split_from_plan.py tests\test_import_loop87_review_evidence_verdicts.py -q
```

Result: `40 passed`.

## Current Decision

The 20w corrected split cache is ready from a coverage and metadata standpoint. This does not authorize new training, Test-10k, or full-test evaluation by itself. The funnel still requires an independent content or external verdict, then quarantine plus fresh same-original-label redraw if a bad row is confirmed, then Val-first reverification.
