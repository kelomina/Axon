# Phase 3 Loop81 Val Complementarity Audit

## Purpose

Loop81 checks whether Loop57 and the probability calibrator make different
Val errors often enough to justify a later Val-only fusion probe.

This is not a training run, not threshold tuning, and not Test/Test-10k access.
Identity fields are audit metadata only. `source_sha256` is used only to align
prediction rows that came from the same original file; it is forbidden as model
evidence, fusion input, threshold logic, or a noise-cleaning decision feature.

## Why sample_index Was Rejected

The first strict audit joined the two prediction files by `sample_index` and
found `9830` label mismatches on the nominal 20k Val overlap. That means the
two CSVs do not share a comparable row/sample numbering scheme. `sample_index`
is therefore unsafe for this audit unless both files are proven to come from the
same manifest and ordering.

This matches the production rule: filenames, paths, extensions, directories,
row order, sample indices, hashes, and split names must never be model evidence.
They can only be used for loading, alignment, cache audit, duplicate detection,
and manual review indexing.

## Command

Guard:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\analyze_loop81_val_complementarity.py `
  --output-json reports\random_20w_split\loop81_val_complementarity_guard.json
```

Result: `decision=pass`.

Audit:

```powershell
.\vnev\Scripts\python.exe scripts\analyze_loop81_val_complementarity.py `
  --loop57-predictions reports\random_20w_split\loop57_fn_overlay_gate_valonly\loop57_fn_overlay_gate_val_predictions.csv `
  --calibrator-predictions reports\random_20w_split\phase2_val_calibrator_predictions.csv `
  --join-key source_sha256 `
  --output-json reports\random_20w_split\loop81_val_complementarity_sha.json `
  --output-overlap-csv reports\random_20w_split\loop81_val_complementarity_sha_overlap.csv `
  --strict
```

Strict mode returned failure by design because the audit found blockers.

## Result

Input rows:

- Loop57 Val predictions: `20000`
- calibrator Val predictions: `20000`
- Loop57 unique `source_sha256`: `20000`
- calibrator unique `source_sha256`: `19998`
- common unique keys: `19907`
- unambiguous joined rows: `19906`
- missing from Loop57 side: `92` rows
- missing from calibrator side: `93` rows

Blockers:

- Val overlap is not exactly `20000` rows.
- The calibrator prediction file has duplicate `source_sha256` values.
- The common `source_sha256` set contains an ambiguous duplicate key.
- The two prediction files do not cover the same `source_sha256` set.

Duplicate-key examples:

- `4c9a98896700e1ec034f17f232565a5e1fcf2286972d813bf2d9fe8c3ba39b1e`
  appears twice in the calibrator file with labels `0` and `1`.
- `3b142b8f4460762d9a0cc78a1b33733b0876e05d3219c08e8b5d3ef77da79db6`
  appears twice in the calibrator file with label `1`.

The first example is direct label-noise evidence: identical file content is
present under conflicting labels. This is exactly the kind of noise that makes
`F1 >= 99.9%` unrealistic unless it is quarantined and replaced by fresh
same-original-label redraws under the 20w protocol.

## Diagnostic Metrics Only

The following numbers were computed on the `19906` unambiguous joined rows, so
they are diagnostic only and must not be used for candidate selection:

- Loop57: F1 `0.9926897656719407`, errors `146`, FP/FN `91 / 55`
- calibrator: F1 `0.9725533406791546`, errors `548`, FP/FN `289 / 259`
- oracle choose-correct-if-either: F1 `0.9955448766080993`, errors `89`
- calibrator-only-correct rows: `57`
- Loop57-only-correct rows: `459`
- both-wrong rows: `89`

The diagnostic oracle says there is some complementarity, but the alignment
blockers prevent using this file pair to approve a fusion probe.

## Verdict

Loop81 is a readiness failure, not a fusion success:

- Do not run Test-10k or full-test from this result.
- Do not train or tune a fusion rule from these two Val prediction files.
- Re-export both Val prediction files from the same corrected 20w manifest, or
  rebuild the calibrator Val predictions from the current Loop79-ready split.
- Quarantine duplicate-content and cross-label cases for Data-Agent noise
  review. Bad files/features must be replaced by fresh same-original-label
  redraws; they must not be used to patch counts.
- Keep `source_sha256` strictly as an alignment/cache-audit identifier.

## Validation

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_analyze_loop81_val_complementarity.py -q
```

Result: `5 passed`.
