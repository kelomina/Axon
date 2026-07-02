# Phase 3 Loop49: Content PE Productization Audit

Date: 2026-07-02

## Objective

Loop28 is still the strongest current-split result, but its 100 content-derived
PE metadata features live in the Stage-2 experiment path. Loop49 checks whether
those features are already covered by the main fixed-v2 PE cache schema, and
which proven content signals still need a stable product schema before the next
model attempt.

This loop is a productization preflight only. It does not train, tune
thresholds, run Test-10k, or evaluate full-test.

## Protocol

Inputs:

- Loop28 content PE schema: `scripts/train_stage2_cache_matrix.py`
- main extractor fixed-v2 schema: `src/kvd_features/extractor.py`
- output report:
  `reports/random_20w_split/loop49_content_pe_productization_audit.json`

New code:

- `fixed_v2_feature_names()` in `src/kvd_features/extractor.py`
- `scripts/audit_content_pe_productization.py`
- `tests/test_fixed_v2_feature_names.py`
- `tests/test_audit_content_pe_productization.py`

Identity policy:

- filename, path, extension, directory, hash, sample id, split, and row order
  are audit-only metadata
- the audit checks feature names through `scripts/identity_feature_guard.py`
- no identity-derived feature is allowed as model evidence

## Result

The current fixed-v2 PE vector has:

- configured PE dimension: `256`
- used fixed-v2 dimensions: `143`
- reserved dimensions: `113`

Loop28 content PE schema has:

- feature count: `100`
- exact name overlaps with fixed-v2: `0`
- known covered or partially covered content features: `20`
- productization gaps: `80`

High-value gap groups:

| Gap group | Missing features |
| --- | ---: |
| `data_directory_size_ratio` | `28` |
| `header_flags` | `12` |
| `section_permission_combo` | `8` |
| `layout_ratio` | `8` |
| `import_shape` | `5` |
| `overlay` | `4` |
| `resource_shape` | `2` |

The large number of reserved fixed-v2 dimensions means there is room to promote
validated content-derived features without changing the 256-dimensional PE
tensor shape, but the column semantics must be explicit and versioned. Silent
reuse of reserved positions would make existing checkpoints ambiguous, so the
next implementation should be a named schema migration, for example
`fixed_v3` or `content_pe_v1`.

## Decision

Do not run Test-10k from Loop49.

Loop49 is not a model candidate; it is schema evidence. Its decision is:

```text
Promote Loop28 content PE as a named stable content-derived schema or fixed_v3
candidate; do not rely on external identity fields.
```

The highest-value implementation target is not more HGB tuning. It is moving
the Loop28-proven content signals into a stable extraction path, then training a
new current-split model or Stage-2 candidate under the normal Val gate.

## Verification

Commands run:

```powershell
.\vnev\Scripts\python.exe -m py_compile src\kvd_features\extractor.py scripts\audit_content_pe_productization.py

.\vnev\Scripts\python.exe -m pytest tests\test_fixed_v2_feature_names.py tests\test_audit_content_pe_productization.py -q

.\vnev\Scripts\python.exe scripts\audit_content_pe_productization.py --output-json reports\random_20w_split\loop49_content_pe_productization_audit.json
```

Result:

- new tests: `4 passed`
- audit output: `reports/random_20w_split/loop49_content_pe_productization_audit.json`
