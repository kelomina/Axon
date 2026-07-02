# Phase 3 Loop46: Certificate ASN.1 Structure Val-Only Probe

Date: 2026-07-02

## Objective

Loop46 tested whether higher-quality Authenticode certificate parsing improves
Loop28 residuals. Loop31 had already tested shallow certificate blob features
and failed at `168` Val errors. Loop46 adds a lightweight ASN.1/TLV structure
parser over the PE Security Directory payload, avoiding filename, path,
extension, directory, hash, sample id, split, and row-order features.

This loop is Val-only. It does not evaluate Test-10k or the 160k full-test.

## Feature Boundary

The new features read only the WIN_CERTIFICATE / PKCS#7 byte content from the PE
Security Directory. Paths and source hashes are used only to open files, align
rows, and cache sidecar features.

Feature groups:

- ASN.1 parse success, malformed count, trailing ratio
- node counts by universal tag class, sequence/set/context-specific counts
- OID count and unique OID count
- selected standard OID presence for PKCS#7, Authenticode, code signing,
  digest/signature algorithms, signing time, message digest, and generic
  certificate subject attributes
- UTC/Generalized time counts and year span
- aggregate string-node statistics, without tokenizing subject/issuer text

The implementation deliberately does not add filename/path-derived fields and
does not tokenize signer/vendor names into high-cardinality features.

## Implementation

New tooling:

- `scripts/train_loop46_cert_structure.py`
- `tests/test_loop46_cert_structure.py`

The parser is bounded and dependency-light:

- reads at most the existing `_read_certificate_blob()` payload
- parses ASN.1 definite-length TLV recursively with a node cap
- returns zeros for unsigned files or unreadable blobs
- treats small sample runs as smoke-only so they cannot be marked eligible for
  Test-10k

## Protocol

Inputs:

- checkpoint: `models/random_20w_8192/best_model.pt`
- train predictions: `reports/random_20w_split/loop27_train_predictions.csv`
- val predictions: `reports/random_20w_split/loop27_val_predictions.csv`
- content PE cache: `reports/random_20w_split/content_pe_cache_v1`
- shallow cert cache: `reports/random_20w_split/content_cert_cache_v1`
- new ASN.1 cache: `reports/random_20w_split/content_cert_structure_cache_v1`

Rows:

- train: `20000/20000`, cache misses `0`
- val: `20000/20000`, cache misses `0`

Coverage:

- train signed/structure-present rows: `6815`
- val signed/structure-present rows: `6936`
- train zero rows: `13185`
- val zero rows: `13064`

## Command

```powershell
.\vnev\Scripts\python.exe scripts\train_loop46_cert_structure.py `
  --checkpoint models\random_20w_8192\best_model.pt `
  --train-predictions reports\random_20w_split\loop27_train_predictions.csv `
  --val-predictions reports\random_20w_split\loop27_val_predictions.csv `
  --output-dir reports\random_20w_split\loop46_cert_structure_valonly `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --content-cert-cache-dir reports\random_20w_split\content_cert_cache_v1 `
  --cert-structure-cache-dir reports\random_20w_split\content_cert_structure_cache_v1 `
  --model-candidates hgb_lr0.06_leaf31_l2_0,hgb_lr0.08_leaf31_l2_1e-3 `
  --noise-modes none,soft_conflict_downweight,trim_extreme_conflict `
  --thresholds 0.35:0.65:0.005 `
  --seed 46
```

## Result

Best Val candidate:

- model: `hgb_lr0.08_leaf31_l2_1e-3`
- noise mode: `none`
- threshold: `0.585`
- Val F1: `0.9909891870`
- Val errors: `180`
- FP/FN: `78/102`
- AUC: `0.9994761600`

Comparison:

| Candidate | Val F1 | Errors | FP/FN | Decision |
| --- | ---: | ---: | ---: | --- |
| Loop28 content PE locked reference | `0.9919048571` | `162` | `87/75` | Current best |
| Loop31 shallow cert blob | `0.9916100679` | `168` | `96/72` | Rejected |
| Loop46 cert ASN.1 structure | `0.9909891870` | `180` | `78/102` | Rejected |

Loop46 reduced FP compared with Loop28 but added too many FN. It is worse than
both Loop28 and the earlier shallow certificate blob experiment.

## Decision

Reject for Test-10k.

The report's `test_gate_decision` is `reject_val_margin_too_small`, with
`test10k_error_gate=152`. No Test-10k or full-test evaluation was run.

This result means that certificate structure parsing alone does not currently
solve the signed-file residual. The next useful direction should not be another
minor certificate-feature variant unless it adds external trust validation or
manual adjudication evidence; otherwise it risks re-running the same weak
signal.

## Artifacts

- Report:
  `reports/random_20w_split/loop46_cert_structure_valonly/loop46_cert_structure_report.json`
- Val predictions:
  `reports/random_20w_split/loop46_cert_structure_valonly/loop46_cert_structure_val_predictions.csv`
- ASN.1 sidecar cache:
  `reports/random_20w_split/content_cert_structure_cache_v1`

Generated model/prediction artifacts are not committed because they are large
experiment outputs.

## Verification

Commands run:

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\train_loop46_cert_structure.py
.\vnev\Scripts\python.exe -m pytest tests\test_loop46_cert_structure.py tests\test_identity_feature_guard.py
```

Result: `6 passed`.
