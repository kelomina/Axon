# Loop34/35 Content PE v2 Group Selection

Date: 2026-07-01

## Objective

Loop32/33 showed that broad content PE v2 features did not beat Loop28. Loop34
and Loop35 tested whether narrower v2 feature groups could avoid that noise.

Rules held:

- Val-only selection.
- No Test-10k unless Val beats Loop28's `162` errors.
- No full-test evaluation.
- No filename/path/extension/source-hash/sample-id model features.

## Code Change

`scripts/train_stage2_cache_matrix.py` now supports:

```powershell
--content-pe-v2-groups "import_dll,api,delay_import,imports,export,resource,section,all"
```

The default remains `all`, so Loop32 is still reproducible. Old Loop28 pickles
remain compatible because v2 fields are optional and read through `getattr`.

Validation coverage:

- `tests/test_stage2_content_pe_v2_features.py` verifies group parsing and that
  the section group only selects section/entrypoint-derived features.
- Existing filename-independence tests still pass.

## Loop34 Fast Full-Val Probe

All rows were still full `20000 train / 20000 val`. The "fast" part only means
the candidate matrix used the two strongest HGB candidates and `noise=none`.

| Groups | Best Val F1 | Errors | FP/FN | Decision |
| --- | ---: | ---: | ---: | --- |
| `imports` | `0.9918049170` | `164` | `88/76` | Reject |
| `export` | `0.9918073734` | `164` | `91/73` | Reject |
| `export,section` | `0.9918163673` | `164` | `102/62` | Reject |
| `api,section` | `0.9917668779` | `165` | `103/62` | Reject |
| `resource` | `0.9917091200` | `166` | `94/72` | Reject |
| `api` | `0.9916092298` | `168` | `95/73` | Reject |
| `section` | `0.9915630772` | `169` | `100/69` | Reject |
| `imports,section` | `0.9915656036` | `169` | `103/66` | Reject |
| `import_dll` | `0.9912443088` | `175` | `81/94` | Reject |

## Loop35 Full Candidate Matrix

The three closest groups were rerun with the full default candidate matrix and
all existing noise modes.

| Groups | Best Val F1 | Errors | FP/FN | Best model/noise | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| `imports` | `0.9918049170` | `164` | `88/76` | `hgb_lr0.08_leaf31_l2_1e-3 / none` | Reject |
| `export` | `0.9918073734` | `164` | `91/73` | `hgb_lr0.08_leaf31_l2_1e-3 / none` | Reject |
| `export,section` | `0.9918163673` | `164` | `102/62` | `hgb_lr0.08_leaf31_l2_1e-3 / none` | Reject |

None beat Loop28's Val `162` errors. Therefore none entered Test-10k.

## Interpretation

Content PE v2 is not the next near-term lever in its current form:

- Full v2: `170` Val errors.
- v2 only: `192` Val errors.
- Best narrow v2 group: `164` Val errors.
- Best narrow v2 group after full candidate/noise matrix: still `164` Val errors.

The residual evidence remains useful, but the current v2 implementation is not
a validated production improvement. The next attempt should move to OOF
stacking, parser-quality improvements, or a genuinely different signal source
instead of continuing v2 group permutations.
