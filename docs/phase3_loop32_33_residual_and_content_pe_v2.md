# Loop32/33 Residual Attribution And Content PE v2

Date: 2026-07-01

## Objective

Loop32/33 followed the Loop28 best model with two goals:

- Build content-only residual attribution for the frozen Loop28 predictions.
- Test whether a broader content PE v2 sidecar improves Val enough to enter Test-10k.

The production feature boundary remains strict: filename, path, directory name,
extension, source hash, sample index, and split are not model inputs. They are
allowed only as cache keys or diagnostic report slices.

## Residual Attribution

New tooling:

- `scripts/analyze_stage2_residual_content.py`
- `tests/test_analyze_stage2_residual_content.py`

Generated reports:

- `reports/random_20w_split/loop32_residual_content_attribution_val/residual_content_attribution_report.json`
- `reports/random_20w_split/loop32_residual_content_attribution_val/content_feature_attribution.csv`
- `reports/random_20w_split/loop32_residual_content_attribution_val/content_feature_slices.csv`
- `reports/random_20w_split/loop32_residual_content_attribution_full/residual_content_attribution_report.json`
- `reports/random_20w_split/loop32_residual_content_attribution_full/content_feature_attribution.csv`
- `reports/random_20w_split/loop32_residual_content_attribution_full/content_feature_slices.csv`

Loop28 Val residuals at the frozen threshold:

- Total: `20000`
- Errors: `162`
- FP/FN: `87/75`
- High-confidence wrong: FP score >= `0.95`: `31`; FN score <= `0.05`: `18`

Loop28 full-test residuals:

- Total: `160000`
- Errors: `1949`
- FP/FN: `1087/862`
- High-confidence wrong: FP score >= `0.95`: `416`; FN score <= `0.05`: `233`

Top content differences are stable between Val and full-test. FN rows are
over-represented in signed/security-directory, overlay, export, DLL, exception,
debug, and non-32-bit/large-address-aware style PE structures. FP rows are
over-represented in high system-DLL ratio, high import counts, high section
entropy, RW sections, and larger files, while being much less often DLLs.

This supports further content-side work, but it does not justify using
filename/path/extension shortcuts.

## Content PE v2

New implementation:

- `scripts/train_stage2_cache_matrix.py`
  - `--content-pe-v2-features`
  - `--content-pe-v2-cache-dir`
- `scripts/build_content_pe_v2_feature_cache.py`
- `tests/test_stage2_content_pe_v2_features.py`

The v2 sidecar adds `182` content-only PE features:

- Specific imported DLL presence and import share.
- More granular API categories such as service, driver, privilege, anti-debug,
  memory, thread, module, process enumeration, persistence, HTTP/socket,
  file mutation, crypto/cert, resource, installer, and COM.
- Export shape, forwarder ratio, ordinal-only ratio, and export pattern flags.
- Resource-tree structure and resource data entropy.
- Section/entrypoint structure, entropy, zero-raw, raw/virtual mismatch, and
  section-name group ratios.

Train/Val cache:

- Input rows: `40000`
- Unique rows: `40000`
- Feature dim: `182`
- Created: `40000`
- Zero features: `0`
- Report: `reports/random_20w_split/stage2_loop32_content_pe_v2_cache_train_val/content_pe_v2_cache_report.json`

## Val Results

| Candidate | Feature setup | Best Val F1 | Val errors | FP/FN | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Loop28 | content PE v1 | `0.9919048571` | `162` | `87/75` | Current best |
| Loop32 | content PE v1 + v2 | `0.9915033986` | `170` | `89/81` | Reject |
| Loop33 | content PE v2 only | `0.9904210736` | `192` | `118/74` | Reject |

Loop32 and Loop33 failed the Val gate. Neither entered Test-10k.

Interpretation: the residual signal is real, but broad PE v2 expansion added
too much redundant/noisy surface for the current Stage-2 candidate set. The
next attempt should be narrower: group-selected v2 subsets, OOF stacking, or
parser-quality improvements, not another wide feature dump.

## Backward Compatibility

After adding v2, the old Loop28 frozen model was re-evaluated on the locked
Test-10k base predictions:

- Report: `reports/random_20w_split/stage2_loop28_content_pe_backward_compat_test10k_eval.json`
- Result: F1 `0.9888677164`, errors `111`, FP/FN `61/50`

This matches the original Loop28 Test-10k result, so v2 code did not break old
Stage-2 model replay.

## Verification

Commands run:

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\train_stage2_cache_matrix.py scripts\evaluate_stage2_cache_model.py scripts\build_content_pe_v2_feature_cache.py scripts\analyze_stage2_residual_content.py
.\vnev\Scripts\python.exe -m pytest tests\test_stage2_content_pe_v2_features.py tests\test_stage2_content_pe_features.py tests\test_analyze_stage2_residual_content.py
```

The leakage regression verifies that identical bytes under different filenames
produce identical content PE v2 features.
