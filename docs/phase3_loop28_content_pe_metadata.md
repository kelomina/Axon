# Loop28 Content PE Metadata Experiment

Date: 2026-07-01

## Decision

Loop28 is a valid improvement over Loop26/Loop27. It adds 100 content-derived
PE metadata features to Stage-2 and does not use filename, extension, directory
name, or path text as model input.

The best frozen candidate is:

- Model: `hgb_lr0.06_leaf31_l2_0__noise_none`
- Feature dim: `1520`
- Threshold: `0.50`, selected only on Val
- Train rows: `20000`
- Val rows: `20000`
- Test rows: `160000`

## Why This Is Not The Rejected Name-Feature Shortcut

The rejected shortcut used deployment-unstable naming signals. Real-world file
names can be arbitrary, so filename/extension/path-derived model inputs are not
valid for production scoring.

Loop28 uses only file content:

- PE header and optional header metadata
- Data directory presence and sizes
- Import/export/resource/TLS/relocation counts
- Import API category ratios
- Overlay size and entropy
- Section permission combinations and entropy summaries

Path is used only to open the file and to locate a sidecar cache entry. It is
not encoded into the feature vector.

## Funnel Results

| Stage | Result |
| --- | --- |
| Content PE cache, train+val | `40000/40000` unique rows, `zero_features=0` |
| Val content-only | F1 `0.9919048571`, errors `162`, FP/FN `87/75` |
| Val content+kNN | F1 `0.9917429815`, errors `165`, FP/FN `74/91` |
| Test-10k frozen content-only | F1 `0.9888677164`, errors `111`, FP/FN `61/50` |
| Full-test frozen content-only | F1 `0.9878358558`, errors `1949`, FP/FN `1087/862` |

Loop26 blend comparison:

| Stage | Loop26 blend | Loop28 content PE | Delta |
| --- | ---: | ---: | ---: |
| Val errors | `223` | `162` | `-61` |
| Test-10k errors | `144` | `111` | `-33` |
| Full-test errors | `2571` | `1949` | `-622` |

## Evidence Paths

- Train/Val content cache report:
  `reports/random_20w_split/stage2_loop28_content_pe_cache_train_val/content_pe_cache_report.json`
- Content-only Val matrix:
  `reports/random_20w_split/stage2_loop28_content_pe_valonly/stage2_cache_matrix_report.json`
- Content+kNN Val matrix:
  `reports/random_20w_split/stage2_loop28_content_pe_knn_valonly/stage2_cache_matrix_report.json`
- Test-10k content cache report:
  `reports/random_20w_split/stage2_loop28_content_pe_cache_test10k/content_pe_cache_report.json`
- Frozen Test-10k evaluation:
  `reports/random_20w_split/stage2_loop28_content_pe_frozen_test10k_eval.json`
- Full-test content cache report:
  `reports/random_20w_split/stage2_loop28_content_pe_cache_full_test/content_pe_cache_report.json`
- Frozen full-test evaluation:
  `reports/random_20w_split/stage2_loop28_content_pe_frozen_full_test_eval.json`

## Interpretation

Content PE metadata is the strongest validated improvement so far. It improves
Val, Test-10k, and full-test in the same direction, which is the key signal that
the gain is not just Val overfitting.

OOF kNN support features did not help after content PE metadata was added. The
best content+kNN Val result had `165` errors, while content-only had `162`
errors. The next iteration should not add kNN by default.

Full-test residual error slices, using extension only for analysis:

| Slice | Errors | FP | FN |
| --- | ---: | ---: | ---: |
| `<none>` | `887` | `849` | `38` |
| `.exe` | `831` | `237` | `594` |
| `.dll` | `218` | `1` | `217` |
| `.sys` | `6` | `0` | `6` |

Compared with Loop26, content PE metadata reduced the largest known blind spots
but did not eliminate them. Extension remains an analysis slice only, not a
production model feature.

## Remaining Gap

Loop28 still has `1949` full-test errors. This remains far from the `F1 >= 99.9%`
target, which would require roughly hundred-level errors on the 160k balanced
test set. The next work should focus on residual error attribution and stable
content-schema integration, not on filename-derived shortcuts or more Val-only
noise replacement.
