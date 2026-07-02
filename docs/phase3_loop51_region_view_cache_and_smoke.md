# Phase 3 Loop51: Region-view Neural Cache And Smoke

Date: 2026-07-02

## Objective

Loop44 showed that semantic PE regions carry real content signal, but the
region n-gram SGD path was too weak. Loop51 prepares a neural version of that
idea: replace the byte prefix input with an 8192-byte view built from PE/content
regions, while keeping PE/stat/lightweight features unchanged.

This loop is still Train/Val only. It does not build Test region cache, run
Test-10k, or evaluate full-test.

## Region View

The generated `byte_sequence` is made from fixed slots over content-derived
regions:

- head
- tail
- entrypoint
- overlay payload
- security directory
- resource directory
- import directory
- export directory
- first executable section
- last section
- max entropy section

For 8192 bytes, the first eight regions receive `745` bytes and the last three
receive `744` bytes. Missing regions are zero-padded. Paths and hashes are used
only to open files, align cache rows, and audit; they are not encoded into the
tensor.

## Cache Result

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop51_region_view_cache.py --splits train,val --output-cache-dir data\.cache_loop51_region_view_8192 --output-json reports\random_20w_split\loop51_region_view_cache_train_val_audit.json --workers 8 --no-skip-existing
```

Result:

- requested rows: `40000`
- written rows: `40000`
- split counts: train `20000`, val `20000`
- label counts: benign `20000`, malicious `20000`
- issue counts: `{}`
- output manifest:
  `data/.cache_loop51_region_view_8192/manifest_38672ba0.json`

Region coverage:

| Region | Present rows |
| --- | ---: |
| head | `40000` |
| tail | `40000` |
| max entropy section | `40000` |
| last section | `39840` |
| resource directory | `37003` |
| first executable section | `35230` |
| entrypoint | `34414` |
| import directory | `34310` |
| security directory | `13830` |
| export directory | `11881` |
| overlay payload | `4191` |

## Loader Verification

Direct `FeatureCacheDataset(cache_dir=...)` verification loaded the region
cache and produced:

- dataset rows: `40000`
- train indices: `20000`
- val indices: `20000`
- item shapes: byte `[8192]`, PE `[256]`, stat `[49]`

The generic `create_split_from_file()` helper rejects this train/val-only cache
because test is intentionally absent. That is expected for this loop and helps
prevent accidental Test use before the Val gate.

## Neural Smoke

New files:

- `scripts/train_loop51_region_view_neural.py`
- `config/random_20w_region_view_8192_seed51.toml`

Smoke command:

```powershell
.\vnev\Scripts\python.exe scripts\train_loop51_region_view_neural.py --config config\random_20w_region_view_8192_seed51.toml --cache-dir data\.cache_loop51_region_view_8192 --epochs 1 --batch-size 16 --max-train-samples 512 --max-val-samples 512 --output-dir models\smoke_loop51_region_view_e1 --summary-json reports\random_20w_split\loop51_region_view_neural_smoke512_summary.json
```

Smoke result:

- train samples: `512`
- val samples: `512`
- test samples: `0`
- best smoke Val F1: `0.6692810458`
- FP/FN at smoke Val: `253/0`

This is not a candidate metric. It only proves that the region-view cache can be
loaded by the neural model and trained through `AxonTrainer`. A real Loop51
candidate still requires full `20000 train / 20000 val` training and must beat
the Loop28 Val gate before any Test-10k run.

## Gate

Reference remains Loop28 content PE metadata:

- Val errors: `162`
- Val F1: `0.9919048571`

For Loop51 to enter Test-10k, it must run full Val and reach the existing gate:

- general candidate: clearly below `162` Val errors
- shallow/override-like candidate: `<=152` Val errors
- no Test-10k if the improvement is only a few Val samples

## Verification

Commands run:

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\build_loop51_region_view_cache.py scripts\train_loop51_region_view_neural.py

.\vnev\Scripts\python.exe -m pytest tests\test_build_loop51_region_view_cache.py tests\test_train_loop51_region_view_neural.py -q
```

Result: `3 passed`.
