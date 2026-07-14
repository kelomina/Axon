# Phase 3 Loop44: Regionized Byte N-gram Val-Only Probe

Date: 2026-07-02

## Objective

Loop44 tested a content-only byte n-gram model that reads semantic PE regions
instead of only the cached file prefix. This follows the Loop38/43 residual
diagnosis: remaining errors concentrate around entrypoint, section, resource,
security/certificate, export/import, and overlay-like structures.

This loop is Val-only. It does not evaluate Test-10k or the 160k full-test.

## Feature Boundary

Identity fields remain audit/load-only. Filename, path, extension, directory,
source hash, sample id, split, and row order are not model features.

The new model uses paths only to open the binary content and uses source hashes
only to align against the cache manifest. Model features are hashed byte
n-grams and small scalar statistics from these content regions:

- file head and tail
- entrypoint window
- overlay payload, with the Authenticode security blob excluded
- security directory blob, handled as PE file offset rather than RVA
- resource/import/export directory windows
- first executable section
- last section
- max-entropy section

## Protocol

Inputs:

- split: `reports/random_20w_split/loop27_corrected_split.csv`
- manifest: `data/.cache/manifest_38672ba0.json`
- checkpoint config: `models/random_20w_8192/best_model.pt`
- train rows: `20000`
- val rows: `20000`
- test rows: `0`

Implementation:

- `scripts/train_loop44_region_byte_ngram.py`
- `tests/test_loop44_region_byte_ngram.py`

The tests verify that same bytes under different filenames produce identical
region slices, region salts separate identical bytes in different PE regions,
security directory handling uses file-offset semantics, overlay payload excludes
the security blob, and feature names remain identity-safe.

## Standalone Val Run

Command:

```powershell
.\vnev\Scripts\python.exe scripts\train_loop44_region_byte_ngram.py `
  --split-csv reports\random_20w_split\loop27_corrected_split.csv `
  --manifest data\.cache\manifest_38672ba0.json `
  --checkpoint models\random_20w_8192\best_model.pt `
  --output-dir reports\random_20w_split\loop44_region_byte_ngram_valonly `
  --n-features 2097152 `
  --prefix-len 4096 `
  --region-window 1024 `
  --tail-window 1024 `
  --ngram-min 2 `
  --ngram-max 5 `
  --ngram-stride 2 `
  --alphas 3e-6,1e-5,3e-5 `
  --epochs 4 `
  --batch-size 256 `
  --thresholds 0.20:0.80:0.005 `
  --include-prefix-features `
  --include-byte-hist `
  --include-cache-features `
  --skip-test-eval `
  --seed 44
```

Best standalone candidate:

- alpha: `3e-6`
- threshold: `0.56`
- Val F1: `0.9703068952`
- Val errors: `596`
- FP/FN: `334/262`
- AUC: `0.9939406700`

Region coverage:

| Region | Train rows | Val rows |
| --- | ---: | ---: |
| head | 20000 | 20000 |
| tail | 20000 | 20000 |
| entrypoint | 17219 | 17195 |
| overlay_payload | 2144 | 2047 |
| security_directory | 6856 | 6974 |
| resource_directory | 18541 | 18462 |
| import_directory | 17170 | 17140 |
| export_directory | 5928 | 5953 |
| first_exec_section | 17594 | 17636 |
| last_section | 19908 | 19932 |
| max_entropy_section | 20000 | 20000 |

The standalone model improves materially over Loop41 stronger byte n-gram
(`944` Val errors), but it remains much weaker than Loop28 content PE (`162`
locked Val errors).

## Val-Only Blend Sweep

Command:

```powershell
.\vnev\Scripts\python.exe scripts\analyze_val_prediction_ensemble.py `
  --prediction loop28=reports\random_20w_split\stage2_loop28_content_pe_valonly\stage2_val_predictions.csv=stage2_prob_malicious `
  --prediction loop44_region=reports\random_20w_split\loop44_region_byte_ngram_valonly\loop44_region_byte_ngram_val_predictions.csv=prob_malicious `
  --thresholds 0.35:0.65:0.001 `
  --weighted-blend loop28:0.99,loop44_region:0.01 `
  ... `
  --weighted-blend loop28:0.70,loop44_region:0.30 `
  --key-column sample_index `
  --output-json reports\random_20w_split\loop44_val_ensemble_loop28_region_ngram\val_prediction_ensemble_fine_sweep.json
```

Results:

| Candidate | Val F1 | Errors | FP/FN |
| --- | ---: | ---: | ---: |
| Loop28, same fine sweep | `0.9919504025` | `161` | `81/80` |
| Loop44 standalone, same fine sweep | `0.9703581926` | `595` | `334/261` |
| Best weighted blend, 95/5 | `0.9919576402` | `161` | `90/71` |

Error overlap:

- Loop28 errors: `161`
- Loop44 errors: `595`
- shared errors: `103`
- union errors: `653`
- Jaccard: `0.1577335375`

The weak model is genuinely different from Loop28, but its corrections are not
clean enough to reduce total Val errors.

## Decision

Reject for Test-10k.

Loop44 is a useful negative result: semantic region bytes are better than the
previous prefix-only byte n-gram model, but the best blend only ties Loop28 on
Val error count and shifts FP/FN trade-off. This does not meet the post-Loop37
gate for promotion. No Test-10k or full-test run was performed.

Next steps should not simply increase the region n-gram hash space. Higher-value
follow-ups are:

- OOF residual gating if region n-gram is reused, so candidate overrides are
  trained without train-score leakage.
- Better parser-quality features around certificate-vs-payload overlay and
  resource data entries.
- True diverse base learners from different neural checkpoints or byte lengths,
  not another shallow blend with a weak model.

## Artifacts

- Standalone report:
  `reports/random_20w_split/loop44_region_byte_ngram_valonly/loop44_region_byte_ngram_report.json`
- Standalone Val predictions:
  `reports/random_20w_split/loop44_region_byte_ngram_valonly/loop44_region_byte_ngram_val_predictions.csv`
- Val ensemble sweep:
  `reports/random_20w_split/loop44_val_ensemble_loop28_region_ngram/val_prediction_ensemble_fine_sweep.json`

Generated model/prediction artifacts are not committed because they are large
experiment outputs.

## Verification

Commands run:

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\train_loop44_region_byte_ngram.py
.\vnev\Scripts\python.exe -m pytest tests\test_loop44_region_byte_ngram.py tests\test_identity_feature_guard.py
```

Result: `8 passed`.
