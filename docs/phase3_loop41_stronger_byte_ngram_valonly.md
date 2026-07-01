# Phase 3 Loop41: Stronger Byte N-gram Val-Only Probe

Date: 2026-07-02

## Objective

Loop37 showed that byte n-gram predictions have low error overlap with Loop28,
but the standalone byte n-gram SGD model was weak and the final full-test blend
reversed. Loop41 tests whether a stronger byte n-gram configuration can become
a useful diverse base learner.

This loop is Val-only. It does not evaluate Test-10k or the 160k full-test.

## Protocol

Same data/evidence lane as Loop37:

- split: `reports/random_20w_split/loop27_corrected_split.csv`
- cache manifest: `data/.cache/manifest_38672ba0.json`
- checkpoint config: `models/random_20w_8192/best_model.pt`
- train rows: `20000`
- val rows: `20000`
- test rows: `0`

Identity fields remain audit/load-only. Filename, path, extension, directory,
source hash, sample id, split, and row order are not model features.

## Stronger Byte N-gram Run

Command:

```powershell
.\vnev\Scripts\python.exe scripts\train_byte_ngram_sgd.py `
  --split-csv reports\random_20w_split\loop27_corrected_split.csv `
  --manifest data\.cache\manifest_38672ba0.json `
  --checkpoint models\random_20w_8192\best_model.pt `
  --output-dir reports\random_20w_split\loop41_byte_ngram_stronger_valonly `
  --n-features 2097152 `
  --prefix-len 4096 `
  --ngram-min 2 `
  --ngram-max 5 `
  --ngram-stride 2 `
  --alphas 3e-6,1e-5,3e-5 `
  --epochs 5 `
  --batch-size 256 `
  --thresholds 0.20:0.80:0.005 `
  --include-byte-hist `
  --include-cache-features `
  --skip-test-eval `
  --seed 43
```

Best standalone result:

- alpha: `3e-6`
- Val F1: `0.9530675152`
- Val errors: `944`
- FP/FN: `529/415`
- AUC: `0.9801784350`

This is substantially stronger than Loop37 standalone byte n-gram:

- Loop37 standalone errors: `1250`
- Loop41 standalone errors: `944`
- improvement: `306` fewer Val errors

## Val-Only Blend Sweep

Command:

```powershell
.\vnev\Scripts\python.exe scripts\analyze_val_prediction_ensemble.py `
  --prediction loop28=reports\random_20w_split\stage2_loop28_content_pe_valonly\stage2_val_predictions.csv=stage2_prob_malicious `
  --prediction byte_ngram_stronger=reports\random_20w_split\loop41_byte_ngram_stronger_valonly\byte_ngram_sgd_val_predictions.csv=prob_malicious `
  --thresholds 0.35:0.65:0.001 `
  --weighted-blend loop28:0.99,byte_ngram_stronger:0.01 `
  ... `
  --weighted-blend loop28:0.70,byte_ngram_stronger:0.30 `
  --key-column sample_index `
  --output-json reports\random_20w_split\loop41_val_ensemble_loop28_stronger_byte_ngram\val_prediction_ensemble_fine_sweep.json
```

Best blend result:

- weights: `0.95 * Loop28 + 0.05 * stronger byte n-gram`
- threshold: `0.486`
- Val F1: `0.9920559580`
- Val errors: `159`
- FP/FN: `87/72`

Loop28 in the same fine threshold sweep:

- Val errors: `161`
- FP/FN: `81/80`

Loop28 locked baseline from Loop28 report:

- Val errors: `162`
- FP/FN: `87/75`

Error overlap:

- Loop28 errors in fine sweep: `161`
- stronger byte n-gram errors: `943`
- shared errors: `83`
- union errors: `1021`
- Jaccard: `0.0812928501`

## Decision

Reject for Test-10k.

The stronger byte n-gram model is a real standalone improvement over Loop37,
but the best blend still reaches only `159` Val errors. That is just `2-3`
errors better than Loop28 depending on the baseline threshold convention. Loop37
already showed that this exact scale of Val/Test-10k improvement can reverse on
full-test.

The gate for future byte n-gram work is therefore stricter:

- standalone byte n-gram should become much stronger than `944` Val errors, or
- a blend must improve Val by at least a wider margin before using Test-10k.

No Test-10k or full-test evaluation was run for Loop41.

## Artifacts

- Byte n-gram report:
  `reports/random_20w_split/loop41_byte_ngram_stronger_valonly/byte_ngram_sgd_report.json`
- Byte n-gram Val predictions:
  `reports/random_20w_split/loop41_byte_ngram_stronger_valonly/byte_ngram_sgd_val_predictions.csv`
- Val ensemble fine sweep:
  `reports/random_20w_split/loop41_val_ensemble_loop28_stronger_byte_ngram/val_prediction_ensemble_fine_sweep.json`
