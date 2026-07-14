# Phase 2 Loop 25: Loop 24 Stage-2 Evaluation

## Scope

Loop 25 evaluates the clean Loop 24 split through the locked funnel:

1. Val-only base checkpoint evaluation.
2. Val-only Stage-2 candidate fitting.
3. Val-only blend weight and threshold selection.
4. One frozen Test-10k confirmation.
5. One frozen 160k full-test evaluation.

No Test-10k or full-test threshold sweep was used. Full-test parameters were
frozen from Val:

- stage2 extended weight: `0.5`
- stage2 kNN weight: `0.5`
- blend threshold: `0.54`

## Active Clean Split

```text
reports\random_20w_split\stage2_corrected_best_val_auto_noise_dedup_corrected_split_loop24.csv
```

The split passed strict checks before evaluation:

- total rows: `200000`
- train/val/test: `20000 / 20000 / 160000`
- per-split label balance preserved
- SHA-like duplicate source groups: `0`
- cache coverage: `200000 / 200000`

## Old Split Comparison Blocked

I attempted to rerun the old `duplicate_source_corrected_split.csv` Val
baseline under the current strict cache loader.

Command:

```powershell
.\vnev\Scripts\python.exe scripts\evaluate_split_from_cache.py `
  --checkpoint models\random_20w_8192\best_model.pt `
  --config config\random_20w_8192.toml `
  --split-csv reports\random_20w_split\duplicate_source_corrected_split.csv `
  --manifest data\.cache\manifest_38672ba0.json `
  --output-json reports\random_20w_split\loop24_old_duplicate_source_corrected_val_eval_sweep.json `
  --split val `
  --threshold 0.5 `
  --sweep-thresholds "0.35,0.385,0.40,0.43,0.45,0.50,0.505,0.525,0.55,0.575,0.60" `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda
```

Result:

- blocked by strict cache label mismatch
- example: expected split label `0`, cached NPZ label `1`

Decision: do not bypass this guard. The old split is not a clean apples-to-apples
training/validation surface after the duplicate and label-conflict findings.

## Base Checkpoint on Clean Val

Command:

```powershell
.\vnev\Scripts\python.exe scripts\evaluate_split_from_cache.py `
  --checkpoint models\random_20w_8192\best_model.pt `
  --config config\random_20w_8192.toml `
  --split-csv reports\random_20w_split\stage2_corrected_best_val_auto_noise_dedup_corrected_split_loop24.csv `
  --manifest data\.cache\manifest_38672ba0.json `
  --output-json reports\random_20w_split\loop24_dedup_corrected_val_eval_sweep.json `
  --split val `
  --threshold 0.5 `
  --sweep-thresholds "0.35,0.385,0.40,0.43,0.45,0.50,0.505,0.525,0.55,0.575,0.60" `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda `
  --output-predictions-csv reports\random_20w_split\loop24_dedup_corrected_val_predictions.csv
```

Fixed threshold `0.5`:

- F1: `0.9328055217`
- AUC: `0.97822714`
- errors: `1324 / 20000`
- FP/FN: `514 / 810`

Best scanned Val threshold:

- threshold: `0.43`
- F1: `0.9335931644`
- errors: `1329 / 20000`
- FP/FN: `671 / 658`

Interpretation: threshold movement alone is not a solution.

## Stage-2 Val Candidates

Train predictions were exported from the same clean split:

```text
reports\random_20w_split\loop24_dedup_corrected_train_predictions.csv
```

### Extended Stage-2

Output:

```text
reports\random_20w_split\stage2_loop24_extended_valonly
```

Selected by Val:

- model: `hgb_lr0.10_leaf63_l2_1e-3`
- noise mode: `none`
- threshold: `0.5`
- Val F1: `0.9867437456`
- Val AUC: `0.9992214`
- errors: `266 / 20000`
- FP/FN: `166 / 100`

### Extended + kNN Stage-2

Output:

```text
reports\random_20w_split\stage2_loop24_knn_extended_valonly
```

Selected by Val:

- model: `hgb_lr0.08_leaf31_l2_1e-3`
- noise mode: `knn_soft_conflict_downweight`
- threshold: `0.41`
- Val F1: `0.9871801267`
- Val AUC: `0.99926429`
- errors: `257 / 20000`
- FP/FN: `152 / 105`

## Val Blend Selection

Output:

```text
reports\random_20w_split\stage2_loop24_blend_val_grid.json
reports\random_20w_split\stage2_loop24_blend_val_best_predictions.csv
```

Selected by Val:

- extended weight: `0.5`
- kNN weight: `0.5`
- threshold: `0.54`
- Val F1: `0.9882`
- Val AUC: `0.99936428`
- errors: `236 / 20000`
- FP/FN: `118 / 118`

This clearly beats the Phase 1 Val gate `0.9687640114`, so it qualified for one
frozen Test-10k confirmation.

## Frozen Test-10k Confirmation

Test-10k is the first `10000` rows of the clean Loop 24 test split. No threshold
sweep was run.

Base checkpoint input:

- F1: `0.9299908285`
- errors: `687 / 10000`
- FP/FN: `270 / 417`

Extended Stage-2:

- F1: `0.9822698588`
- errors: `177 / 10000`
- FP/FN: `100 / 77`

kNN Stage-2:

- F1: `0.9833667335`
- errors: `166 / 10000`
- FP/FN: `93 / 73`

Frozen blend:

- threshold: `0.54`
- F1: `0.9842258615`
- AUC: `0.9986420183`
- errors: `157 / 10000`
- FP/FN: `75 / 82`

Decision: the frozen blend passed Test-10k confirmation and was eligible for one
full-test evaluation.

## Frozen Full-Test Result

Base checkpoint:

- rows: `160000`
- F1: `0.9284814254`
- AUC: `0.9766111998`
- errors: `11295`
- FP/FN: `4613 / 6682`

Extended Stage-2:

- F1: `0.9806573752`
- AUC: `0.9978295374`
- errors: `3106`
- FP/FN: `1842 / 1264`

kNN Stage-2:

- F1: `0.9828778010`
- AUC: `0.9984093748`
- errors: `2747`
- FP/FN: `1591 / 1156`

Frozen blend:

- threshold: `0.54`
- F1: `0.9832264030`
- AUC: `0.9984619640`
- errors: `2685 / 160000`
- FP/FN: `1379 / 1306`

## Target Feasibility

The current frozen full-test F1 is `0.9832264030`, below the `0.999` target.

At 16w test size, `F1 >= 0.999` requires roughly low-hundreds total errors, not
thousands. This run has `2685` errors. The target is therefore not reachable by
small threshold tweaks, probability calibration, or a shallow Stage-2 blend
alone.

This is also supported by the balanced FP/FN profile:

- false positives: `1379`
- false negatives: `1306`
- high-confidence FP `>=0.90`: `634`
- very-low-score FN `<0.10`: `339`

High-confidence mistakes indicate real label/source noise and feature blind
spots, not merely a decision threshold issue.

## Error Attribution

Val error package:

```text
reports\random_20w_split\stage2_loop24_blend_val_error_analysis
```

Val errors:

- total: `236`
- FP/FN: `118 / 118`
- white-list FP average score: `0.7967`
- blacklist FN average score: `0.2674`

Full-test diagnostic package:

```text
reports\random_20w_split\stage2_loop24_blend_full_test_error_analysis
```

Full-test errors:

- total: `2685`
- FP/FN: `1379 / 1306`
- white-list FP average score: `0.8380`
- blacklist FN average score: `0.2706`

Main full-test concentrations:

- FP source: `待加入白名单`
- FN source: `待拉黑`
- extension `<none>`: `1171` errors, mostly FP (`1136`)
- extension `.exe`: `1155` errors, mostly FN (`913`)
- extension `.dll`: `342` errors, almost all FN (`341`)
- FN month hotspots: `2026-03=159`, `2020-11=128`, `2021-09=83`, `2026-02=79`

## Decision

Do not treat the model path as solved.

The correct next phase is another data/noise loop, focused on:

1. Val FP review of white-list no-extension files.
2. Val FN review of black `.exe/.dll` files, especially very-low-score FN.
3. Feature audit for extensionless benign files and PE metadata that may be
   over-triggering malicious signatures.
4. Source/date family audit for FN hotspots.
5. Only after a new Val improvement should another Test-10k/full-test pass be
   allowed.
