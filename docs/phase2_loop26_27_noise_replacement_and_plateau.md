# Phase 2 Loop 26/27: Automatic Noise Replacement and Plateau

## Scope

This report covers the post-Loop24 continuation:

1. Loop26 conservative automatic Val-noise replacement.
2. Strict 20w split/cache/duplicate audits.
3. Loop26 Val -> Test-10k -> full-test funnel.
4. Loop27 follow-up replacement attempt and plateau decision.

No Test-10k or full-test result was used to tune thresholds, blend weights, or
replacement policy. Full-test was used only after Loop26 passed Val and
Test-10k.

## Loop26 Data Action

Input split:

```text
reports\random_20w_split\stage2_corrected_best_val_auto_noise_dedup_corrected_split_loop24.csv
```

Evidence input:

```text
reports\random_20w_split\stage2_loop24_blend_val_p0_p1_neighbor_audit.csv
```

Automatic replacement plan:

```text
reports\random_20w_split\loop26_auto_noise_replacement_plan.csv
reports\random_20w_split\loop26_auto_noise_replacement_plan.json
```

Policy:

- support bucket: `neighbors_support_model_prediction`
- priority: `<= 1`
- model confidence: `>= 0.95`
- opposite-label neighbor ratio: `>= 0.80`
- relabeling: disabled
- test actions: disabled

Plan result:

- planned rows: `14`
- split: all `val`
- labels: `0=8`, `1=6`
- action: `exclude_and_replace`

Corrected split:

```text
reports\random_20w_split\loop26_corrected_split.csv
```

Strict audits:

| Audit | Result |
| --- | --- |
| Replacement integrity | PASS, `14` excluded rows removed, `14` fresh same-label replacements inserted |
| Shape and balance | PASS, `200000`, `20000/20000/160000`, per-split `1:1` labels |
| SHA duplicate audit | PASS, duplicate groups `0` |
| Cache before recovery | expected miss `14` |
| Cache recovery | extracted `14`, uncompressed, fixed-v2 hash `38672ba0` |
| Cache after recovery | PASS, `200000/200000`, missing `0` |

Key files:

```text
reports\random_20w_split\loop26_replacement_integrity.json
reports\random_20w_split\loop26_duplicate_source_audit.json
reports\random_20w_split\loop26_cache_recovery.json
reports\random_20w_split\loop26_cache_ready_after_recovery.json
```

## Loop26 Evaluation Funnel

Base Val:

- F1 @ 0.5: `0.9332994924`
- errors: `1314`
- FP/FN: `507 / 807`

Stage-2 extended selected by Val:

- model: `hgb_lr0.06_leaf31_l2_0`
- noise mode: `none`
- threshold: `0.60`
- Val F1: `0.9872519122`
- Val errors: `255`

Stage-2 kNN selected by Val:

- model: `hgb_lr0.06_leaf31_l2_0`
- noise mode: `knn_trim_exact_opposite`
- threshold: `0.555`
- Val F1: `0.9878372291`
- Val errors: `243`

Val-only blend:

- extended weight: `0.5`
- kNN weight: `0.5`
- threshold: `0.485`
- Val F1: `0.9888639201`
- Val errors: `223`
- FP/FN: `124 / 99`

This improved over Loop24 blend Val:

- Loop24 Val F1: `0.9882`
- Loop24 Val errors: `236`
- Loop26 delta: `-13` Val errors

## Loop26 Frozen Test-10k Confirmation

Test-10k used the unchanged Loop24/Loop26 test rows. The base neural-network
predictions were reused because train/test digests matched exactly; only Val
changed.

Frozen blend result:

- threshold: `0.485`
- rows: `10000`
- F1: `0.9855769231`
- AUC: `0.9986825789`
- errors: `144`
- FP/FN: `84 / 60`

This improved over Loop24 blend Test-10k:

- Loop24 Test-10k F1: `0.9842258615`
- Loop24 errors: `157`
- Loop26 delta: `-13` Test-10k errors

Decision: passed Test-10k confirmation and qualified for one full-test run.

## Loop26 Frozen Full-Test Result

Frozen blend:

- threshold: `0.485`
- rows: `160000`
- F1: `0.9839728205`
- AUC: `0.9984116707`
- errors: `2571`
- FP/FN: `1493 / 1078`

Comparison:

| Model | Full-test F1 | Errors | FP / FN |
| --- | ---: | ---: | ---: |
| Loop24 blend | `0.9832264030` | `2685` | `1379 / 1306` |
| Loop26 blend | `0.9839728205` | `2571` | `1493 / 1078` |

Loop26 reduced full-test errors by `114`. The trade-off shifted toward fewer
FN but more FP:

- FP increased by `114`
- FN decreased by `228`

## Loop26 Residual Error Diagnosis

Val residual errors:

- total: `223`
- FP/FN: `124 / 99`
- suspected noise/hard examples: `76`
- severe FP `>=0.99`: `6`
- severe FN `<=0.01`: `5`

Full-test residual errors:

- total: `2571`
- FP/FN: `1493 / 1078`
- suspected noise/hard examples: `1160`
- severe FP `>=0.99`: `204`
- severe FN `<=0.01`: `88`

Full-test concentration:

| Dimension | Error pattern |
| --- | --- |
| extension `<none>` | `1261` errors, `1234` FP |
| extension `.exe` | `987` errors, `729` FN |
| extension `.dll` | `306` errors, `305` FN |
| data dir `待加入白名单` | `1493` FP |
| data dir `待拉黑` | `1078` FN |
| month hotspots | `2026-03`, `2020-11`, `2021-09`, `2026-02` |

Interpretation: residual error is not mainly threshold calibration. The model
still has hard blind spots around extensionless benign files and malicious
DLL/sys/exe families.

## Loop27 Follow-Up

Loop26 Val P0/P1 neighbor audit:

```text
reports\random_20w_split\stage2_loop26_blend_val_p0_p1_neighbor_audit.csv
```

Selected rows:

- P0/P1 review rows: `65`
- support model prediction: `16`
- support dataset label: `5`
- mixed: `44`

The same strict automatic policy produced only `2` replacement rows:

- split: `val`
- labels: `0=1`, `1=1`

Loop27 corrected split:

```text
reports\random_20w_split\loop27_corrected_split.csv
```

Strict audits:

- replacement integrity: PASS
- SHA duplicate audit: PASS
- cache recovery: extracted `2`
- cache after recovery: `200000/200000`, missing `0`

Loop27 Val result:

| Candidate | Val F1 | Errors |
| --- | ---: | ---: |
| extended | `0.9873518972` | `253` |
| kNN | `0.9878878879` | `242` |
| blend | `0.9889133040` | `222` |

Loop27 blend improved over Loop26 blend by only `1` Val error. This is below
the practical improvement gate, so Loop27 did not enter Test-10k.

## Plateau Decision

Conservative Val-noise replacement is no longer the main lever:

- Loop26 gave a real full-test improvement: `-114` errors.
- Loop27 gave only `-1` Val error and no Test-10k entry.
- Remaining full-test errors are still `2571`, far above the rough `<=160`
error scale needed for `F1 >= 99.9%` on 160k balanced test rows.

Next work should shift from small Val-noise replacement to feature and model
coverage:

1. Add explicit extensionless/SHA-name benign features and high-value benign
   holdouts.
2. Add DLL/sys-focused PE features: exports, subsystem, service/driver hints,
   section permission combinations, TLS, import category granularity.
3. Move Stage-2 from one checkpoint to OOF stacking across multiple base
   checkpoints/seeds.
4. Keep noise replacement as a hygiene loop, not as the primary route to
   `99.9%`.
