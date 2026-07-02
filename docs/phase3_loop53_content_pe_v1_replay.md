# Phase 3 Loop53: Content PE v1 Replay

日期：2026-07-02

## 目标

Loop53 复验 Loop52 的产品化改动是否保持 Loop28 语义不变。它不是新冲榜候选，不做 Test-10k，也不做 full-test。

关键问题只有两个：

- 已有 `content_pe_cache_v1` 是否在新审计口径下仍是完整、有效的 Train/Val sidecar cache。
- `scripts/train_stage2_cache_matrix.py` 改为引用 `src/kvd_features/content_pe_v1.py` 后，Loop28 Val-only 最佳点是否可复现。

## 身份字段规则

本轮继续执行硬规则：filename、path、extension、directory、`source_sha256`、`cache_path`、`sample_index`、`split` 和行顺序只允许用于加载、缓存对齐、审计、去重和人工复核，不能作为模型特征或阈值捷径。

`content_pe_v1` 中的 `content_dir_*` 表示 PE Data Directory，不是文件系统目录。

## Cache Audit

命令：

```powershell
.\vnev\Scripts\python.exe scripts\build_content_pe_feature_cache.py `
  --predictions reports\random_20w_split\loop27_train_predictions.csv reports\random_20w_split\loop27_val_predictions.csv `
  --cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --workers 8 `
  --output-json reports\random_20w_split\loop53_content_pe_v1_train_val_cache_audit.json
```

结果：

| 字段 | 值 |
| --- | ---: |
| input rows | 40000 |
| deduplicated rows before limit | 40000 |
| limit | null |
| smoke | false |
| unique rows | 40000 |
| feature dim | 100 |
| exists | 40000 |
| created | 0 |
| refreshed invalid | 0 |
| zero features | 0 |

这次审计会打开已有 `.npz` 并验证 `features` shape 是 100 维且数值有限。结果说明 Train/Val sidecar cache 完整可用，没有截断、坏缓存或零向量问题。

## Val-Only Replay

命令：

```powershell
.\vnev\Scripts\python.exe scripts\train_stage2_cache_matrix.py `
  --checkpoint models\random_20w_8192\best_model.pt `
  --train-predictions reports\random_20w_split\loop27_train_predictions.csv `
  --val-predictions reports\random_20w_split\loop27_val_predictions.csv `
  --output-dir reports\random_20w_split\loop53_content_pe_v1_productized_replay_valonly `
  --feature-set extended `
  --content-pe-features `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --thresholds 0.05:0.95:0.005 `
  --noise-modes none,soft_conflict_downweight,trim_extreme_conflict `
  --test-val-f1-gate 0.989
```

结果复现 Loop28：

| 指标 | 值 |
| --- | ---: |
| selected model | `hgb_lr0.06_leaf31_l2_0__noise_none` |
| feature dim | 1520 |
| train rows | 20000 |
| val rows | 20000 |
| threshold | 0.50 |
| Val F1 | 0.9919048571 |
| Val errors | 162 |
| FP / FN | 87 / 75 |
| test | null |

## 决策

Loop53 通过产品化回归复验。`content_pe_v1` 模块化后仍复现 Loop28 的 Val-only 最佳点，没有新增身份特征，也没有触碰 Test-10k。

后续新候选必须在此稳定 schema 上继续走完整 Train/Val 漏斗。除非 Val 明显超过 Loop28 参考门槛，否则不进入 Test-10k。
