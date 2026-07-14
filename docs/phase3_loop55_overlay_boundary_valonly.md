# Phase 3 Loop55: Overlay / Security Boundary Val-Only Probe

日期：2026-07-02

## 目标

Loop55 针对 Error-Agent 指出的 signed/overlay 残差主题，测试一组窄内容特征：把 PE Security Directory 的证书 blob 与真实 overlay payload 拆开，避免把“签名证书尾部”和“附加 payload”混成同一个 `overlay_present` 信号。

本轮只跑 Train/Val，不触碰 Test-10k 或 16 万 full-test。

## 身份字段规则

filename、path、extension、directory、`source_sha256`、`sample_index`、`split` 和行顺序只用于打开文件、缓存对齐和审计，不作为模型特征或阈值捷径。

Loop55 特征只来自文件内容和 PE 结构：

- Security Directory 文件偏移与大小
- overlay 起点与大小
- overlay 扣除证书区后的真实 payload segments
- payload 熵、头尾熵差、payload 是否在证书后
- overlay/security 与最后 section 的间隙和触碰关系
- 最后 section 熵与 raw/virtual size 差异

注意：PE Security Directory 的 `VirtualAddress` 在 PE 规范中是文件偏移，不按普通 RVA 转换。

## 实现

新增：

- `scripts/train_loop55_overlay_boundary.py`
- `tests/test_loop55_overlay_boundary.py`

先加入 cache-only 模式并行构建 sidecar cache，避免训练阶段单进程现场提取过慢。测试覆盖：

- security directory 从 overlay 中扣除后，payload segment 位置正确。
- 同一内容在不同文件名下提取结果一致。
- feature names 通过 identity guard。
- cache builder 可写出 `.npz` sidecar。

## Cache

命令：

```powershell
.\vnev\Scripts\python.exe scripts\train_loop55_overlay_boundary.py `
  --checkpoint models\random_20w_8192\best_model.pt `
  --train-predictions reports\random_20w_split\loop27_train_predictions.csv `
  --val-predictions reports\random_20w_split\loop27_val_predictions.csv `
  --output-dir reports\random_20w_split\loop55_overlay_boundary_cache_build `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --overlay-boundary-cache-dir reports\random_20w_split\loop55_overlay_boundary_cache_train_val `
  --build-cache-only `
  --cache-workers 8 `
  --cache-report-json reports\random_20w_split\loop55_overlay_boundary_cache_train_val_report.json
```

结果：

| 字段 | 值 |
| --- | ---: |
| input rows | 40000 |
| unique rows | 40000 |
| feature dim | 32 |
| processed | 40000 |
| zero features | 43 |

`zero_features=43` 主要表示 PE 解析失败或没有可用结构。训练仍完整保留 `20000 train / 20000 val`。

## Val-Only

命令：

```powershell
.\vnev\Scripts\python.exe scripts\train_loop55_overlay_boundary.py `
  --checkpoint models\random_20w_8192\best_model.pt `
  --train-predictions reports\random_20w_split\loop27_train_predictions.csv `
  --val-predictions reports\random_20w_split\loop27_val_predictions.csv `
  --output-dir reports\random_20w_split\loop55_overlay_boundary_valonly `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --overlay-boundary-cache-dir reports\random_20w_split\loop55_overlay_boundary_cache_train_val `
  --model-candidates hgb_lr0.06_leaf31_l2_0,hgb_lr0.08_leaf31_l2_1e-3 `
  --noise-modes none,soft_conflict_downweight,trim_extreme_conflict `
  --thresholds 0.35:0.65:0.005 `
  --seed 55
```

Best Val candidate:

| 字段 | 值 |
| --- | --- |
| model | `hgb_lr0.06_leaf31_l2_0__noise_none` |
| threshold | `0.365` |
| Val F1 | `0.9913208300` |
| Val errors | `174` |
| FP / FN | `111 / 63` |
| delta vs Loop28 | `+12` errors |

Comparison:

| Candidate | Val F1 | Errors | FP/FN | Decision |
| --- | ---: | ---: | ---: | --- |
| Loop28 locked reference | `0.9919048571` | `162` | `87 / 75` | Current best |
| Loop55 overlay boundary | `0.9913208300` | `174` | `111 / 63` | Rejected |

Loop55 reduced FN compared with Loop28 (`75 -> 63`) but increased FP too much (`87 -> 111`), so the total error count worsened.

## 决策

Reject for Test-10k.

The overlay/security boundary signal is content-valid, but direct concatenation into Stage-2 HGB is not enough. It behaves like an FP/FN tradeoff rather than a net improvement. The feature may still be useful for residual stratification or a very conservative FP/FN-specific gate, but it should not be advanced as a standalone candidate.

## 验证

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\train_loop55_overlay_boundary.py
.\vnev\Scripts\python.exe -m pytest tests\test_loop55_overlay_boundary.py tests\test_identity_feature_guard.py -q
```

Result: `8 passed`.

## Artifacts

- Cache report:
  `reports/random_20w_split/loop55_overlay_boundary_cache_train_val_report.json`
- Val report:
  `reports/random_20w_split/loop55_overlay_boundary_valonly/loop55_overlay_boundary_report.json`
- Large generated artifacts not committed:
  `loop55_overlay_boundary_selected_model.pkl`
  `loop55_overlay_boundary_val_predictions.csv`
