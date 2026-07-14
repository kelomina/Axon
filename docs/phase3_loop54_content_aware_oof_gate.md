# Phase 3 Loop54: Content-Aware OOF Residual Gate

日期：2026-07-02

## 目标

Loop54 复测 Loop42 的严格 OOF residual gate，但让 gate 额外看到内容矩阵：PE/stat/lightweight/byte summary/content PE v1。Loop42 的 gate 只看 base/candidate 分数，曾达到 `160` Val errors，但相对 Loop28 locked reference 的 `162` errors 只有 2 个错误的薄 margin，不允许进入 Test-10k。

本轮问题是：gate 多看内容侧特征后，能否更可靠地区分“candidate 能修正 Loop28”与“candidate 会添乱”。

本轮仍是 Val-only，不传 Test-10k，也不跑 16 万 full-test。

## 身份字段规则

filename、path、extension、directory、`source_sha256`、`sample_index`、`split` 和行顺序只用于加载、对齐和审计，不作为模型特征或阈值捷径。`--alignment-key-column sample_index` 只用于确认 baseline Val prediction 对齐，不进入 gate feature。

## 命令

```powershell
.\vnev\Scripts\python.exe scripts\train_loop42_oof_residual_gate.py `
  --checkpoint models\random_20w_8192\best_model.pt `
  --train-predictions reports\random_20w_split\loop27_train_predictions.csv `
  --val-predictions reports\random_20w_split\loop27_val_predictions.csv `
  --output-dir reports\random_20w_split\loop54_oof_residual_gate_content_valonly `
  --content-pe-features `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --drop-base-prob-features `
  --base-model-candidate hgb_lr0.06_leaf31_l2_0 `
  --candidate-model-candidates hgb_lr0.08_leaf31_l2_1e-3,extra_trees_300_leaf1 `
  --include-byte-ngram `
  --byte-ngram-n-features 2097152 `
  --byte-ngram-prefix-len 4096 `
  --byte-ngram-min 2 `
  --byte-ngram-max 5 `
  --byte-ngram-stride 2 `
  --byte-ngram-alpha 3e-6 `
  --byte-ngram-epochs 3 `
  --byte-ngram-batch-size 256 `
  --byte-ngram-include-byte-hist `
  --byte-ngram-include-cache-features `
  --neutral-weight 0.05 `
  --thresholds 0.35:0.65:0.005 `
  --gate-thresholds 0.50:0.95:0.01 `
  --folds 5 `
  --seed 54 `
  --baseline-val-predictions reports\random_20w_split\stage2_loop28_content_pe_valonly\stage2_val_predictions.csv `
  --baseline-probability-column stage2_prob_malicious `
  --alignment-key-column sample_index `
  --gate-content-features
```

## 结果

Rows:

- train: `20000/20000`
- val: `20000/20000`
- cache misses: `0`
- gate content features: `true`

Best Val candidate:

| 字段 | 值 |
| --- | --- |
| candidate | `extra_trees_300_leaf1` |
| gate | `gate_logreg_balanced_c0.25` |
| base train threshold | `0.44` |
| candidate train threshold | `0.465` |
| gate threshold | `0.60` |
| Val F1 | `0.9917676994` |
| Val errors | `165` |
| FP / FN | `104 / 61` |
| overrides | `99` |

Comparison:

| Candidate | Val F1 | Errors | FP/FN | Decision |
| --- | ---: | ---: | ---: | --- |
| Loop28 locked reference | `0.9919048571` | `162` | `87 / 75` | Current best |
| Loop42 score-only gate | `0.9920143741` | `160` | `98 / 62` | Rejected, margin too thin |
| Loop54 content-aware gate | `0.9917676994` | `165` | `104 / 61` | Rejected |

The content-aware gate improved its internal base by `16` errors, but it is `3` errors worse than the locked Loop28 reference. It also remains far above the shallow gate/blend Test-10k entry requirement of about `<=152` Val errors.

## 决策

Reject for Test-10k.

This closes the immediate “just let the residual gate see the full content matrix” branch. The result suggests the gate can reduce some FN, but it pays too many FP and does not create a stable enough margin.

下一步应转向 Error-Agent 指出的内容结构缺口：security directory 与真实 overlay payload 的边界、DLL/export/exception/TLS 组合、以及 signed/overlay 恶意 FN 与复杂正常 FP 的结构区分。不要继续围绕同一批 score/gate 参数做薄 margin 调参。

## Artifacts

- Report:
  `reports/random_20w_split/loop54_oof_residual_gate_content_valonly/loop42_oof_residual_gate_report.json`
- Large generated artifacts not committed:
  `loop42_oof_residual_gate_selected_model.pkl`
  `loop42_oof_residual_gate_val_predictions.csv`
