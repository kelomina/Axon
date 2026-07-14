# Phase 3 Loop69: Nested OOF Materialization

日期：2026-07-03

## 目标

Loop69 接在 Loop68 后面，解决一个协议缺口：Loop57/61/62 不能直接作为第三层 residual learner 的训练素材，因为它们没有落盘整条流水线的 train final OOF 行级输出。

本轮新增 Loop61-style override-only pipeline 的 nested OOF 物化脚本。它的作用不是提高分数，而是生成合格训练素材：每个 train 样本的上一层分数都必须来自没有见过该样本的外层 fold。

## 身份字段规则

输出 CSV 会保留 `source_path`、`cache_path`、`source_sha256`、`sample_index`、`split`，但这些列只用于对齐、缓存审计和复核，不允许进入模型特征。真正可用于后续第三层训练的是 OOF 分数、fold、阈值和内容特征。

脚本不导出 `correct` 列，因为它直接包含“预测是否命中标签”的答案信息，容易被后续训练误用。

## 实现

新增：

- `scripts/materialize_loop69_nested_oof_override.py`
- `tests/test_materialize_loop69_nested_oof_override.py`

核心输出列：

- `oof_fold`
- `base_oof_prob_malicious`
- `candidate_oof_prob_malicious`
- `allow_oof_prob`
- `final_oof_prob_malicious`
- `final_oof_prediction`
- `oof_override_flag`
- `possible_override_flag`
- `candidate_threshold`
- `allow_threshold`

## Smoke 验证

真实命令：

```powershell
.\vnev\Scripts\python.exe scripts\materialize_loop69_nested_oof_override.py `
  --checkpoint models\random_20w_8192\best_model.pt `
  --train-predictions reports\random_20w_split\loop27_train_predictions.csv `
  --output-dir reports\random_20w_split\loop69_nested_oof_override_smoke `
  --max-train-rows 400 `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --overlay-boundary-cache-dir reports\random_20w_split\loop55_overlay_boundary_cache_train_val `
  --drop-base-prob-features `
  --outer-folds 2 `
  --inner-folds 2 `
  --thresholds 0.45:0.55:0.05 `
  --allow-thresholds 0.30:0.90:0.20 `
  --base-model-candidate hgb_lr0.06_leaf31_l2_0 `
  --candidate-model-candidate extra_trees_300_leaf1 `
  --override-model-candidate override_logreg_balanced_c1
```

Smoke 结果：

- rows: `400`
- OOF final errors: `20`
- OOF final F1: `0.9497487437`
- override count: `5`

这个 smoke 分数不是模型结论，只说明 nested OOF pipeline 可以完整执行并落盘。

随后用 Loop68 readiness gate 复验 smoke 产物：

```powershell
.\vnev\Scripts\python.exe scripts\audit_loop68_residual_oof_readiness.py `
  --candidate "reports\random_20w_split\loop69_nested_oof_override_smoke\loop69_nested_oof_override_report.json" `
  --output-json reports\random_20w_split\loop69_nested_oof_override_smoke\loop68_readiness_on_loop69_smoke.json `
  --expected-train-rows 400 `
  --expected-val-rows 0
```

结果：

- `overall_decision=third_layer_residual_training_allowed`
- `ready_candidate_count=1`

## 决策

Loop69 smoke 通过。下一步可以运行完整 train `20000/20000` nested OOF 物化。完整输出仍不是 Test 候选，也不触碰 Val/Test；它只是为后续第三层 residual learner 提供合格的 train-only 输入。

## 完整 Train OOF 物化

真实命令：

```powershell
.\vnev\Scripts\python.exe scripts\materialize_loop69_nested_oof_override.py `
  --checkpoint models\random_20w_8192\best_model.pt `
  --train-predictions reports\random_20w_split\loop27_train_predictions.csv `
  --output-dir reports\random_20w_split\loop69_nested_oof_override_full_train `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --overlay-boundary-cache-dir reports\random_20w_split\loop55_overlay_boundary_cache_train_val `
  --drop-base-prob-features `
  --outer-folds 5 `
  --inner-folds 5 `
  --thresholds 0.35:0.65:0.005 `
  --allow-thresholds 0.05:0.99:0.005 `
  --base-model-candidate hgb_lr0.06_leaf31_l2_0 `
  --candidate-model-candidate extra_trees_300_leaf1 `
  --override-model-candidate override_logreg_balanced_c1
```

完整结果：

| Check | Result |
| --- | ---: |
| rows | `20000` |
| records kept | `20000/20000` |
| label counts | `10000 / 10000` |
| fold counts | `4000` each |
| missing scores | `0` |
| OOF F1 | `0.9874489491` |
| OOF errors | `252` |
| FP/FN | `165 / 87` |
| possible overrides | `120` |
| actual overrides | `36` |
| override label1/label0 | `18 / 18` |

Loop68 readiness 复验：

```powershell
.\vnev\Scripts\python.exe scripts\audit_loop68_residual_oof_readiness.py `
  --candidate "reports\random_20w_split\loop69_nested_oof_override_full_train\loop69_nested_oof_override_report.json" `
  --output-json reports\random_20w_split\loop69_nested_oof_override_full_train\loop68_readiness_on_loop69_full_train.json `
  --expected-train-rows 20000 `
  --expected-val-rows 0
```

结果：

- `overall_decision=third_layer_residual_training_allowed`
- `ready_candidate_count=1`

这说明完整 `20000` train OOF CSV 已经满足第三层 residual learner 的输入协议。注意：这仍然不是模型收益结论，因为没有跑 Val；它只是把“能合法训练下一层”的数据准备好了。

## Artifacts

- Smoke report: `reports/random_20w_split/loop69_nested_oof_override_smoke/loop69_nested_oof_override_report.json`
- Smoke OOF CSV: `reports/random_20w_split/loop69_nested_oof_override_smoke/loop69_nested_oof_override_train_predictions.csv`
- Smoke readiness report: `reports/random_20w_split/loop69_nested_oof_override_smoke/loop68_readiness_on_loop69_smoke.json`
- Full train report: `reports/random_20w_split/loop69_nested_oof_override_full_train/loop69_nested_oof_override_report.json`
- Full train OOF CSV: `reports/random_20w_split/loop69_nested_oof_override_full_train/loop69_nested_oof_override_train_predictions.csv`
- Full train readiness report: `reports/random_20w_split/loop69_nested_oof_override_full_train/loop68_readiness_on_loop69_full_train.json`

Generated reports are not committed.

## 验证

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_materialize_loop69_nested_oof_override.py tests\test_audit_loop68_residual_oof_readiness.py tests\test_identity_feature_guard.py -q
.\vnev\Scripts\python.exe -m py_compile scripts\materialize_loop69_nested_oof_override.py scripts\audit_loop68_residual_oof_readiness.py
```

Latest local result: `8 passed`.
