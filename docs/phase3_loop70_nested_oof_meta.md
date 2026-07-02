# Phase 3 Loop70: Nested OOF Meta Layer

日期：2026-07-03

## 目标

Loop70 使用 Loop69 生成的完整 `20000` 行 train nested OOF CSV，训练第三层 meta layer，并在完整 Val 上验证。它要回答的问题是：如果第三层 residual learner 的训练输入已经合规，继续堆 score-level meta 是否能超过 Loop57？

结论：不能。

## 协议

- Train 输入来自 Loop69 nested OOF，每个 train 行的上一层分数都来自没见过该样本的外层 fold。
- Val 输入由上游 base/candidate/override 模型在全 train 上拟合后冻结生成。
- Val 只用于选择第三层 meta model 和阈值。
- 不触碰 Test-10k 或 full-test。
- `source_path`、`cache_path`、`source_sha256`、`sample_index`、`split` 只用于对齐和审计，不进入 meta 特征。

## 实现

新增：

- `scripts/train_loop70_nested_oof_meta.py`
- `tests/test_train_loop70_nested_oof_meta.py`

Meta 特征只包含 score/flag 类可部署信号：

- base/candidate/allow/final score
- score delta 与 logit delta
- score confidence
- previous prediction
- previous override flag
- possible override flag

不包含 fold id、路径、hash、文件名、目录、扩展名、split 或 `correct`。

## 真实命令

```powershell
.\vnev\Scripts\python.exe scripts\train_loop70_nested_oof_meta.py `
  --checkpoint models\random_20w_8192\best_model.pt `
  --train-predictions reports\random_20w_split\loop27_train_predictions.csv `
  --val-predictions reports\random_20w_split\loop27_val_predictions.csv `
  --train-oof-predictions reports\random_20w_split\loop69_nested_oof_override_full_train\loop69_nested_oof_override_train_predictions.csv `
  --output-dir reports\random_20w_split\loop70_nested_oof_meta_valonly `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --overlay-boundary-cache-dir reports\random_20w_split\loop55_overlay_boundary_cache_train_val `
  --drop-base-prob-features `
  --thresholds 0.35:0.65:0.005 `
  --allow-thresholds 0.05:0.99:0.005 `
  --meta-model-candidates meta_logreg_balanced_c0.1,meta_logreg_balanced_c1,meta_hgb_leaf7,meta_hgb_leaf15 `
  --reference-val-errors 147 `
  --min-val-error-improvement 10
```

## 结果

Records:

- train: `20000/20000`
- val: `20000/20000`

Previous layer metrics under Loop70 train-derived thresholds:

| Split | F1 | Errors | FP/FN |
| --- | ---: | ---: | ---: |
| Train OOF | `0.9877909005` | `245` | `156 / 89` |
| Val | `0.9907111466` | `186` | `105 / 81` |

Meta candidates:

| Candidate | Val F1 | Errors | FP/FN |
| --- | ---: | ---: | ---: |
| `meta_logreg_balanced_c0.1` | `0.9917173935` | `166` | `104 / 62` |
| `meta_hgb_leaf15` | `0.9917157401` | `166` | `102 / 64` |
| `meta_hgb_leaf7` | `0.9916209476` | `168` | `109 / 59` |
| `meta_logreg_balanced_c1` | `0.9914827913` | `171` | `124 / 47` |

Best candidate:

- `meta_logreg_balanced_c0.1`
- threshold: `0.46`
- Val errors: `166`
- Delta vs Loop57 reference: `+19` errors

Decision:

- `test_gate_decision=reject_val_margin_too_small`
- No Test-10k.

## 结论

Loop70 是一个干净的负面实验。Loop68/69 解决了协议问题，但当协议正确后，第三层 score-level meta 没有超过 Loop57，甚至退回到 Loop28/54 附近的错误量级。

因此不要继续沿同一组 base/candidate/allow/final 分数做 stack/gate/threshold 微调。下一步应转向：

- 新的独立内容证据；
- 或者高置信冲突的人工/外部证据噪声复核；
- 或者先物化真正不同视角的 OOF base，而不是继续重排同一路线分数。

## Artifacts

- Report: `reports/random_20w_split/loop70_nested_oof_meta_valonly/loop70_nested_oof_meta_report.json`
- Val predictions: `reports/random_20w_split/loop70_nested_oof_meta_valonly/loop70_nested_oof_meta_val_predictions.csv`
- Selected model: `reports/random_20w_split/loop70_nested_oof_meta_valonly/loop70_nested_oof_meta_selected_model.pkl`

Generated artifacts are not committed.

## 验证

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_train_loop70_nested_oof_meta.py tests\test_materialize_loop69_nested_oof_override.py tests\test_identity_feature_guard.py -q
.\vnev\Scripts\python.exe -m py_compile scripts\train_loop70_nested_oof_meta.py scripts\materialize_loop69_nested_oof_override.py
```
