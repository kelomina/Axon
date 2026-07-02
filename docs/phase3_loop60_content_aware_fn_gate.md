# Phase 3 Loop60: Content-Aware FN Gate

日期：2026-07-03

## 目标

Loop60 针对 Loop59 的结论，把 Loop57 的 FN-specific overlay gate 扩展为 content-aware gate。区别是：Loop57 gate 只看 score + overlay boundary；Loop60 gate 额外看到去身份化后的 Stage-2 content/cache 矩阵，希望学会区分“值得修复的 FN”与“会误伤的正常软件”。

本轮只跑 Train/Val，不触碰 Test-10k 或 full-test。

## 实现

改动：

- `scripts/train_loop57_fn_overlay_gate.py` 新增 `--gate-content-features`
- `scripts/evaluate_loop57_fn_overlay_gate.py` 支持读取 payload 中的 `include_gate_content_features`
- `tests/test_loop57_fn_overlay_gate.py` 增加 gate content feature alias 测试

默认行为保持 Loop57 兼容：不传 `--gate-content-features` 时，gate 仍只看 score + overlay boundary。开启后，content 特征会用 `gate_content_feature_N` 这类匿名 alias 加入，并通过 `identity_feature_guard` 检查，避免 filename/path/hash/split/row order 泄漏。

## Protocol

命令核心参数：

```powershell
.\vnev\Scripts\python.exe scripts\train_loop57_fn_overlay_gate.py `
  --checkpoint models\random_20w_8192\best_model.pt `
  --train-predictions reports\random_20w_split\loop27_train_predictions.csv `
  --val-predictions reports\random_20w_split\loop27_val_predictions.csv `
  --baseline-val-predictions reports\random_20w_split\stage2_loop28_content_pe_valonly\stage2_val_predictions.csv `
  --output-dir reports\random_20w_split\loop60_content_aware_fn_gate_valonly `
  --content-pe-cache-dir reports\random_20w_split\content_pe_cache_v1 `
  --overlay-boundary-cache-dir reports\random_20w_split\loop55_overlay_boundary_cache_train_val `
  --drop-base-prob-features `
  --gate-content-features `
  --base-model-candidate hgb_lr0.06_leaf31_l2_0 `
  --candidate-model-candidates hgb_lr0.06_leaf31_l2_0,hgb_lr0.08_leaf31_l2_1e-3,extra_trees_300_leaf1 `
  --gate-model-candidates gate_logreg_balanced_c0.25,gate_logreg_balanced_c1,gate_hgb_leaf7 `
  --neutral-weight 0.02 `
  --thresholds 0.35:0.65:0.005 `
  --gate-thresholds 0.50:0.99:0.005 `
  --folds 5 `
  --seed 60
```

## Val Results

Loop57 reference:

| Candidate | F1 | Errors | FP/FN | Overrides |
| --- | ---: | ---: | ---: | ---: |
| Loop57 FN gate | `0.9926635724` | `147` | `92 / 55` | `25` |

Loop60 best:

| Candidate | Gate | F1 | Errors | FP/FN | Overrides |
| --- | --- | ---: | ---: | ---: | ---: |
| `extra_trees_300_leaf1` | `gate_logreg_balanced_c0.25` | `0.9926657686` | `147` | `95 / 52` | `31` |

Loop60 与 Loop57 错误数相同，但 FP 更多、FN 更少。它是更偏低漏报的取舍，不是整体新 best。

## 决策

Reject for Test-10k.

原因：

- Val errors 没有低于 Loop57 的 `147`；
- FP 从 `92` 增到 `95`，正好违背 Loop59 后要削减新增 FP 的目标；
- 该候选没有形成足够新信息，不应消耗 Test-10k 次数。

下一步如果继续二级 guard，应减少泛化 content 矩阵直接喂给线性 gate 的做法，改为更明确地建模“正常软件结构”信号，例如 import/resource/section 的 benign-like summary，或单独训练 override-only classifier。

## 验证

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\train_loop57_fn_overlay_gate.py scripts\evaluate_loop57_fn_overlay_gate.py
.\vnev\Scripts\python.exe -m pytest tests\test_loop57_fn_overlay_gate.py tests\test_loop42_oof_residual_gate.py tests\test_loop55_overlay_boundary.py tests\test_identity_feature_guard.py -q
```

Result: `20 passed`.

