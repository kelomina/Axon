# Phase 3 Loop68: Residual OOF Readiness Audit

日期：2026-07-03

## 目标

Loop68 回答一个很具体的问题：在 Loop57、Loop61、Loop62 之后，是否可以继续训练第三层 residual learner？

结论先放前面：现在不可以。

不是因为这些候选本身不合规，而是因为第三层学习需要更强证据。上一层模型如果已经在训练集上训练过 gate/override，再让第三层看同一批 train 行的最终预测，就像拿已经批改过的卷子当模拟考试分数，会把噪声和拟合痕迹学进去。安全条件是：必须有整条上一层流水线的逐行 train OOF final prediction，也就是每个训练样本的 final score 都来自没有见过它的完整流水线。

## 身份字段规则

`source_path`、`cache_path`、`source_sha256`、`sample_index`、`split` 只允许用于加载、对齐、审计和复核。它们不能作为第三层模型特征，也不能作为自动改标或阈值捷径。

Loop68 同时检查报告和模型 payload 中的 feature names。真实审计里，Loop57、Loop61、Loop62 都没有发现身份字段泄漏。

## 实现

新增：

- `scripts/audit_loop68_residual_oof_readiness.py`
- `tests/test_audit_loop68_residual_oof_readiness.py`

真实命令：

```powershell
.\vnev\Scripts\python.exe scripts\audit_loop68_residual_oof_readiness.py `
  --candidate "reports\random_20w_split\loop57_fn_overlay_gate_valonly\loop57_fn_overlay_gate_report.json::reports\random_20w_split\loop57_fn_overlay_gate_valonly\loop57_fn_overlay_gate_selected_model.pkl" `
  --candidate "reports\random_20w_split\loop61_override_classifier_valonly\loop61_override_classifier_report.json::reports\random_20w_split\loop61_override_classifier_valonly\loop61_override_classifier_selected_model.pkl" `
  --candidate "reports\random_20w_split\loop62_override_content_classifier_valonly\loop61_override_classifier_report.json::reports\random_20w_split\loop62_override_content_classifier_valonly\loop61_override_classifier_selected_model.pkl" `
  --output-json reports\random_20w_split\loop68_residual_oof_readiness_audit.json
```

## 结果

| Candidate | Val F1 | Val errors | FP/FN | Payload identity violations | Train final OOF artifact | Readiness |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Loop57 FN overlay gate | `0.9926635724` | `147` | `92 / 55` | `0` | missing | blocked |
| Loop61 override-only classifier | `0.9930139721` | `140` | `90 / 50` | `0` | missing | blocked |
| Loop62 content override classifier | `0.9926096075` | `148` | `87 / 61` | `0` | missing | blocked |

审计输出：

- `candidate_count=3`
- `ready_candidate_count=0`
- `overall_decision=third_layer_residual_training_blocked`
- 三个候选共同缺口：`missing_row_level_train_final_whole_pipeline_oof_predictions`

## 决策

阻断第三层 residual training。

这个阻断是协议保护，不是路线放弃。Loop61 虽然 Val 从 Loop57 的 `147` errors 降到 `140` errors，但冻结 Test-10k 与 Loop57 同为 `102` errors，只是 FP/FN 交换不同。这说明当前小幅 Val 收益已经容易反转；如果再叠第三层而没有 nested OOF，只会更容易把噪声当规律。

## 下一步

若继续做模型侧残差学习，先实现 nested OOF export：

- 输出 train `20000/20000` 行；
- 每行包含 `label`、`oof_fold`、`base_oof_prob_malicious`、`candidate_oof_prob_malicious`、`gate_oof_prob_override` 或 `allow_oof_prob`、`final_oof_prob_malicious`、`final_oof_prediction`、`oof_override_flag`；
- 可包含 `source_path`、`source_sha256`、`sample_index` 等对齐列，但这些列必须保持 alignment-only；
- 每行 final prediction 必须来自没见过该样本的完整 Loop57/61 风格流水线。

如果不做 nested OOF，本阶段更稳的方向是继续人工/外部证据复核高置信冲突，或引入真正独立的新内容证据，而不是继续在同一批 score/overlay gate 上堆层。

## Artifacts

- Summary: `reports/random_20w_split/loop68_residual_oof_readiness_audit.json`

Generated reports are not committed.

## 验证

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_audit_loop68_residual_oof_readiness.py tests\test_identity_feature_guard.py -q
.\vnev\Scripts\python.exe -m py_compile scripts\audit_loop68_residual_oof_readiness.py scripts\identity_feature_guard.py
```

Latest local result: `6 passed`.
