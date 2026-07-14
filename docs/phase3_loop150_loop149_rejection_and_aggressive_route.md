# Phase 3 Loop150 Loop149 Rejection and Aggressive Route

更新时间：2026-07-07

## 目标

本轮回应“测试成绩进展不佳时要激进”的要求：不再温吞地继续加码 32768-byte 神经网络训练，而是把 Loop149 aggressive long-context 候选做严格 Val 复验、做一次轻量互补性证伪，然后把路线切到更高杠杆的噪声治理和新证据方向。

本轮仍遵守身份字段禁令：`source_path`、文件名、目录、后缀、hash、`source_sha256`、`sample_index`、split、row order 只允许用于 cache lookup、对齐和审计，不能作为模型证据、verdict 证据、feature mask、阈值或融合输入。

## 当前 strict best

Loop136 仍是当前 strict best：

| Split | F1 | Errors | FP | FN |
|---|---:|---:|---:|---:|
| Val | `0.9910789933` | `179` | `122` | `57` |
| Test-10k | `0.9916958479` | `83` | `54` | `29` |
| Full-test | `0.9903723842` | `1544` | `958` | `586` |

因此任何新候选进入 Test-10k 前，Val 至少需要达到：`errors <= 169`、`FP <= 122`、`FN <= 57`，并且 F1 明确高于 Loop136。

## Loop149 strict Val 复验

Loop149 使用 `32768` byte、`active_mean_detached`、FP32、`batch_size=4`、`learning_rate=5e-5`、`epochs=2`。模型产物：

- `models/loop149_active_mean_detached_32768_lr5e5_ep2/best_model.pt`
- `reports/logs/loop149_active_mean_detached_32768_lr5e5_ep2/training_history.json`

训练日志最佳 Val 仍很弱：

| Epoch | Val F1 | Errors | FP | FN |
|---:|---:|---:|---:|---:|
| 1 | `0.9216616920` | `1601` | `1019` | `582` |
| 2 | `0.9231154771` | `1534` | `743` | `791` |

严格 cache Val sweep 结果：

| Threshold | F1 | Errors | FP | FN |
|---:|---:|---:|---:|---:|
| `0.45` | `0.9243755897` | `1523` | `831` | `692` |
| `0.50` | `0.9231154771` | `1534` | `743` | `791` |

结论：Loop149 距离 Loop136 的 `179` Val errors 太远，不允许进入 Test-10k。

主要 artifact：

- `reports/phase3_loop149/loop149_32768_lr5e5_ep2_strict_val_sweep.json`
- `reports/phase3_loop149/loop149_32768_lr5e5_ep2_strict_val_predictions.csv`
- `reports/phase3_loop149/loop149_32768_lr5e5_ep2_strict_train_t050.json`
- `reports/phase3_loop149/loop149_32768_lr5e5_ep2_strict_train_t050_predictions.csv`

## 互补性证伪

为了避免错杀一个“整体弱但能修少量 Loop136 错误”的候选，本轮又做了 Loop136 vs Loop149 的 Val-only 互补性检查。

直接对齐结果显示：

| Item | Count |
|---|---:|
| Both right | `18407` |
| Both wrong | `120` |
| Loop136 wrong, Loop149 right | `59` |
| Loop136 right, Loop149 wrong | `1414` |
| Loop136 wrong total | `179` |
| Loop149 wrong total | `1534` |

也就是说 Loop149 修复 `59` 个 Loop136 错误时，会新增 `1414` 个 Loop136 原本正确的错误；这不是可学习互补池，而是整体退化。

随后尝试用 `scripts/train_loop135_pairwise_selector.py` 做 Train-only fit / Val-only selection：

1. 带 string sidecar 的版本被阻断，因为 `reports/random_20w_split/content_string_cache_v1` 对当前 Val 存在缺失 sidecar 行。该失败不改变模型结论，只说明这条 sidecar 输入不能直接用于本次 selector。
2. 去掉 string sidecar 后，严格 gate 版本没有任何 selector candidate 被选中。
3. 放宽到 `min_val_error_reduction=1`、`min_val_f1_delta=0` 后，仍然没有任何 selector candidate 被选中。

结论：Loop149 不值得再训练、不值得做 Test-10k、不值得围绕它继续调 selector 或 calibrator。

## 激进路线切换

继续调 32768 LR、epoch、阈值或小 selector 已经不是激进，而是低收益消耗。真正激进的下一步是把工作切到两个方向：

1. 噪声治理放大：Loop145 原本只选 Top 300，本轮已经扩展为全量 `neighbors_support_model_prediction` 高冲突包，共 `784` 行，其中 FP/FN 为 `554/230`。这批样本代表当前静态特征空间里最强的标签噪声、灰区或不可分信号。
2. Val-only 噪声治理入口：本轮同时输出 Loop136 Val 的高冲突 `86` 行，FP/FN 为 `65/21`。这批更适合后续训练前治理，因为它不会把 final full-test 错误身份当成模型选择信号。

新增 focus artifact：

- `reports/phase3_loop150/loop150_loop136_full_noise_focus_all784_blinded.csv`
- `reports/phase3_loop150/loop150_loop136_full_noise_focus_all784_private_map.csv`
- `reports/phase3_loop150/loop150_loop136_full_noise_focus_all784_summary.json`
- `reports/phase3_loop150/loop150_loop136_val_noise_focus_all86_blinded.csv`
- `reports/phase3_loop150/loop150_loop136_val_noise_focus_all86_private_map.csv`
- `reports/phase3_loop150/loop150_loop136_val_noise_focus_all86_summary.json`

这些 focus 文件不是 verdict，不能自动改标签、不能自动替换、不能作为模型特征。只有独立内容证据或外部证据确认 `label_wrong`、`feature_broken`、`out_of_scope` 后，才允许 quarantine，并从 locked manifest 的同原始标签池 fresh redraw。最终 split 必须仍然是 `200000`，一个都不能少。

## Review Gate 接入

为了让 Loop150 噪声治理不是停在 CSV，本轮把 Val/full 高冲突包接入现有 Loop126 review template 和 annotation preflight。

新增模板：

- `reports/phase3_loop150/loop150_loop136_val_noise_focus_all86_annotations_template.csv`
- `reports/phase3_loop150/loop150_loop136_val_noise_focus_all86_template.json`
- `reports/phase3_loop150/loop150_loop136_full_noise_focus_all784_annotations_template.csv`
- `reports/phase3_loop150/loop150_loop136_full_noise_focus_all784_template.json`

No-op 预检：

| Package | Rows | Annotated | Blockers | Ready for private mapping |
|---|---:|---:|---:|---|
| Val high-conflict | `86` | `0` | `0` | `false` |
| Full high-conflict | `784` | `0` | `0` | `false` |

这里 `ready_for_private_mapping=false` 是正确状态：模板还没有独立 verdict，因此不能 unblind、不能 redraw、不能训练、不能 Test-10k。结构预检的价值是确认模板字段安全、行数正确、没有身份/模型字段泄漏；后续只需要填 `manual_label_verdict`、`manual_verdict_note`、`recommended_action`，再重跑 preflight。

对应 artifact：

- `reports/phase3_loop150/loop150_loop136_val_noise_focus_all86_preflight_noop.csv`
- `reports/phase3_loop150/loop150_loop136_val_noise_focus_all86_preflight_noop.json`
- `reports/phase3_loop150/loop150_loop136_full_noise_focus_all784_preflight_noop.csv`
- `reports/phase3_loop150/loop150_loop136_full_noise_focus_all784_preflight_noop.json`

## 数据合同

当前 fixed-v2 数据合同仍有效：

- strict split：`reports/random_20w_split/loop127_full_duplicate_corrected_split.csv`
- 8192 fixed-v2 cache：`data/.cache/manifest_38672ba0.json`
- 32768 fixed-v2 cache：`data/.cache_loop145_fixedv2_32768_uncompressed/manifest_2205ded7.json`

已知状态：

- split 正好 `200000` 行。
- train/val/test 为 `20000/20000/160000`。
- 每个 split 黑白平衡。
- 当前 8192 与 32768 fixed-v2 cache 对 strict split 均为 `200000/200000` 覆盖、missing `0`、metadata failure `0`。
- 8192 主 manifest 是 split 超集，不能把 manifest 总条数误读成训练样本数；必须以 locked split 行数为准。

## Decision

Loop149 拒绝，Loop136 继续保持 strict best。

短期不再投入 32768 神经网络训练，除非引入真正新证据源。下一轮应优先推进：

1. 对 `loop150_loop136_val_noise_focus_all86_blinded.csv` 做独立内容/外部证据复核。
2. 若确认 bad rows，执行 same-original-label fresh redraw，并重跑 20w split/cache readiness。
3. 引入正交证据源作为新模型分支，例如真实 Authenticode trust chain、publisher/reputation、动态行为或 sandbox 行为摘要。没有这些新证据，不应把 `F1 >= 99.9%` 当作下一轮工程 gate。

## Verification

已运行：

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_build_loop145_loop136_blinded_noise_focus.py tests\test_evaluate_strict_split_from_cache.py tests\test_loop135_pairwise_selector.py -q
```

结果：`22 passed`。

补充运行：

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_loop126_review_package.py tests\test_build_loop145_loop136_blinded_noise_focus.py tests\test_evaluate_strict_split_from_cache.py tests\test_loop135_pairwise_selector.py -q
```

结果：`29 passed`。

Pre-run guard：

- `reports/phase3_loop149/pre_run_guard_loop149_strict_eval.json`
- `reports/phase3_loop150/pre_run_guard_loop150_loop136_vs_loop149_pairwise.json`
- `reports/phase3_loop150/pre_run_guard_loop150_expand_loop136_noise_focus784.json`
- `reports/phase3_loop150/pre_run_guard_loop150_review_templates.json`
