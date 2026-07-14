# Phase 3 Loop135 R5/OOF Noise Pairwise Selector Report

更新时间：2026-07-05

## 目标

Loop134 证明 `Noise-aware OOF fixed-v2 + content string` 与 R5 有互补性：它在 Val 上修复了部分 R5 错误，但整体弱于 R5。本轮只在 Train/Val 上训练一个低风险 selector：默认保留 R5，只有 selector 判断候选更可信时才接管 OOF noise 的预测。

身份字段仍只用于对齐和缓存定位；模型输入只包含冻结预测概率/分歧方向，以及 content PE v1/v2/string 数值特征。

## 产物

- 训练脚本：`scripts/train_loop135_pairwise_selector.py`
- 冻结评估脚本：`scripts/evaluate_loop135_pairwise_selector.py`
- OOF stacker 冻结评估脚本：`scripts/evaluate_stage2_oof_stacker.py`
- Val 报告：`reports/phase3_loop135/r5_oof_noise_pairwise_selector_valonly/loop135_pairwise_selector_report.json`
- Test-10k 报告：`reports/phase3_loop135/r5_oof_noise_pairwise_selector_test10k_eval.json`
- OOF noise Test-10k 报告：`reports/phase3_loop135/oof_noise_test10k_eval.json`

## Val 结果

| Model | F1 | Errors | FP | FN |
|---|---:|---:|---:|---:|
| R5 baseline | 0.9907342832 | 186 | 130 | 56 |
| OOF noise candidate | 0.9905796740 | 189 | 126 | 63 |
| Loop135 selector | 0.9911733905 | 177 | 115 | 62 |

Val 上 selector 修复了 R5 的 FP，代价是 FN 上升。总错误 `186 -> 177`，达到进入 Test-10k 的条件。

## Test-10k 结果

| Model | F1 | Errors | FP | FN |
|---|---:|---:|---:|---:|
| R5 baseline | 0.9915983197 | 84 | 56 | 28 |
| OOF noise candidate | 0.9910991099 | 89 | 59 | 30 |
| Loop135 selector | 0.9913913914 | 86 | 53 | 33 |

Test-10k 上 selector 虽然继续减少 FP，但新增 FN 更多，整体从 R5 的 `84` errors 退化到 `86` errors。因此 Loop135 被拒绝，不进入 16 万 full-test。

## 结论

Loop135 证明 R5 和 OOF noise 的互补性是真实存在的，但当前 selector 学到的接管边界仍不稳定，尤其会把部分恶意样本从黑翻白。后续如果继续做 selector，必须把目标函数改成 recall-aware 或 harmful-FN-constrained，而不是只按总错误/F1 选择。

后续 Loop136 已按这个方向加入 Val FN 约束，并通过 Test-10k 与 full-test，成为新的 strict best。见 `docs/phase3_loop136_recall_pairwise_selector_full_test_report.md`。
