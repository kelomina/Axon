# Phase 3 Loop140 Loop136/OOF Direction-aware Selector Report

更新时间：2026-07-05

## 目标

Loop139 在 R5 与 OOF-noise 之间做方向阈值，Val 改善但 Test-10k 失败。Loop140 换一个更窄的接管关系：以当前 strict best Loop136 为 baseline，只在 Loop136 与 OOF-noise 仍然分歧的样本上训练第二层 direction-aware selector。目标是恢复 Loop136 的部分 FN，同时不牺牲 Test-10k 总错误。

本实验仍只用 Train/Val 选择模型和阈值；Test-10k 只做冻结确认，不扫阈值、不反向调参。身份字段只用于对齐与缓存定位，不作为模型证据。

## Val 结果

| Model | F1 | Errors | FP | FN |
|---|---:|---:|---:|---:|
| Loop136 baseline | `0.9910789933` | `179` | `122` | `57` |
| OOF noise candidate | `0.9905796740` | `189` | `126` | `63` |
| Loop140 selector | `0.9913831748` | `173` | `125` | `48` |

Val 选择：

- model：`selector_extra_trees_leaf2`
- `threshold_0to1 = 0.24`
- `threshold_1to0 = 0.84`
- train disagreements：`94`
- val disagreements：`46`
- val accepted candidate rows：`12`

Val 上 Loop140 相比 Loop136 减少 `6` 个错误，FN 从 `57` 降到 `48`，因此进入 Test-10k。

## Test-10k 结果

| Model | F1 | Errors | FP | FN | Decision |
|---|---:|---:|---:|---:|---|
| Loop136 baseline | `0.9916958479` | `83` | `54` | `29` | current strict best gate |
| OOF noise candidate | `0.9910991099` | `89` | `59` | `30` | diagnostic only |
| Loop140 selector | `0.9911026692` | `89` | `61` | `28` | reject |

Loop140 在 Test-10k 上虽然少了 `1` 个 FN，但 FP 从 Loop136 的 `54` 增加到 `61`，总错误从 `83` 退化到 `89`。因此拒绝，不进入 full-test。

## Decision

拒绝 Loop140。Loop136 仍是当前 strict best。

这进一步确认：在同一 R5/OOF-noise 分歧空间里，Val 上看起来能恢复 FN 的选择边界不稳定，到了 Test-10k 会明显外溢成 FP。下一步应转向 Loop138 建议的噪声建模、新特征或 support-aware 证据，而不是继续围绕同一个小分歧集扫阈值。

## Artifacts

- Pre-run guard：`reports/phase3_loop140/pre_run_guard_loop136_vs_oof_directional_valonly.json`
- Val report：`reports/phase3_loop140/loop136_vs_oof_noise_directional_valonly/loop135_pairwise_selector_report.json`
- Val predictions：`reports/phase3_loop140/loop136_vs_oof_noise_directional_valonly/loop135_pairwise_selector_val_predictions.csv`
- Test-10k guard：`reports/phase3_loop140/pre_run_guard_loop136_vs_oof_directional_test10k.json`
- Test-10k eval：`reports/phase3_loop140/loop136_vs_oof_noise_directional_test10k_eval.json`
- Test-10k predictions：`reports/phase3_loop140/loop136_vs_oof_noise_directional_test10k_predictions.csv`
