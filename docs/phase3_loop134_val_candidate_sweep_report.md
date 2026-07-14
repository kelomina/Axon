# Phase 3 Loop134 Val-first 候选筛选报告

更新时间：2026-07-05

## 目标

Loop133 审计确认 R5 后仍有大量高置信错误，所以本轮回到 Train/Val 做候选验证，不使用 full-test 调参。所有候选都遵守：路径、文件名、目录、后缀、hash、sample index 和 row order 只用于加载、对齐、缓存定位和审计，不作为模型证据。

当前门槛是 Loop130 R5 的 Val：

| Candidate | Val F1 | Errors | FP | FN | 决策 |
|---|---:|---:|---:|---:|---|
| Loop130 R5 baseline | 0.9907342832 | 186 | 130 | 56 | current strict best |

## 候选结果

| Candidate | Val F1 | Errors | FP | FN | 结论 |
|---|---:|---:|---:|---:|---|
| Noise-aware OOF fixed-v2 + content string | 0.9905796740 | 189 | 126 | 63 | 低于 R5，拒绝 |
| Train-only kNN local support + content string | 0.9892086331 | 216 | 116 | 100 | 低于 R5，拒绝 |
| Overlay/security boundary specialist | 0.9882670128 | 235 | 132 | 103 | 低于 R5，拒绝 |

本轮没有任何候选满足“Val 明显优于 R5”的进入条件，因此全部不进入 Test-10k，更不进入 full-test。

## 主要证据

- OOF noise report：`reports/phase3_loop134/oof_fixed_v2_string_noise_valonly/stage2_oof_stacker_report.json`
- OOF noise predictions：`reports/phase3_loop134/oof_fixed_v2_string_noise_valonly/stage2_oof_stacker_val_predictions.csv`
- kNN report：`reports/phase3_loop134/stage2_knn_content_string_valonly/stage2_cache_matrix_report.json`
- kNN predictions：`reports/phase3_loop134/stage2_knn_content_string_valonly/stage2_val_predictions.csv`
- Overlay report：`reports/phase3_loop134/overlay_boundary_valonly/loop55_overlay_boundary_report.json`
- Overlay predictions：`reports/phase3_loop134/overlay_boundary_valonly/loop55_overlay_boundary_val_predictions.csv`
- Error overlap：`reports/phase3_loop134/val_overlap_vs_r5_all_candidates.json`

## 解释

OOF noise 是最接近 R5 的候选。它相比 R5 少了 `4` 个 FP，但多了 `7` 个 FN，净增 `3` 个错误。它修复了 `26` 个 R5 Val 错误，同时新增 `29` 个错误，说明它有互补性，但直接替换 R5 不成立。

kNN local support 和 overlay specialist 都显著降低 FP，但 FN 代价过高。kNN 相比 R5 修复 `47` 个错误但新增 `77` 个错误；overlay 相比 R5 修复 `28` 个错误但新增 `77` 个错误。这个现象和 Loop133 的结论一致：当前特征空间确实有可分片段，但用整体替换模型会把漏报放大。

## 下一步

1. 不再把 kNN/overlay 作为整体替换模型推进。
2. 把 OOF noise 的互补样本作为 Train/Val-only 研究对象，设计“只在低风险区域接管 R5”的 selector，但 selector 必须在 Train/Val 上预注册和验证。
3. 继续做 R5 harmful flip 与高置信错误的数值邻域/人工证据审计，先判断是标签噪声、静态不可分、模型盲区还是特征坏/缺失。
