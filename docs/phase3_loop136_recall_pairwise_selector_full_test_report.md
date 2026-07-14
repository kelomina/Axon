# Phase 3 Loop136 Recall-aware Pairwise Selector Full-test Report

更新时间：2026-07-05

## 目标

Loop135 的普通 pairwise selector 在 Val 上明显改善，但 Test-10k 上因 FN 增长被拒绝。Loop136 在同一 Train/Val 协议下加入 recall-aware 约束：候选在 Val 上的 FN 不能比 R5 多超过 `2` 个。该约束只用 Train/Val 选择，Test-10k 和 full-test 只做冻结确认。

## 漏斗结果

| Split | Model | F1 | Errors | FP | FN | Decision |
|---|---|---:|---:|---:|---:|---|
| Val | R5 baseline | 0.9907342832 | 186 | 130 | 56 | baseline |
| Val | Loop136 selector | 0.9910789933 | 179 | 122 | 57 | enter Test-10k |
| Test-10k | R5 baseline | 0.9915983197 | 84 | 56 | 28 | baseline |
| Test-10k | Loop136 selector | 0.9916958479 | 83 | 54 | 29 | enter full-test |
| Full-test | R5 baseline | 0.9902567651 | 1563 | 991 | 572 | previous best |
| Full-test | Loop136 selector | 0.9903723842 | 1544 | 958 | 586 | new strict best |

Loop136 full-test 相比 R5：总错误 `-19`，FP `-33`，FN `+14`。它仍然是偏向降低误报的策略，但在 Val 和 Test-10k 都通过了冻结确认，full-test 也净改善。

## 错误交换

Full-test overlap：`reports/phase3_loop136/loop136_full_overlap_vs_r5.json`

| Item | Count |
|---|---:|
| R5 errors | 1563 |
| Loop136 errors | 1544 |
| Shared errors | 1528 |
| Fixed R5 errors | 35 |
| New Loop136 errors | 16 |

这说明 Loop136 是小幅稳定改善，不是大规模换错。距离 99.9% 仍然很远，下一阶段仍需要噪声审计和更强特征，而不是继续堆微小 selector。

## Artifacts

- Val selector report：`reports/phase3_loop136/r5_oof_noise_pairwise_selector_recall_valonly/loop135_pairwise_selector_report.json`
- Test-10k eval：`reports/phase3_loop136/r5_oof_noise_pairwise_selector_recall_test10k_eval.json`
- Full-test eval：`reports/phase3_loop136/r5_oof_noise_pairwise_selector_recall_full_test_eval.json`
- Full-test predictions：`reports/phase3_loop136/r5_oof_noise_pairwise_selector_recall_full_test_predictions.csv`
- OOF noise full-test predictions：`reports/phase3_loop136/oof_noise_full_test_predictions.csv`

## Decision

Adopt Loop136 as the current strict best by full-test F1/errors. Business trade-off remains important: compared with R5 it reduces FP but increases FN. If recall is the dominant risk, R5 remains a safer fallback; if total error/F1 is the strict objective, Loop136 is now ahead.
