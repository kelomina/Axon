# Phase 3 Loop138 Noise / Upper-bound Audit Report

更新时间：2026-07-05

## 目标

Loop136 是当前 strict best，但距离 full-test F1 `99.9%` 仍有明显差距。本轮不调阈值、不选择新模型，只对 Loop136 的 Val/full-test 错误做噪声与静态特征可分性审计，判断后续优化是否还应继续堆小规则，还是需要转向噪声建模与新证据。

身份字段约束不变：`source_path`、文件名、目录、后缀、`source_sha256`、`cache_path`、`sample_index`、split 与 row order 只用于对齐、cache lookup 和人工复核，不作为模型证据。

## 当前基线

| Split | Model | F1 | Errors | FP | FN |
|---|---|---:|---:|---:|---:|
| Val | Loop136 selector | `0.9910789933` | `179 / 20000` | `122` | `57` |
| Full-test | Loop136 selector | `0.9903723842` | `1544 / 160000` | `958` | `586` |

Full-test 的 `99.9%` F1 目标大致只允许约 `160` 个错误量级；Loop136 当前还有 `1544` 个错误，需要净减少约 `1384` 个错误才接近目标。

## 高置信错误

Loop136 full-test 错误不是普通的近阈值摇摆：

| Item | Count |
|---|---:|
| Total errors | `1544` |
| High-confidence FP (`>=0.90`) | `471` |
| High-confidence FN (`<0.10`) | `292` |
| Near-threshold errors (`0.45-0.55`) | `74` |

Val 也同向：`179` 个错误里，高置信 FP `62`、高置信 FN `31`，近阈值错误只有 `10`。因此全局阈值微调不是主解法。

## 邻域审计

邻域审计使用冻结 Stage2 kNN memory 作为数值特征空间参照，只看内容/PE/stat/byte summary 等数值证据。审计对象是 Loop136 的官方错误队列。

Full-test `1544` 个错误全部命中缓存：

| Neighbor bucket | Count | FP | FN | Interpretation |
|---|---:|---:|---:|---|
| `neighbors_support_model_prediction` | `784` | `554` | `230` | 邻居更支持模型当前预测，优先视为标签噪声、灰区样本或静态特征不可分 |
| `neighbors_mixed` | `541` | `275` | `266` | 邻域混杂，边界重叠，需要新特征或噪声建模 |
| `neighbors_support_dataset_label` | `219` | `129` | `90` | 邻居支持数据标签，更像可学习的模型缺口 |

Val `179` 个错误也一致：

| Neighbor bucket | Count | FP | FN |
|---|---:|---:|---:|
| `neighbors_support_model_prediction` | `86` | `65` | `21` |
| `neighbors_mixed` | `68` | `39` | `29` |
| `neighbors_support_dataset_label` | `25` | `18` | `7` |

Full-test 中 `opposite_label_ratio >= 0.8` 的错误有 `784` 个，`>= 0.9` 的有 `552` 个。这是强噪声/强混叠信号。

## 上限判断

如果只修复 `neighbors_support_dataset_label` 的 `219` 个更像模型缺口的错误，full-test F1 约只能到 `0.991736`。如果进一步把 `neighbors_mixed` 的 `541` 个混杂错误也全部修掉，只剩 `784` 个邻域支持模型预测的强冲突错误，full-test F1 约为 `0.995110`。

这不是严格 Bayes error 证明，但足以说明：在当前静态特征和标签口径下，`99.9%` F1 大概率不由小 selector、小阈值或 feature-mask 微调支撑。下一阶段必须引入更正交的证据，或先做标签/灰区样本治理。

## Decision

Loop136 保持当前 strict best；Loop138 不替换模型。

后续方向：

1. 停止把主要希望放在全局阈值和单规则 FN recovery 上。
2. 优先做 `support-aware / noise-aware` 候选：只用数值内容证据和邻域统计，不用身份字段。
3. 将 `neighbors_support_dataset_label` 作为最可学习错误池，先在 Val 上证明能减少这类错误且不增加 FN。
4. 若目标仍坚持 `99.9%`，需要新增行为/动态/签名证书/publisher 等更正交特征，或对强冲突样本做人工标签复核。

## Artifacts

- Full-test error audit：`reports/phase3_loop138/loop136_error_audit_summary.json`
- Full-test review queue：`reports/phase3_loop138/loop136_error_review_queue.csv`
- Full-test neighbor audit：`reports/phase3_loop138/loop136_full_errors_neighbor_audit.json`
- Val error audit：`reports/phase3_loop138/loop136_val_error_audit_summary.json`
- Val neighbor audit：`reports/phase3_loop138/loop136_val_errors_neighbor_audit.json`
- Pre-run guard：`reports/phase3_loop138/pre_run_guard_loop136_error_audit.json`
