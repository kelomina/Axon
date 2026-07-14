# Phase 3 Loop142 Cert-OOF Negative Report

更新时间：2026-07-05

## 目标

本轮按用户纠偏回到模型改进主线：内存/资源 guard 只作为训练前安全闸，不把任务转成内存泄漏专项。

Loop142 测试一个新的证据方向：在 fixed-v2/content PE 已有特征之外，加入 PE Security Directory / Authenticode 证书 blob 的内容特征，验证它能否改善当前 strict best Loop136 的 Val 表现。该方向不使用 `source_path`、文件名、目录、后缀、hash、`source_sha256`、`sample_index`、split 或 row order 作为模型证据；这些字段只用于对齐、缓存定位和审计。

## 直接 OOF 候选

候选：`fixed-v2 all + content PE v1 + content PE v2 + content_cert + with_logreg OOF stacker`，不启用 content string。训练仍只用 Train 的 OOF 预测训练 meta 层，并只用 Val 选择模型和阈值。

| Item | Value |
|---|---:|
| Train kept | `20000 / 20000` |
| Val kept | `20000 / 20000` |
| Feature dim | `1751` |
| Selected meta model | `meta_logreg_l2_c1` |
| Threshold | `0.41` |
| Val F1 | `0.9899915351` |
| Val errors | `201` |
| Val FP/FN | `142 / 59` |

对照当前门槛：

| Model | Val F1 | Errors | FP | FN | Decision |
|---|---:|---:|---:|---:|---|
| Loop136 strict best | `0.9910789933` | `179` | `122` | `57` | baseline |
| R5 fallback | `0.9907342832` | `186` | `130` | `56` | fallback |
| Loop142 cert-OOF | `0.9899915351` | `201` | `142` | `59` | reject |

Loop142 cert-OOF 比 Loop136 多 `22` 个 Val 错误，也比 R5 多 `15` 个 Val 错误，因此没有资格进入 Test-10k。

## 二层 Pairwise 诊断

为确认证书 OOF 是否至少能在少量样本上提供互补信号，又跑了一个 Val-only 诊断：默认保留 Loop136，只在 selector 认为 cert-OOF 更可信时接管。

| Model | Val F1 | Errors | FP | FN | Accepted rows |
|---|---:|---:|---:|---:|---:|
| Loop136 baseline | `0.9910789933` | `179` | `122` | `57` | - |
| Cert-OOF candidate | `0.9899915351` | `201` | `142` | `59` | - |
| Loop136 vs cert-OOF selector | `0.9911292734` | `178` | `122` | `56` | `1` |

该 selector 只改善 `1` 个 Val 错误且只接管 `1` 行，不满足 post-Loop139/140 的稳定性门槛（另一个 selector 家族至少应达到 `Val errors <= 169` 且有足够接管支持）。因此也拒绝进入 Test-10k。

## 非决策运行

早先误启动过一个 `fixed-v2 + string + cert` 重候选，前台超时且未完成可用报告；它不作为模型决策证据。随后一次 pairwise 诊断带 content string selector feature 时因缺少 string sidecar 直接失败，没有拟合模型，也不作为结果证据；成功的 pairwise 诊断只使用 content PE / fixed-v2 selector 特征。

## 结论

拒绝 Loop142。证书 blob 特征在当前严格 split 下不是足够强的新证据；它没有补上 Loop136 的主要错误池，反而扩大了 FP。结合此前 Loop31/Loop46 对浅层证书 blob 和 ASN.1 结构的负结果，下一步不应继续堆证书字段的小变体，除非引入真正新的外部信任链验证、信誉数据或经过治理的标签证据。

当前 strict best 保持 Loop136：

- Full-test F1：`0.9903723842`
- Full-test errors：`1544 / 160000`
- FP/FN：`958 / 586`

## Artifacts

- Direct cert-OOF report：`reports/phase3_loop142/oof_fixed_v2_all_cert_with_logreg_valonly/stage2_oof_stacker_report.json`
- Direct cert-OOF Val predictions：`reports/phase3_loop142/oof_fixed_v2_all_cert_with_logreg_valonly/stage2_oof_stacker_val_predictions.csv`
- Pairwise diagnostic report：`reports/phase3_loop142/loop136_vs_cert_oof_pairwise_valonly/loop135_pairwise_selector_report.json`
- Pairwise diagnostic Val predictions：`reports/phase3_loop142/loop136_vs_cert_oof_pairwise_valonly/loop135_pairwise_selector_val_predictions.csv`
- Pre-run guard：`reports/phase3_loop142/pre_run_guard_oof_fixedv2_all_cert_with_logreg_valonly.json`
- Pairwise pre-run guard：`reports/phase3_loop142/pre_run_guard_loop136_vs_cert_oof_pairwise_valonly.json`

## Next

继续改进模型时，应停止 R5/OOF 小 selector 与证书小变体的消耗，把精力转到更正交的证据：外部签名信任状态/信誉、行为或动态特征、以及 high-confidence FP/FN 的标签治理。若只能保留当前静态特征合同，Loop138 的邻域审计已经显示 99.9% 目标缺少足够统计支撑，需要进入指标降级或数据治理决策。
