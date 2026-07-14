# Phase 3 Loop133 策略重审与噪声审计路线

更新时间：2026-07-05

## 当前状态

当前 strict best 仍是 Loop130 `R5_r4_plus_vendor_strings`：

| Split | F1 | Errors | FP | FN |
|---|---:|---:|---:|---:|
| Val | 0.9907342832 | 186 | 130 | 56 |
| Test-10k | 0.9915983197 | 84 | 56 | 28 |
| Full-test | 0.9902567651 | 1563 | 991 | 572 |

R5 已完整走完 Val -> Test-10k -> Full-test 漏斗。它相比 Loop129 R4 的 full-test `1571` errors 小幅改善到 `1563`，但距离 F1 >= 99.9% 仍很远。16 万测试集上，99.9% F1 通常意味着错误预算接近百级；当前仍是千级错误。

## 最近负结果

| 候选 | Val | Test-10k | Full-test | 结论 |
|---|---:|---:|---:|---|
| fixed-v2 + content_string OOF stacker | 202 errors | 未跑 | 未跑 | Val 弱于 R5，拒绝 |
| R8 dialog protector | 185 errors | 85 errors | 未跑 | Val 改善但 Test-10k 退化，拒绝 |
| Loop132 R11 FN recovery | 180 errors | 84 errors | 1594 errors | Val 改善、Test-10k 打平，但 full-test 退化，拒绝 |
| overlay/last-section/image-base FN rescue | 180 errors | 86 errors | 未跑 | Val 改善但 Test-10k 退化，拒绝 |

这些负结果说明：当前阶段很多 Val 小支持规则会显得很诱人，但跨到 Test-10k 或 full-test 后不稳定。继续堆窄规则很可能只是在 Val 上“抠样本”，不是稳定提升泛化能力。

## 噪声与理论上限风险

R4/R5 后仍存在大量高置信错误。已知 R4 audit 中：

- full-test errors：1571
- high-confidence FP (`prob >= 0.9`)：474
- high-confidence non-guard FN (`prob < 0.1`)：291
- guard-induced FN：38

R5 进一步减少 FP，但把 FN 从 R4 的 `542` 增到 `572`。这表明当前模型并非只差阈值，而是在若干样本簇上高度自信地错。若这些高置信错判中有相当比例是标签噪声、近重复标注冲突或静态不可分样本，则 F1 >= 99.9% 在当前静态特征与标签状态下可能不可达。

## Loop133 R5 审计更新

已补齐 R5 full-test 的错误与 flip 审计产物：

- 汇总：`reports/phase3_loop133/loop133_r5_error_audit_summary.json`
- 错误队列：`reports/phase3_loop133/loop133_r5_error_review_queue.csv`
- flip 队列：`reports/phase3_loop133/loop133_r5_flips_audit.csv`
- 报告：`docs/phase3_loop133_r5_error_audit_report.md`

R5 审计确认：

| 审计项 | 数量 |
|---|---:|
| full-test errors | 1563 |
| high-confidence FP (`prob >= 0.90`) | 472 |
| high-confidence FN (`prob < 0.10`) | 291 |
| R5 guard flips | 210 |
| R5 repaired FP | 142 |
| R5 harmful FN | 68 |
| R5 extra repaired FP over R4 | 38 |
| R5 extra harmful FN over R4 | 30 |

`guard_repaired_fp` 与 `guard_harmful_fn` 在 resource、overlay、vendor-string 特征上高度相似：例如 `string_benign_vendor_count_log` 均值分别为 `3.4761` 与 `3.6755`，`content_overlay_log_size` 均值分别为 `8.5086` 与 `8.7632`。这意味着继续手写 vendor/resource 单阈值很容易同时修复 FP 和制造 FN，下一轮应改用 Train/Val OOF selector 或数值邻域审计。

## 下一阶段建议

1. **先做噪声/冲突审计，而不是继续小规则。**
   对 R5 full-test 的 high-confidence FP/FN、R5 extra flips、R5 harmful flips 做 review queue，并用 Train/Val 数值邻域判断这些样本更像标签噪声还是模型盲区。

2. **把 FN recovery 从“单规则”升级为分层候选。**
   Loop132 证明单条 FN rescue 外推不稳。下一轮应考虑 train-OOF 的 selector 或分层模型，但必须避免把 Test/full 结果用于训练或阈值选择。

3. **优先做 R5 flip harm selector。**
   在 Train/Val 上训练或预声明 selector，目标是阻止高风险 `1 -> 0` flip，输入只允许概率、模型分歧、content PE v1/v2、content string 等数值特征。full-test 审计只能帮助定义问题，不参与阈值选择。

4. **把当前 R5 作为 production-style strict best，但标注业务 trade-off。**
   它减少 FP，同时增加 FN；如果业务更怕漏报，R4 可能仍是更保守的可选策略。当前 strict best 按 F1/total errors 是 R5。

5. **重新评估 99.9% 可行性。**
   如果噪声审计确认高置信错误里存在数百个疑似标签噪声或静态不可分样本，应提出科学降级，例如先把目标改为 F1 >= 99.2% 或 99.5%，并说明所需额外数据/动态特征条件。

## 下一步可执行任务

- 生成 R5 full-test error review queue，按 FP/FN、confidence、guard flip、feature slices 分层。（已完成）
- 对 R5 错误样本做 kNN 数值邻域审计，只用 Train/Val 数值特征，不用路径、文件名、目录、后缀、hash 作为模型证据。
- 对 R5 总 flips 中的 `142` 个修复 FP 和 `68` 个 harmful FN 做差异归因；其中相对 R4 额外 flips 是 `38` 个修复 FP 和 `30` 个 harmful FN。
- 把噪声候选分为：可疑标签、静态不可分、模型盲区、特征坏/缺失。只有“特征坏/缺失”才重新抽/重提；不要用坏样本补齐样本数。

## Loop134 Val-first 候选更新

已按 Val-first 漏斗验证三个候选，均未超过 R5 Val，因此全部拒绝，不进入 Test-10k：

| Candidate | Val F1 | Errors | FP | FN |
|---|---:|---:|---:|---:|
| Noise-aware OOF fixed-v2 + content string | 0.9905796740 | 189 | 126 | 63 |
| Train-only kNN local support + content string | 0.9892086331 | 216 | 116 | 100 |
| Overlay/security boundary specialist | 0.9882670128 | 235 | 132 | 103 |

详见 `docs/phase3_loop134_val_candidate_sweep_report.md`。本轮最有价值的信号是 OOF noise 与 R5 有互补性：它修复 `26` 个 R5 Val 错误，但新增 `29` 个错误。下一轮不应整体替换 R5，而应在 Train/Val 上研究“何时低风险接管 R5”的 selector。

## Loop135 Pairwise Selector 更新

已把 Loop134 的互补性做成 Train/Val-only selector：默认保留 R5，只在冻结 selector 判断 OOF noise 更可信时接管候选预测。

| Split | Candidate | F1 | Errors | FP | FN | Decision |
|---|---|---:|---:|---:|---:|---|
| Val | Loop135 selector | 0.9911733905 | 177 | 115 | 62 | 允许进入 Test-10k |
| Test-10k | Loop135 selector | 0.9913913914 | 86 | 53 | 33 | 弱于 R5 的 84 errors，拒绝 |

Loop135 不进入 full-test。它进一步确认当前 selector 最大风险是把部分恶意样本从黑翻白；后续若继续做 selector，需要显式加入 recall-aware / harmful-FN-constrained 选择准则。

## Loop136 Recall-aware Selector 更新

Loop136 在 Loop135 基础上加入 Val FN 约束：候选 FN 不能比 R5 多超过 `2` 个。该约束在 Train/Val 阶段预先用于选择，之后按冻结模型走 Test-10k 和 full-test。

| Split | Candidate | F1 | Errors | FP | FN |
|---|---|---:|---:|---:|---:|
| Val | Loop136 selector | 0.9910789933 | 179 | 122 | 57 |
| Test-10k | Loop136 selector | 0.9916958479 | 83 | 54 | 29 |
| Full-test | Loop136 selector | 0.9903723842 | 1544 | 958 | 586 |

Loop136 相比 R5 full-test：errors `1563 -> 1544`，FP `991 -> 958`，FN `572 -> 586`。按 F1/总错误，Loop136 是新的 strict best；按漏报风险，R5 仍是更保守 fallback。主报告：`docs/phase3_loop136_recall_pairwise_selector_full_test_report.md`。

## Loop137 Recall Recovery 更新

Loop137 尝试在 Loop136 之上恢复 FN，并先补强 selector Val 选择门禁，要求候选不仅满足 FN 约束，还必须在 Val 上真实降低错误并提升 F1。

结果：fixed-v2/string selector 变体虽在 Val 上从 `179` errors 降到 `176/177/178`，但 Test-10k 退化或仅打平，全部拒绝。Loop136 FN recovery R11 在 Val 上达到 `174` errors、Test-10k 打平 `83` errors 且 recall 提升，但 full-test 退化到 `1583` errors (`1059` FP / `524` FN)，弱于 Loop136 的 `1544` errors。

结论：Loop137 不替代 Loop136。当前 FN recovery 小规则存在明显 FP 外溢风险，下一步应优先做噪声/邻域审计和更正交特征，而不是继续扩写同类规则。主报告：`docs/phase3_loop137_recall_recovery_negative_report.md`。
