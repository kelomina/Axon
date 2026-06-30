# Eval Protocol Audit

更新时间：2026-07-01

## 结论

当前项目存在流程性 test 泄漏风险。这里的“泄漏”不是指脚本已经明确把 test 数据拿去训练，而是指若持续查看 test10k 的阈值扫描、候选模型排行榜或融合权重结果，再反过来调整阈值、特征、模型或融合权重，test10k 就会从“确认集”变成“调参集”。这会让最终指标看起来更好，但真实泛化能力会被高估。

## 已发现的风险点

1. `evaluate_split_from_cache.py` 支持对任意 split 做 `--sweep-thresholds`。当前已有 test10k sweep 报告，后续不得根据 test10k sweep 选择阈值。
2. 历史 stage2 报告中存在多个候选同时展示 test10k 分数的情况。后续不得把这些 test10k 分数作为候选排序依据。
3. byte n-gram 实验脚本会在同一报告中输出 test 结果。如果反复运行并根据 test 结果改参数，也会形成泄漏。
4. `max_rows 10000` 的 test10k 是测试集前 10000 行，不是随机抽样，可能存在顺序偏差。它只能作为冻结候选的快速确认，不能代表最终 16w 验收。

## 后续强制协议

1. Cache gate：20w split 必须保持 `200000/200000` cache 覆盖。
2. NN export gate：同一个 checkpoint、split、manifest、feature mask 的神经网络概率只导出一次，后续优先复用 CSV。
3. Model selection gate：calibrator、stage2、byte n-gram、融合权重只能使用 train 拟合、val 选择。
4. Candidate freeze gate：进入 test10k 前必须冻结候选，包括 checkpoint、特征集、模型类型、阈值、所有超参和随机种子。
5. Test10k smoke gate：冻结候选只允许做一次 test10k 确认，不允许因为 test10k 结果再调阈值或换候选。
6. Final gate：只有 test10k 确认通过后，才跑完整 16w test。最终报告必须使用 val 选出的阈值，不做 test sweep。

## 当前指标解释

- 裸 baseline：Val F1 约 `0.9297`，Test10k F1 约 `0.9299`。
- 概率校准器：Val F1 约 `0.9688`，Test10k F1 约 `0.9724`。
- 历史 stage2 extended 结果来自替换前旧 split，不能作为纠正后 split 的有效证据。
- 当前所有已确认结果都距离 `16w test F1 >= 99.9%` 很远，必须继续执行错误归因和噪声审计，而不能直接冲全量 test。

## 执行要求

后续所有候选报告必须把 `selected_by_val` 作为唯一选择依据。test10k 字段只能出现在冻结候选的一次性确认报告里，不能出现在日常候选矩阵排行榜里。
