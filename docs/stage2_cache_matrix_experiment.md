# Stage2 Cache Matrix Experiment

更新时间：2026-07-01

## 协议

本实验遵守 Val-first 漏斗：

1. 使用已导出的神经网络预测 CSV 和 20w cache，不重新跑神经网络。
2. 候选矩阵只在 Train/Val 上训练和选择。
3. Test10k 只对 Val Top-1 冻结候选执行一次确认，不做阈值扫描，不重新选择候选。
4. 因 Test10k 仍远低于 `F1 >= 99.9%`，本轮不进入 16w full test。

## Cache 抽样审计

新增 1% 随机抽样审计：

- 脚本：`scripts/audit_cache_random_sample.py`
- 报告：`reports/random_20w_split/cache_random_sample_audit_1pct_seed42.json`
- 抽样：`2000 / 200000`
- NPZ 检查：`2000 / 2000`
- 源文件 SHA256 检查：`2000 / 2000`
- 失败样本：`0`
- 结论：抽样范围内 cache 与 split、manifest、源文件一致。

## Val-only 候选结果

### Tabular

- 报告：`reports/random_20w_split/stage2_cache_matrix_replaced_tabular_valonly/stage2_cache_matrix_report.json`
- 特征维度：`311`
- Val Top-1：`hgb_lr0.06_leaf31_l2_0__noise_none`
- 阈值：`0.545`
- Val F1：`0.9810568295`
- Val 错误：`380 / 20000`
- Test：未运行

### Extended

- 报告：`reports/random_20w_split/stage2_cache_matrix_replaced_extended_valonly/stage2_cache_matrix_report.json`
- 特征维度：`1420`
- Val Top-1：`hgb_lr0.08_leaf31_l2_1e-3__noise_none`
- 阈值：`0.515`
- Val F1：`0.9818199930`
- Val 错误：`365 / 20000`
- Test：Val Top-1 冻结后单次确认

Extended 比 Tabular 的 Val F1 高约 `0.000763`，因此本轮冻结 Extended Top-1 进入 Test10k。

## 冻结候选 Test10k 确认

- 冻结模型：`reports/random_20w_split/stage2_cache_matrix_replaced_extended_valonly/stage2_selected_model.pkl`
- 冻结评估脚本：`scripts/evaluate_stage2_cache_model.py`
- Test10k 报告：`reports/random_20w_split/stage2_cache_matrix_replaced_extended_frozen_test10k_eval.json`
- 阈值：`0.515`
- Test10k F1：`0.9821285141`
- Test10k AUC：`0.9981620811`
- Test10k 错误：`178 / 10000`
- FP：`92`
- FN：`86`

## 错误与噪声

冻结候选 Val：

- 错误：`365 / 20000`
- FP：`221`
- FN：`144`
- 疑似噪声/硬样本：`147`
- 严重 FP 冲突，`label=0` 且 `prob>=0.99`：`28`
- 严重 FN 冲突，`label=1` 且 `prob<=0.01`：`8`

冻结候选 Test10k：

- 错误：`178 / 10000`
- FP：`92`
- FN：`86`
- 疑似噪声/硬样本：`67`
- 严重 FP 冲突，`label=0` 且 `prob>=0.99`：`12`
- 严重 FN 冲突，`label=1` 且 `prob<=0.01`：`2`

## 决策

本轮不进入 16w full test。理由很直接：Test10k 仍有 `178/10000` 错误，按比例外推到 16w 会是数千级错误；而 `F1 >= 99.9%` 只允许非常低的错误量。继续跑 16w full test 只会消耗算力，不能带来新的有效选择信号。

## 下一轮建议

1. 回到 Phase 2：优先复核 Val 中 `28` 个严重 FP 和 `8` 个严重 FN。
2. 对白名单 `<none>` 扩展名高置信 FP 做家族/来源聚类，判断是标签噪声还是模型把无扩展 PE/packed benign 当成恶意。
3. 对恶意 `.exe/.dll` FN 按月份和 family 归因，判断是否存在批次分布偏移。
4. 下一轮 Model-Agent 不应继续只做浅层 stage2 堆叠，需要引入更强的 hard-example 训练或噪声鲁棒训练；否则提升会进入平台期。
