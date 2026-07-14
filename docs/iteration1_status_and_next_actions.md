# Iteration 1 Status And Next Actions

更新时间：2026-07-01

## 当前最好结果

当前最强候选仍是 stage2 extended：

- 分支：`exp/model-agent-stage2-cache-matrix`
- 提交：`5180b8c`
- Val F1：`0.9818199930`
- Val 错误：`365 / 20000`
- 冻结 Test10k F1：`0.9821285141`
- Test10k 错误：`178 / 10000`
- 决策：未达到进入 16w full test 的门槛。

## 已失败/未通过 gate 的模型实验

Hardweight stage2：

- 分支：`exp/model-agent-hardweight-stage2`
- 提交：`651bac7`
- Val Top-1：`hgb_lr0.06_leaf31_l2_0__noise_near_threshold_upweight`
- Val F1：`0.9815534949`
- Val 错误：`370 / 20000`
- 决策：低于当前最好 Val F1 `0.9818199930`，未进入 Test10k。

结论：简单 train-only hard-example 重加权未突破平台期。

## 当前噪声/错误证据链

1. 1% cache 随机审计通过：
   - `2000 / 2000`
   - NPZ 可读、字段形状、label、源文件 SHA256 一致。
2. Val 错误复核队列：
   - `365` 个 Val 错误。
   - P0 严重冲突：`36`
   - P1 高置信冲突：`57`
3. Train-neighbor 审计：
   - P0/P1 中 `61` 个更支持模型预测。
   - `6` 个更支持原标签。
   - `27` 个混合。
4. PE 元数据审计：
   - P0/P1 `93` 个全部是可解析 PE。
   - 高熵 section 样本：`24`
   - writable + executable section 样本：`11`
   - overlay > 1MB 样本：`10`

这些证据说明：当前错误不是简单阈值问题，也不是 cache 文件坏了。高置信错误中存在明显的标签可信度/样本语义冲突。

## Top20 复核规则

按 Error-Agent 建议，下一步 Top20 不应只取概率最高，而应分层：

1. 全部 `severe_fn_conflict_prob_le_0.01`：`8` 条。
2. `severe_fp_conflict_prob_ge_0.99` 中概率最高 `8` 条，至少包含 `2` 条 `.exe`。
3. `2020-08/2022-08-*` 路径异常 FN 中概率最低 `2` 条。
4. `黑文件1/samples/samples` 家族路径 FN 中最接近阈值的 `2` 条。

目标是同时覆盖黑名单疑似污染、白名单疑似污染、目录异常和真实模型盲区。

## 是否触发 99.9% 降级讨论

目前还不直接降级，因为还缺少人工/多源复核结论。

但风险已经很高：P0/P1 中 `61` 个近邻支持模型预测，且 P0/P1 全部是有效 PE。如果 Top20 复核确认其中多数标签不可信，则应进入科学降级讨论。原因是 `F1 >= 99.9%` 的容错率约在 `0.1%` 量级，而当前 Val 中仅 P0 严重冲突就有 `36/20000 = 0.18%`。

## 下一步

1. 构建 Top20 人工复核包。
2. 对每个样本记录：当前标签是否可信、是否应作为训练噪声处理、是否属于模型盲区。
3. 不删除 Val 样本，不用 clean Val 作为主指标。
4. 若 Top20 多数为噪声，启动可行性降级报告。
5. 若 Top20 多数为模型盲区，下一轮实验转向新特征或更强模型，而不是继续 hardweight/HGB 微调。
