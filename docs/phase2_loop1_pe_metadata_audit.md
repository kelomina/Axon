# Phase 2 Loop 1 PE Metadata Audit

更新时间：2026-07-01

## 目的

本报告对 stage2 extended 冻结候选的 P0/P1 高置信 Val 错误做轻量 PE 元数据审计。它回答一个很具体的问题：这些高置信错误是不是因为样本不是有效 PE、cache 坏了、或者文件结构异常。

本轮仍只使用 Val 复核队列，不使用 Test10k 调参。

## 输入与输出

- 输入队列：`reports/random_20w_split/stage2_extended_val_error_review_queue.csv`
- 脚本：`scripts/audit_pe_metadata_queue.py`
- 输出 CSV：`reports/random_20w_split/stage2_extended_val_p0_p1_pe_metadata.csv`
- 输出 JSON：`reports/random_20w_split/stage2_extended_val_p0_p1_pe_metadata.json`

## 总体结果

- 审计样本：`93`
- 可解析 PE：`93`
- 解析失败：`0`
- 有高熵 section 的样本：`24`
- 有 writable + executable section 的样本：`11`
- overlay 大于 1MB 的样本：`10`

这说明 P0/P1 错误不是由“非 PE 文件混入”造成的。它们都是结构上可解析的 PE 文件，剩下的问题更可能是标签可信度、样本族分布混杂、packed/overlay/section 异常，或者模型对某些合法 PE 形态掌握不足。

## 分组观察

### 严重 FN：`label=1` 且模型概率 `<=0.01`

- 数量：`8`
- 扩展名：`.dll` 3、`.exe` 4、`.sys` 1
- 平均文件大小：约 `4.64MB`
- 最大文件大小：约 `20.12MB`
- 平均 section 数：`6.25`
- 有高熵 section 的比例较低，但 overlay 平均较大，平均约 `1.71MB`

解释：严重 FN 里存在 DLL/SYS 和大 overlay 样本。模型把它们判得极白，可能是这些样本在当前特征上非常接近白样本，或者标签来自目录继承而没有进一步确认。

### 高置信 FN：`label=1` 且模型概率 `<=0.05`

- 数量：`19`
- 扩展名：`.exe` 15、`.dll` 4
- 平均文件大小：约 `3.20MB`
- 最大文件大小：约 `36.33MB`
- 有高熵 section 的样本存在，但不是主导特征

解释：这批样本更像“模型盲区 + 批次分布偏移”的组合，尤其需要按月份和家族复核。

### 严重 FP：`label=0` 且模型概率 `>=0.99`

- 数量：`28`
- 扩展名：无扩展名 22、`.exe` 6
- 平均文件大小：约 `2.97MB`
- 最大文件大小：约 `18.99MB`
- 有高熵 section 的样本：存在
- writable + executable section 平均高于 FN 组

解释：这批白名单样本的 PE 结构不像普通干净文件那么稳定。大量无扩展名但可解析 PE，被模型判黑有结构依据。它们应优先进入“白名单标签可信度复核”。

### 高置信 FP：`label=0` 且模型概率 `>=0.95`

- 数量：`38`
- 扩展名：无扩展名 26、`.exe` 12
- 平均文件大小：约 `2.33MB`
- 最大文件大小：约 `24.13MB`
- overlay 和高熵 section 信号均存在

解释：这批样本可能包含 packed benign、灰色软件、签名缺失的正常程序，或者白名单污染。仅靠当前静态特征很难无损区分。

## 与近邻审计的联合结论

结合 `docs/phase2_loop1_noise_adjudication.md`：

- P0/P1 中 `61` 个样本的 train 近邻更支持模型预测。
- 本 PE 审计显示 P0/P1 全部是可解析 PE，且不少有高熵、overlay、可写可执行 section 等风险形态。

因此，当前最强模型的错误不是简单“模型没训练好”或“阈值没调好”。高置信错误里存在真实的数据语义冲突，必须先复核标签可信度。

## 下一步

1. 对 P0 FP 白名单样本优先查来源、签名、业务可信度，尤其是无扩展名 PE。
2. 对 P0 FN 黑样本优先查是否只是目录继承标签，尤其是 DLL/SYS 和大 overlay 文件。
3. 不建议删除 Val 中这些样本来抬分；Val 必须保持完整。
4. 如果人工复核确认 P0/P1 中相当比例是噪声，必须启动 99.9% 可行性降级讨论。
