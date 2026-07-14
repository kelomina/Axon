# Loop167 Phase A Source Closure

## 结论

Loop167 的 Phase A 已通过静态语义闭环，但这不是训练结果、不是 F1 提升，也不打开 Phase B。Loop151 仍是唯一 research champion，legacy full-test `F1=0.9908541911`；`F1 >= 0.9997` 仍未达成。

本轮没有打开 raw 样本、checkpoint 或 prediction rows，没有训练、拟合、阈值搜索，也没有访问 Val、Test-10k、legacy full-test 或任何 sealed window。

## 已冻结事实

- EMBER2024 v3 的真实拼接顺序已以固定源码 SHA 绑定：Header 位于 `[696,770)`，Section 位于 `[770,994)`，而不是旧维数说明中的块顺序。
- 2568 列严格分类为 exact `49`、partial `487`、genuinely novel `292`、forbidden/unstable `1740`。
- 292 列 novel 来自有序首字节 `4`、16x16 local byte-entropy histogram `256`、未覆盖 Header/DOS 字段 `31`、Rich pair count `1`；data directories、Authenticode、hash blocks 和 warnings 没有被冒充为 novel。
- DataDirectories 的 reserved pair `[2435,2437)` 已被识别为 pinned official loop bound 导致的死列；Exports 的 `[2276]` 是 hash-vector length sentinel，不是 export count。
- B0 初始 Axon inventory 为 `572` 个有名结构列，只有 `content_file_log_size` 与 `fixed_v2_log_size` 被证明 bit-equivalent 并去重，因此冻结 allowlist 为 `571` 列。
- 项目原生 `extract_novel_delta(bytes)` 只接收已打开的 bytes，不接收路径、标签、样本标识、hash 或模型分数；空输入与 PE 解析失败都返回有限向量和明确 missing reason。

## 证据

- `semantic_delta_mapping.json`：2568 个逐列分类与源语义。
- `frozen_deduplicated_baseline_allowlist.json`：572→571 的名称级基线清单。
- `phase_a_source_semantics_addendum.json`：真实顺序、缺失策略、死列和后续阻塞项。
- `phase_a_source_closure.json`：源文件与上述合同的 SHA 绑定。
- `phase_a_static_decision.json`：本轮结论与验证计数。

验证使用项目固定 `vnev/Scripts/python.exe`：`11 passed`，Ruff、`py_compile`、mapping/allowlist/addendum/source-closure 的 `--check` 均通过。

## 下一门

Phase B 仍然严格关闭。下一项工作不是再跑一次数据，而是实现独立的 one-pass `RawFeatureContext` 与 overlap-control extractor：同一已打开字节流只能解析一次、同时生成 B0/B1/M/A/CF 所需内容、记录 missing reason，且不导入 Loop164 controller。

在此之后还必须完成 pinned Authenticode contract、不可复用 lease、runtime lock、资源 guard 和三-seed 规则修订。当前 HGB 配置在已审计的 synthetic matrix 上让 seed `41/42/43` 产生 bit-identical 输出，因此未来不能把它们称为三个独立稳健性实验，除非新授权明确改变机制。
