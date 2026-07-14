# Loop168 Capability Semantic Expert Proposal

## 目的

Loop168 是 Loop167 的后备静态专家，不是与其并行抢占 raw 预算的第二次试跑。只有 Loop167 的 EMBER-v3 novel-delta 因果门关闭或证明不可行动时，才考虑推进此路线。

它使用 Mandiant capa 的固定 capability 规则语义，提取“程序能做什么”的静态能力，而不是继续堆 PE 统计、阈值、旧预测分数或目录名称。这个方向与现有结构投影、字节摘要和 signer guard 更正交，但没有任何理由承诺单独把 Loop151 的 `1466` 个 legacy 错误压到 `F1 >= 0.9997` 所需的约 `48` 个错误量级。

## 已核验来源

- [Mandiant capa](https://github.com/mandiant/capa) 在 `f850024b5604443170de502ea0a6a598e82fe1c9` 固定；官方 README 说明它从可执行文件的静态证据识别 capability，并提供 standalone binary。
- [capa-rules](https://github.com/mandiant/capa-rules) 在 `aed45e2571ebf7d2330e3daddbb5c472cc54966e` 固定。
- 本机未发现 `capa` 可执行文件，也未在项目虚拟环境发现 `flare-capa`。本提案不安装软件、不下载 rules、不执行外部代码，也不打开样本。

## 未来合同

Phase A 只能是新的 SHA-bound `256` 行 Train-only coverage gate：固定 binary 和 ruleset hash、无网络隔离、单次 parse、解析前后 SHA 复验、显式 success/missing/timeout/unsupported/packed receipt。覆盖低于 `0.95`、任何 silent drop 或无法明确的 timeout/unsupported 都关闭路线；缺失回退 B0，绝不视为良性。

Phase B 才能在独立授权、resource guard 和 one-shot lease 后做同一分区的四臂 Train-only 对照：B0、capability-only、B0+capability、以及 fit-fold 内 capability 置乱反事实。三个固定 seed、五折、固定阈值和无超参搜索下，每个 seed 都必须同时满足净减错、repair、precision、折稳定性、component LCB、FP/FN 与 `M-CF` 因果门；任何失败直接关闭，不调 rules、阈值、seed 或 heldout。

## 红线

- 禁止路径、hash、目录、标签、review verdict、历史 score 和动态 sandbox 结果进入特征。
- 禁止把 packed、超时、解析失败或不支持格式当作良性；它们只能是 explicit missing 并回退 B0。
- 静态 capability 解析必须 no-network、不会执行样本、不持久化 raw byte 或 match location。
- Loop168 的任何 Train-only 正结果都不是 F1、Val、Test-10k、legacy Full-test、promotion 或认证证据。

真正的 `99.97%` 需要新标签/provenance、time-family-source 隔离、至少两个正交专家在同一 decision-aligned OOF 的稳定纠错，以及双 sealed-window 的最终验证。完整 machine-readable 合同见 `manifests/roadmap_9997/loop168_capability_semantic_expert/proposal.json`。
