# Loop166 Phase B1 Nonfinite Closure

## 结论

Loop166 当前 `code-section BPE-1024 + MLM` 配方已经永久关闭。Phase A 只证明可执行代码段提取可行；Phase B1 在完整完成一次授权的 Train outer-fit raw scan 后，于训练阶段触发 `B1FatalError: B1 training produced a non-finite gradient norm`。因此 B1 未完成，Phase C 五折没有解锁，Loop151 继续是唯一 research champion。

这不是 Loop28 回退，也不是 supervisor、超时或外部重启导致的失败。v2 supervisor 正常创建 Windows Job，先 suspended-create、assign、verify、写入并 fsync launch receipt，再 resume；最终 controller 返回 `1`，Job tree 清空，source closure 未漂移。

## 可复查证据

- v2 one-shot marker SHA-256：`bb684dbe772caf95c2fa959663a71fec6fe9ccf1ee473f77adb9956dd8a3d612`。lease 已消费，禁止删除、覆盖、改名、复用、resume、rescan 或同 lease retry。
- raw progress ledger SHA-256：`14a2f2a11ac157e27d13cf24aae37e310ad91ea3735fea4169c0c8570e1e5a2c`。hash chain 状态为 `complete`，`32,002` lines、`16,000` terminal records、raw opens/successes `15,988/15,988`、读取 `19,239,582,561` bytes；outer holdout raw opens/bytes 均为 `0/0`。
- ledger final record SHA-256：`51ee9446ac3af97590b3a940b4114eff8c9b9da05d58dc873b58073ac4fc4dc4`；corpus commitment：`880b9cb7a79f5253b4d7562280baeb9e8543ccae6f89791a8a97a3da91159ffc`。
- 部分 checkpoint SHA-256：`9077217b48c062733c5f505bbd3668f5f9ab5031e61285fbaa13208906301704`。它只到 step `8192/28768`、cursor `32768/115072`、prefix original bytes `16,405,658`、cumulative wall `7053.1366561s`。weights-only 结构/有限值检查与 fresh restore synthetic logits bit-exact 检查通过，但这不是 final checkpoint 或 final verify receipt。
- stderr SHA-256：`efd75b921c63c177ca10eec14c8030cbddbdc737cb93e183b0c7f5e1435b376e`；exit receipt SHA-256：`53349114f0d7bd4b339307f3b2e0bc89d6ea40af7ce623841ab46fd46600631d`。
- 成功 report 与 final verify receipt 均不存在，且不得事后补写。失败发生在 step `8192` checkpoint 之后、step `12288` checkpoint 之前；精确失败 step 不可证明。

机器事故记录位于 `manifests/roadmap_9997/loop166_code_section_foundation/phase_b1_step4096_recovery_v2_nonfinite_failure.json`，程序决策位于 `manifests/roadmap_9997/loop166_code_section_foundation/phase_b1_nonfinite_decision.json`。

## Raw Pass 口径

- original confirmed pass：`1`；
- v1 physical pass：`0..1`，但 charged `1`；
- v2 confirmed complete pass：`1`，charged `1`；
- 当前 physical completed total：`2..3`，不是精确值；
- 当前 charged full-pass-equivalents：`3`。

不得把 v1 unknown 写成零，也不得使用带有 `successful_*` 含义的字段描述这次失败。

## 止损与下一路线

预注册的 nonfinite kill condition 已触发，所以不能通过降 scaler、改 FP32、换 optimizer/LR/schedule、补 checkpoint、调 threshold 或查看 heldout 来延续相同 lineage。这些变化只能属于全新实验，不能包装成 v2 recovery。

下一路线固定为独立 Loop167 `EMBER2024 v3 novel-delta structural control`。它先做不打开 raw 的 semantic-delta freeze，再在新的 source-closed authorization 与 one-shot lease 下决定是否进行 Train-only OOF。Loop167 不是 Loop166 修复，也不是 champion。

本地 Train-only 治理不需要 public key。public key 仍只属于未来独立 A2/A3 custody/certification，不是当前项目运行前提。

