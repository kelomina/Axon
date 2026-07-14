# Loop167 Phase-B v6 Windows Job ABI Remediation

## 结论

Loop167 Phase-B v5 在 lease 前的 Windows Job ABI 边界失败后，已被永久封存，不能重试、复用 lease 或复用输出。v6 是新的 ABI remediation 链，不是对 v5 的续跑；它目前只完成 raw-free 的静态信任链和 Windows Job 验证，不能据此宣称训练、OOF、Val、Test-10k、legacy Full-test 或 F1 改善。

本地 Train-only 路径不需要 public key。`phase_b_protocol.json` 明确 `public_key_required=false`，且禁止把 public key 变成运行前提；无 pinned CMS parser 的签名输入遵循零向量加 missing 指标的既定合同。

## 静态证据链

以下 v6 产物均为新写入、不可覆盖的 canonical JSON，目录与 v5 隔离：

| 产物 | SHA-256 |
| --- | --- |
| `phase_b_v5_windows_job_abi_prelease_attestation.json` | `8dca8a405bb1f6da0f57fc15b4d7f96cb76c6fd9c875774610857ab6ccc6441a` |
| `phase_b_execution_contract_v6.json` | `f44d1267db3cc945170d6fb1375fa4b8540945127c5b34a7fab96efe0182dc79` |
| `phase_b_runtime_lock_v6.json` | `cf30a13e76d327096a63a159b3398fa6cece3de92b798a92e2d09f57240cfca8` |
| `phase_b_source_closure_v6.json` | `585e5c637b337fc71d439808042cfd7a15fed7c2531bcddf226a3383baf5e633` |

它们位于 `manifests/roadmap_9997/loop167_ember_v3_novel_delta/phase_b_v6_windows_job_abi_remediation/`。source closure 覆盖控制面、数据面、Loop166 proof bridge、v6 脚本和相关测试；其中任一受控源码发生变化时，此 v6 链必须视为 drift，不能改写或重用，应另起新链。

## 验证

- v6 suite 加 Loop166 code-section tests：`53 passed, 5 skipped`；跳过项仅为当前 Windows 无法创建 symlink 的平台限制。
- v6 scope Ruff 与 `compileall` 均通过。
- Loop166 proof bridge 增加加载期替换回归：loader 执行期间改写 proof 后，post-load SHA-256 复验拒绝并清理动态模块。
- supervisor 和 controller 的 static preflight 均通过，分别报告 `raw_open_attempts=0`。
- zero-input suspended-child probe 通过：创建暂停子进程、`AssignProcessToJobObject`、`IsProcessInJob`、恢复后退出，树内活跃进程数为 `0`，且 `raw_open_attempts=0`。

预检的 process argv 是严格字节级合同。运行入口必须保留相对 POSIX 形式 `scripts/run_loop167_phase_b_*_v6.py` 与 `-I`；Windows 中等价的反斜杠脚本参数会被 fail-closed 拒绝，而不会打开 raw。

## 当前阻断与恢复顺序

2026-07-14T05:24:41Z 的 Windows live snapshot 显示可用物理内存 `6,610,083,840` bytes (`6.16 GiB`)；v6 sealed floor 是 `12 GiB`，当时缺口为 `6,274,818,048` bytes (`5.84 GiB`)。该值会波动，任何恢复前必须重新采样，且至少达到 floor。

随后实际运行 fresh v6 resource-guard builder；它以 exit code `2` 返回 `available_memory_below_sealed_launch_floor`，并明确报告 guard path 未写入。该结果没有产生 guard、authorization、lease 或 raw open。

在该条件满足前，禁止写入 resource guard、run authorization 或 lease；禁止启动 raw pass、fit、Val、Test-10k、legacy Full-test 或 promotion。资源就绪后仍须按以下顺序进行：fresh passing resource guard、立即校验其 300 秒有效期、fresh authorization、supervisor-contained execute。任一失败均消耗对应尝试，不得以同一 lease retry、resume 或 rescan。

Loop151 仍是唯一 strict research champion，legacy Full-test F1 仍为 `0.9908541911`；`F1 >= 0.9997` 尚未接近，本轮没有产生任何模型质量声明。
