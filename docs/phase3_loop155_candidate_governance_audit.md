# Phase 3 Loop155 Candidate Governance Audit

更新时间：2026-07-08

## 目标

Loop151 周边已经出现几个“看起来更激进”的组合：有的 Val 很强但 Test-10k 外溢，有的 full-test 错误数甚至低于 Loop151，但 Val 先失败。Loop155 的目标不是训练新模型，而是把这些候选统一放回漏斗规则里审计，避免因为看到 full-test 局部好看就反向选择模型。

本轮新增：

- `scripts/build_loop155_candidate_governance_audit.py`
- `tests/test_build_loop155_candidate_governance_audit.py`
- `reports/phase3_loop155/loop155_candidate_governance_audit.json`
- `reports/phase3_loop155/loop155_candidate_governance_audit.md`

身份字段策略不变：`source_path`、`cache_path`、`source_sha256`、`sample_index`、split、row order 只用于物流、对齐和审计，不作为模型、阈值、verdict、replacement sampling 或生产推理证据。

## 审计结果

| Candidate | Val | Test-10k | Full-test | Decision |
|---|---:|---:|---:|---|
| Loop151 trusted signer guard | `162` errors | `78` errors | `1466` errors | `adopted_current_strict_best` |
| Loop144 union + trusted signer | `150` errors | `81` errors | `not_run` | `reject_test10k_gate` |
| OOF-noise/R5 + trusted signer | `173` errors | `not_run` | `1460` errors | `reject_val_gate_full_test_mirage` |
| Loop154 threshold `0.995` | `162` errors | `78` errors | `1466` errors | `reject_equivalent_to_current_best` |

最容易误判的是 OOF-noise/R5 + trusted signer：它的 full-test 是 `1460` errors，比 Loop151 少 `6` 个，但 Val 是 `173` errors，比 Loop151 多 `11` 个。这个结果只能作为诊断证据，不能替代 Loop151；否则就是用 full-test 反向选模型。

Loop144 union + trusted signer 也不能采用：它 Val 从 Loop151 的 `162` errors 降到 `150`，但 Test-10k 从 `78` 退到 `81`。这说明同类 recall union 在 Val 上的收益仍会外溢成 Test 误差，不能进入 full-test。

## 决策

Loop151 仍是当前可部署 strict best：Val `162` errors、Test-10k `78` errors、full-test `1466 / 160000` errors。Loop155 把“full-test 更低但 Val 失败”的候选标成 full-test mirage，后续 agent 不应把 OOF-noise/R5 + signer 的 `1460` full-test errors 当作新的 best。

下一步继续两条路：

1. Val-first 正交证据：外部信誉、多引擎、行为沙箱或签名/发布者人工复核，不能从 full-test 观察追加规则。
2. Loop153 噪声治理：当前-best Val `73` 行 focus 等待独立 verdict；确认坏行后才 same-original-label fresh redraw，并保持严格 `200000 = 20000/20000/160000`。

## 验证

```powershell
.\vnev\Scripts\python.exe scripts\build_loop155_candidate_governance_audit.py --output-json reports\phase3_loop155\loop155_candidate_governance_audit.json --output-md reports\phase3_loop155\loop155_candidate_governance_audit.md

.\vnev\Scripts\python.exe -m pytest tests\test_build_loop155_candidate_governance_audit.py -q
```

结果：`1 passed`。
