# Phase 3 Loop161 Test-10k Promotion Margin Guard

更新时间：2026-07-08

## 目标

Loop160 证明了一个重要问题：候选在 Test-10k 上只少 `1` 个错误，仍然可能在 16 万 full-test 上反转。Loop161 把“Test-10k 确认”从口头判断变成明确闸门：Val 赢了以后，Test-10k 也必须至少少 `3` 个错误，才值得进入 full-test。

这个闸门只读聚合指标，不训练、不调阈值、不改 split/cache、不使用 full-test 选择模型。路径、hash、`source_sha256`、`sample_index` 和 row order 仍只允许做对齐和审计。

## 产物

新增脚本和测试：

- `scripts/build_loop161_test10k_promotion_margin_guard.py`
- `tests/test_build_loop161_test10k_promotion_margin_guard.py`

真实输出：

- `reports/phase3_loop161/loop161_test10k_promotion_margin_guard.json`
- `reports/phase3_loop161/loop161_test10k_promotion_margin_guard.md`

## 真实审计结果

闸门参数：

- Val 至少减少 `3` 个错误。
- Test-10k 至少减少 `3` 个错误。

| Candidate | Val delta errors | Test-10k delta errors | Decision |
|---|---:|---:|---|
| `loop151_trusted_signer_guard` | `-17` | `-5` | `allow_full_test_confirmation` |
| `loop144_union_trusted_signer` | `-12` | `+3` | `reject_test10k_margin_too_small` |
| `loop159_r11_only_trusted_signer` | `-7` | `0` | `reject_test10k_margin_too_small` |
| `loop160_lowprob_r11_gate` | `-3` | `-1` | `reject_test10k_margin_too_small` |

## 决策

Loop161 不改变当前 best；它改变的是后续 full-test 使用纪律。Loop151 这种 Val `-17`、Test-10k `-5` 的候选可以进入 full-test；Loop160 这种 Test-10k 只少 `1` 个错误的候选，以后应先停在 Test-10k，不再直接消耗 full-test。

这条规则也解释了当前路线为什么开始停滞：R11 类 recall rescue 在 Val 上很诱人，但到了 Test-10k 的收益不足够宽；如果继续把每个 0-1 个样本级收益都送 full-test，会把 full-test 变成调参反馈，这是必须避免的。

## 验证

```powershell
.\vnev\Scripts\python.exe scripts\build_loop161_test10k_promotion_margin_guard.py --loop151-val-eval reports\phase3_loop151\loop151_trusted_signer_guard_val_eval.json --loop151-test10k-eval reports\phase3_loop151\loop151_trusted_signer_guard_test10k_eval.json --loop144-val-eval reports\phase3_loop151\loop151_trusted_signer_guard_on_loop144_union_val_eval.json --loop144-test10k-eval reports\phase3_loop151\loop151_trusted_signer_guard_on_loop144_union_test10k_eval.json --loop159-audit reports\phase3_loop159\loop159_r11_only_candidate_audit.json --loop160-audit reports\phase3_loop160\loop160_lowprob_r11_gate_audit.json --output-json reports\phase3_loop161\loop161_test10k_promotion_margin_guard.json --output-md reports\phase3_loop161\loop161_test10k_promotion_margin_guard.md --min-val-error-improvement 3 --min-test10k-error-improvement 3

.\vnev\Scripts\python.exe -m pytest tests\test_build_loop161_test10k_promotion_margin_guard.py -q
```

结果：Loop161 `decision=guard_active`；单测 `1 passed`。
