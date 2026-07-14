# Phase 3 Loop160 Low-Probability R11 Gate

更新时间：2026-07-08

## 目标

Loop159 证明“全量接受 R11 0->1 rescue”会变成高召回 trade-off：FN 少，但 FP 增加更多。Loop160 因此做一个更保守的 Val-only 门控：只在 Loop151 当前预测为 benign、R11 候选预测为 malicious 的行里，根据非身份概率列 `baseline_prob_malicious` 选择一个很低的阈值，试图只吃最稳的少量 FN rescue。

选择规则是预先固定的：只用 Val，找“满足 Val 至少减少 `3` 个错误且 FP 不增加”的最小阈值。路径、hash、`source_sha256`、`sample_index` 和 split 只用于对齐，不参与阈值选择或模型证据。

## 产物

新增脚本和测试：

- `scripts/build_loop160_lowprob_r11_gate.py`
- `tests/test_build_loop160_lowprob_r11_gate.py`

真实输出：

- `reports/phase3_loop160/loop160_lowprob_r11_gate_audit.json`
- `reports/phase3_loop160/loop160_lowprob_r11_gate_audit.md`
- `reports/phase3_loop160/loop160_lowprob_r11_val_predictions.csv`
- `reports/phase3_loop160/loop160_lowprob_r11_test10k_predictions.csv`
- `reports/phase3_loop160/loop160_lowprob_r11_full_predictions.csv`

## 漏斗结果

Val 选择出的阈值为 `baseline_prob_malicious <= 0.2487261742`。

| Split | Accepted rows | Correct / Wrong accepted | Errors | FP | FN | Delta errors vs Loop151 | Delta FP/FN vs Loop151 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Val | `3` | `3 / 0` | `159` | `105` | `54` | `-3` | `0 / -3` |
| Test-10k | `1` | `1 / 0` | `77` | `49` | `28` | `-1` | `0 / -1` |
| Full-test | `41` | `18 / 23` | `1471` | `902` | `569` | `+5` | `+23 / -18` |

## 决策

Loop160 通过 Val 和 Test-10k，但 frozen full-test 失败。它说明 Test-10k 上一个样本级的改善仍可能是抽样波动，不能作为替代 strict best 的充分证据。Loop151 仍是当前 strict best。

这条路线的科学价值是缩小了问题：低概率 R11 rescue 在 Val/Test-10k 上看起来很干净，但到 16 万 full-test 仍出现 FP 外溢。后续若继续 recall rescue，不能只靠概率阈值，必须引入更强的内容/外部证据来过滤那 `23` 个 full-test wrong accepted rows。

## 验证

```powershell
.\vnev\Scripts\python.exe scripts\build_loop160_lowprob_r11_gate.py --val-base-csv reports\phase3_loop151\loop151_trusted_signer_guard_val_predictions.csv --val-candidate-csv reports\phase3_loop159\loop159_r11_only_trusted_signer_val_predictions.csv --test10k-base-csv reports\phase3_loop151\loop151_trusted_signer_guard_test10k_predictions.csv --test10k-candidate-csv reports\phase3_loop159\loop159_r11_only_trusted_signer_test10k_predictions.csv --full-base-csv reports\phase3_loop151\loop151_trusted_signer_guard_full_predictions.csv --full-candidate-csv reports\phase3_loop151\loop151_trusted_signer_guard_on_r11_filtered_full_predictions.csv --output-dir reports\phase3_loop160 --output-json reports\phase3_loop160\loop160_lowprob_r11_gate_audit.json --output-md reports\phase3_loop160\loop160_lowprob_r11_gate_audit.md --min-val-error-improvement 3 --max-val-fp-delta 0

.\vnev\Scripts\python.exe -m pytest tests\test_build_loop160_lowprob_r11_gate.py -q
```

结果：Loop160 audit `decision=reject_full_test_confirmation`；单测 `1 passed`。
