# Phase 3 Loop159 R11-Only Candidate Audit

更新时间：2026-07-08

## 目标

Loop144 union + trusted signer 在 Val 上有 `150` errors，但 Test-10k 退化到 `81` errors。Loop159 把这个激进候选拆开，只保留 R11-filtered 分支，再叠加 Loop151 已冻结的 trusted signer guard，目的是确认“只走 R11 recall recovery”是否能保住泛化。

这不是重新训练，也不是阈值搜索。路径、hash、`source_sha256`、`sample_index` 和 split 只用于对齐；模型选择仍按 Val -> Test-10k -> frozen full-test 漏斗执行。

## 产物

新增脚本和测试：

- `scripts/build_loop159_r11_only_candidate_audit.py`
- `tests/test_build_loop159_r11_only_candidate_audit.py`

真实输出：

- `reports/phase3_loop159/loop159_r11_only_trusted_signer_val_eval.json`
- `reports/phase3_loop159/loop159_r11_only_trusted_signer_test10k_eval.json`
- `reports/phase3_loop159/loop159_r11_only_candidate_audit.json`
- `reports/phase3_loop159/loop159_r11_only_candidate_audit.md`

## 漏斗结果

| Split | Candidate F1 | Errors | FP | FN | Delta errors vs Loop151 | Delta FP/FN vs Loop151 |
|---|---:|---:|---:|---:|---:|---:|
| Val | `0.9922720247` | `155` | `106` | `49` | `-7` | `+1 / -8` |
| Test-10k | `0.9921968788` | `78` | `52` | `26` | `0` | `+3 / -3` |
| Full-test | `0.9907064008` | `1491` | `962` | `529` | `+25` | `+83 / -58` |

## 决策

Loop159 不替代 Loop151。它在 Val 上确实少 `7` 个错误，Test-10k 总错误不退化但也没有减少，最后在 frozen full-test 上比 Loop151 多 `25` 个错误，F1 更低。它的价值是暴露了一个清晰 trade-off：FN 少 `58`，但 FP 多 `83`。

因此 Loop159 只能保留为 high-recall trade-off 记录，不进入 strict best。当前 strict best 仍是 Loop151 trusted signer guard；如果业务明确愿意用更多 FP 换更少 FN，Loop159 可以作为后续“高召回模式”的候选之一重新按产品口径评审。

## 验证

```powershell
.\vnev\Scripts\python.exe scripts\evaluate_authenticode_trusted_signer_guard.py --predictions-csv reports\phase3_loop144\loop136_r11_filtered_valonly\loop135_pairwise_selector_val_predictions.csv --signature-csv reports\phase3_loop151\loop151_loop136_authenticode_val_predpos_signatures.csv --reference-errors 162 --min-error-improvement 1 --output-json reports\phase3_loop159\loop159_r11_only_trusted_signer_val_eval.json --output-predictions-csv reports\phase3_loop159\loop159_r11_only_trusted_signer_val_predictions.csv --trusted-term "Microsoft Corporation" --trusted-term "Microsoft Windows" --trusted-term "Seagate Technology" --trusted-term "FinalWire" --trusted-term "NetEase" --trusted-term "Beijing Sogou" --trusted-term "Beijing Kingsoft" --trusted-term "Beijing Qihu" --trusted-term "Wondershare" --trusted-term "IObit" --trusted-term "Yozosoft" --trusted-term "Huya"

.\vnev\Scripts\python.exe scripts\evaluate_authenticode_trusted_signer_guard.py --predictions-csv reports\phase3_loop144\loop136_r11_filtered_test10k_predictions.csv --signature-csv reports\phase3_loop151\loop151_loop136_authenticode_test10k_predpos_signatures.csv --reference-errors 78 --min-error-improvement 1 --output-json reports\phase3_loop159\loop159_r11_only_trusted_signer_test10k_eval.json --output-predictions-csv reports\phase3_loop159\loop159_r11_only_trusted_signer_test10k_predictions.csv --trusted-term "Microsoft Corporation" --trusted-term "Microsoft Windows" --trusted-term "Seagate Technology" --trusted-term "FinalWire" --trusted-term "NetEase" --trusted-term "Beijing Sogou" --trusted-term "Beijing Kingsoft" --trusted-term "Beijing Qihu" --trusted-term "Wondershare" --trusted-term "IObit" --trusted-term "Yozosoft" --trusted-term "Huya"

.\vnev\Scripts\python.exe scripts\build_loop159_r11_only_candidate_audit.py --val-eval-json reports\phase3_loop159\loop159_r11_only_trusted_signer_val_eval.json --test10k-eval-json reports\phase3_loop159\loop159_r11_only_trusted_signer_test10k_eval.json --full-eval-json reports\phase3_loop151\loop151_trusted_signer_guard_on_r11_filtered_full_eval.json --output-json reports\phase3_loop159\loop159_r11_only_candidate_audit.json --output-md reports\phase3_loop159\loop159_r11_only_candidate_audit.md

.\vnev\Scripts\python.exe -m pytest tests\test_build_loop159_r11_only_candidate_audit.py -q
```

结果：Loop159 audit `decision=reject_full_test_confirmation_not_strict_best`；单测 `1 passed`。
