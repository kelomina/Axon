# Phase 3 Loop154 Trusted Signer Threshold Tightening Report

更新时间：2026-07-08

## 目标

Loop151 trusted signer guard 在 full-test 上减少 `78` 个错误，但引入 `1` 个 FN。本轮做一个保守复验：不扩展 signer term，不使用 full-test 选择发布者，只把 `score_threshold` 从 `1.0` 收紧到 `0.995`。如果 Val 和 Test-10k 仍保持 Loop151 收益，而 full-test 避免那个新增 FN，就可以得到一个无新规则的微改进；如果三段完全等价，就把这条路关闭，避免继续盲扫阈值。

规则仍然只使用 Authenticode `Valid` 状态和冻结 signer subject term。路径、文件名、hash、`source_sha256`、`sample_index`、split、row order、模型分数和概率只用于对齐、审计或阈值门控，不作为身份捷径或 replacement 证据。

## 结果

`score_threshold=0.995` 与 Loop151 的 `1.0` 在 Val、Test-10k、full-test 三段预测完全一致：

| Split | F1 | Errors | FP | FN | Fixed FP | Introduced FN |
|---|---:|---:|---:|---:|---:|---:|
| Val | `0.9919193935` | `162` | `105` | `57` | `17` | `0` |
| Test-10k | `0.9921921922` | `78` | `49` | `29` | `5` | `0` |
| Full-test | `0.9908541911` | `1466` | `879` | `587` | `79` | `1` |

Equivalence audit:

| Split | Rows | Prediction diffs vs Loop151 |
|---|---:|---:|
| Val | `20000` | `0` |
| Test-10k | `10000` | `0` |
| Full-test | `160000` | `0` |

## 决策

Loop154 不替代 Loop151。它证明 `0.995` 收紧不会改变任何实际预测，也不能修掉 Loop151 的 full-test 新增 FN。当前 canonical strict best 仍是 Loop151 `score_threshold=1.0` 版本。

后续不要继续在 `0.995` 到 `1.0` 附近扫 signer score threshold。只有当出现新的 score 来源，或有外部/人工预批准的新 signer term list，才值得重新打开这一类实验，并且必须重新从 Val 开始。

## 产物

- `reports/phase3_loop154/loop154_trusted_signer_guard_t0995_val_eval.json`
- `reports/phase3_loop154/loop154_trusted_signer_guard_t0995_test10k_eval.json`
- `reports/phase3_loop154/loop154_trusted_signer_guard_t0995_full_eval.json`
- `reports/phase3_loop154/loop154_trusted_signer_guard_t0995_val_predictions.csv`
- `reports/phase3_loop154/loop154_trusted_signer_guard_t0995_test10k_predictions.csv`
- `reports/phase3_loop154/loop154_trusted_signer_guard_t0995_full_predictions.csv`

## 验证

```powershell
.\vnev\Scripts\python.exe scripts\evaluate_authenticode_trusted_signer_guard.py --predictions-csv reports\phase3_loop136\r5_oof_noise_pairwise_selector_recall_valonly\loop135_pairwise_selector_val_predictions.csv --signature-csv reports\phase3_loop151\loop151_loop136_authenticode_val_predpos_signatures.csv --score-column stage2_prob_malicious --score-threshold 0.995 --reference-errors 179 --min-error-improvement 10 --trusted-term "Microsoft Corporation" --trusted-term "Microsoft Windows" --trusted-term "Seagate Technology" --trusted-term "FinalWire" --trusted-term "NetEase" --trusted-term "Beijing Sogou" --trusted-term "Beijing Kingsoft" --trusted-term "Beijing Qihu" --trusted-term "Wondershare" --trusted-term "IObit" --trusted-term "Yozosoft" --trusted-term "Huya" --output-json reports\phase3_loop154\loop154_trusted_signer_guard_t0995_val_eval.json --output-predictions-csv reports\phase3_loop154\loop154_trusted_signer_guard_t0995_val_predictions.csv
```

同一冻结 term list 继续跑了 Test-10k 和 full-test。最终用 prediction diff 复核三段与 Loop151 输出完全一致。
