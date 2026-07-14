# Phase 3 Loop151 Trusted Signer Guard Report

更新时间：2026-07-08

## 目标

Loop150 证明继续调 32768-byte long-context 神经候选收益很低，本轮改走真正正交的外部证据：Windows Authenticode trust status 与 signer certificate subject。路径、文件名、目录、后缀、hash、`source_sha256`、`sample_index`、split 和 row order 只用于打开文件、对齐和审计，不作为模型证据。

本轮新增一个冻结规则评估器：

- `scripts/evaluate_authenticode_trusted_signer_guard.py`
- `tests/test_evaluate_authenticode_trusted_signer_guard.py`

规则语义：默认保留 Loop136；只有当当前预测为 malicious、`Get-AuthenticodeSignature` 返回 `Valid`，且 signer subject 命中预声明 trusted publisher term 时，才把预测降级为 benign。该规则不改标签、不改 split/cache、不训练、不使用 full-test 选择 signer term。

## Trusted Terms

本轮冻结的 signer terms：

`Microsoft Corporation`, `Microsoft Windows`, `Seagate Technology`, `FinalWire`, `NetEase`, `Beijing Sogou`, `Beijing Kingsoft`, `Beijing Qihu`, `Wondershare`, `IObit`, `Yozosoft`, `Huya`

这些是外部证书 subject 中的发布者文本，不是文件名或路径文本。后续如果扩展 term list，必须重新从 Val 开始走完整漏斗，不能根据 full-test 结果追加。

## 漏斗结果

| Split | Model | F1 | Errors | FP | FN | Decision |
|---|---|---:|---:|---:|---:|---|
| Val | Loop136 baseline | `0.9910789933` | `179` | `122` | `57` | baseline |
| Val | Trusted signer guard | `0.9919193935` | `162` | `105` | `57` | enter Test-10k |
| Test-10k | Loop136 baseline | `0.9916958479` | `83` | `54` | `29` | baseline |
| Test-10k | Trusted signer guard | `0.9921921922` | `78` | `49` | `29` | enter full-test |
| Full-test | Loop136 baseline | `0.9903723842` | `1544` | `958` | `586` | previous strict best |
| Full-test | Trusted signer guard | `0.9908541911` | `1466` | `879` | `587` | new strict best |

Full-test 相比 Loop136 净少 `78` 个错误：修复 `79` 个 FP，同时引入 `1` 个 FN。该规则是明显的 precision-side 改进，不是 99.9% 级别突破。

## Negative Controls

本轮同时复验了几个“看起来也许能更激进”的组合，均拒绝：

| Candidate | Gate | Errors | FP | FN | Decision |
|---|---|---:|---:|---:|---|
| Authenticode valid-status only on Loop136 | Val | `179` | `122` | `57` | no improvement |
| Cert-OOF frozen candidate | Test-10k | `134` | `49` | `85` | reject |
| Loop136 vs Cert-OOF pairwise | Test-10k | `84` | `55` | `29` | reject |
| R11 + fixed + cert union | Test-10k | `85` | `59` | `26` | reject |
| Trusted signer guard on Loop144 union | Test-10k | `81` | `55` | `26` | reject vs trusted Loop136 |

这说明真正有效的是“已验证发布者的保守 FP guard”，不是证书 blob 小特征、valid-status 粗规则，或再叠加 R11 recall union。

## Artifacts

- Val signature cache：`reports/phase3_loop151/loop151_loop136_authenticode_val_predpos_signatures.csv`
- Val eval：`reports/phase3_loop151/loop151_trusted_signer_guard_val_eval.json`
- Test-10k signature cache：`reports/phase3_loop151/loop151_loop136_authenticode_test10k_predpos_signatures.csv`
- Test-10k eval：`reports/phase3_loop151/loop151_trusted_signer_guard_test10k_eval.json`
- Full signature cache：`reports/phase3_loop151/loop151_loop136_authenticode_full_predpos_signatures.csv`
- Full eval：`reports/phase3_loop151/loop151_trusted_signer_guard_full_eval.json`
- Full predictions：`reports/phase3_loop151/loop151_trusted_signer_guard_full_predictions.csv`

## Verification

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_evaluate_authenticode_trusted_signer_guard.py tests\test_probe_loop73_authenticode_val.py -q
```

结果：`4 passed`。

## Decision

Adopt Loop151 trusted signer guard as the current strict best by full-test F1/errors, with the business caveat that it trades `79` fewer FP for `1` additional FN. If recall is the dominant product risk, keep Loop136 as fallback; if F1 / total error is the objective, Loop151 is ahead.

下一步仍然不是继续根据 full-test signer subject 加词，而是：

1. 把 trusted term list 冻结，做生产前安全 review。
2. 继续推进 Loop150 Val `86` 行高冲突复核，确认坏行后 same-original-label fresh redraw。
3. 若要继续冲击 99.9%，必须引入更强的外部信誉、多引擎、行为沙箱或动态证据；当前静态 + signer guard 仍剩 `1466 / 160000` full-test errors。
