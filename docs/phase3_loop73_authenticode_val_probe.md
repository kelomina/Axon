# Phase 3 Loop73: Authenticode Trust Val Probe

日期：2026-07-03

## 目标

Loop73 尝试一个真正不同的信息源：Windows Authenticode 签名/信任状态。它回答的问题是：对 Loop57 Val 中已经判恶意的样本，如果文件签名链是 `Valid`，能否作为保守 FP guard 把一部分预测降级为良性？

结论：不能作为自动规则进入 Test-10k。

最佳 Val-only 规则只把错误从 `147` 降到 `143`，净改善 `4` 个样本，低于 `>=10` errors 的 Test-10k 门槛；无分数限制地把所有 `Valid` 签名预测恶意样本降级，会把错误提高到 `176`。

## 协议

- 只使用 Loop57 Val predictions。
- 只对 Val 中 `prediction=1` 的行采集 Authenticode 状态，共 `10037` 行。
- 路径只用于打开文件读取签名状态，不执行样本。
- 不读取 Test-10k，不读取 full-test。
- 不训练模型，不自动改标，不改 split。
- `filename`、`path`、`extension`、`directory`、`source_sha256`、`cache_path`、`sample_index`、`split` 和行顺序只用于加载/对齐/审计，不是模型证据。
- `loop57_*` 分数只用于 Val 内规则扫描，不能从本轮结果反推 Test 阈值。

## 实现

新增：

- `scripts/probe_loop73_authenticode_val.py`
- `tests/test_probe_loop73_authenticode_val.py`

真实命令：

```powershell
.\vnev\Scripts\python.exe scripts\probe_loop73_authenticode_val.py `
  --predictions-csv reports\random_20w_split\loop57_fn_overlay_gate_valonly\loop57_fn_overlay_gate_val_predictions.csv `
  --build-signature-cache-csv reports\random_20w_split\loop73_authenticode_val_predpos_signatures.csv `
  --output-json reports\random_20w_split\loop73_authenticode_val_report.json `
  --output-predictions-csv reports\random_20w_split\loop73_authenticode_val_predictions.csv `
  --reference-val-errors 147 `
  --min-val-error-improvement 10
```

实现细节：PowerShell 子进程显式使用 Windows PowerShell module path，避免继承 PowerShell 7 module path 后 `Microsoft.PowerShell.Security` 类型数据冲突，导致 `Get-AuthenticodeSignature` 加载失败。

## 结果

Authenticode status on Val predicted-positive rows:

| Status | Rows |
| --- | ---: |
| `NotSigned` | `9204` |
| `HashMismatch` | `397` |
| `UnknownError` | `367` |
| `Valid` | `69` |

Baseline Loop57 Val:

| F1 | Errors | FP/FN |
| ---: | ---: | ---: |
| `0.9926635724` | `147` | `92 / 55` |

Best Val-only Authenticode rule:

| Rule | Threshold | F1 | Errors | FP/FN | Fixed FP | Introduced FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Valid` and `final_prob <= threshold` -> benign | `0.65` | `0.9928582131` | `143` | `83 / 60` | `9` | `5` |

Full `Valid` downgrade:

| Rule | F1 | Errors | FP/FN | Fixed FP | Introduced FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| all `Valid` predicted-positive -> benign | `0.9911858974` | `176` | `72 / 104` | `20` | `49` |

Decision:

- `reject_val_margin_too_small`
- No Test-10k.

## 解释

Valid Authenticode 签名确实能修掉一部分 FP，但也会误放一批真实恶意样本。当前 Val 里 `Valid` 且预测恶意的 `69` 行中，只有 `20` 行是 FP，`49` 行是 TP。换句话说，签名有效不是“良性”的充分证据，攻击者或灰产软件可以使用有效签名，正常签名链也不代表业务上安全。

因此 Authenticode 状态可以进入人工/外部证据复核上下文，例如帮助解释某些 FP 为什么值得优先查证；但不能作为自动降级规则接入模型，也不能进入 Test-10k。

## Artifacts

- Signature cache: `reports/random_20w_split/loop73_authenticode_val_predpos_signatures.csv`
- Report: `reports/random_20w_split/loop73_authenticode_val_report.json`
- Val predictions with probe columns: `reports/random_20w_split/loop73_authenticode_val_predictions.csv`

Generated reports are not committed.

## 验证

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_probe_loop73_authenticode_val.py tests\test_identity_feature_guard.py -q
.\vnev\Scripts\python.exe -m py_compile scripts\probe_loop73_authenticode_val.py
```
