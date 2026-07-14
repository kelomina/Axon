# Phase 3 Loop78: 1% Cache Sample Integrity Audit

日期：2026-07-03

## 目标

Loop78 把“20w cache 必须抽样审计”落成可重复脚本。它是只读审计，不训练、不评估、不扫阈值、不改 cache，也不把路径、hash 或 sample_index 当成模型证据。

新增：

- `scripts/audit_loop78_cache_sample_integrity.py`
- `tests/test_audit_loop78_cache_sample_integrity.py`

审计内容：

- strict split shape：`200000 = 20000 train + 20000 val + 160000 test`
- strict label balance：train/val/test 内部 `0/1` 平衡
- deterministic 1% sample：默认 `sample_fraction=0.01`
- manifest 对齐
- cache 文件存在
- NPZ 必需字段存在
- `byte_sequence / pe_features / stat_features / lightweight_features` shape
- label 与 split/manifest/cache 一致
- NPZ `source_sha256` 与 manifest 一致
- PE/stat/lightweight 数值 finite

## 真实 1% 审计

运行前使用 Loop77 guard，显式允许 `npz_array_load`，因为 Loop78 的目的就是抽样打开 NPZ：

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\audit_loop78_cache_sample_integrity.py `
  --output-json reports\random_20w_split\loop78_real_audit_guard.json `
  --allow-risk npz_array_load
```

审计命令：

```powershell
.\vnev\Scripts\python.exe scripts\audit_loop78_cache_sample_integrity.py `
  --split-csv reports\random_20w_split\loop27_corrected_split.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --sample-fraction 0.01 `
  --seed 7801 `
  --output-json reports\random_20w_split\loop78_cache_sample_integrity_1pct.json `
  --detail-output-csv reports\random_20w_split\loop78_cache_sample_integrity_1pct_detail.csv
```

结果：

| Metric | Value |
| --- | ---: |
| Guard ready | `true` |
| Sampled rows | `2000` |
| Sample split counts | `train=200 / val=200 / test=1600` |
| Sample label counts | `0=1000 / 1=1000` |
| Manifest matches | `2000` |
| NPZ/cache pass rows | `2000` |
| Failed rows | `0` |
| Audit ready | `true` |

Declared shapes:

| Field | Shape |
| --- | ---: |
| `byte_sequence` | `8192` |
| `pe_features` | `256` |
| `stat_features` | `49` |
| `lightweight_features` | `256` |

## 解释

Loop78 证明 sampled cache 的内部完整性：这 2000 个抽样缓存文件都能打开，字段完整，shape 正确，标签/source SHA 一致，且数值有限。

它不能替代全量 coverage/readiness。当前历史事实仍是 fixed-v2 manifest 覆盖 `199870/200000`，剩余 `130` 行来自严格 PE 提取失败。训练或 corrected split 前仍必须运行对应的全量 cache readiness gate。

## 验证

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_audit_loop78_cache_sample_integrity.py -q
```

结果：`6 passed`。

Generated reports are not committed.
