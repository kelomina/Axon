# Phase 3 Loop162 Loop160 Failure Posthoc

更新时间：2026-07-08

## 目标

Loop160 在 Val 和 Test-10k 上都看起来干净，但 full-test 反转。Loop162 只做失败归因：统计 Loop160 实际接受的 R11 rescue 行在 Val、Test-10k、full-test 上到底有多少是正确 rescue、多少是错误 FP 外溢。

这份报告是 posthoc 诊断，不是选择信号。full-test 行不能用于调阈值、扩展 signer term、训练 selector、采样 replacement 或生产推理规则。公开 CSV 使用合成 `loop162_focus_id` 和分桶字段；真实路径、hash、`source_sha256`、`sample_index` 只保存在 private map 里用于审计。

## 产物

新增脚本和测试：

- `scripts/build_loop162_loop160_failure_posthoc.py`
- `tests/test_build_loop162_loop160_failure_posthoc.py`

真实输出：

- `reports/phase3_loop162/loop162_loop160_failure_posthoc.json`
- `reports/phase3_loop162/loop162_loop160_failure_posthoc.md`
- `reports/phase3_loop162/loop162_loop160_failure_posthoc_public.csv`
- `reports/phase3_loop162/loop162_loop160_failure_posthoc_private_map.csv`

## 真实归因

| Split | Rows | Accepted | Correct | Wrong | Label counts | Score buckets |
|---|---:|---:|---:|---:|---|---|
| Val | `20000` | `3` | `3` | `0` | `1=3` | `(0.20,0.25]=3` |
| Test-10k | `10000` | `1` | `1` | `0` | `1=1` | `(0.20,0.25]=1` |
| Full-test | `160000` | `41` | `18` | `23` | `0=23, 1=18` | `(0.10,0.20]=1, (0.20,0.25]=40` |

关键结论：Val/Test-10k 接受样本太少，无法估计 full-test 的 FP 外溢风险。到了 16 万 full-test，错误接受行 `23` 多于正确 rescue 行 `18`，这就是 Loop160 总错误反转的直接原因。

## 决策

Loop162 不改变当前 best，也不授权任何新规则。它只说明一件事：继续做 R11 rescue 不能只靠低概率阈值，因为在同一个低分桶里，full-test 的 wrong accepted 和 correct accepted 混在一起。后续如果还要救 FN，必须先引入独立的 Val-side 内容证据或外部证据，不能把这份 full-test posthoc 反向用于筛选。

## 验证

```powershell
.\vnev\Scripts\python.exe scripts\build_loop162_loop160_failure_posthoc.py --val-predictions-csv reports\phase3_loop160\loop160_lowprob_r11_val_predictions.csv --test10k-predictions-csv reports\phase3_loop160\loop160_lowprob_r11_test10k_predictions.csv --full-predictions-csv reports\phase3_loop160\loop160_lowprob_r11_full_predictions.csv --output-json reports\phase3_loop162\loop162_loop160_failure_posthoc.json --output-public-csv reports\phase3_loop162\loop162_loop160_failure_posthoc_public.csv --output-private-map-csv reports\phase3_loop162\loop162_loop160_failure_posthoc_private_map.csv --output-md reports\phase3_loop162\loop162_loop160_failure_posthoc.md

.\vnev\Scripts\python.exe -m pytest tests\test_build_loop162_loop160_failure_posthoc.py -q
```

结果：Loop162 `decision=posthoc_failure_record_only`；单测 `1 passed`。
