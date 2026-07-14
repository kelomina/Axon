# Phase 3 Loop71: Target Gap and Noise Review ROI

日期：2026-07-03

## 目标

Loop71 不训练模型、不调阈值、不生成候选方案。它只回答一个业务决策问题：在当前 best full-test 结果下，要达到 `F1 >= 99.9%`，还需要消灭多少错误？现有噪声复核队列是否足够覆盖这个缺口？

结论先放前面：不够。

当前 best 仍是 Loop57。它在 16 万 full-test 上的 F1 是 `0.9883629658`，错误 `1868` 个，FP/FN 为 `1195 / 673`。即使按最有利情况优先修复 FN，要达到 `0.999` F1 也至少需要修复 `1708` 个错误，占当前错误的 `91.43%`。这意味着只靠小批量人工复核、薄阈值、同路线 stack 或 gate 微调，量级上都不够。

## 身份字段规则

Loop71 延续硬规则：`filename`、`path`、`extension`、`directory`、`source_sha256`、`cache_path`、`sample_index`、`split` 和行顺序只允许用于加载、对齐、缓存审计、重复内容复核和人工复核定位。它们不是模型证据，不能驱动阈值、自动改标、特征工程或生产推理。

这里尤其要避免一个误解：训练集目录名或文件名最多只是早期人工语料造册时的标签来源，就像把样本先放进“待拉黑/待加入白名单”的文件夹；这不是文件内容本身的恶意证据。实战文件命名和训练集命名不是同一分布，攻击者也能随时改名，所以模型只能学习字节、PE 结构、统计特征、证书、overlay 等内容证据。

## 实现

新增：

- `scripts/audit_loop71_target_gap_noise_roi.py`
- `tests/test_audit_loop71_target_gap_noise_roi.py`

真实命令：

```powershell
.\vnev\Scripts\python.exe scripts\audit_loop71_target_gap_noise_roi.py `
  --loop57-eval-json reports\random_20w_split\loop57_fn_overlay_gate_frozen_full_test_eval.json `
  --loop63-summary-json reports\random_20w_split\loop63_persistent_error_review_queue_summary.json `
  --loop63-queue-csv reports\random_20w_split\loop63_persistent_error_review_queue.csv `
  --loop65-summary-json reports\random_20w_split\loop65_A_lane_review_batch_summary.json `
  --loop65-batch-csv reports\random_20w_split\loop65_A_lane_review_batch.csv `
  --loop50-summary-json reports\random_20w_split\loop50_conflict_content_audit\loop50_conflict_content_audit_summary.json `
  --loop64-summary-json reports\random_20w_split\loop64_manifest_sha_duplicate_audit.json `
  --output-json reports\random_20w_split\loop71_target_gap_noise_roi.json `
  --target-f1 0.999
```

## 结果

Current best:

| Source | Samples | F1 | Errors | FP/FN |
| --- | ---: | ---: | ---: | ---: |
| Loop57 full-test | `160000` | `0.9883629658` | `1868` | `1195 / 673` |

Target gap:

| Target F1 | Minimum fixed errors, best case | Ratio of current errors | Remaining FP/FN, best case |
| ---: | ---: | ---: | ---: |
| `0.999` | `1708` | `91.43%` | `160 / 0` |

Review ROI:

| Review source | Rows | Best-case F1 if all rows confirmed/fixed | Remaining errors |
| --- | ---: | ---: | ---: |
| Loop65 selected batch | `62` | `0.9887535495` | `1806` |
| Loop63 A-lane high-conflict errors | `643` | `0.9923990941` | `1225` |
| Loop50 objective issues | `0` | `0.9883629658` | `1868` |
| Loop64 duplicate detail rows | `4` | `0.9883881739` | `1864` |

现有客观自动审计没有找到足够多的坏样本：Loop50 的 objective issue rows 是 `0`，Loop64 针对 focus queue 的重复明细只有 `4` 行。这说明噪声真实存在，但当前自动证据不足以支撑大规模自动改标或自动替换。

## 决策

1. 停止继续同路线的 score/gate/threshold 薄调。Loop70 已经证明即使 nested OOF 协议修正，同一组分数再堆一层也不能超过 Loop57。
2. 若继续挑战 `99.9%`，下一步必须转向两类更强证据：
   - 扩大人工/外部证据复核，优先覆盖 Loop63 的 `1868` 个 current-best 错误，尤其是高置信冲突；
   - 引入真正独立的新检测视角，而不是重排 Loop57/61/70 的同一组分数。
3. 一旦人工或外部证据确认 `label_wrong`、`feature_broken` 或 `out_of_scope`，只能触发 fresh same-original-label redraw。坏样本不补齐，坏样本只触发重新抽样；新 split 必须继续严格保持 `200000 = 20000 train / 20000 val / 160000 test`。

## Artifacts

- Summary: `reports/random_20w_split/loop71_target_gap_noise_roi.json`

Generated reports are not committed.

## 验证

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_audit_loop71_target_gap_noise_roi.py tests\test_identity_feature_guard.py -q
.\vnev\Scripts\python.exe -m py_compile scripts\audit_loop71_target_gap_noise_roi.py
```
