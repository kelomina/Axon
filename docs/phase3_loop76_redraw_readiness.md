# Phase 3 Loop76: Redraw Readiness Orchestration Gate

日期：2026-07-03

## 目标

Loop76 把 Loop74/75 strict external verdict import、fresh same-original-label redraw、replacement integrity、cache readiness 和 Val-first 漏斗串成一个只读总门禁。

它不是模型候选，不训练、不评估、不扫阈值、不改 split、不提取 cache、不扫描原始数据。它只读取 JSON/CSV 元数据并回答一个问题：当前是否允许进入下一步。如果不允许，阻断原因是什么。

## 实现

新增：

- `scripts/build_loop76_redraw_readiness.py`
- `tests/test_build_loop76_redraw_readiness.py`

核心输出：

- `decision`
- `ready_for`
- `blocked_reasons`
- `next_step`
- `loop75_import`
- `redraw_requirements`
- `corrected_split_integrity`
- `cache_readiness`
- `val_first_policy`
- `memory_leak_profile`

Loop76 的 `memory_leak_profile` 明确为低风险只读元数据脚本：不加载模型、不用 CUDA、不读 NPZ 特征数组、不扫描 raw data。但运行前仍必须执行资源/静态泄漏守卫。

## 真实 no-op 复验

命令：

```powershell
.\vnev\Scripts\python.exe scripts\build_loop76_redraw_readiness.py `
  --strict-import-json reports\random_20w_split\loop75_empty_external_verdict_import.json `
  --adjustment-plan-json reports\random_20w_split\loop75_empty_external_adjustment_plan.json `
  --split-csv reports\random_20w_split\loop27_corrected_split.csv `
  --plan-csv reports\random_20w_split\loop75_empty_external_adjustment_plan.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --output-prefix reports\random_20w_split\loop76_noop `
  --output-json reports\random_20w_split\loop76_noop_redraw_readiness.json `
  --output-md reports\random_20w_split\loop76_noop_redraw_readiness.md
```

结果：

| Metric | Value |
| --- | ---: |
| Decision | `await_external_verdicts` |
| Next step | `no_redraw_required_until_actionable_verdicts` |
| Strict failures | `[]` |
| Review rows | `1868` |
| Sample-index matches | `1868` |
| Replacement required | `0` |
| Training policy rows | `0` |
| Ready for train/val | `false` |
| Ready for Test-10k | `false` |
| Ready for full-test | `false` |

这个结果是期望行为：Loop72 空 verdict 表没有任何 actionable evidence，因此不能重抽、不能训练、不能进入 Test-10k 或 full-test。当前唯一正确动作是等待外部/人工证据填回。

## 严格规则

- strict import 必须 `import_ready=true`，`invalid_rows=0`，`training_policy_rows=0`。
- Loop75 必须完整 `1868` 行，`sample_index_match_count=1868`。
- `manual_verdict_note` 不能是 identity-only 或 score-only 伪证据。
- adjustment plan 不能有 unresolved rows、duplicate review rows、unknown verdict rows 或 training-policy rows。
- replacement required 大于 `0` 时，必须先构建 candidate pool，且 `replacement_shortfall={}`。
- corrected split 必须仍是严格 `200000 = 20000/20000/160000`。
- replacement audit 和 cache readiness 的最终命令默认带 `--enforce-label-balance`。
- cache readiness 必须 `cache_ready=true`，`missing_rows=0`，`coverage_ratio=1.0`。
- 只有 corrected split、replacement integrity 和 cache readiness 全部通过后，才允许回到 Train/Val 漏斗。
- Test-10k 仍只允许在 Val 明显优于 baseline 后进入；full-test 只允许在冻结 Test-10k 通过后进入。

## 验证

执行前已做资源/静态泄漏守卫，确认无 Python 训练进程，且相关脚本无模型/CUDA/NPZ 大读/并发池/无限循环风险。

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\build_loop76_redraw_readiness.py scripts\import_loop72_external_verdicts.py scripts\build_corrected_split_from_plan.py scripts\audit_corrected_split_replacements.py scripts\audit_corrected_split_cache_ready.py scripts\build_corrected_split_cache_recovery_plan.py
.\vnev\Scripts\python.exe -m pytest tests\test_build_loop76_redraw_readiness.py tests\test_import_loop72_external_verdicts.py tests\test_build_corrected_split_from_plan.py tests\test_audit_corrected_split_replacements.py tests\test_audit_corrected_split_cache_ready.py tests\test_build_corrected_split_cache_recovery_plan.py -q
```

结果：`43 passed`。

Generated reports are not committed.
