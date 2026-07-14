# Phase 3 Loop114 Loop112 Redraw Readiness Bridge

更新时间：2026-07-03

## 目的

Loop114 新增 `scripts/run_loop114_loop112_redraw_readiness.py`，把 Loop112 的严格外部 focus verdict 输出接到 Loop76 redraw readiness。

它只做三件事：

1. 读取 Loop112 summary，定位 Loop110 summary、Loop87 validated CSV 和 Loop87 import JSON。
2. 将 Loop87 中已通过内容/外部证据验证的坏行转换成严格 adjustment plan。
3. 调用 Loop76 readiness，判断下一步是否可以进入同原始标签 fresh redraw。

## 证据边界

文件名、路径、目录、后缀、`source_sha256`、`sample_index`、split、row order 和模型分数只允许用于加载、对齐、重复检测、审计或排队优先级。

它们不能作为：

- verdict 证据
- 模型证据
- 特征选择证据
- 阈值选择证据
- 生产推理输入

可执行 verdict 必须来自内容或外部证据，例如 PE 解析、NPZ/特征异常、header/section/import/export/resource/overlay/entropy/signature/certificate、sandbox、VT、多引擎、YARA 等，并且必须写入 `manual_verdict_note`。

## 重抽规则

Loop114 对正式 20w 协议采用 full-error data-governance 策略：

- `label_wrong`
- `feature_broken`
- `out_of_scope`
- 其它 Loop87 判定为需要 replacement 的坏行

都会统一生成：

- `plan_action=exclude_and_replace`
- `replacement_required=true`
- `replacement_label=<original locked split label>`
- `usable_for_training_policy=false`

也就是说，坏行只会被隔离，然后从锁定候选池重新抽一个同原始标签的有效样本。Loop114 不做直接 relabel，不允许坏样本自填补数，也不采样 replacement。

## 真实 no-op 复验

输入：

- Loop112 summary: `reports/random_20w_split/loop113_to_loop112_noop_summary.json`
- split: `reports/random_20w_split/random_20w_split.csv`

输出：

- `reports/random_20w_split/loop114_loop112_redraw_readiness_noop_summary.json`
- `reports/random_20w_split/loop114_loop112_redraw_readiness_noop.md`
- `reports/random_20w_split/loop114_loop112_redraw_readiness_noop/loop114_strict_adjustment_plan.csv`
- `reports/random_20w_split/loop114_loop112_redraw_readiness_noop/loop114_strict_adjustment_plan.json`
- `reports/random_20w_split/loop114_loop112_redraw_readiness_noop/loop114_loop76_readiness.json`
- `reports/random_20w_split/loop114_loop112_redraw_readiness_noop/loop114_loop76_readiness.md`

真实结果：

- Loop87 rows: `1868`
- Loop112 actionable rows: `0`
- replacement_required: `0`
- training_policy_rows: `0`
- decision: `await_external_verdicts`
- Train/Val、Test-10k、full-test 均不授权

## 验证

资源守卫：

```powershell
.\vnev\Scripts\python.exe scripts/pre_run_resource_leak_guard.py --target-script scripts/run_loop114_loop112_redraw_readiness.py --target-script tests/test_run_loop114_loop112_redraw_readiness.py --output-json reports/random_20w_split/loop114_redraw_readiness_guard.json
```

结果：`guard_ready=true`。

回归测试：

```powershell
.\vnev\Scripts\python.exe -m pytest tests/test_run_loop114_loop112_redraw_readiness.py tests/test_build_loop76_redraw_readiness.py tests/test_run_loop112_external_focus_verdict_pipeline.py tests/test_import_loop87_review_evidence_verdicts.py -q
```

结果：`30 passed`。

覆盖点包括：

- no-op verdict 不授权训练或测试
- `feature_broken` 生成 fresh same-label replacement request
- `label_wrong` 不会生成 direct relabel
- Loop112 上游阻断时 Loop114 不进入 Loop76
- 身份字段或模型分数作为唯一 evidence 时阻断

## 当前结论

Loop114 已经把“这些文件不合格就重新抽，而不是拿这些样本补齐”的规则固化成只读门控。

当前真实状态仍没有独立 actionable verdict，因此不能进入 redraw、训练、Test-10k 或 full-test。下一步仍是收集独立内容/外部证据标注，再跑 Loop112 和 Loop114；只有出现 confirmed bad row，才允许进入同原始标签 fresh redraw 候选池构建。

## Loop115 下游 hash 严格化

Loop115 补强了 Loop114 之后的 corrected split 链路：

- `build_corrected_split_from_plan.py` 输出 corrected split 时保留 `source_sha256`
- replacement row 会从 candidate pool 带入内容 hash
- `audit_corrected_split_replacements.py` 的 detail 输出包含 `source_sha256`
- `audit_corrected_split_cache_ready.py` 在 metadata validation 开启时，若 corrected split 行缺少有效 `source_sha256`，会产生 `split_missing_source_sha256` 或 `split_invalid_source_sha256` 并阻断 cache_ready

这里的 hash 仍然不是恶意/良性证据，只是为了证明 split、manifest 和 NPZ 指向同一个实际内容，避免路径或文件名变化导致误对齐。

验证：

```powershell
.\vnev\Scripts\python.exe scripts/pre_run_resource_leak_guard.py --target-script scripts/build_corrected_split_from_plan.py --target-script scripts/audit_corrected_split_replacements.py --target-script scripts/audit_corrected_split_cache_ready.py --target-script tests/test_build_corrected_split_from_plan.py --target-script tests/test_audit_corrected_split_replacements.py --target-script tests/test_audit_corrected_split_cache_ready.py --output-json reports/random_20w_split/loop115_hash_strict_redraw_guard.json --allow-risk npz_array_load
```

结果：`guard_ready=true`。

```powershell
.\vnev\Scripts\python.exe -m pytest tests/test_build_corrected_split_from_plan.py tests/test_audit_corrected_split_replacements.py tests/test_audit_corrected_split_cache_ready.py tests/test_run_loop114_loop112_redraw_readiness.py tests/test_build_loop76_redraw_readiness.py -q
```

结果：`53 passed`。

真实 no-op 复跑：

- `reports/random_20w_split/loop115_loop112_redraw_readiness_noop_summary.json`
- decision: `await_external_verdicts`
- replacement_required: `0`
- Train/Val、Test-10k、full-test 仍全部不授权

## Loop116 hash-first redraw e2e

Loop116 新增 `tests/test_redraw_hash_e2e.py`，把下游链路用同一个 replacement candidate 串起来验证：

1. `build_corrected_split_from_plan.py` 从 candidate CSV 选择 fresh replacement，并把 candidate 的 `source_sha256` 写入 corrected split
2. `audit_corrected_split_replacements.py` 识别该行是 fresh replacement，并在 detail CSV 中保留 `source_sha256`
3. `audit_corrected_split_cache_ready.py` 通过 `source_sha256` 命中 manifest，并确认 split、manifest、NPZ 三者 hash 一致

同时覆盖两个负例：

- corrected split 的 replacement 行 hash 被清空：即使路径能匹配 manifest，也会因 `split_missing_source_sha256` 阻断
- corrected split 的 replacement 行 hash 漂移：即使路径能匹配 manifest，也会因 `source_sha256_mismatch_split_manifest` 阻断

验证：

```powershell
.\vnev\Scripts\python.exe scripts/pre_run_resource_leak_guard.py --target-script tests/test_redraw_hash_e2e.py --target-script scripts/build_corrected_split_from_plan.py --target-script scripts/audit_corrected_split_replacements.py --target-script scripts/audit_corrected_split_cache_ready.py --output-json reports/random_20w_split/loop116_redraw_hash_e2e_guard.json --allow-risk npz_array_load
```

结果：`guard_ready=true`。

```powershell
.\vnev\Scripts\python.exe -m pytest tests/test_redraw_hash_e2e.py tests/test_build_corrected_split_from_plan.py tests/test_audit_corrected_split_replacements.py tests/test_audit_corrected_split_cache_ready.py tests/test_run_loop114_loop112_redraw_readiness.py tests/test_build_loop76_redraw_readiness.py -q
```

结果：`56 passed`。

真实 no-op 复跑：

- `reports/random_20w_split/loop116_loop112_redraw_readiness_noop_summary.json`
- decision: `await_external_verdicts`
- replacement_required: `0`
- Train/Val、Test-10k、full-test 仍全部不授权

## Loop117 cache recovery hash-first

Loop117 继续补强 corrected split 后的 cache recovery 环节：

- `audit_corrected_split_cache_ready.py` 写出的 missing-cache CSV 现在包含 `source_sha256`
- `build_corrected_split_cache_recovery_plan.py` 汇总 `missing_source_sha256_rows`、`invalid_source_sha256_rows` 和 `recovery_input_ready`
- `recover_missing_feature_cache.py` 在实际恢复前会计算源文件内容 hash，并与 missing CSV 的 expected `source_sha256` 比对
- 如果 expected hash 缺失，返回 `missing_expected_source_sha256`
- 如果路径对应文件的内容 hash 与 expected hash 不一致，返回 `source_sha256_mismatch`，并且不写 cache
- 文件 hash 分块读取从 `while True` 改为 `iter(lambda: read(...), b"")`，避免资源守卫把它识别成潜在无限循环
- cache recovery plan 不再输出后缀分布这类命名派生统计，恢复决策只看缺失行、split/label 物流字段、缺失原因和内容 hash 完整性

这保证 cache recovery 不能只因为路径存在就恢复缓存，仍必须证明它是 corrected split 中那一个实际内容。

验证：

```powershell
.\vnev\Scripts\python.exe scripts/pre_run_resource_leak_guard.py --target-script scripts/audit_corrected_split_cache_ready.py --target-script scripts/build_corrected_split_cache_recovery_plan.py --target-script scripts/recover_missing_feature_cache.py --target-script tests/test_build_corrected_split_cache_recovery_plan.py --target-script tests/test_recover_missing_feature_cache.py --target-script tests/test_audit_corrected_split_cache_ready.py --output-json reports/random_20w_split/loop117_cache_recovery_hash_guard.json --allow-risk npz_array_load --allow-risk process_pool --allow-risk thread_pool --allow-risk torch_import
```

结果：`guard_ready=true`。

```powershell
.\vnev\Scripts\python.exe -m pytest tests/test_build_corrected_split_cache_recovery_plan.py tests/test_recover_missing_feature_cache.py tests/test_audit_corrected_split_cache_ready.py tests/test_redraw_hash_e2e.py tests/test_run_loop114_loop112_redraw_readiness.py tests/test_build_loop76_redraw_readiness.py -q
```

结果：`43 passed`。

真实 no-op 复跑：

- `reports/random_20w_split/loop117_loop112_redraw_readiness_noop_summary.json`
- decision: `await_external_verdicts`
- replacement_required: `0`
- Train/Val、Test-10k、full-test 仍全部不授权

## Loop118 strict split metadata gate

Loop118 把“不能根据命名训练/判定”先固化为可独立执行的 split/cache metadata 审计 gate：

- 新增 `scripts/audit_strict_split_metadata.py`
- split CSV 必须同时包含 `label` 和 `source_sha256`
- split CSV 的 `label` 必须与 cache manifest 中的显式标签一致
- split CSV 的 `source_sha256` 必须与 cache manifest 中的内容 hash 一致
- 默认校验 NPZ metadata，要求 NPZ 中的 `label/source_sha256` 与 split/manifest 一致
- `source_path` 只用于定位和报告，不作为通过条件；hash 也只用于内容身份一致性，不是恶意/良性证据

路径仍只用于“split 行和 dataset 行的对齐”，不是标签证据。文件名、目录、后缀、路径文本不参与模型输入、verdict、阈值、feature mask 或训练标签决策。

当前工作树中还验证了训练/评估入口 strict gate，但 `scripts/main.py` 和 `src/dataset.py` 在本轮前已有大量历史未提交改动；为避免混入无关改动，本次先提交独立审计 gate。入口层 gate 后续应在清洁基线中单独提交。

验证：

```powershell
.\vnev\Scripts\python.exe scripts/pre_run_resource_leak_guard.py --target-script src/dataset.py --target-script scripts/main.py --target-script tests/test_split_file_dataset.py --target-script tests/test_security_hardening.py --target-script tests/test_training_seed.py --target-script tests/test_audit_corrected_split_cache_ready.py --target-script tests/test_redraw_hash_e2e.py --output-json reports/random_20w_split/loop118_strict_split_metadata_guard.json --allow-risk npz_array_load --allow-risk process_pool --allow-risk thread_pool --allow-risk torch_import --allow-risk torch_dataloader --allow-risk cuda_usage
```

结果：`guard_ready=true`。

```powershell
.\vnev\Scripts\python.exe scripts/pre_run_resource_leak_guard.py --target-script scripts/audit_strict_split_metadata.py --target-script tests/test_audit_strict_split_metadata.py --output-json reports/random_20w_split/loop118_strict_split_metadata_tool_guard.json --allow-risk npz_array_load --allow-risk torch_import
```

结果：`guard_ready=true`。

```powershell
.\vnev\Scripts\python.exe -m pytest tests/test_audit_strict_split_metadata.py tests/test_split_file_dataset.py tests/test_security_hardening.py tests/test_training_seed.py tests/test_audit_corrected_split_cache_ready.py tests/test_redraw_hash_e2e.py tests/test_run_loop114_loop112_redraw_readiness.py tests/test_build_loop76_redraw_readiness.py -q
```

结果：`62 passed`。

真实 no-op 复跑：

- `reports/random_20w_split/loop118_loop112_redraw_readiness_noop_summary.json`
- decision: `await_external_verdicts`
- replacement_required: `0`
- Train/Val、Test-10k、full-test 仍全部不授权

## Loop119 split metadata gate wired into readiness

Loop119 把 Loop118 的独立 split/cache metadata 审计接入 Loop76/Loop114 readiness：

- Loop76 新增可选 `--split-metadata-json`
- corrected split 和 cache readiness 都通过后，仍必须先提供 `audit_strict_split_metadata.py` 的结果
- split metadata audit 必须满足：
  - `audit_ready=true`
  - `validate_npz=true`
  - `expect_20w=true`
  - `row_issue_count=0`
  - `metadata_issue_counts={}`
  - `shape_failures=[]`
- 缺少该审计时，Loop76 返回 `needs_strict_split_metadata_audit`
- 审计失败时，Loop76 返回 `blocked_split_metadata`
- 只有 split/cache 显式 label+hash 一致性审计也通过，才允许进入 `ready_for_val_first_reverification`
- Test-10k 和 full-test 仍保持关闭，必须继续遵守 Val-first 漏斗

验证：

```powershell
.\vnev\Scripts\python.exe scripts/pre_run_resource_leak_guard.py --target-script scripts/build_loop76_redraw_readiness.py --target-script scripts/run_loop114_loop112_redraw_readiness.py --target-script scripts/audit_strict_split_metadata.py --target-script tests/test_build_loop76_redraw_readiness.py --target-script tests/test_audit_strict_split_metadata.py --output-json reports/random_20w_split/loop119_split_metadata_readiness_guard.json --allow-risk npz_array_load --allow-risk torch_import
```

结果：`guard_ready=true`。

```powershell
.\vnev\Scripts\python.exe -m pytest tests/test_run_loop114_loop112_redraw_readiness.py tests/test_build_loop76_redraw_readiness.py tests/test_audit_strict_split_metadata.py tests/test_audit_corrected_split_cache_ready.py tests/test_redraw_hash_e2e.py -q
```

结果：`42 passed`。

真实 no-op 复跑：

- `reports/random_20w_split/loop119_loop112_redraw_readiness_noop_summary.json`
- decision: `await_external_verdicts`
- ready_for.split_metadata_audit: `false`
- Train/Val、Test-10k、full-test 仍全部不授权
