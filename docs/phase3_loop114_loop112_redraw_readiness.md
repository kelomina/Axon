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
