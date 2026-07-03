# Phase 3 Loop112 External Focus Verdict Pipeline

更新时间：2026-07-03

## 目的

Loop112 新增 `scripts/run_loop112_external_focus_verdict_pipeline.py`，把 Loop111 外部 focus 标注导入和 Loop110 严格 verdict pipeline 串成一个入口：

1. Loop111：只允许外部文件按 `blind_review_id` 写入 `manual_label_verdict/manual_verdict_note/recommended_action`
2. Loop109：导入后立即检查 focus 标注质量
3. Loop110：只有导入和 preflight 通过，才继续 merge、unblind、Loop87 import

这样外部标注文件如果混入 filename、path、hash、`source_sha256`、`sample_index`、split、model score、probability、threshold 等字段，或者 note 只引用这些伪证据，会在进入 merge/unblind 前被阻断。

## 不变的边界

Loop112 不训练、不评估模型、不调阈值、不加载 checkpoint、不打开 NPZ 数组、不采样 replacement、不修改 split/cache。它只是把外部证据入口串成单命令，减少人为绕过顺序门禁的机会。

文件名、路径、目录、后缀、hash、`source_sha256`、`sample_index`、split、row order 和模型分数仍然只能用于加载、对齐、cache audit、重复检测或盲审索引，不能作为模型证据、verdict 证据、feature mask、阈值/融合或 replacement sampling 依据。

## 真实 no-op 复验

输入：

- focus: `reports/random_20w_split/loop106_content_review_focus_top240.csv`
- external annotations: `reports/random_20w_split/loop111_external_annotations_noop.csv`
- full blinded: `reports/random_20w_split/loop96_full_queue_blinded_review.csv`
- private map: `reports/random_20w_split/loop96_full_queue_private_map.csv`

输出：

- `reports/random_20w_split/loop112_external_focus_pipeline_noop_summary.json`
- `reports/random_20w_split/loop112_external_focus_pipeline_noop/`

真实结果：

- external rows: `0`
- imported rows: `0`
- post-import actionable rows: `0`
- Loop87 rows: `1868`
- Loop87 actionable rows: `0`
- Loop87 replacement required rows: `0`
- Loop87 training policy rows: `0`
- decision: `ready_noop_no_actionable_verdicts`
- Train/Val、Test-10k、full-test 仍全部不授权

## 验证

资源守卫：

- `reports/random_20w_split/loop112_external_focus_pipeline_guard.json`
- `reports/random_20w_split/loop112_real_noop_pipeline_guard.json`

测试：

```powershell
.\vnev\Scripts\python.exe -m pytest tests/test_run_loop112_external_focus_verdict_pipeline.py tests/test_import_loop111_focus_external_annotations.py tests/test_run_loop110_focus_verdict_pipeline.py tests/test_preflight_loop106_focus_annotations.py -q
```

结果：`21 passed`。

覆盖点包括：

- no-op 外部文件会跑完整 Loop112，但不产生 actionable verdict
- 外部文件出现身份/模型字段时，Loop110 不运行
- 外部 note 只引用 filename / model score 时，Loop110 不运行
- 合法内容证据 verdict 可以进入 Loop87，但仍只进入 redraw preflight review-only 状态
- 全链路继续禁止训练、Test-10k 和 full-test

## 当前结论

Loop112 把外部 verdict 入口变成了唯一推荐路径。当前真实状态仍没有独立 actionable verdict，所以不能进入 redraw、训练、Test-10k 或 full-test。下一步只能收集独立内容/外部证据标注，再跑 Loop112；确认 bad-row verdict 后，才允许进入 quarantine + locked manifest 同原始标签 fresh redraw 的预检。
