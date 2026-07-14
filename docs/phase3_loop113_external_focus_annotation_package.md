# Phase 3 Loop113 External Focus Annotation Package

更新时间：2026-07-03

## 目的

Loop113 新增 `scripts/export_loop113_external_focus_annotation_package.py`，把 Loop106 focus 表导出成适合外部内容复核的安全包：

- `context_csv`：只含 `blind_review_id` 和内容派生字段
- `annotation_template_csv`：只含 Loop111 允许的四列表头
- `reviewer_guide_json`：说明允许的 verdict/action 和禁止证据

这一步解决一个实际风险：Loop106 focus 表虽然已经盲化，但仍带有 `loop106_focus_rank` 和 `loop106_focus_score`。这些字段可以做内部排序，却不能作为外部 verdict 证据。Loop113 导出的 context 包会移除 rank/score，只保留内容上下文。

## 安全边界

Loop113 不读 private map、不 unblind、不训练、不调阈值、不加载模型、不打开 NPZ 数组、不采样 replacement、不修改 split/cache。

导出的 context 不包含：

- filename / path / directory / extension
- hash / `source_sha256` / `sample_index` / split / row order
- model score / probability / prediction / threshold
- `loop57` / `loop39` 字段
- `loop106_focus_rank` / `loop106_focus_score`
- manual verdict 字段

外部提交文件仍必须走 Loop111/112；Loop113 不接受、生成或推断任何 verdict。

## 真实导出结果

输入：

- `reports/random_20w_split/loop106_content_review_focus_top240.csv`

输出：

- `reports/random_20w_split/loop113_external_focus_context.csv`
- `reports/random_20w_split/loop113_external_annotation_template.csv`
- `reports/random_20w_split/loop113_external_reviewer_guide.json`
- `reports/random_20w_split/loop113_external_focus_package_summary.json`

真实结果：

- rows: `240`
- label counts: `0=207`, `1=33`
- context field count: `25`
- forbidden focus columns: `[]`
- context header violations: `[]`
- context value violation count: `0`
- decision: `ready_for_external_content_annotation`

随后用 Loop113 的 header-only 模板跑 Loop112 no-op：

- `reports/random_20w_split/loop113_to_loop112_noop_summary.json`

结果仍然：

- external rows: `0`
- imported rows: `0`
- Loop87 rows: `1868`
- Loop87 actionable rows: `0`
- replacement required rows: `0`
- training policy rows: `0`
- Train/Val、Test-10k、full-test 均不授权

## 验证

资源守卫：

- `reports/random_20w_split/loop113_external_focus_package_guard.json`
- `reports/random_20w_split/loop113_to_loop112_noop_guard.json`

测试：

```powershell
.\vnev\Scripts\python.exe -m pytest tests/test_export_loop113_external_focus_annotation_package.py tests/test_run_loop112_external_focus_verdict_pipeline.py tests/test_import_loop111_focus_external_annotations.py tests/test_run_loop110_focus_verdict_pipeline.py tests/test_preflight_loop106_focus_annotations.py -q
```

结果：`25 passed`。

覆盖点包括：

- context 包不导出 rank/score、身份字段、模型字段或 manual 字段
- annotation template 只有 Loop111 允许的四列表头
- focus 输入混入身份或模型列会阻断
- context 内容值里出现身份/模型词会阻断
- duplicate `blind_review_id` 会阻断
- Loop113 模板接 Loop112 仍保持 no-op，不产生 actionable verdict

## 当前结论

Loop113 让外部证据收集有了安全输入包，但它仍不产生 verdict。当前合法下一步仍是：在 `loop113_external_annotation_template.csv` 的格式基础上填入独立内容/外部证据 verdict，然后跑 Loop112。只有 Loop87 接受 actionable bad-row verdict 后，才允许进入 quarantine + 同原始标签 fresh redraw 的后续预检。
