# Phase 3 Loop111 Focus External Annotation Import

更新时间：2026-07-03

## 目的

Loop111 新增 `scripts/import_loop111_focus_external_annotations.py`，用于把外部或人工复核结果受控导入 Loop106 focus 表。它解决的是“外部 reviewer 不能直接改任意 CSV、也不能把文件名/路径/hash/模型分数混进 verdict”的入口问题。

这一步仍然是只读治理链路，不训练、不调阈值、不加载模型、不读 private map、不 unblind、不采样 replacement、不改 split/cache。

## 严格输入边界

外部标注文件只允许四列：

- `blind_review_id`
- `manual_label_verdict`
- `manual_verdict_note`
- `recommended_action`

任何其它列都会被拒绝；其中 `filename`、`path`、`directory`、`extension`、`source_sha256`、`sample_index`、`split`、`probability`、`score`、`prediction`、`threshold`、`loop57` 等身份或模型分数字段会被明确标记为身份/模型字段违规。

`blind_review_id` 只用于把外部结论写回盲化 focus 行。它不是模型证据，也不能推导真实路径、hash、sample index 或 split。导入器不会读取 `reports/random_20w_split/loop96_full_queue_private_map.csv`。

## 导入后门禁

导入器写出新的 focus CSV 后，会立即调用 Loop109 的 `preflight_focus_annotations()` 做后置验证。也就是说，即使外部文件字段合法，如果 `manual_verdict_note` 只写“filename 证明”“source_path 证明”“loop57 probability 证明”这类说明，也会被后置 preflight 拦截为无效。

合法 actionable verdict 仍必须引用内容或外部证据，例如 PE parse、NPZ feature mismatch、overlay、section、import、signature、sandbox、VT/multi-engine、YARA 等。

## 真实 no-op 复验

本轮使用空外部标注模板：

- `reports/random_20w_split/loop111_external_annotations_noop.csv`

导入命令输出：

- `reports/random_20w_split/loop111_focus_annotations_noop_imported.csv`
- `reports/random_20w_split/loop111_focus_external_annotation_import_noop_summary.json`
- `reports/random_20w_split/loop111_focus_annotations_noop_preflight.csv`
- `reports/random_20w_split/loop111_focus_annotations_noop_preflight.json`

真实结果：

- focus rows: `240`
- imported rows: `0`
- post preflight annotated rows: `0`
- post preflight actionable rows: `0`
- post preflight invalid rows: `0`
- decision: `ready_noop_no_external_annotations`

随后把 Loop111 输出接入 Loop110：

- `reports/random_20w_split/loop111_to_loop110_focus_pipeline_noop_summary.json`

结果保持：

- Loop87 rows: `1868`
- Loop87 actionable rows: `0`
- replacement required rows: `0`
- training policy rows: `0`
- decision: `ready_noop_no_actionable_verdicts`
- Train/Val、Test-10k、full-test 仍全部不授权

## 验证

资源守卫：

- `reports/random_20w_split/loop111_external_annotation_import_guard.json`
- `reports/random_20w_split/loop111_real_noop_import_guard.json`
- `reports/random_20w_split/loop111_to_loop110_pipeline_guard.json`

测试：

```powershell
.\vnev\Scripts\python.exe -m pytest tests/test_import_loop111_focus_external_annotations.py tests/test_preflight_loop106_focus_annotations.py tests/test_run_loop110_focus_verdict_pipeline.py -q
```

结果：`17 passed`。

覆盖点包括：

- 空外部文件不产生 verdict
- 缺少四列表头的空外部文件会被阻断
- 合法内容证据导入后通过 Loop109 preflight
- JSONL 导入
- 身份/模型字段列拒绝
- unknown `blind_review_id` 拒绝
- duplicate `blind_review_id` 拒绝
- 只引用 filename / loop57 probability 的 note 被后置 preflight 拒绝
- 空 manual 字段行拒绝

## 当前结论

Loop111 只是把外部证据入口做成受控阀门，并没有创造新的独立 verdict。当前可执行的下一步仍是：由独立内容/外部证据填入 focus 外部标注文件，再跑 Loop111 和 Loop110。只有出现通过 Loop87 的 actionable bad-row verdict，才允许进入 quarantine + locked manifest 同原始标签 fresh redraw 的下一道预检。
