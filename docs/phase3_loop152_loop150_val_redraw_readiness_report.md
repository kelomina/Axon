# Phase 3 Loop152 Loop150 Val Redraw Readiness Report

更新时间：2026-07-08

## 目标

Loop151 已把 strict best 从 Loop136 推到 trusted signer guard，但剩余 full-test 错误仍有 `1466 / 160000`。继续盲目叠 selector 的收益很低，下一步必须把噪声治理入口做实：如果 Val focus 行被独立内容/外部证据确认 `label_wrong`、`feature_broken` 或 `out_of_scope`，只能 quarantine 后从 locked manifest 的同原始标签池 fresh redraw，不能直接改标，也不能让坏样本自己补齐名额。

本轮补上 Loop150 Val `86` 行 focus 包到 Loop76 redraw readiness 的桥接脚本：

- `scripts/run_loop152_loop150_val_focus_redraw_readiness.py`
- `tests/test_run_loop152_loop150_val_focus_redraw_readiness.py`

该脚本只读，不训练、不评估、不采样 replacement、不改 split/cache。路径、文件名、目录、后缀、hash、`source_sha256`、`sample_index`、split、row order、`review_focus_id` 和模型分数只用于对齐、审计和找到原始 split 行，不作为 verdict、模型、阈值、GA mask、replacement sampling 或生产推理证据。

## 当前真实状态

先重新复核 Loop150 Val `86` 行标注模板：

- 输出：`reports/phase3_loop152/loop152_loop150_val86_preflight.json`
- rows：`86`
- annotated rows：`0`
- replacement required：`0`
- blockers：`[]`
- ready for private mapping：`false`

随后运行 Loop152 bridge：

- 输出：`reports/phase3_loop152/loop152_loop150_val86_redraw_readiness_summary.json`
- strict adjustment plan：`reports/phase3_loop152/loop152_strict_adjustment_plan.json`
- Loop76 readiness：`reports/phase3_loop152/loop152_loop76_readiness.json`

结果：

| Item | Value |
|---|---:|
| Review rows | `86` |
| Planned rows | `0` |
| Replacement required | `0` |
| Training policy rows | `0` |
| Decision | `await_external_verdicts` |
| Next step | `no_redraw_required_until_actionable_verdicts` |
| Train/Val allowed | `false` |
| Test-10k allowed | `false` |
| Full-test allowed | `false` |

Split 形状在 bridge 中重新对齐检查，仍是严格 `200000 = 20000/20000/160000`，且每个 split 保持 `0/1` 平衡。

## 决策

Loop152 不替代 Loop151，也不触发训练或评估。它把后续噪声治理的安全入口补齐：只有当 Val `86` 行中出现独立内容/外部证据确认的坏行，Loop152 才会生成 `exclude_and_replace` plan，并交给 Loop76 决定下一步是否允许 building replacement candidate pool。当前因为 `annotated_rows=0`，正确决策是等待独立 verdict。

## 验证

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_run_loop152_loop150_val_focus_redraw_readiness.py tests\test_run_loop114_loop112_redraw_readiness.py tests\test_build_loop76_redraw_readiness.py -q
```

结果：`26 passed`。

真实 no-op 复验：

```powershell
.\vnev\Scripts\python.exe scripts\preflight_loop126_review_annotations.py --annotations-csv reports\phase3_loop150\loop150_loop136_val_noise_focus_all86_annotations_template.csv --output-csv reports\phase3_loop152\loop152_loop150_val86_preflight.csv --output-json reports\phase3_loop152\loop152_loop150_val86_preflight.json --expected-rows 86

.\vnev\Scripts\python.exe scripts\run_loop152_loop150_val_focus_redraw_readiness.py --preflight-csv reports\phase3_loop152\loop152_loop150_val86_preflight.csv --preflight-json reports\phase3_loop152\loop152_loop150_val86_preflight.json --private-map-csv reports\phase3_loop150\loop150_loop136_val_noise_focus_all86_private_map.csv --split-csv reports\random_20w_split\loop127_full_duplicate_corrected_split.csv --manifest-json data\.cache\manifest_38672ba0.json --output-dir reports\phase3_loop152 --output-json reports\phase3_loop152\loop152_loop150_val86_redraw_readiness_summary.json --output-md reports\phase3_loop152\loop152_loop150_val86_redraw_readiness_summary.md
```

第一条命令返回非零是预期 no-op 状态：`ready_for_private_mapping=false`，因为当前没有任何人工/外部标注。
