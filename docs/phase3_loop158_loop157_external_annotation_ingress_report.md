# Phase 3 Loop158 Loop157 External Annotation Ingress Report

更新时间：2026-07-08

## 目标

Loop157 已经给外部 reviewer 准备好 `162` 行当前 best Val 错误的安全上下文和四列表头模板。Loop158 补上返回入口：外部返回文件不能直接进 private map，也不能直接触发重抽；它必须先经过一个严格导入闸门，只接受四列 `review_focus_id/manual_label_verdict/manual_verdict_note/recommended_action`。

Loop158 的作用是把“外部说法”变成“内部可审计输入”。它先检查列、ID、重复行、空 manual 行，以及 note 里是否出现路径、hash、`source_sha256`、`sample_index`、模型概率、prediction、threshold 等身份或模型痕迹。只有这一步通过后，才按 `review_focus_id` 合回 Loop157 context 的 `current_label`，再调用 Loop126 preflight；Loop126 通过后，才允许内部调用 Loop152/Loop76 readiness。整个过程不训练、不评估、不调阈值、不自动 verdict、不直接 relabel、不采样 replacement、不改 split/cache。

## 产物

新增脚本和测试：

- `scripts/import_loop158_current_best_val_external_annotations.py`
- `tests/test_import_loop158_current_best_val_external_annotations.py`

真实 no-op 输出：

- `reports/phase3_loop158/loop158_loop157_external_annotation_import_summary.json`
- `reports/phase3_loop158/loop158_loop157_external_annotation_import_summary.md`

## 当前真实状态

| Item | Value |
|---|---:|
| Context rows | `162` |
| Context fields | `33` |
| Context label 0 / 1 | `105 / 57` |
| Returned annotation rows | `0` |
| Annotated rows | `0` |
| Forbidden columns | `0` |
| Note identity/model term rows | `0` |
| Blockers | `[]` |
| Private join performed | `false` |
| Decision | `ready_noop_no_external_annotations` |

当前使用的是 Loop157 header-only annotation template，所以没有真实 verdict rows。这个状态是安全 no-op：不读 private map、不生成 replacement plan、不授权 fresh redraw、Train/Val、Test-10k 或 full-test。

## 决策

Loop158 成为 Loop157 返回标注的唯一入口。后续如果有真实人工或外部证据 verdict，只能先跑 Loop158；任何额外列、未知/重复 `review_focus_id`、空 manual 行、身份字段、模型分数字段，或 note 中把路径/hash/概率/prediction/threshold 当证据的写法，都必须被阻断。

若独立证据最终确认 `label_wrong`、`feature_broken` 或 `out_of_scope`，Loop158 也只会把它交给 Loop152/Loop76 生成 same-original-label fresh redraw readiness；坏样本不能直接改标，不能自己补齐名额，最终仍必须严格保持 `200000 = 20000/20000/160000`。

## 验证

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_import_loop158_current_best_val_external_annotations.py -q

.\vnev\Scripts\python.exe scripts\import_loop158_current_best_val_external_annotations.py --returned-annotations-csv reports\phase3_loop157\loop157_loop151_val_all_errors_annotation_template.csv --context-csv reports\phase3_loop157\loop157_loop151_val_all_errors_external_context.csv --private-map-csv reports\phase3_loop156\loop156_loop151_val_all_errors_private_map.csv --split-csv reports\random_20w_split\loop127_full_duplicate_corrected_split.csv --manifest-json data\.cache\manifest_38672ba0.json --output-dir reports\phase3_loop158 --output-json reports\phase3_loop158\loop158_loop157_external_annotation_import_summary.json --output-md reports\phase3_loop158\loop158_loop157_external_annotation_import_summary.md --expected-rows 162
```

结果：pytest `6 passed`；真实导入闸门 `decision=ready_noop_no_external_annotations`、`blockers=[]`、`private_join_performed=false`。
