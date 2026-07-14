# Phase 3 Loop153 Current-Best Val Noise Focus Report

更新时间：2026-07-08

## 目标

Loop151 trusted signer guard 已经成为当前 strict best：Val `162` errors、Test-10k `78` errors、full-test `1466 / 160000` errors。Loop150 的 Val focus 包仍然基于旧 Loop136 的 `179` 个 Val 错误，因此本轮把噪声治理入口重建到当前 best 上：只从 Loop151 仍然判错的 Val 行中筛出高冲突复核队列。

Loop153 不训练、不评估、不改阈值、不采样 replacement、不改 split/cache。它只生成一个盲化复核包。路径、文件名、目录、后缀、hash、`source_sha256`、`sample_index`、split、row order、模型分数和概率字段只能用于加载、对齐、审计、去重和 private map，不允许作为 verdict、模型、阈值、GA mask、replacement sampling 或生产推理证据。

## 实现

新增脚本和测试：

- `scripts/build_loop153_current_best_val_noise_focus.py`
- `tests/test_build_loop153_current_best_val_noise_focus.py`

脚本读取 Loop151 Val predictions，按 `trusted_signer_guard_prediction` 找出当前错误，再用 `source_sha256` 仅做对齐，把旧 Loop136 Val neighbor/content 证据过滤到同一批当前错误。随后复用 Loop145 的盲化包构建逻辑，只保留 `support_bucket=neighbors_support_model_prediction` 的高冲突行。

## 当前真实状态

主要输出：

- `reports/phase3_loop153/loop153_loop151_val_noise_focus_summary.json`
- `reports/phase3_loop153/loop153_loop151_val_noise_focus_blinded.csv`
- `reports/phase3_loop153/loop153_loop151_val_noise_focus_private_map.csv`
- `reports/phase3_loop153/loop153_loop151_val_noise_focus_preflight.json`
- `reports/phase3_loop153/loop153_loop151_val_noise_focus_redraw_readiness_summary.json`

结果：

| Item | Value |
|---|---:|
| Loop151 Val prediction rows | `20000` |
| Loop151 Val current errors | `162` |
| Current Val FP / FN | `105 / 57` |
| Filtered neighbor rows | `162 / 179` |
| Filtered content rows | `162 / 179` |
| Missing current-error evidence rows | `0` |
| Focus rows | `73` |
| Focus FP / FN | `52 / 21` |
| Critical / High | `3 / 70` |
| Annotated rows | `0` |
| Replacement required | `0` |
| Decision | `await_external_verdicts` |

Review lane 分布：

| Lane | Rows |
|---|---:|
| `benign_trust_or_label_quality_review` | `48` |
| `malware_blindspot_or_label_quality_review` | `14` |
| `content_evidence_review` | `11` |

当前 preflight 是正确的 no-op 状态：`rows=73`、`annotated_rows=0`、`replacement_required_rows=0`、`blockers=[]`、`ready_for_private_mapping=false`。Redraw readiness 也保持阻断：Train/Val、Test-10k 和 full-test 全部 `false`。

## 决策

Loop153 取代 Loop150 的 Val `86` 行包，成为当前 best 口径下的优先噪声复核队列。它不是 verdict，也不是自动 redraw 许可。只有独立内容/外部证据确认 `label_wrong`、`feature_broken` 或 `out_of_scope` 后，才能生成 `exclude_and_replace` 计划，并从 locked manifest 的同原始标签池 fresh redraw。坏行不能直接改标，不能自己补齐名额，也不能让 split 少于严格 `200000 = 20000/20000/160000`。

Full-test focus 行仍然不能用于模型选择、阈值选择、trusted signer term 扩展或 GA mask 选择。若要继续激进推进，应优先补独立外部证据源，例如多引擎信誉、行为沙箱或签名/发布者人工复核，然后回到 Val-first 漏斗。

## 验证

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_build_loop153_current_best_val_noise_focus.py -q
```

结果：`1 passed`。

后续整体验证命令：

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_build_loop153_current_best_val_noise_focus.py tests\test_build_ml_recommendation_status.py tests\test_run_loop152_loop150_val_focus_redraw_readiness.py -q
```
