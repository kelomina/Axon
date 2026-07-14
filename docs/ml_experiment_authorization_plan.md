# Axon 机器学习建议执行授权计划

更新时间：2026-06-30

## 这份文档解决什么问题

`docs/ml_improvement_recommendations.md` 已经把 Axon 机器学习方向的建议、证据和风险写清楚了。现在剩下的问题不是“还能想到什么新招”，而是把仍开放的建议逐项跑到可以下结论的程度：确认实用且完成的，从待办建议中移除；确认不实用的，显眼保留为负面记录；还没证据的，继续标成开放项。A/B 两个重操作包已经完成严格复验，当前文档只保留它们的完成记录和证据入口。

这件事像给候选方案安排同一场考试。概率校准和 GA 特征掩码这两个 A/B 包已经考完。C 包 byte noise / near-threshold 多 seed 也已考完，结论是不建议默认启用。D 包 hard-example balanced replay 也已于 2026-06-29 完成严格 source-group 隔离复验，当前配方不实用，不再视为待执行项。当前仅 E 包 SpeakeasyX 二阶段复核仍保留为 P2 边缘探索。

本计划只写执行方案和授权边界，不自动启动训练、不重建 cache、不安装依赖。所有会大量消耗 GPU/CPU、改数据 cache、改核心训练脚本或产生新模型的动作，都需要你单独明确授权。

对应的机器可读清单位于：

- `reports/model_review/final_model_selection/ml_experiment_authorization_plan.json`

cache 缺口的非破坏性恢复计划曾用于 A/B 施工边界，当前保留为历史记录：

- `reports/model_review/final_model_selection/cache_recovery_plan.md`
- `reports/model_review/final_model_selection/cache_recovery_plan.json`
- `reports/model_review/final_model_selection/high_value_benign_manifest.csv`
- `reports/model_review/final_model_selection/high_value_benign_manifest_summary.json`

这份恢复计划只读旧审计和缺失 CSV，不重建 cache、不删除 cache、不训练模型。A/B 复验后，当前权威 cache 结论改看 `reports/model_review/final_model_selection/cache_coverage_audit.json` 和 `reports/model_review/final_model_selection/ab_strict_reverification_report.md`：10 个关键检查面全部 `missing=0`。

2026-06-29 又补充了 A/B 专项严格复验报告：`reports/model_review/final_model_selection/ab_strict_reverification_report.md` 和 `ab_strict_reverification_report.json`。它确认 A/B 两个包都已经完成，preflight 现在只把未完成包当成待授权对象。

2026-06-30 按你的新授权执行了 cache 侧重操作：清空 `data\.cache`，并重新提取 random 20w、8192 字节、fixed-v2 PE256 的未压缩 cache。重建报告在 `reports/random_20w_split/random_20w_8192_uncompressed_cache_rebuild_full_current_split.json`，覆盖审计在 `reports/random_20w_split/random_20w_8192_uncompressed_cache_coverage_audit.json`。结果是 `199870/200000 = 99.935%` 已覆盖，剩余 `130` 条为严格 PE 特征提取失败，不是 cache 匹配或路径污染问题。这个动作已经完成，不再需要重复授权。

## 当前状态

| 建议项 | 状态 | 真正卡点 |
| --- | --- | --- |
| fixed-v2 20w 未压缩 cache | 已完成 | 已清空并重建 `data\.cache`；覆盖 `199870/200000`，剩余 `130` 条是严格 PE 提取失败 |
| 概率校准 | 已确认实用且已完成 | 严格 full test、hard-FN、hard-error 和高价值白样本都已复验；保留为标准流程 |
| GA 特征掩码 | 已确认低漏报方向有价值，但不能默认启用 | 完整 hard-holdout 和高价值白样本都已复验；保留为高安全模式候选 |
| byte noise / near-threshold weighting | ⚠️ 实验验证：不实用 | 三 seed cache-covered group-isolated 复验已完成；mean test F1 低于 baseline，mean FN 更高，不建议默认启用 |
| hard-example replay | ⚠️ 实验验证：不实用 | 严格 source-group 隔离复验显示 FN holdout 退化，当前配方无单一阈值同时改善 FP/FN；保留为负面记录 |
| SpeakeasyX 动态行为特征 | 只能作为 P2 二阶段复核信号 | 固定 timeout filter 会减少 FP 但新增 FN；只能做小样本、val-first 的保守复核研究 |

已经完成或已有负面结论的项不再安排重实验：统一模型评审闸门已完成；概率校准已完成并移出待办；GA 特征掩码已复验完成但仍只保留为高安全模式候选；byte noise / near-threshold 多 seed 复验已确认不适合作为默认训练技巧；RL 主线扩大、SWA/EMA/all combined、当前 gated/residual fusion 路线都保留为负面记录。

## 授权包 0：fixed-v2 20w 未压缩 cache 重建完成记录

### 目的

这次授权解决的是“cache 证据是否可靠、是否还被压缩存储拖慢”的问题。通俗讲，cache 就像把样本预先做成可直接读取的半成品菜，评估时不用每次从原始文件重新切菜、洗菜。未压缩版则是把半成品放成更容易直接拿取的包装，减少读取时再拆压缩包的动作。

### 已执行动作

- 清空范围：仅 `data\.cache`。
- 重建口径：`reports/random_20w_split/random_20w_split.csv`，checkpoint/config 来自 `models/random_20w_8192/best_model.pt`。
- 产物 manifest：`data/.cache/manifest_38672ba0.json`。
- 存储格式：`uncompressed`，并已抽查 NPZ zip type 为 `ZIP_STORED`。

### 结果

- 输入行数：`200000`。
- cache 命中：`2998`。
- 新提取：`196872`。
- 严格 PE 特征提取失败：`130`。
- manifest 样本数：`199870`。
- 覆盖率：`99.935%`。
- 缺失标签：白样本 `125`、黑样本 `5`。
- 缺失 split：train `12`、val `19`、test `99`。

### 结论

fixed-v2 cache 覆盖已经不是当前阻塞项。剩余缺口是严格 PE 解析失败，属于样本内容/解析规则问题，不是施工没补齐。后续如果要完整评估 random 20w 模型，应该先优化评估脚本和推理流水线，而不是继续清 cache。

### 评估吞吐补充

未压缩 cache 之后，最初的 `scripts/evaluate_split_from_cache.py` 仍然无法在当前时间预算内完成 10k 或 1k test 评估：10k 尝试 30 分钟超时，1k 尝试 10 分钟超时。后续 profile 发现主要慢点之一是 manifest lookup 构建，修正路径索引后 lookup 从约 `75.7` 秒降到约 `4.4` 秒。新的 10k test 评估已经完成，输出为 `reports/random_20w_split/random_20w_8192_uncompressed_test10k_after_lookup_opt_eval.json`。

10k 评估口径：checkpoint `models/random_20w_8192/best_model.pt`，split `reports/random_20w_split/random_20w_split.csv`，manifest `data/.cache/manifest_38672ba0.json`，device `cuda`，batch size `64`，未使用概率校准器，未使用 GA feature mask。`10000` 行 raw test 中 `9994` 行预测成功，`6` 行 missing cache。

默认阈值 `0.50` 结果：Accuracy `0.9315`，Precision `0.9441`，Recall `0.9166`，F1 `0.9302`，AUC `0.9782`，FP `270`，FN `415`。阈值 `0.45` 的 F1 略高为 `0.9320`，但 FP/FN 变为 `330/346`；阈值 `0.65` 则 FP `146`、FN `746`。结论：20 万随机样本目标 `F1 >= 99.9%` 当前没有达到，主要瓶颈不是 cache 缺失，而是模型能力和 FP/FN trade-off。因为这 10k slice 已经在最佳已扫阈值下产生 `676` 个错误，即使剩余约 150k test 全部预测正确，完整 test 的理论 F1 上界也只有约 `0.9958`，仍低于 `0.999`。完整 160k test 精确指标可在评估流水线进一步优化后再跑；阈值仍然必须用 val 选择，不能用 test 反复调参。

## 授权包 A：概率校准严格评审完成记录

### 目的

概率校准已经有强正证据：历史完整 test 上 F1 从 `0.9514` 到 `0.9720`，错误从 `1549` 降到 `898`。当前严格 full test、hard-FN、hard-error 和高价值白样本都已经完成，不再保留为待执行项。

这里的“校准”可以理解成给模型分数重新标尺。主模型已经会打分，但它说 `0.8` 不一定真的代表 80% 风险；校准器用训练/验证集把刻度调准。它通常不需要重训主模型，成本比重新训练一个大模型低得多。

### 已完成的严格评估

```powershell
cd "E:\Project\python\Axon_v2.6Exp"; & "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\evaluate_probability_calibrator.py --model "models\clean_train_logreg_calibrator_no_metadata_scripted.pkl" --predictions "reports\hard_family_finetune\clean_hyperparam_search\baseline_test_predictions_threshold053_current.csv" --baseline-threshold 0.53 --output-json "reports\model_review\final_model_selection\probability_calibrator_test_strict_full.json"
```

```powershell
cd "E:\Project\python\Axon_v2.6Exp"; & "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\export_sample_predictions.py --checkpoint "models\group_isolated_rare_weighted_ft_rebuilt_cache\best_model.pt" --config "config\default_config.toml" --data-dir "data" --samples "reports\hard_family_finetune\hard_fn_finetune_threshold055\hard_error_holdout_samples.csv" --decision-threshold 0.53 --output "reports\model_review\final_model_selection\baseline_full_hard_fn_holdout_predictions_threshold053.csv" --batch-size 32 --device cuda
```

```powershell
cd "E:\Project\python\Axon_v2.6Exp"; & "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\export_sample_predictions.py --checkpoint "models\group_isolated_rare_weighted_ft_rebuilt_cache\best_model.pt" --config "config\default_config.toml" --data-dir "data" --samples "reports\hard_family_finetune\hard_error_finetune_threshold055\hard_error_holdout_samples.csv" --decision-threshold 0.53 --output "reports\model_review\final_model_selection\baseline_full_hard_error_holdout_predictions_threshold053.csv" --batch-size 32 --device cuda
```

```powershell
cd "E:\Project\python\Axon_v2.6Exp"; & "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\evaluate_probability_calibrator.py --model "models\clean_train_logreg_calibrator_no_metadata_scripted.pkl" --predictions "reports\model_review\final_model_selection\baseline_full_hard_fn_holdout_predictions_threshold053.csv" --baseline-threshold 0.53 --output-json "reports\model_review\final_model_selection\probability_calibrator_hard_fn_holdout_strict_full.json"
```

```powershell
cd "E:\Project\python\Axon_v2.6Exp"; & "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\evaluate_probability_calibrator.py --model "models\clean_train_logreg_calibrator_no_metadata_scripted.pkl" --predictions "reports\model_review\final_model_selection\baseline_full_hard_error_holdout_predictions_threshold053.csv" --baseline-threshold 0.53 --output-json "reports\model_review\final_model_selection\probability_calibrator_hard_error_holdout_strict_full.json"
```

### 结果

- cache 覆盖：专项报告里 `10/10` 个检查面都是 `missing=0`。
- official test：`1549 -> 898` 错误。
- hard-FN：`20 -> 6` 错误。
- hard-error：`300 -> 132` 错误。
- 高价值白样本：`604 -> 406` 误报。

结论：概率校准已经完成，保留为标准流程的一部分，不再占用 P1 待办。

## 授权包 B：GA 特征掩码严格评审完成记录

### 目的

GA 掩码已经显示出“降低漏报”的价值：20k 评估中总错误下降，FN 大幅下降，但 FP 上升。它像一个“高安全模式”的候选开关，适合宁可多报警也少漏报的场景，不适合无条件默认启用。现在完整 hard-holdout 和高价值白样本也已经复验完成，所以它的定位已经足够明确。

### 结果

- cache 覆盖：GA hard-FN、hard-error、高价值白样本 baseline/mask 导出全部 `missing=0`。
- 20k 评估：总错误 `1340 -> 1210`，FN `958 -> 670`，FP `382 -> 540`。
- hard-FN current subset：`19 -> 18` 错误。
- hard-error current subset：`288 -> 286` 错误。
- 高价值白样本：`604 -> 638` 误报。

结论：GA 掩码保留为高安全模式候选，不默认启用，也不再列为待办重实验项。

## 授权包 C：byte noise / near-threshold 多 seed 复验完成记录

### 目的

byte noise 和 near-threshold weighting 曾在单 seed group-isolated split 下有小幅 test F1 收益，但 val F1 低于 baseline。这个级别的差异很容易是随机波动，所以需要多 seed 确认。

byte noise 可以理解成训练时给输入做一点轻微扰动，让模型别只背死某些字节位置；near-threshold weighting 是让模型多关注“刚好卡在报警线附近”的样本。两者都不是大改模型，而是训练方式的微调。

### 已完成的前置修复

`script/run_generalization_group_split.py` 已支持 `--seeds` 和 `--cache-manifest`。本轮还补了 manifest 协议校验，避免用 64-byte 或 8192-byte cache 去跑 512-byte 主线配置；转换后的 split 也改成用原始文件路径作为 `source_path`，保证训练 dataset 能正确匹配。

固定复验口径：

- split：`reports/raw_group_diagnostics/group_isolated_split.csv`
- output：`models/generalization_group_isolated_seed_confirm/`
- manifest：`data/.cache/manifest_ee122d6c.json`
- cache-covered rows：`20,000/40,000`
- converted train/val/test：`2,997 / 778 / 16,225`
- seeds：`42,43,44`
- 候选：baseline、byte_noise、near_threshold

### 结果

`models/generalization_group_isolated_seed_confirm/summary.md` 已生成同 seed baseline 对比和多 seed 汇总：

- baseline：test F1 mean `0.9444`，std `0.0047`，FP mean `271.0`，FN mean `623.7`。
- byte noise：test F1 mean `0.9260`，std `0.0555`，相对 baseline `-0.0184`；FP mean `293.3`，FN mean `851.7`。
- near-threshold：test F1 mean `0.9200`，std `0.0515`，相对 baseline `-0.0244`；FP mean `275.3`，FN mean `958.7`。

seed43 是主要反证：byte noise 比同 seed baseline 多 `1105` 个 FN，near-threshold 多 `1164` 个 FN。这个结果说明它们不是稳定收益。

### 结论

⚠️ 实验验证：不实用。C 包已完成，不再需要授权训练；byte noise / near-threshold weighting 保留为负面记录，不默认启用。

## 授权包 D：hard-example replay 配方完成

### 状态

⚠️ 实验验证：不实用。

hard-example 微调曾经证明能修复一批漏报，但 2026-06-29 的严格 source-group 隔离复验显示当前 replay 配方（4 epoch、1e-5 LR、FP/FN weight 4x）无法同时改善两类错题：阈值 0.63 时 FN holdout 的正确数从 39 降到 15，FP 从 41 升到 79；总净增益仅 +4/423。阈值 0.50 时交换反转，无单一阈值平衡两类错误。具体复验数据在 `docs/ml_improvement_recommendations.md` 中已增补。

保留为负面记录，不从文档移除。

可以把它理解成错题本训练。只练“漏掉的恶意样本”，模型会更敏感，但可能把正常文件也错判成恶意；加入 hard-FP 和普通样本回放，就是防止它偏科。

### 建议前置

优先使用当前推荐候选 `hard_fn_candidate@0.63` 的错题，而不是旧模型 `threshold 0.55` 的错题。已有当前候选错题目录：

- `reports/hard_family_finetune/model_selection_final/hard_fn_original_test_error_analysis_threshold063/false_positives.csv`
- `reports/hard_family_finetune/model_selection_final/hard_fn_original_test_error_analysis_threshold063/false_negatives.csv`

但最终选择报告没有直接记录 `hard_fn_candidate` 的 checkpoint 路径。训练前必须确认对应 checkpoint，否则只能构建包，不能启动微调。

### 已执行的轻量准备命令

以下命令只构建 CSV 包和建议命令，不启动训练。

已于 2026-06-28 完成第一步（不隔离）：，并生成 `reports/hard_family_finetune/balanced_replay_from_current_candidate_threshold063/`：

```powershell
cd "E:\Project\python\Axon_v2.6Exp"; & "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\build_hard_error_finetune_package.py --source-split "reports\hard_family_finetune\hard_family_finetune_split.csv" --false-positives "reports\hard_family_finetune\model_selection_final\hard_fn_original_test_error_analysis_threshold063\false_positives.csv" --false-negatives "reports\hard_family_finetune\model_selection_final\hard_fn_original_test_error_analysis_threshold063\false_negatives.csv" --output-dir "reports\hard_family_finetune\balanced_replay_from_current_candidate_threshold063" --error-focus both --hard-train-ratio 0.60 --hard-val-ratio 0.20 --fp-weight 4.0 --fn-weight 4.0 --decision-threshold 0.63 --model-output-dir "models\balanced_replay_from_current_candidate_threshold063" --epochs 4 --learning-rate 0.00001 --batch-size 32 --device cuda
```

产物校验：

- hard FP/FN 来源：当前候选 `threshold 0.63` 错题。
- hard error types：`FN=720`、`FP=680`。
- hard rows by split：train `859`、val `291`、holdout `415`。
- weighted train samples：`840`。
- 包内 README 和建议命令已对齐 `threshold 0.63`，threshold sweep 包含 `0.63`。

### 已执行的重操作（2026-06-29）

1. ✅ 确认初始化 checkpoint：`models/group_isolated_rare_weighted_ft_rebuilt_cache/best_model.pt`（8192 fixed-v2 PE256）。
2. ✅ 已生成严格 source-group 隔离的 replay 包：`reports/hard_family_finetune/balanced_replay_strict_source_group_threshold063/`。
3. ✅ 已启动微调训练：4 epoch、1e-5 LR、batch 32、cuda。
4. ✅ 已在 hard-error holdout 上完成 baseline vs replay 对比评估。

复验结论：当前配方未实现 balance improvement，不推荐进入下一步。

### 结论标准（未满足，实验已终止）

- 不接受“只修 hard-FN 但 FP 明显增加”的方案。 ❌ 未满足：FN 反而更差。
- 不接受“original test 好看但 hard-error holdout 回退”的方案。 ❌ 未满足：holdout 净增益仅 +4。
- 只有 original test、hard-FN、hard-error 三者综合改善，且高价值白样本 FP 可控，才算 replay 配方完成。当前配方未满足。

## 授权包 E：SpeakeasyX 只做二阶段复核小实验

### 目的

SpeakeasyX 已经证明有信号，但固定 timeout filter 在 test confirmation 子集上把 FP 从 `122` 降到 `0` 的同时，把 FN 从 `120` 增到 `168`。这不适合直接并入主分类器。

下一步只能做更保守的二阶段复核研究：它不负责改主模型判断，只在“模型已经报恶意、但不算极高置信”的样本上提示“这可能是误报，需要复核”。

### 可选小样本验证命令

```powershell
cd "E:\Project\python\Axon_v2.6Exp"; & "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" reports\hard_family_finetune\clean_hyperparam_search\run_speakeasy_feature_probe.py --role-source calibrator --per-role 5 --random-val-count 20 --sample-timeout 12 --emu-timeout 6 --output-dir "reports\hard_family_finetune\clean_hyperparam_search\speakeasy_conservative_fp_triage_probe"
```

### 结论标准

- 只能在 val 上选规则，不能直接在 test 上调规则。
- 任何规则如果新增 FN 明显，就只能作为人工调查信号，不能进入自动降级。
- `.NET unsupported` 不允许简单当作 benign，因为历史证据显示它常出现在恶意侧。

## 推荐执行顺序

1. D 包已执行完成（负面结论）。
2. E 包 SpeakeasyX 仅当有明确小样本 FP 复核需求时再启动。

## 每轮执行后的文档处理规则

- 实验证实实用且彻底完成：从 `docs/ml_improvement_recommendations.md` 的待办建议中移除，只保留完成记录和报告入口。
- 实验证实不实用：显眼保留，不移除，避免以后重复投入。
- 证据不足：保留开放项，并写明缺的是数据覆盖、多 seed、白样本还是 replay 配方。
- 每轮必须同步更新：
  - `docs/ml_improvement_recommendations.md`
  - `reports/hard_family_finetune/experiment_journal.md`
  - `reports/model_review/final_model_selection/ml_recommendation_status.json`
  - 必要时更新 `reports/model_review/final_model_selection/model_review_summary.md`
