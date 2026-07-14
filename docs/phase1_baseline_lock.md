# Phase 1 Baseline Lock

更新时间：2026-07-01

## 结论

Phase 1 的数据与缓存基线已经跑通：当前 20 万样本 split 使用 `1:1:8` 的训练、验证、测试比例，缓存覆盖率为 `200000/200000`，缺失样本数为 `0`。当前 baseline 模型距离最终目标 `16w test F1 >= 99.9%` 仍有明显差距，因此后续不能直接冲全量测试，必须继续执行 Val 优先、Test10k 确认的漏斗式验证。

## 数据与缓存

- Split 文件：`reports/random_20w_split/random_20w_split.csv`
- Manifest：`data/.cache/manifest_38672ba0.json`
- 覆盖审计报告：`reports/random_20w_split/random_20w_8192_uncompressed_cache_coverage_audit_replaced_130.json`
- 样本总数：`200000`
- 缓存覆盖：`200000`
- 缺失缓存：`0`
- 覆盖率：`1.0`
- 备注：此前 130 个 strict PE 特征失败样本已经按用户要求重新抽样替换，而不是用坏样本补齐。

## Baseline 模型

- Checkpoint：`models/random_20w_8192/best_model.pt`
- Config：`config/random_20w_8192.toml`
- Train 预测：`reports/random_20w_split/random_20w_8192_replaced_train_predictions.csv`
- Val 预测：`reports/random_20w_split/random_20w_8192_replaced_val_predictions.csv`
- Test10k 预测：`reports/random_20w_split/random_20w_8192_replaced_test10k_predictions.csv`

## 初始指标

| 数据集 | 样本数 | F1 | AUC | 错误数 |
| --- | ---: | ---: | ---: | ---: |
| Train | 20000 | 0.9284412955 | 0.9776277200 | 1414 |
| Val | 20000 | 0.9297231518 | 0.9757015150 | 1386 |
| Test10k | 10000 | 0.9298531811 | 0.9780854563 | 688 |

## 已验证的校准器

概率校准器已经在纠正后的 split 上重新训练和验证：

- 模型：`models/random_20w_8192/random20w_replaced_logreg_calibrator.pkl`
- Train/Val 报告：`reports/random_20w_split/random_20w_8192_replaced_calibrator_train_val.json`
- Test10k 报告：`reports/random_20w_split/random_20w_8192_replaced_calibrator_test10k_eval.json`
- Val F1：`0.9687640114`
- Test10k F1：`0.9724110356`
- 结论：显著强于裸 baseline，但仍远低于 `99.9%`，不能进入全量 16w 最终评估。

## Git 状态

- 仓库已初始化：是
- 当前 Phase 1 分支：`codex/phase1-baseline-lock`
- 原始分支：`master`
- 风险：进入本分支前，工作区已有大量未提交/未跟踪改动。为避免污染 baseline，本次 baseline lock 只提交 Phase 1 审计证据，不把历史脏改动混入同一个提交。

## 下一阶段 Gate

进入 Phase 2/3 前必须遵守：

1. 候选方案先完整跑 Val。
2. 只有 Val 明显超过当前校准器基线 `0.9687640114`，才允许进入 Test10k。
3. 只有 Test10k 也显著提升，才允许申请全量 16w test。
4. 噪声审计必须和错误归因同步进行，尤其关注 Val/Test10k 的 FP/FN 与疑似标签冲突样本。
