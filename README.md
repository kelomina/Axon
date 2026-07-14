# Axon v2.6 Experiment

Axon v2.6 是一个面向 Windows PE 文件的二分类恶意软件检测实验项目。主链路同时使用：

- 原始文件字节序列（`byte_seq`）
- PE 结构特征（`pe_features`）
- 统计特征（`stat_features`）

字节序列由 MHDSRA2/DSRA 流式编码器处理，PE 与统计特征分别投影后融合，最后输出 benign/malicious 二分类 logits。项目也包含特征缓存、固定 split、阈值扫描、特征重要性分析、嵌套压缩包预测和独立的 RL 实验分支。

## 当前状态

当前代码库同时维护三个不同的事实边界，不能混为一个“已部署模型”：

| 范围 | 当前状态 |
| --- | --- |
| 研究冠军 | Loop151 trusted signer guard，legacy full-test F1 `0.9908541911`，`1466/160000` errors |
| Native 离线参考 | Loop28，raw executable，但 Python/native parity 仍 blocked，不是质量冠军 |
| Connected system | 尚不存在 |

Loop151 的 Val、Test-10k 和 legacy full-test 指标来自冻结的开发评测协议，不代表未来时间/家族/来源隔离 sealed test，也不代表已经完成生产部署。当前冠军注册表位于 `manifests/roadmap_9997/champion_registry.json`，研究结论和限制记录在 `docs/ml_improvement_recommendations.md`。

## 环境要求

- Python `>=3.10`
- PyTorch `>=2.1`
- Windows 是默认开发环境；PE 解析、Authenticode 相关实验和部分 native 工具依赖 Windows
- CUDA 可选；没有 CUDA 时使用 `--device cpu`
- Rust toolchain 仅在构建嵌套扫描器或 DLL 时需要

项目原生依赖定义在 `pyproject.toml`，完整依赖列表在 `requirements.txt`。推荐使用虚拟环境解释器运行所有命令：

```powershell
& ".\vnev\Scripts\python.exe" -m pip install -r requirements.txt
```

开发测试依赖：

```powershell
& ".\vnev\Scripts\python.exe" -m pip install -e ".[dev]"
```

## 项目结构

```text
src/
  config.py                 实验、模型、训练和增强配置
  model.py                  AxonMalwareModel 与 HybridLightGBMModel
  dataset.py                原始文件、NPZ、缓存和 split 数据集
  trainer.py                训练、评估、AMP、SWA、EMA 和阈值扫描
  security.py               受限 checkpoint 加载
  feature_mask.py           PE/stat 输入特征掩码
  kvd_features/             字节、PE、统计和轻量级特征提取
  dsra/                     MHDSRA2、分页记忆和兼容层
scripts/main.py             train/eval/predict/extract/importance 入口
config/                     主配置和实验配置
tests/                      pytest 测试
Pro/rl_axon/                独立的一阶 bandit/policy-gradient 实验
tools/archive_scanner/      Rust 嵌套压缩包扫描器
tools/predict_dll/          Rust DLL 外部调用封装
manifests/                  实验授权、来源和结果合同
reports/                    评测、审计和实验报告
```

## 数据约定

### 原始文件目录

默认从目录名推断标签，推荐使用明确的目录结构：

```text
data/
  benign/       # label 0
  malicious/    # label 1
```

也可以使用文件名推断，但对正式实验建议使用显式 split 文件或带 `source_sha256` 的缓存 manifest。未知标签在默认的 `strict_unknown_labels = true` 下会被拒绝，而不是静默归类。

### NPZ 与缓存

训练可读取原始 PE 文件，也可读取已经准备好的 NPZ/cache。训练样本至少需要：

```text
byte_sequence
pe_features
label
```

`stat_features` 可选，缺失时使用零向量；`lightweight_features` 和 `source_sha256` 可用于缓存与审计。`extract` 命令生成的 NPZ 不一定包含 `label`，因此不能默认视为可直接训练的数据集。

缓存位于 `data/.cache`，使用配置哈希、manifest 和 `source_sha256` 校验来源。正式评估时应优先使用已经冻结并审计过的 cache/split，而不是临时重新扫描目录。

### 特征 schema

- 默认配置使用 `pe_schema_version = "fixed_v2"`、`pe_feature_dim = 256`、`pe_fixed_section_slots = 32`
- 兼容配置 `legacy_v3_candidate.toml` 使用旧的动态 PE schema，通常为 1500 维
- 统计特征默认 49 维，由 `stat_segment_count` 和 `stat_chunk_count` 通过配置公式派生
- 轻量级特征默认 256 维，当前主要用于缓存和辅助实验

不要混用不同 PE schema 的 checkpoint、cache 和配置；模型输入维度和 schema 必须一致。

## 快速开始

先运行不访问真实数据的冒烟测试：

```powershell
& ".\vnev\Scripts\python.exe" scripts/smoke_test_simple.py
```

训练主配置：

```powershell
& ".\vnev\Scripts\python.exe" scripts/main.py train `
  --config config/default_config.toml `
  --data-dir data `
  --device cuda `
  --resource-guard-json reports/resource_guard_train.json
```

训练、评估、特征提取和重要性分析属于重操作，`scripts/main.py` 要求先提供资源闸门回执。生成回执时，`--receipt-command` 必须与实际执行的命令逐 token 对齐，例如：

```powershell
& ".\vnev\Scripts\python.exe" scripts/pre_run_resource_leak_guard.py `
  --target-script scripts/main.py `
  --output-json reports/resource_guard_train.json `
  --receipt-command=train `
  --receipt-command=--config `
  --receipt-command=config/default_config.toml `
  --receipt-command=--data-dir `
  --receipt-command=data `
  --receipt-command=--device `
  --receipt-command=cuda `
  --receipt-command=--resource-guard-json `
  --receipt-command=reports/resource_guard_train.json
```

快速模式只适合功能验证和回归，不是正式指标：

```powershell
& ".\vnev\Scripts\python.exe" scripts/main.py train `
  --config config/default_config.toml `
  --data-dir data `
  --fast `
  --samples-per-class 200 `
  --epochs 2 `
  --device cpu `
  --resource-guard-json reports/resource_guard_fast.json
```

## 命令入口

统一入口是 `scripts/main.py`，当前提供五个子命令：

### `train`

训练 `AxonMalwareModel`，支持 Adam/AdamW/SGD、Cosine/Step 调度、AMP、梯度裁剪、早停、SWA、EMA、近阈值加权、稀有相似族群加权和 checkpoint resume。

```powershell
& ".\vnev\Scripts\python.exe" scripts/main.py train --help
```

`--resume` 会恢复优化器、调度器和训练状态；`--init-checkpoint` 只初始化模型权重并重新开始训练状态。`--partial-init` 允许非严格权重初始化，应只在明确知道结构差异时使用。

### `eval`

对指定 checkpoint 评估 `train`、`val`、`test` 或 `all` split，并输出 Accuracy、Precision、Recall、F1、AUC、TP/TN/FP/FN、FPR/FNR 等指标。

```powershell
& ".\vnev\Scripts\python.exe" scripts/main.py eval `
  --checkpoint models/best_model.pt `
  --data-dir data `
  --split val `
  --output reports/eval_val.json `
  --resource-guard-json reports/resource_guard_eval.json
```

阈值扫描只能在 `val` split 执行。不要用 `test` 或 `all` 反向选择阈值；选定阈值后，再对 test 做一次冻结评估。

### `predict`

对单个 PE 文件进行预测。该命令默认不要求资源闸门；使用 `--scan-nested` 时会启用嵌套扫描限制和资源闸门。

```powershell
& ".\vnev\Scripts\python.exe" scripts/main.py predict `
  --file samples/example.exe `
  --checkpoint models/best_model.pt `
  --device cpu
```

### `extract`

从原始目录提取字节、PE、统计和轻量级特征并写入输出目录：

```powershell
& ".\vnev\Scripts\python.exe" scripts/main.py extract `
  --data-dir data `
  --output-dir data/extracted `
  --max-workers 4 `
  --resource-guard-json reports/resource_guard_extract.json
```

注意：CLI 暴露了 `--max-workers`，但当前 `extract_command()` 尚未把它接入实际提取并发流程；不要把它当作已经生效的性能控制开关。

### `importance`

使用 checkpoint 对 PE/stat 输入做梯度重要性排名，输出 JSON 和 CSV：

```powershell
& ".\vnev\Scripts\python.exe" scripts/main.py importance `
  --checkpoint models/best_model.pt `
  --data-dir data `
  --split val `
  --output-json reports/feature_importance.json `
  --output-csv reports/feature_importance.csv `
  --resource-guard-json reports/resource_guard_importance.json
```

## 固定 split 与族群隔离

如果数据中存在近重复或相似族群，随机切分可能造成 train/test 泄漏。正式实验应使用已经审计过的 split 文件，例如：

```powershell
& ".\vnev\Scripts\python.exe" scripts/main.py train `
  --config config/default_config.toml `
  --data-dir data `
  --split-file reports/raw_group_diagnostics/group_isolated_split.csv `
  --output-dir models/group_isolated `
  --resource-guard-json reports/resource_guard_group_isolated.json
```

训练时可用 `--rare-group-weighting` 提高 singleton 和小相似族群的 loss 权重。它不会修改数据标签、移动文件或改变验证/测试指标的计算方式。

## 嵌套压缩包预测

`predict --scan-nested` 可扫描 zip、7z、cab、msi 等容器中的内层 PE，并按“任一内层 PE 判为 malicious，则外层对象报警”的运行时规则汇总结果。RAR 当前只识别并阻断，不依赖本机 7-Zip 静默解包。

扫描器源码位于 `tools/archive_scanner`，构建：

```powershell
cd tools/archive_scanner
cargo build --release
```

训练标签与运行时报警规则不同：内层样本默认不能继承外层容器标签进入训练缓存，应由独立数据流程确认标签。

## 预测 DLL

`tools/predict_dll` 提供面向 C/C++/C#/Rust 等调用方的 Rust DLL 外壳，复用现有 Python/PyTorch 预测链路，不是另一套模型实现。

```powershell
cd tools/predict_dll
cargo build --release
```

构建前请确认 Rust 工具链和 `AXON_PROJECT_ROOT`、`AXON_PYTHON` 环境变量。C 头文件和示例位于 `tools/predict_dll/include/axon_predict.h` 与 `tools/predict_dll/examples/predict_example.c`。

## RL 实验分支

`Pro/rl_axon` 是独立的一阶 bandit/policy-gradient 实验，不属于主分类器训练链路。它复用 `AxonMalwareModel` 作为策略网络，把动作定义为：

- `0`：判为 benign
- `1`：判为 malicious

入口和说明见 `Pro/README.md`。不要把 RL 实验结果直接与主分类器的监督学习指标混合比较。

## 安全与实验纪律

- checkpoint 通过 `src/security.py` 的受限加载路径读取，不应对不可信 checkpoint 做任意反序列化。
- 默认 `strict_pe_parsing = true`、`allow_pe_fallback = false`；关键特征提取失败会拒绝样本，而不是生成不明来源的替代特征。
- `use_swanlab`/`enable_swanlab` 默认关闭；显式开启前确认路径、配置和指标的上传边界。
- 测试集只用于确认，不用于阈值、特征 mask 或候选策略选择。
- 研究冠军、native parity 和 connected deployment 是三个独立状态，不能用一个范围的证据替代另一个范围。

## 测试与代码质量

```powershell
& ".\vnev\Scripts\python.exe" -m pytest
& ".\vnev\Scripts\python.exe" -m ruff check .
```

涉及具体模块时，优先运行对应测试文件，再扩大到完整测试集。完整项目测试可能依赖 Windows、CUDA、缓存 manifest 或外部实验 artifact；缺失这些前提时，应把结果记录为环境限制，而不是伪造模型结论。

## 重要文档

- `docs/ml_improvement_recommendations.md`：当前研究路线、冠军和候选治理
- `manifests/roadmap_9997/champion_registry.json`：机器可读冠军注册表
- `docs/eval_protocol_audit.md`：评估协议与泄漏边界
- `docs/identity_feature_policy.md`：身份字段与模型证据边界
- `docs/ml_experiment_authorization_plan.md`：实验授权和数据准入纪律
- `Pro/README.md`：RL 实验分支

## License

Apache License 2.0. See [LICENSE](LICENSE).
