# Axon v2.6 Experiment

## 概述

Axon v2.6 是一个混合恶意软件检测模型，结合了 DSRA（Dynamic Slot-based Retrieval Attention）流式注意力机制和 KVD（Knowledge-based Vector Descriptor）特征提取器。

### 架构特点

- **DSRA 流式注意力**：基于动态槽位检索的注意力机制，支持长序列处理
- **KVD 特征提取**：从 PE 文件中提取丰富的结构特征和统计特征
- **特征融合**：支持 concat、add、attention 三种融合策略
- **端到端训练**：完整的训练流程，支持早停、学习率调度等

### 项目结构

```
├── src/              # 核心源码
│   ├── axon_exp.py   # 模块入口
│   ├── config.py     # 配置模块
│   ├── model.py      # 模型定义
│   ├── trainer.py    # 训练器
│   ├── dataset.py    # 数据集
│   ├── dsra/         # DSRA 核心模块
│   └── kvd_features/ # KVD 特征提取
├── config/           # 配置文件
├── scripts/          # 工具脚本
├── data/             # 数据目录
└── reports/          # 报告目录
```

### 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行冒烟测试
python scripts/smoke_test_simple.py

# 启动训练（主配置：fixed_v2 PE256，默认读取 data 目录）
python scripts/main.py train --config config/default_config.toml

# 小规模回归旧 PE1500 legacy 候选
python scripts/main.py train --config config/legacy_v3_candidate.toml --fast
```

默认数据切分为 16% 训练集、4% 验证集、80% 测试集。验证集用于训练过程中选择最佳模型和早停，测试集只在训练结束后评估一次。
快速训练模式默认每类最多取 10000 个样本（白 10000 + 黑 10000，总计 2 万）并训练 10 个 epoch。可以用 `--samples-per-class` 临时覆盖每类样本数。

### 相似族群隔离与小族群加权训练

当原始样本里存在大量相似族群时，随机切分可能让同一族群的近亲样本同时出现在训练集和测试集，导致评估分数偏高。项目支持用相似度诊断生成的 `group_isolated_split.csv` 作为固定切分清单，让同一个相似族群只进入 train、val、test 之一：

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts/main.py train --config config/default_config.toml --data-dir data --split-file reports/raw_group_diagnostics/group_isolated_split.csv --output-dir models/group_isolated_fast512 --fast --samples-per-class 20000 --epochs 50
```

如果已经有一个无泄漏切分训练出的 `best_model.pt`，并希望继续加强小族群，可以使用 `--init-checkpoint` 做权重初始化。它只加载模型参数，优化器、学习率进度和早停状态会重新开始，适合“换训练策略后的微调”。这和 `--resume` 不同：`--resume` 会连优化器和早停状态一起恢复，主要用于训练中断后的断点续训。

小族群加权训练推荐从保守权重开始。它不会复制、删除或移动数据，只是在训练 loss 里让小族群样本更有存在感；验证集和测试集仍然按普通指标计算：

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts/main.py train --config config/default_config.toml --data-dir data --split-file reports/raw_group_diagnostics/group_isolated_split.csv --output-dir models/group_isolated_rare_weighted_ft --fast --samples-per-class 20000 --epochs 8 --init-checkpoint models/group_isolated_fast512/best_model.pt --rare-group-weighting --singleton-group-weight 1.8 --rare-group-weight 1.5 --medium-group-weight 1.2
```

权重含义：

- `singleton-group-weight`: 只有 1 个样本的族群，默认建议 `1.8`。
- `rare-group-weight`: 2-5 个样本的小族群，默认建议 `1.5`。
- `medium-group-weight`: 6-20 个样本的中小族群，默认建议 `1.2`。
- 21 个及以上样本的族群保持 `1.0`，避免大族群继续挤压小族群。

### 安全默认行为

模型 checkpoint 只应加载可信来源文件。评估、预测、恢复训练和诊断导出都会使用受限的 PyTorch checkpoint 加载方式，并要求 checkpoint 包含模型参数和配置字段。

SwanLab 实验追踪默认关闭，不会自动上传路径、配置或指标。需要上传时，在配置中设置 `enable_swanlab = true`，或训练时显式添加 `--enable-swanlab`。

### 嵌套压缩包/MSI 检测

项目包含一个 Rust 编写的前置解包器，位置在 `tools/archive_scanner`。它用于扫描 zip、rar、7z、msi、cab 这类高风险外壳，找出里面的 PE、MSI 或二次嵌套压缩包，再把内层 PE 交给 Axon 现有预测链路。

这个解包器不依赖本机安装的 7-Zip、NanaZip 或 `7z.exe`。ZIP、7z、CAB 由 Rust 库在扫描器内部解包；MSI 会读取安装包里的内嵌 binary stream，并把抽出的 PE/CAB/压缩包继续递归审计。RAR 第一版只做格式识别并在报告里标记为 `blocked`，原因是当前没有足够可靠的纯 Rust RAR 解压后端；这里选择安全阻断，而不是偷偷调用本地 7z。

先构建 Rust 解包器：

```powershell
cd "E:\Project\python\Axon_v2.6Exp\tools\archive_scanner"; cargo build --release
```

调试时只看嵌套树：

```powershell
cd "E:\Project\python\Axon_v2.6Exp"; .\tools\archive_scanner\target\release\axon-archive-scanner.exe --input ".\sample.zip" --output text
```

对压缩包或 MSI 做嵌套预测：

```powershell
cd "E:\Project\python\Axon_v2.6Exp"; & "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts/main.py predict --file ".\sample.msi" --checkpoint ".\models\group_isolated_rare_weighted_ft_rebuilt_cache\best_model.pt" --device cpu --scan-nested
```

运行时报警规则是：只要任一内层 PE 被 Axon 判为恶意，外层压缩包/MSI 就报警。训练规则不同：MSI 可能同时包含白文件和黑文件，所以解包器报告里的内层候选默认标记为 `unknown_training_label`，不能自动继承父目录黑白标签进入训练缓存。

### 预测 DLL 接口

项目提供一个 Rust 编写的 DLL 外壳，位置在 `tools/predict_dll`。它不是重写一套模型，而是给 C/C++/C#/Rust 等外部程序一个稳定的 DLL 入口：外部程序传入 JSON 请求，DLL 调用项目现有 Python/PyTorch 预测链路，再返回 JSON 结果。这样第一版能保持预测结果和 `scripts/main.py predict` 一致。

先构建 DLL：

```powershell
cd "E:\Project\python\Axon_v2.6Exp\tools\predict_dll"; cargo build --release
```

生成文件：

```text
tools\predict_dll\target\release\axon_predict.dll
tools\predict_dll\target\release\axon_predict.dll.lib
tools\predict_dll\include\axon_predict.h
```

外部程序调用前建议设置两个环境变量：

```powershell
$env:AXON_PROJECT_ROOT = "E:\Project\python\Axon_v2.6Exp"
$env:AXON_PYTHON = "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe"
```

请求 JSON 示例：

```json
{
  "file": "E:/samples/sample.exe",
  "checkpoint": "E:/Project/python/Axon_v2.6Exp/models/group_isolated_rare_weighted_ft_rebuilt_cache/best_model.pt",
  "device": "cpu",
  "scan_nested": false
}
```

如果要对压缩包/MSI 做嵌套预测，把 `scan_nested` 改成 `true`。DLL 返回的字符串由 DLL 分配，调用方必须用 `axon_string_free` 释放。C/C++ 程序可以链接 `axon_predict.dll.lib`，也可以用 `LoadLibrary` / `GetProcAddress` 运行时加载 DLL。C 头文件和示例在 `tools/predict_dll/include/axon_predict.h` 与 `tools/predict_dll/examples/predict_example.c`。

### 性能指标

| 指标 | 预期值 |
|------|--------|
| Accuracy | > 98% |
| Precision | > 97% |
| Recall | > 97% |
| F1 | > 97% |
| AUC | > 0.99 |

### 文档

详细的模块架构文档请参考 `agents.md` 文件。

---

**License**: MIT
