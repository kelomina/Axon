# Axon

面向 Windows PE 文件的机器学习恶意软件检测引擎。源自 Axon v2.6-EXP 实验，当前为 **v3.0-Pre** 预发布版本（pyproject `3.0.0`）。

> **仓库命名约定**：仓库名不携带版本号（曾用名 `Axon_V2.6Exp`，2026-08-12 起改名 `Axon`，旧地址自动重定向）。版本号由 git tag 与 GitHub Release 承载。

## 项目定位

Axon 对 Windows PE 文件做 benign/malicious 二分类。主链路融合三种视角：

- **原始文件字节序列**（`byte_seq`，训练时截断至 4096）
- **PE 结构特征**（`pe_features`，生产主干用 legacy_dynamic 1500 维）
- **统计特征**（`stat_features`，49 维）

字节序列由 **MHDSRA2/DSRA** 流式编码器处理，PE 与统计特征分别投影后融合，进入分类头。最终决策由 **Stage-2 内容特征 HGB 纠错层**完成。

## 当前状态（诚实口径）

生产主干、研究线程、部署链路是三个独立事实边界，指标不可互换：

| 范围 | 状态 |
| --- | --- |
| **生产主干**（739k/813k 基座 + Stage-2） | 冻结测试集 F1 **0.99199**（fp 976 / fn 860，errors 1836）；best 产物已发布 |
| **研究线程** | Loop151 trusted signer guard（开发期 leaderboard，legacy full-test F1 0.99085），见 `manifests/roadmap_9997/champion_registry.json` |
| **部署链路** | 代码路径存在（`predict_api` → Rust DLL / ONNX DLL，单文件 ~355 ms），但未完成严格生产认证 |

测试集是冻结的 7:1:2 分层划分（seed 42），属开发期协议，**不代表**未来时间/家族/来源隔离的 sealed test，也不代表已上线生产环境。

### 关键指标（冻结测试集）

| 指标 | 值 |
| --- | --- |
| Stage-2 test F1（threshold 0.55） | 0.99199（val_best F1 0.99608） |
| Stage-2 Precision / Recall | 0.99149 / 0.99250 |
| 基座 test F1（良性扩充后重训，epoch19） | 0.97648（FPR 0.0716，AUC 0.99289） |
| 基座 FPR 变化 | 良性扩充 74k 后 0.09695 → 0.0716（−26%） |

## 架构

### 基座模型：`AxonMalwareModel`（`src/model.py`）

- 三路输入：`byte_seq`（ByteEmbedding 256→128）+ `pe_features`（PEFeatureProjector 1500→256→128）+ `stat_features`（stat_projector 49→128）→ concat 384 → classifier
- 编码器：**MHDSRA2**（`src/dsra/mhdsra2/improved_dsra_mha.py`），默认 2 层堆叠；DSRA 配置 dim=128 / heads=4 / slots=128 / read_topk=8 / write_topk=4
- 可训练参数 ≈ 696k（另有 8M+ 非训练正弦位置编码 buffer）

### Stage-2 内容特征 HGB 纠错层

- 331 维内容特征：base-prob derived + content_pe_v1 + content_pe_v2 + content_string
- 3-seed HistGradientBoosting ensemble，是**最终决策层**，对基座输出纠错（FP/FN 大幅下降）
- 训练：`scripts/stage2_739k_v2_benign.py`；SKLearn 模型带 SHA-256 trust manifest 绑定（`manifests/roadmap_9997/p0_raw_replay/pickle_sha256_allowlist.json`）

### 硬约束

- **FP32 强制**：DSRA 在 FP16 下前向溢出产生 NaN（`mixed_precision=False` 是硬约束）；bf16 实测无 NaN 但 kernel-bound 无加速；TF32 尾数截断导致 loss 轨迹偏移，已否决

## 数据与语料

- 特征缓存 **813,098** 样本（739k 全量 + 2026-08-10 良性扩充 +74k）
- 划分：7:1:2（train/val/test）分层，seed 42；字节序列训练截断至 4096
- 特征 schema：`byte_seq`（截断 4096）/ PE `legacy_dynamic` 1500 维 / stat 49 维
- 缓存与 manifest：`data/.cache/` + 带 `source_sha256` 的 manifest，用配置哈希校验来源
- **注意**：`config/default_config.toml` 是另一条 `fixed_v2` 256 维 PE 的研究/开发配置轨道，与生产主干 1500 维**不兼容**；不要混用两套 checkpoint/cache/config

### 标签治理

- 跨树冲突清理（同一 sha256 同时存在于良性树与恶意树）用 `corpus-conflict-cleanup` skill 物理归位 + 重建缓存
- 标签与数据准入遵循 `docs/ml_experiment_authorization_plan.md`

## 训练与重训

生产主干脚本（`scripts/`）：

```powershell
& ".\vnev\Scripts\python.exe" -u scripts\train_739k_full.py            # 全量重训（20 epochs ≈ 27h）
& ".\vnev\Scripts\python.exe" -u scripts\train_739k_benign_hardneg.py  # k=5.0 良性难例加权微调
& ".\vnev\Scripts\python.exe" -u scripts\stage2_739k_v2_benign.py      # Stage-2 HGB 训练
```

- 关键训练配置：batch 64、LR 8e-5、label_smoothing 0.03、focal γ=1.0 α=0.55、diversity_loss_weight 0.03、gradient_clip 0.75
- **Windows 多进程加载必须限制 BLAS 线程**：脚本顶部须在 import numpy/torch 前设 `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1`，否则 spawn worker 线程栈分配失败卡死
- best-model 选择用 `best_metric='goal'`（F1 − 5×FPR），显式惩罚误报
- 产物：`models/full_739k_benign/best_model_739k.pt`（epoch19）、`models/full_739k_benign_hardneg/final_model_739k_hardneg.pt`

### 性能

- `torch.compile(mode='reduce-overhead')`（CUDA graph）：训练真实提速 −46%，推理单样本 −94%（约 3.9 ms/sample）
- 需 `triton-windows`；FP32 恒定

## 推理与部署

生产推理入口是 `src/predict_api.py`（`predict_file` / `predict_json`，输出稳定 JSON），外部程序复用；`scripts/main.py predict` 提供命令行单文件预测：

```powershell
& ".\vnev\Scripts\python.exe" scripts\main.py predict --file <exe> --checkpoint models\full_739k_benign\best_model_739k.pt --device cuda
```

部署形态：

| 形态 | 路径 |
| --- | --- |
| 单文件推理 API | `src/predict_api.py`（基座 prob + Stage-2 + FamilyClassifier） |
| Rust DLL | `tools/predict_dll`（复用 predict_api，面向 C/C++/C#/Rust 调用方） |
| ONNX 单模型 | `scripts/export_onnx_model.py` → `tools/axon_onnx_dll`（ONNX Runtime CPU，单文件 ~355 ms） |
| Stage-2 免 sklearn 导出 | `scripts/export_stage2_hgb_json.py`（`axon_stage2_hgb_json_v1`，C++ 可读） |

## 命令入口

### `scripts/main.py`（通用 CLI，五个子命令）

`train` / `eval` / `predict` / `extract` / `importance`。重操作（训练/评估/提取/重要性）要求先提供资源闸门回执：

```powershell
& ".\vnev\Scripts\python.exe" scripts\pre_run_resource_leak_guard.py `
  --target-script scripts/main.py --output-json reports/resource_guard_train.json `
  --receipt-command=train --receipt-command=--config `
  --receipt-command=config/default_config.toml --receipt-command=--data-dir --receipt-command=data

& ".\vnev\Scripts\python.exe" scripts\main.py train `
  --config config/default_config.toml --data-dir data --device cuda `
  --resource-guard-json reports/resource_guard_train.json
```

## 实验轨道（不是主模型）

以下均为独立实验，各自 checkpoint + 收据，**未接入生产主干/部署路径**，README 不应把它们描述为主模型：

- **loop2xx 专家**：`src/loop202_whole_file_streamer`、`loop208_rich_header_fusion`、`loop216_graph_expert`、`loop222_stream_gnn_fusion` 等，由 `scripts/run_loopXXX.py` 单独训练评测（如 Loop222 自标注 val F1 0.6742 / 0.6618）
- **Flash+Pro 级联**：`src/axon_flash_pro_cascade.py`（Flash=HistGBDT 快判 + Pro=Speakeasy-X 动态仿真，uncertainty 区间升级；loop233-243 是 Speakeasy-X 多入口/TLS/C2 沙箱实验）
- **RL 分支**：`Pro/rl_axon`（独立一阶 bandit/policy-gradient，不要与监督学习指标混比）

## 工具

- `tools/archive_scanner`：Rust 嵌套压缩包扫描（zip/7z/cab/msi），`predict_api` 复用其迭代 PE 预测目标
- `tools/predict_dll`：Rust DLL 外壳，复用 Python/PyTorch 预测链路
- `tools/axon_onnx_dll`：C++ ONNX Runtime DLL（免 Python 部署形态）
- `tools/axon_onnx_fidelity` / `tools/axon_tiny_pytorch_native`：ONNX 保真 / native parity 诊断

## 发布约定

- **模型/数据大文件走 GitHub Releases，不走 git**（`.gitignore` 排除 `models/`、`*.pt` 等）；已发布 `v2.6-739k-models`（4 个 .pt，附 SHA-256）
- 发布流程见 `.claude/skills/git-commit-push-release`（gh CLI 缺失时从 Windows Credential Manager 提 token + curl）
- 语料冲突清理流程见 `.claude/skills/corpus-conflict-cleanup`

## 环境要求

- Python `>=3.10`；PyTorch `>=2.1`；Windows 为默认开发环境；CUDA 可选（无则 `--device cpu`）
- `torch.compile` 需要 `triton-windows`
- 虚拟环境：`.\vnev\Scripts\python.exe`

```powershell
& ".\vnev\Scripts\python.exe" -m pip install -r requirements.txt
& ".\vnev\Scripts\python.exe" -m pip install -e ".[dev]"
```

## 项目结构

```text
src/
  model.py                    AxonMalwareModel（基座）与 HybridLightGBMModel
  dsra/mhdsra2/               MHDSRA2 编码器正式实现（improved_dsra_mha.py / paged_exact_memory.py）
  kvd_features/               字节 / PE / 统计 / 内容特征提取
  predict_api.py              生产推理 API（predict_file / predict_json）
  axon_flash_pro_cascade.py   Flash+Pro 级联实验
  loop2xx_*.py                独立实验专家（不接入主干）
  config.py / trainer.py / dataset.py / security.py / feature_mask.py
scripts/
  main.py                     通用 CLI（train/eval/predict/extract/importance + 资源闸门）
  train_739k_full.py          生产全量训练
  train_739k_benign_hardneg.py  良性难例加权微调
  stage2_739k_v2_benign.py    Stage-2 HGB 训练
  export_onnx_model.py / export_stage2_hgb_json.py   部署导出
  run_loopXXX.py              实验轨道脚本
config/                       主配置 + 实验配置（fixed_v2 256 研究轨 vs 生产 legacy_dynamic 1500）
tools/                        archive_scanner / predict_dll / axon_onnx_dll 等
Pro/rl_axon/                  独立 RL 实验
manifests/roadmap_9997/       冠军注册表 / 审计 manifests / trust 清单
reports/                      评测 / 审计 / 实验报告
docs/                         交接文档 / 提案 / 案例复盘
```

## 安全与实验纪律

- checkpoint 走 `src/security.py` 受限加载，不对不可信 checkpoint 做任意反序列化
- 默认 `strict_pe_parsing=true`、`allow_pe_fallback=false`；关键特征提取失败拒绝样本
- **FP32 硬约束**：不要为提速改 `mixed_precision`（FP16 产生 NaN）
- 测试集只用于确认，不用于阈值 / 特征 mask / 候选选择；阈值扫描只能在 val 上执行
- 生产主干、研究冠军、部署链路是三个独立事实边界，不能用单一范围的证据替代另一个

## 测试与代码质量

```powershell
$env:PYTHONPATH="src"; & ".\vnev\Scripts\python.exe" -m pytest
& ".\vnev\Scripts\python.exe" -m ruff check .
```

（项目 `src` 在 `PYTHONPATH` 上；直接 `pytest` 会 `ModuleNotFoundError`。完整测试可能依赖 Windows / CUDA / 缓存 manifest。）

## 重要文档

- `docs/handoff_full_739k_training_2026-08-07.md`：739k 全量训练交接（硬约束 / 踩坑 / 产物）
- `docs/ml_improvement_recommendations.md`：研究路线与候选治理
- `manifests/roadmap_9997/champion_registry.json`：机器可读冠军注册表
- `docs/code_project_case_studies.md`：项目案例复盘
- `docs/release_notes_axon_v2_6_exp_checkpoint_260702.md`：v2.6 发布说明（v3 定位）

## License

Apache License 2.0. See [LICENSE](LICENSE).
