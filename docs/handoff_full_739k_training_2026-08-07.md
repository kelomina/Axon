# Axon 739k 全量训练交接文档

交接时间：2026-08-07（训练完成于 2026-08-07 06:03，Asia/Shanghai）

## 1. 任务背景与最终状态

**任务目标**（用户原始需求）：
1. 使用现有缓存以 7:1:2（训练:验证:测试）比例执行全量训练
2. 启用 PyTorch 多进程数据加载（num_workers=4~8）+ pin_memory=True + persistent_workers=True
3. Batch Size 从 32 调大到 64 或 128

**最终状态：全部完成，全量训练已跑通并产出模型。**

| 指标 | 值 |
|---|---|
| Test F1 | 0.9795 |
| Test Acc | 0.9679 |
| Test AUC | 0.9921 |
| Precision / Recall | 0.9723 / 0.9867 |
| FPR / FNR | 0.0969 / 0.0133 |
| 训练时长 | 27.23 h（20 epochs，早停未触发） |

## 2. 产物与位置

| 产物 | 路径 |
|---|---|
| best 检查点 | `models/full_739k/best_model_739k.pt`（41824805 B，2026-08-07 05:47） |
| final 检查点 | `models/full_739k/final_model_739k.pt` |
| 训练日志 | `reports/full_739k/train_full.log`（844 KB，完整 20 epochs） |
| 训练错误日志 | `reports/full_739k/train_full_err.log`（仅含收尾 receipt 序列化错误，已修复） |
| 训练 receipt | `reports/full_739k/train_739k_receipt.json`（post-hoc 补生成，含 test 全套指标） |
| 数据缓存 | `data/.cache/`（738,983 个 NPZ + manifest_a807341e.json，pe=1500/stat=49/legacy_dynamic/65536） |

## 3. 改动文件清单

| 文件 | 改动 | 状态 |
|---|---|---|
| `scripts/train_739k_full.py` | 主训练脚本：数据加载优化 + 序列截断 + 稳定训练配置 + receipt 序列化修复 | untracked（git 未跟踪，live 文件为准） |
| `tests/test_train_739k_full.py` | 新增 9 个单元测试（比例/DataLoader 参数/截断/分层划分） | untracked |
| `.gitignore` | 追加 `docs/` 忽略 | tracked 已修改 |
| `docs/code_project_case_studies.md` | 项目案例复盘（困难/失败/成功方案/有效命令，最详细） | untracked（被忽略） |

**train_739k_full.py 关键配置（当前生效值）**：
- `BATCH_SIZE = 64`（实测训练峰值显存 2.24GB，128 估算 4.3GB 会 OOM）
- `NUM_WORKERS = 8` + `persistent_workers=True` + `pin_memory=True`（`_build_dataloaders()`）
- `TRUNCATE_BYTE_LENGTH = 4096`（序列截断，`_TruncatedByteDataset` 包装）
- `MAX_EPOCHS = 20`、`LR = 8e-5`、`gradient_clip = 0.75`、`label_smoothing = 0.03`、`focal_gamma = 1.0`、`focal_alpha = 0.55`、`diversity_loss_weight = 0.03`
- 脚本顶部（import numpy/torch 前）：`OMP_NUM_THREADS=1` + `OPENBLAS_NUM_THREADS=1`

## 4. 关键决策与硬约束（接手前必读）

这些是本次踩坑后的**事实约束**，全部实测验证：

1. **Windows 多进程加载必须限制 BLAS 线程**：脚本顶部 `os.environ.setdefault("OMP_NUM_THREADS","1")` / `OPENBLAS_NUM_THREADS("1")` 必须在 import numpy/torch 之前。否则 8 个 spawn worker 各初始化 32 线程 OpenBLAS，线程栈分配失败，训练卡死（"OpenBLAS error: Memory allocation still failed"）。
2. **不要增大 dsra_chunk_size 加速**：DSRA 是 chunk 内全量注意力，chunk_size 翻倍 → 显存 4 倍（512→2.24GB，1024→4.23GB，2048→8.25GB 已超 8GB 上限），耗时反增。
3. **不要用 FP16**：脚本 `mixed_precision=False` 是硬约束（DSRA FP16 产生 NaN）。bf16 实测无 NaN 但无加速收益（瓶颈是 Python chunk 循环），未采用。
4. **全量训练必须用稳定配置**：裸配置（LR=1e-4、无 label_smoothing/focal）在 1:3.45 类别不平衡（benign 165,916 : malware 573,067）下，模型快速过度自信 → logits 溢出 NaN（FloatingPointError 崩溃，浪费 3h）。当前配置对齐 `config/default_config.toml` 已验证 20 epochs 无 NaN。
5. **序列截断是决定性加速**：65536→4096 单步 9.2s→0.5s（18 倍）。`FeatureCacheDataset` 的 cache hash 包含 max_byte_length，不能改数据集参数，必须用 `_TruncatedByteDataset` 包装类在 split 后截断。
6. **训练速度基准**（batch=64、截断 4096、OMP 单线程）：单步 ~0.5s，1 epoch ≈ 1.6h，20 epochs ≈ 27h。
7. **训练配置复现公式**：739k 全量训练可用 `& "vnev\Scripts\python.exe" -u scripts\train_739k_full.py` 后台运行（Start-Process + 日志重定向），结果落盘 `models/full_739k/` 与 `reports/full_739k/`。
8. **GPU 实际规格**：RTX 4070 Laptop 8GB（用户口述的 5.4GB 不准确），决策前实测显存。

## 5. 验证证据

- 9 个单元测试全通过：`& "vnev\Scripts\python.exe" -m pytest tests\test_train_739k_full.py -q --tb=short`
- ruff 通过：`& "vnev\Scripts\python.exe" -m ruff check scripts\train_739k_full.py tests\test_train_739k_full.py`
- 多进程加载端到端：8 worker + persistent_workers，epoch 2 加载仅 0.1s（worker 常驻 + OS 文件缓存命中）
- 小样本链路试跑：512 样本 7:1:2 划分（360/50/102）6 epochs 无 NaN，loss 0.187→0.143
- 全量训练：20 epochs 27.23h，Val F1 0.929→0.9800 持续上升（早停 patience=8 未触发），Test 复评与训练时结果一致（F1 0.9795）

## 6. 剩余风险与下一步建议

**剩余风险**：
- receipt 为 post-hoc 补生成（`mode: posthoc_receipt`），训练时主流程的 `results`（train/val 全历史）未落盘；如需逐 epoch 指标历史，需从 `train_full.log` 解析或重跑评估。
- 测试集 FPR 0.0969（FP 3217 / 147,796），若业务对误报敏感，可考虑 `threshold_sweep` 调低判定阈值或后续 calibration。
- 训练用截断 4096 字节序列，与推理时全量 65536 字节语义存在差异（模型只见过前 4KB）；若部署推理走 65536，建议评估一致性或对齐推理输入。

**下一步建议**：
1. 用 `scripts/main.py eval --checkpoint models/full_739k/best_model_739k.pt --data-dir data` 独立复评并做阈值扫描（`--sweep-thresholds`）
2. 如需续训：`AxonTrainer.load_checkpoint` 支持恢复（`_resumed_epoch`），可写续训脚本从 best 检查点继续
3. 评估模型在真实 PE 文件上的单文件推理：`scripts/main.py predict --file <exe> --checkpoint models/full_739k/best_model_739k.pt`
4. 若考虑全序列（65536）训练，需先解决单步 9s 的速度问题（当前无低成本方案，chunk_size 增大已否决）

## 7. 环境与约定

- 虚拟环境：`E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe`（Python 3.14.4，torch 2.12.0+cu132，pytest 9.0.3）
- Git：`C:\Program Files\Git\bin\git.exe`；PowerShell 串联用 `;` 不用 `&&`
- 机器：32GB RAM，16 核 32 线程，RTX 4070 Laptop 8GB
- Windows spawn 约束：多进程 DataLoader 代码必须存为 .py 文件执行（不能 stdin），迭代须在 `if __name__ == "__main__"` 保护块内
- `train_739k_full.py` 与 `tests/test_train_739k_full.py` 尚未提交 git（untracked），接手时以 live filesystem 为准；如需提交请用户明确授权
- 详细踩坑记录见 `docs/code_project_case_studies.md`
