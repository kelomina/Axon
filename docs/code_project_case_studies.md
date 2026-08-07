# Axon v2.6 项目案例复盘记录

本文件记录每次处理本项目时遇到的困难、失败尝试、成功方案、有效命令、环境注意事项与架构约定，供后续任务优先参考。本文件位于 docs/ 目录，默认不提交 git。

---

## 2026-08-05 - 全量训练数据加载优化（num_workers/persistent_workers/batch size）

### 任务背景
- 用户需求：使用现有缓存以 7:1:2（训练:验证:测试）比例执行全量训练；启用 num_workers=4~8 + pin_memory=True + persistent_workers=True；Batch Size 从 32 调大到 64 或 128。
- 涉及模块：scripts/train_739k_full.py（唯一满足 7:1:2 的全量训练脚本）、src/dataset.py（FeatureCacheDataset/SubDataset/create_stratified_split）、tests/test_train_739k_full.py（新增）。
- 涉及文件：scripts/train_739k_full.py、tests/test_train_739k_full.py、.gitignore、docs/code_project_case_studies.md。

### 关键事实（实测确认）
- data/.cache 存在 738,983 个 NPZ 缓存文件，manifest_a807341e.json 配置：pe_feature_dim=1500、stat_feature_dim=49、max_byte_length=65536、pe_schema_version=legacy_dynamic。
- GPU 是 NVIDIA GeForce RTX 4070 Laptop GPU，总显存 8GB（不是用户所说的 5.4GB，也不是台式 4070 的 12GB），空闲约 3GB。
- 显存实测（FP32 训练步，含 backward）：
  - batch=32: ~1.1GB（推断）／batch=64: 峰值 2.24GB，空闲余量 4.45GB／batch=128: 估算峰值 ~4.3GB，会顶到 8GB 上限，OOM 风险高。
  - 结论：batch_size=64 是安全选择；128 有 OOM 风险。
- 训练速度硬瓶颈（与数据加载无关）：batch=64 时 DSRA 模型训练模式单步 forward 约 56.7~68.9s（真实缓存样本 68.9s），backward 仅 0.3s。全量 738,983 样本 / 64 = 11,547 步/epoch，一个 epoch 约 221 小时（~9.2 天）。推理模式（no_grad）快得多，差异来自训练模式需要保留全部 chunk 激活图。
- 多进程数据加载验证（Windows spawn）：epoch 1 74.4s（含 8 个 spawn 进程启动 + 磁盘读），epoch 2 0.1s（persistent_workers 常驻 + OS 文件缓存命中）。

### 遇到的困难
- 困难 1：Windows 下 PyTorch DataLoader 默认 spawn 启动子进程；若主程序通过 stdin（`python -`）执行，子进程重新导入主模块时路径无效（`OSError: [Errno 22] Invalid argument: '<stdin>'`）。
- 困难 2：`torch.utils.data.DataLoader` 对象不暴露 `shuffle` 属性（该属性只影响内部 sampler 创建），测试断言 `loader.shuffle` 直接 AttributeError；必须通过 `isinstance(loader.sampler, RandomSampler/SequentialSampler)` 验证。
- 困难 3：PowerShell 控制台显示 UTF-8 中文为乱码，容易误判文件编码；实际文件是合法 UTF-8，乱码只是控制台输出编码问题。
- 困难 4（重要）：Windows 下 8 个 spawn worker 各自初始化 OpenBLAS 线程池（默认 = 逻辑核数 32 线程），并发加载 NPZ + 训练同时运行时出现大量 "OpenBLAS error: Memory allocation still failed after 10 retries"，训练卡死（>1 小时无进展）。
  - 根因：不是物理内存不足（机器 32GB），而是 8 进程 × 32 线程的线程栈/内部缓冲分配失败。
  - 修复：在脚本 import numpy/torch 之前设置 `os.environ.setdefault("OMP_NUM_THREADS", "1")` 和 `os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")`；数据加载是 I/O 密集，worker 内不需要多线程 BLAS。spawn 的 worker 会重导入主模块，模块级 env 设置自动对 worker 生效。
- 困难 5：Windows spawn 下创建 DataLoader/迭代必须放在 `if __name__ == "__main__":` 保护块内，否则 worker 重导入主模块时触发 "An attempt has been made to start a new process before the current process has finished its bootstrapping phase"。
- 困难 6：`SubDataset` 不暴露 `label_list` 属性，对它二次调用 `create_stratified_split` 会 AttributeError（真实脚本是对 FeatureCacheDataset 直接划分，无此问题；试跑脚本需自建带 label_list 的子集包装）。

### 失败尝试
- 尝试方案 1：用 `python -`（stdin）执行多进程 DataLoader 验证脚本。
- 失败原因 1：Windows spawn 模式下子进程执行 `runpy.run_path` 重新加载主模块，stdin 没有可寻址路径，spawn 失败。
- 证据 1：`OSError: [Errno 22] Invalid argument: 'E:\\Project\\python\\Axon_v2.6Exp\\<stdin>'`，进程挂起直到超时。
- 成功替代 1：把验证代码写成临时 .py 文件（C:\Users\Saika\AppData\Local\Temp\opencode\verify_mp_loader.py）后执行，spawn 可正常重导入主模块。
- 尝试方案 2：直接以 8 worker + 训练跑小样本试跑（未限制 OpenBLAS 线程）。
- 失败原因 2：worker OpenBLAS 多线程初始化失败，训练卡死。
- 证据 2：持续输出 "OpenBLAS error: Memory allocation still failed after 10 retries"（来自各 worker 进程 stderr），1 小时无进度。
- 成功替代 2：限制 OMP/OpenBLAS 线程为 1（shell env 或脚本内 setdefault），重跑通过。

### 成功方案
- 最终采用：修改 scripts/train_739k_full.py —— BATCH_SIZE 32→64、新增 NUM_WORKERS=8、提取模块级函数 `_build_dataloaders()`（三个 DataLoader 统一加 persistent_workers=True，pin_memory 按 device 传递）、receipt 增加 num_workers 字段、脚本顶部（import numpy/torch 前）设置 OMP_NUM_THREADS=1/OPENBLAS_NUM_THREADS=1；新增 tests/test_train_739k_full.py 6 个测试；.gitignore 追加 docs/ 忽略。
- 关键修改：`_build_dataloaders()` 使 DataLoader 参数可被单元测试直接验证（最小重构，不改变 main 行为）。
- 为什么该方案能成功：
  1) FeatureCacheDataset/SubDataset 状态全部可 pickle（Path/int 列表/np.ndarray），Windows spawn 下 8 进程并行读 NPZ 实测成功；
  2) persistent_workers=True 消除每 epoch 重复 spawn 的开销（epoch 2 仅 0.1s）；
  3) batch=64 显存峰值 2.24GB，在 8GB 显卡上留足余量，避免 128 的 OOM 风险；
  4) OMP/OpenBLAS 单线程化后 worker 内存分配稳定，8 worker 与训练共存无崩溃。

### 性能实测数据（重要修正）
- 修正前：batch=64 训练单步 forward ~69s（未限制 CPU 线程，torch intra-op 默认 32 线程，多进程线程竞争开销巨大）。
- 修正后（OMP_NUM_THREADS=1）：完整训练步（取数据+前向+反向+step）8.7~10.2s，快约 7 倍。
- 全量估算修正：738,983 样本 / 64 = 11,547 步/epoch × ~9s ≈ 28 小时/epoch。50 epochs 仍不现实，但不再是 9 天/epoch。
- 小样本端到端试跑：512 样本（每类 256）按 7:1:2 划分（360/50/102），8 worker 加载 + 2 epochs 训练 + val/test 评估 + 检查点保存 + receipt 写出，11.9 分钟全部完成，链路验证通过。

### 序列截断加速（决定性方案）
- 背景：9s/步 × 11,547 步 = 28h/epoch 仍不可行。
- 尝试 1（失败）：增大 dsra_chunk_size（512→1024/2048）。DSRA 是 chunk 内全量注意力，chunk_size 翻倍 → 注意力矩阵 4 倍 → 显存 4 倍（512: 2.24GB → 1024: 4.23GB → 2048: 8.25GB 超 8GB 上限），2048 时耗时暴涨 7 倍。此路线不可行。
- 尝试 2（成功）：缩短输入序列长度。模型支持变长输入，缓存 NPZ 加载后截断（FeatureCacheDataset 的 hash 包含 max_byte_length，不能改数据集参数；改为在 split 后用 _TruncatedByteDataset 包装截断）。
- 实测（batch=64, fp32）：65536→9.2s / 8192→1.06s / 4096→0.50s / 2048→0.23s。选择 4096（保留 PE 头+入口代码上下文，1 epoch ≈ 1.6h）。
- bf16 实测：无 NaN（loss 与 fp32 一致），但加速有限（瓶颈是 Python chunk 循环非精度）；显存 2.24→1.42GB。未采用，保持 fp32。
- 生产实现：_TruncatedByteDataset 包装类（可 pickle，split 之后包 train/val/test 三个子集）+ TRUNCATE_BYTE_LENGTH=4096 常量 + MAX_EPOCHS 50→20。

### 全量训练 NaN 崩溃与修复（关键教训）
- 现象：全量训练在 epoch 2（LR 升至峰值前）loss 降到 ~0.08 后 logits 溢出，trainer 抛 FloatingPointError: Non-finite training loss（logits_finite=False）进程退出。
- 根因分析：train_739k_full.py 用裸训练配置（LR=1e-4、无 label_smoothing/focal、gradient_clip 默认 1.0），在 1:3.45 类别不平衡（benign 165,916 : malware 573,067）的 517k 训练集上，模型快速对多数类过度自信 → logits 极端 → 溢出。
- 修复：对齐项目主链路 config/default_config.toml 的稳定配置：LR 8e-5、gradient_clip 0.75、label_smoothing 0.03、focal_gamma 1.0、focal_alpha 0.55、diversity_loss_weight 0.03。
- 验证：512 样本 6-epoch 试跑 loss 平滑下降（0.187→0.143）、LR 轨迹正常（1e-6→8e-5→cosine 衰减）、无 NaN。
- 最终全量训练：20 epochs 27.23h 完成，早停未触发（F1 持续创新高），Val F1 0.929→0.9800，Test F1 0.9795 / Acc 0.9679 / AUC 0.9921 / FPR 0.0969 / FNR 0.0133。
- 收尾 bug：train_739k_full.py 的 receipt 代码对 TrainingMetrics（dataclass 无 to_dict）直接 json.dump 抛 TypeError 导致 receipt 未保存；已修复（递归 _jsonable 转换），并用 post-hoc 脚本（加载 best checkpoint + trainer.evaluate(test)）补生成 receipt。

### 有效命令
```bash
# 语法编译检查
& "vnev\Scripts\python.exe" -m py_compile scripts\train_739k_full.py tests\test_train_739k_full.py

# ruff 静态检查（项目已有规则）
& "vnev\Scripts\python.exe" -m ruff check scripts\train_739k_full.py tests\test_train_739k_full.py

# 单元测试（新增 + 相关）
& "vnev\Scripts\python.exe" -m pytest tests\test_train_739k_full.py tests\test_training_seed.py -q --tb=short

# 多进程 DataLoader 端到端验证（必须写临时 .py 文件执行，不能用 stdin）
& "vnev\Scripts\python.exe" "C:\Users\Saika\AppData\Local\Temp\opencode\verify_mp_loader.py"

# env 修复验证（无 shell env 依赖，复现 OpenBLAS 失败场景）
& "vnev\Scripts\python.exe" "C:\Users\Saika\AppData\Local\Temp\opencode\verify_env_fix.py"

# 小样本端到端试跑（每类 256 / 2 epochs / 输出到临时目录）
& "vnev\Scripts\python.exe" "C:\Users\Saika\AppData\Local\Temp\opencode\smoke_train_739k.py"

# 全量训练（后台，日志重定向；Start-Process 命令本身会挂起直到超时，但训练进程独立存活）
Start-Process -FilePath "vnev\Scripts\python.exe" -ArgumentList "-u","scripts\train_739k_full.py" -WorkingDirectory "." -RedirectStandardOutput "reports\full_739k\train_full.log" -RedirectStandardError "reports\full_739k\train_full_err.log" -WindowStyle Hidden

# 从 best checkpoint 补生成 receipt（收尾失败时用）
& "vnev\Scripts\python.exe" -u "C:\Users\Saika\AppData\Local\Temp\opencode\write_receipt_posthoc.py"
```

### 环境、依赖、路径或平台注意事项
- 虚拟环境：vnev\Scripts\python.exe（Python 3.14.4，torch 2.12.0+cu132）。
- 机器规格：32GB RAM，16 核 32 线程，RTX 4070 Laptop GPU 8GB 显存。
- Windows spawn：多进程 DataLoader 验证/训练代码必须保存为 .py 文件执行（不能走 stdin），且 DataLoader 迭代必须在 `if __name__ == "__main__":` 保护块内。
- Windows + num_workers>0：必须设置 OMP_NUM_THREADS=1 与 OPENBLAS_NUM_THREADS=1（脚本内 import numpy/torch 前 setdefault），否则 OpenBLAS 线程栈分配失败卡死训练。
- pytest 9.0.3 已安装；ruff 可用（`python -m ruff`）。
- DataLoader 无 shuffle 属性，验证 shuffle 需检查 sampler 类型。
- 脚本 scripts/train_739k_full.py 是 untracked 新文件（git 未跟踪），修改不体现在 git diff 中。

### 后续再次处理本项目时应优先参考的结论
- 全量 739k 训练最终方案（已验证）：序列截断 4096 + 稳定训练配置（LR 8e-5 + label_smoothing 0.03 + focal 1.0/0.55 + clip 0.75 + diversity 0.03）+ OMP/OpenBLAS 单线程 + 8 worker persistent + batch 64，20 epochs ≈ 27h，Test F1 0.9795。
- 训练脚本中 mixed_precision=False 是硬约束（DSRA FP16 会 NaN；bf16 实测无 NaN 但无加速收益）。
- 不要增大 dsra_chunk_size 加速：chunk 内全量注意力，显存平方级增长，2048 即 OOM。
- 全量训练首次跑通的坑：无正则配置下模型过度自信 → logits 溢出 NaN；必须用主链路稳定配置。
- trainer.train() 返回 {train/val/test: [TrainingMetrics]}，TrainingMetrics 是 dataclass 无 to_dict，序列化需递归转换（_jsonable）。
- 用户提到"RTX 4070 5.4GB 显存"与实际不符（实际 RTX 4070 Laptop 8GB），决策前应实测显存，不要直接采信。
- 数据缓存 manifest 配置（pe=1500/stat=49/legacy_dynamic/max_len=65536）与 AxonExperimentConfig 默认值一致，训练脚本无需改维度；FeatureCacheDataset 的 cache hash 包含 max_byte_length，改长度必须用包装类而非改数据集参数。
- 任何新增的 Windows 多进程 DataLoader 脚本都应内置 OMP/OpenBLAS 线程限制，这是本项目环境的事实约束。

---
