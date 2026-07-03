# Phase 3 Loop77: Pre-run Resource and Static Leak Guard

日期：2026-07-03

## 目标

Loop77 把“任何重操作前先检查内存泄漏/资源风险”产品化为一个可复用脚本。它服务于后续训练、评估、cache recovery、corrected split 复验等步骤的前置门禁。

新增：

- `scripts/pre_run_resource_leak_guard.py`
- `tests/test_pre_run_resource_leak_guard.py`

这个 guard 是低风险只读脚本：

- 不导入训练模块
- 不加载模型
- 不导入 torch/CUDA
- 不读 NPZ 特征数组
- 不扫描 raw data
- 不启动 worker pool

## 检查内容

运行时资源：

- 系统内存使用率
- Python/Python3 进程数量和 RSS
- NVIDIA GPU 显存、利用率和 Python compute apps

静态风险：

- `while True`
- `torch` / CUDA 使用
- `np.load`
- process/thread pool
- PyTorch `DataLoader`
- `persistent_workers=True`
- 无界子进程启动

静态扫描会忽略 Python 字符串和注释，避免把测试 fixture 或规则说明误报为真实代码风险。

## 示例

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop76_redraw_readiness.py `
  --target-script tests\test_build_loop76_redraw_readiness.py `
  --output-json reports\random_20w_split\loop77_pre_run_guard.json
```

如果 `guard_ready=false`，不得继续执行目标脚本。若某项风险确实是有意为之，必须显式传入 `--allow-risk`，并在实验文档里说明原因。

## 自检结果

自检命令：

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\pre_run_resource_leak_guard.py `
  --target-script tests\test_pre_run_resource_leak_guard.py `
  --output-json reports\random_20w_split\loop77_guard_self_check.json `
  --max-gpu-python-apps 99
```

真实结果：

| Metric | Value |
| --- | ---: |
| Guard ready | `true` |
| Static findings | `0` |
| Heavy Python processes | `0` |
| Python GPU compute apps | `0` |

默认 GPU Python 进程限制也已复验：

| Metric | Value |
| --- | ---: |
| Guard ready | `true` |
| Max GPU Python apps | `0` |
| Python GPU compute apps | `0` |
| Static findings | `0` |

## 验证

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_pre_run_resource_leak_guard.py -q
```

结果：`7 passed`。

随后用 Loop77 guard 保护 Loop77/76/75/74 与 corrected split/cache gate 的相关测试：

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_pre_run_resource_leak_guard.py tests\test_build_loop76_redraw_readiness.py tests\test_import_loop72_external_verdicts.py tests\test_build_corrected_split_from_plan.py tests\test_audit_corrected_split_replacements.py tests\test_audit_corrected_split_cache_ready.py tests\test_build_corrected_split_cache_recovery_plan.py -q
```

结果：`50 passed`。

Generated reports are not committed.
