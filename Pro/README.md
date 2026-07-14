# Axon Pro RL Branch

这个目录是一个独立的强化学习实验分支，不会修改当前主训练链路。

## 为什么这里用 contextual bandit

恶意软件检测本质上是“看一个样本，然后判断白文件或黑文件”。它不像游戏那样有很多连续步骤，所以这里采用最轻量的强化学习形式：contextual bandit。可以把它理解成“每次只做一道判断题”：模型看到样本状态，选择动作 `0=判为白文件` 或 `1=判为黑文件`，环境再根据真实标签给奖励或惩罚。

当前业务偏好是“误报比漏报更严重”，所以默认奖励函数会对 false positive 施加更大的惩罚，并且默认使用 `decision_threshold=0.65`。这表示只有“黑文件概率”达到 0.65 以上才判黑，模型会比普通 0.50 阈值更保守。

## 文件结构

- `rl_axon/config.py`：RL 奖励和训练配置。
- `rl_axon/environment.py`：把恶意软件分类包装成 RL 环境，并提供 smoke 用的合成数据集。
- `rl_axon/agent.py`：把现有 `AxonMalwareModel` 包成策略网络。
- `rl_axon/trainer.py`：策略梯度训练循环。
- `smoke_test_rl.py`：最小可运行 smoke 测试。

## Smoke 测试命令

PowerShell 中运行：

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" Pro\smoke_test_rl.py --device cpu
```

如果想用显卡：

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" Pro\smoke_test_rl.py --device cuda
```

看到 `[OK] RL smoke test passed` 表示这个实验分支的模型前向、奖励计算、反向传播和评估都能跑通。

如果要临时调低或调高判黑阈值：

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" Pro\smoke_test_rl.py --device cuda --decision-threshold 0.70
```

脚本会自动输出一个 threshold sweep 表格，用同一个模型查看不同判黑阈值下的 reward、precision、recall、误报率和漏报率。

## 真实缓存小规模对照

这个命令会直接读取 `data/.cache`，不会重新扫描原始 PE 文件。默认每类 200 个缓存样本、2 个 epoch，用同一批样本对比普通监督训练和 Pro RL 训练。真实缓存对照默认 `decision_threshold=0.80`，因为当前业务更怕误报：

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" Pro\compare_rl_cache.py --device cuda --samples-per-class 200 --epochs 2 --batch-size 16 --decision-threshold 0.80 --output-json reports\pro_rl_cache_compare.json
```

输出里的 `fp_rate` 是误报率，`fn_rate` 是漏报率。当前业务更怕误报，所以优先看 reward、precision 和 fp_rate，再看 recall。
