# Loop164 本地 whole-file 工程可行性 probe

## 结论

Loop164 的首个 `user_directed_local_custody` probe 已按冻结边界完成。它证明当前 RTX 4070 Laptop 环境能以 FP32、单进程和有界内存运行论文尺度的 MalConv2-style GCG 全文件训练路径；它没有计算质量指标，也不构成候选晋级、A2 Train-OOF 或 99.97% 认证证据。

## 实现与修正

- `src/loop164/whole_file_gcg.py` 使用 257-token 语义，raw byte `0..255` 映射到 `1..256`，PAD 固定为 `0`。两支 GLU 后加入官方结构中的 `1x1 channel share + LeakyReLU`，GCG 仍使用独立 winner receptive-field 重算，避免拼接不连续区域制造伪窗口。
- 模型强制 `stride <= receptive field`、每块不超过冻结 output count，并要求每遍 output-coordinate chunks 精确覆盖声明的全文件输出位置。dense/chunk oracle 统一冻结为 first-winner `.max` tie 语义。
- `src/loop164/authorized_input.py` 每遍只顺序读取新 raw bytes；重叠区保留在有界 buffer，不重复计入 SHA。每个成功文件恰好两遍，每遍独立核对 byte count、SHA 和前后文件指纹。
- `scripts/build_loop164_local_probe_bundle.py` 只在 canonical train role 内选样。它把旧 canonical path 按相对路径映射到 hardlink worktree，不跟随 `data/待*` 的 repo 外 symlink；manifest 中 4 个与 canonical train 无关的 cross-label SHA 只进入 aggregate audit，不阻断 train-only bundle。
- `scripts/run_loop164_feasibility_probe.py` 不使用通用 Dataset/DataLoader、cache、NPZ、checkpoint 或预测文件，不写模型状态，不计算 F1、accuracy、概率、预测或阈值。

## 冻结配置

| 项目 | 值 |
|---|---:|
| 样本 | canonical train 每类 128，共 256 |
| 模型 | embedding 8，channels 256，RF 256，stride 64 |
| output chunk | 4093 outputs / 262144 bytes |
| 精度与设备 | FP32 / CUDA |
| 训练 | 1 epoch，batch 1，accumulate 8，AdamW |
| 硬上限 | 256 optimizer steps，30 分钟，全文件 8 MiB |
| loader | 单进程，逐文件两遍，逐遍 SHA |

## 真实结果

- denominator/success/missing：`256 / 256 / 0`；五类 missingness 全为 `0`。
- completed scans / verified SHA passes：`512 / 512`。
- raw bytes read：`382001358`，精确等于 selected source bytes 的两倍。
- backward microbatches / optimizer steps：`256 / 32`；discarded accumulation `0`。
- timeout / OOM / nonfinite：`0 / 0 / 0`。
- elapsed：`14.092734s`；peak process RSS `1745612800` bytes。
- peak CUDA allocated/reserved：`104785920 / 125829120` bytes。
- checkpoint/model-state/prediction written：全部 `false`；`metrics_computed=[]`，threshold operations `0`。

## 证据

- receipt：`reports/roadmap_9997/loop164/local_feasibility_probe_receipt.json`，SHA-256 `cfb299be80c7c1b535d4bf1d61f86ddf76ddd78c94fd747d8c87158a0a8f15e1`。
- bundle：`reports/roadmap_9997/loop164/local_probe_bundle.jsonl`，SHA-256 `90961bfed0460787e261965a3180e1b0569df0f9d275f9693daad1ccf53dc233`。
- bundle summary：`reports/roadmap_9997/loop164/local_probe_bundle_summary.json`，SHA-256 `3ab978be18a3fa6a91dc34bded3c51dee337e48903a457bd49c1616066c6db91`。
- resource guard：`reports/roadmap_9997/loop164/local_feasibility_resource_guard.json`，SHA-256 `ec528798a217046e9483c4ca9e2113955e1b7e3940a79d695f41997caa0e47ea`，decision `pass`。
- config：`config/loop164_whole_file.toml`，SHA-256 `3da7c075ae561eb68b0dfe4083549b403a16d0eacf33f8c4a8fa4284de25a312`。

定向验证为 `40 passed`，相关 Ruff 与 `py_compile` 通过。

## 决策

工程路线通过，不再把 whole-file 全文件读取、两遍 exact pooling 或本机显存视为首要风险。下一主风险转为统计与数据边界：建立可信的 Train-OOF partition、生成 Loop151-equivalent base OOF、训练 whole-file OOF，并只在两者均为 OOF 时拟合 residual fusion。Loop151 继续是唯一 research champion；本 probe 权重未保存、不可复用。
