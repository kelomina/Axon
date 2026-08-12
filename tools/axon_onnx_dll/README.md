# Axon ONNX C++ Prediction DLL

这个目录提供 C++ 版预测 DLL。它遵循 KoloVirusDetector 预测器 DLL 的调用习惯：

- `kvd_create` 创建预测器句柄
- `kvd_scan_path` / `kvd_scan_bytes` 扫描文件或字节
- `kvd_free` 释放返回的 JSON 字符串
- `kvd_destroy` 销毁句柄
- 可选：`family_classifier_json_path` 启用 Axon 家族归属输出

所有 `const char*` 路径参数都使用 UTF-8。`allowed_scan_root` 和目标路径都会做 canonical 比较；通过目录链接扫描时必须显式配置链接展开后的物理根。这样不会把根目录内可替换的目录链接自动扩展成任意外部目录权限。raw replay runner 会先把已授权、SHA 校验通过的样本复制到私有临时目录，再把该目录作为 native allowed root。

## 方案说明

Python 只负责离线训练和导出模型资产。第三方运行时不需要 Python、不需要 PyTorch。
C++ DLL 使用 ONNX Runtime 加载 `.onnx` 基础模型，并可加载 Loop28 Stage-2 JSON：

- 前 `8192` 个字节组成的 `byte_seq`
- `fixed_v2` 的 `256` 维 PE 特征
- 前 `8192` 个字节对应的 `49` 维统计特征
- Loop28 Stage-2 的 `1520` 维融合特征

运行时可以随 DLL 附带这些原生/数据资产：

- `onnxruntime.dll`
- `axon_loop28_base.onnx`
- `loop28_stage2_hgb.json`
- `family_classifier.json`
- `axon-archive-scanner.exe`，用于 MSI/ZIP/7z/CAB 嵌套解包

这相当于把“训练厨房”和“上菜窗口”分开：Python 负责做菜谱和训练，C++ DLL 负责按菜谱快速出结果。

## Loop151 Native Champion

Loop151 使用同一套 native-only 交付边界，但把 Stage-2 及 selector 权重导出为
纯 JSON 数值资产，不再加载 Python、PyTorch、pickle 或 sklearn。构建的
`axon_loop151_champion.dll` 通过 `runtime/loop151_native_runtime.json` 绑定
base ONNX、primary/conservative/content-cross/noise/selector 五个权重文件；该
配置文件和模型目录可以整体复制到另一台 Windows x64 机器。`onnxruntime.dll`
必须和 champion DLL 放在同一 `bin` 目录，ONNX `.data` 文件必须和 `.onnx` 相邻。

```powershell
.\build\bin\Release\axon_loop151_example.exe `
  --dll .\build\bin\Release\axon_loop151_champion.dll `
  --runtime-config ..\..\dist\axon_loop151_native_champion\runtime\loop151_native_runtime.json `
  --target C:\samples\sample.exe
```

## 家族归属

Axon 的相似样本分组工具可以离线生成 `family_classifier.json`。DLL 加载它以后，会在样本被判定为恶意时追加：

```json
"malware_family": {
  "family_name": "axon_group_12",
  "cluster_id": 12,
  "is_new_family": false,
  "distance": 0.123,
  "threshold": 0.456
}
```

生成流程示例：

```powershell
cd "E:\Project\python\Axon_v2.6Exp"
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts/analyze_raw_similarity.py --config config/default_config.toml --data-dir data --output-dir reports/raw_similarity_fixed512
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts/analyze_raw_groups.py --config config/default_config.toml --data-dir data --raw-report-dir reports/raw_similarity_fixed512 --output-dir reports/raw_group_diagnostics_fixed512
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts/export_family_classifier.py --config config/default_config.toml --data-dir data --group-members reports/raw_group_diagnostics_fixed512/group_members.csv --output resources/axon_family/family_classifier.json
```

这一步不是训练模型。它只是把调试阶段的“相似样本组”整理成部署时能查的“家族地图”。

## 关闭 base ONNX（`base_onnx_enabled`）

`runtime/loop151_native_runtime.json` 支持一个可选布尔字段：

```json
{
  "schema": "axon_loop151_native_runtime_v1",
  "base_onnx_enabled": false,
  ...
}
```

**字段缺省等于 `true`**，已发布的运行时配置行为不变。

设为 `false` 时，DLL 不加载 base ONNX，也不执行它的推理，并跳过 `content_cross`。
这样做的依据是 Stage-2 资产自身的元数据：`primary`、`conservative`、`noise` 三个
stack 都带 `drop_base_prob_features: true` 且 `dropped_feature_count: 6`，
原生打分器在评分前会把 base 概率派生的那 6 列（`prob`、`prob²`、`|prob-0.5|`、
`log(prob)`、`log(1-prob)`、`logit`）整段 erase 掉。也就是说这三个主力模型从不
消费 base 概率，`content_cross` 是它唯一的下游。

`content_cross` 只通过 `possible` 的 OR 分支参与判决；去掉它以后判据退化为
`primary == 1 && conservative == 0`，比原式更难满足，因此规则触发的
「恶意 → 良性」翻转只会变少，不会凭空多出漏报。

关闭以后基础 ONNX 及其 `.onnx.data` 就不必随包分发。

### 原生模型加载

Stage-2 资产是几个非常大的扁平数字数组：`loop151_noise.native.json` 一个文件就有
1143 万个数字，而字符串只有 2486 个、数组 178 个、对象 49 个。加载器针对这个形状
做了三处处理，都不改变任何数值：

- **不装箱纯数字数组。** 全部元素都是数字的数组直接存进 `std::vector<double>`，
  不为每个数字建 JSON 节点。三个大资产合计 1919 万个节点，按每节点约 96 字节算
  是 1.72 GiB，加上 vector 扩容拷贝正是此前 4.8 GiB 加载峰值的来源。
- **不做序列化往返。** 嵌套模型此前会被 `json_encode` 重新序列化成 17 位精度的
  文本再解析一遍；每个 stack 有 15 个基模型，这是加载耗时的主要来源。现在直接
  传递已解析的节点。
- **用 `std::from_chars` 代替 `std::strtod`**，避免逐个数字查询 C locale。

累计效果：初始化 117 s → 5.1 s，加载峰值 4876 MiB → 607 MiB，稳态常驻
1041 MiB → 238 MiB。判决完全不变，`tests/axon_loop151_number_parse_test.cpp`
用真实资产的 1917 万个数字逐位验证了新旧解析结果一致。

### 诊断用环境变量

这两个开关只影响输出内容，不改变 ABI，也不改变判决：

- `AXON_LOOP151_TIMING=1`：在结果 JSON 里追加 `timing_ms`，按阶段给出
  `base_onnx` / `stage2_features` / `content_features` / 各 stack / `authenticode`
  的耗时。
- `AXON_LOOP151_NO_ONNX_SHADOW=1`：在开启 ONNX 的前提下，额外计算一份
  「假如关闭 ONNX」的判决并写入 `no_onnx` 字段，用于在同一次扫描里做配对对比。

配套的基准程序是 `axon_loop151_bench.exe`，它把一次性初始化、常驻内存和单文件
延迟分开测量，并可通过 `--split-csv` 读入带标签的切分直接计算 F1：

```powershell
.\build\bin\Release\axon_loop151_bench.exe `
  --dll .\build\bin\Release\axon_loop151_champion.dll `
  --runtime-config ..\..\dist\axon_loop151_native_champion\runtime\loop151_native_runtime.json `
  --split-csv ..\..\manifests\roadmap_9997\corpus_712_split\split_712.csv `
  --split test --count 2000
```

## 重要限制

Loop28 兼容 DLL 当前面向 `fixed_v2 + byte_length=8192 + pe_feature_dim=256 + stat_feature_dim=49`。
Loop151 native DLL 还会执行五阶段原生 stack/selector 决策；其交付包中的 README
和 parity receipt 明确标注了验证范围，不把有限样本 parity 当成 full-test 质量声明。

## 导出 ONNX

```powershell
cd "E:\Project\python\Axon_v2.6Exp"
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts/export_onnx_model.py --checkpoint ".\models\random_20w_8192\best_model.pt" --output ".\models\random_20w_8192\axon_loop28_base.onnx"
```

如果环境里安装了 `onnxruntime`，可以加 `--verify` 做 PyTorch 与 ONNX 的零输入对比：

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts/export_onnx_model.py --checkpoint ".\models\random_20w_8192\best_model.pt" --output ".\models\random_20w_8192\axon_loop28_base.onnx" --verify
```

## 导出 Loop28 Stage-2 JSON

```powershell
cd "E:\Project\python\Axon_v2.6Exp"
& ".\vnev\Scripts\python.exe" scripts/export_stage2_hgb_json.py --input ".\reports\random_20w_split\stage2_loop28_content_pe_valonly\stage2_selected_model.pkl" --output ".\models\random_20w_8192\loop28_stage2_hgb.json" --verify
```

## 编译 DLL

默认复用 Kolo 项目里的 ONNX Runtime：

```text
E:\Project\python\KoloVirusDetector_ML_V2-main\onnxruntime-win-x64-1.24.4
```

编译命令：

```powershell
cd "E:\Project\python\Axon_v2.6Exp\tools\axon_onnx_dll"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

生成文件通常在：

```text
tools\axon_onnx_dll\build\bin\Release\axon_onnx_predict.dll
tools\axon_onnx_dll\build\bin\Release\onnxruntime.dll
tools\axon_onnx_dll\build\lib\Release\axon_onnx_predict.lib
```

## 自测调用

```powershell
cd "E:\Project\python\Axon_v2.6Exp\tools\axon_onnx_dll"
.\build\bin\Release\axon_onnx_selftest.exe --dll ".\build\bin\Release\axon_onnx_predict.dll" --onnx "E:\Project\python\Axon_v2.6Exp\models\random_20w_8192\axon_loop28_base.onnx" --stage2 "E:\Project\python\Axon_v2.6Exp\models\random_20w_8192\loop28_stage2_hgb.json" --target "E:\samples\sample.exe"
```

带家族归属：

```powershell
.\build\bin\Release\axon_onnx_selftest.exe --dll ".\build\bin\Release\axon_onnx_predict.dll" --onnx "E:\Project\python\Axon_v2.6Exp\models\random_20w_8192\axon_loop28_base.onnx" --stage2 "E:\Project\python\Axon_v2.6Exp\models\random_20w_8192\loop28_stage2_hgb.json" --family "E:\Project\python\Axon_v2.6Exp\resources\axon_family\family_classifier.json" --target "E:\samples\sample.exe"
```

嵌套 MSI/压缩包扫描：

```powershell
.\build\bin\Release\axon_onnx_selftest.exe --dll ".\build\bin\Release\axon_onnx_predict.dll" --onnx "E:\Project\python\Axon_v2.6Exp\models\random_20w_8192\axon_loop28_base.onnx" --stage2 "E:\Project\python\Axon_v2.6Exp\models\random_20w_8192\loop28_stage2_hgb.json" --family "E:\Project\python\Axon_v2.6Exp\resources\axon_family\family_classifier.json" --archive_scanner "E:\Project\python\Axon_v2.6Exp\tools\archive_scanner\target\release\axon-archive-scanner.exe" --nested --target "E:\samples\installer.msi"
```

正常会输出两段 JSON：第一段是模型校验结果，第二段是预测结果。

## 常见报错

- `onnx_model_main_missing`：ONNX 文件路径不对，或者文件不存在。
- `onnx_model_main_invalid`：ONNX Runtime 能找到文件，但无法加载。常见原因是导出的模型不兼容、文件损坏，或 ONNX Runtime DLL 没放在同目录。
- `family_classifier_missing`：传入了家族 JSON 路径，但文件不存在。
- `family_classifier_invalid`：家族 JSON 格式不对，或维度和 DLL 当前 fixed_v2 特征不匹配。
- `file_read_failed`：目标样本路径不对，或者当前程序没有读取权限。
- `path_not_allowed`：设置了 `allowed_scan_root`，但目标文件不在这个目录下。
- `invalid_model_output`：ONNX 模型输出不是两个分类 logits，需要检查导出脚本和 checkpoint。

最有用的报错信息是自测程序打印的整段 JSON，以及命令行里 `LoadLibrary failed` / `GetProcAddress failed` / `kvd_create failed` 这几行。
