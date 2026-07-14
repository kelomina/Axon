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

## 重要限制

DLL 当前面向 Loop28：`fixed_v2 + byte_length=8192 + pe_feature_dim=256 + stat_feature_dim=49`。
Stage-2 C++ 版会复刻主要内容特征，但少量深层 PE 目录计数字段使用稳定兜底值，因此和 Python 研究脚本的分数可能存在很小差异。

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
