# Axon 7:1:2 漏斗实验交接文档

更新时间：2026-07-27 22:35（Asia/Shanghai）

## 1. 本轮目标与证据边界

目标数据源：

- 良性：`G:\私人\良性文件\待加入白名单`
- 恶意：`G:\私人\恶意\MB\unziped`

目标协议：平衡总样本 `1k -> 5k -> 10k -> full`，每阶段使用 `7:1:2` 的 train/val/test 比例，阶段之间严格嵌套。

最终双硬门：

1. 同一个 DLL 对单个文件的 warm scan 必须小于 `500 ms`。
2. 冻结候选在完整测试集上的 F1 必须大于 `99.9%`。

证据纪律：

- `1k/5k/10k` 阶段只允许使用 train 和 val 做选择、阈值调整和晋级判断。
- `1k/5k/10k` 的 test 行已保留，但到当前为止没有做 test 推理或读取 test 指标。
- full-test 只能在模型、阈值、DLL 和协议全部冻结后做一次最终确认。
- 当前只证明了 1k 学习阶段候选可以晋级到 5k；没有证明最终 F1 大于 `99.9%`。

## 2. 工作区与安全约束

- 当前仓库 HEAD：`ceb5e49`。
- 工作区非常脏，包含大量用户或历史实验改动。不要 reset、checkout 或回滚不属于本漏斗任务的文件。
- 虚拟环境：`E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe`。
- Git：`C:\Program Files\Git\bin\git.exe`。
- PowerShell 命令用 `;` 串联，不使用 `&&`。
- 当前实验产物和新增脚本大多尚未纳入 Git；接手时以 live filesystem 和本交接文档为准。

## 3. 数据集和漏斗 manifest

canonical 漏斗 receipt：

- `manifests/roadmap_9997/corpus_712_funnel/funnel_receipt.json`
- source split SHA256：`4f95ca881cfc9a98122d424071048f2d88bcf77834c2fafccc08927e011b25a8`
- selection seed：`9997`
- 选择算法：`sha256(seed,split,label,source_sha256)_prefix`
- 类别平衡：`1:1`
- nesting audit：`1k subset 5k subset 10k subset full`，全部 `missing_rows = 0`

各阶段真实规模：

| 阶段 | train | val | test | 总计 | 每类 |
|---|---:|---:|---:|---:|---:|
| 1k | 700 | 100 | 200 | 1,000 | 500 |
| 5k | 3,500 | 500 | 1,000 | 5,000 | 2,500 |
| 10k | 7,000 | 1,000 | 2,000 | 10,000 | 5,000 |
| full | 209,144 | 29,876 | 59,758 | 298,778 | 149,389 |

manifest 哈希：

- `split_1k.csv`: `bb17657de444f421372ff404f00db7586b498a77f44a1208a2fb5be449e96388`
- `split_5k.csv`: `9831acc0531063907018e5b9b61371b022af0f088f77ade012d24a3329514321`
- `split_10k.csv`: `6ac69a85eae55e33f6cb147b3bd41957495904ce2bc0123ef89cb998f06e1fb4`
- `split_full.csv`: `2dfa25f800368a84412d6f4be16f1d6b6ad4f42bc7fb56cb3984901e590da7c0`

重要口径：当前 `full=298,778` 是平衡后的完整可用漏斗，不是把恶意目录中约 594k 个候选全部纳入。它使用全部通过验证的 `149,389` 个良性样本和等量恶意样本。进入 full 阶段前，必须由用户确认“完整数据集”采用这个平衡口径，还是要求包含所有恶意候选并改成不平衡协议。

## 4. 冻结配置与 1k cache

实验配置：`config/funnel_712_fixedv2.toml`

关键参数：

- seed `41`
- `max_byte_length = 8192`
- `pe_feature_dim = 256`
- `stat_feature_dim = 49`
- `lightweight_feature_dim = 256`
- `pe_schema_version = fixed_v2`
- `pe_fixed_section_slots = 32`
- strict PE parsing，禁止 fallback
- train batch size `32`，最多 `20` epochs

cache 目录：`data/.cache_712_fixedv2`

1k cache 已闭合：

- receipt：`reports/roadmap_9997/funnel_712/cache_1k_receipt.json`
- rows/success/failures：`1000/1000/0`
- cache config hash：`38672ba0`
- manifest：`data/.cache_712_fixedv2/manifest_38672ba0.json`
- 冻结 1k manifest SHA256：`3a867436e9db6b39933fe0635648ec216f6602ecb118617eec37b7a50d3a301c`

## 5. 1k 基础模型实验

### 5.1 B0

- checkpoint：`models/funnel_712_fixedv2/1k_b0/best_model.pt`
- checkpoint SHA256：`01b01343fcdbb7a96ea8139a4c759982734082d802239bdd71a5a285df3086db`
- epoch：`10`
- Val：F1 `0.9411764706`，AUC `0.9764`，FP/FN `4/2`
- 报告：`reports/roadmap_9997/funnel_712/val_1k_b0.json`

ONNX：

- `models/funnel_712_fixedv2/1k_b0/axon_1k_b0.onnx`
- `models/funnel_712_fixedv2/1k_b0/axon_1k_b0.onnx.data`
- checker 已通过；三个输入为 `byte_seq`、`pe_features`、`stat_features`，输出为 `logits`。

### 5.2 alpha1

- checkpoint：`models/funnel_712_fixedv2/1k_alpha1/best_model.pt`
- Val：F1 `0.8627450980`，AUC `0.9428`，FP/FN `8/6`
- 报告：`reports/roadmap_9997/funnel_712/val_1k_alpha1.json`
- 结论：明显劣于 B0，已淘汰。

## 6. DLL 延迟调查和失败实验

基础 ONNX DLL 的 raw 文件路径太慢：

- 单线程约 `822-2887 ms`。
- 线程矩阵最佳约 8 线程，但 p50 约 `509 ms`，max 约 `1296 ms`，仍不满足硬门。
- graph optimization 的 disabled/basic/extended/all 都没有使 max 小于 `500 ms`。
- 报告位于 `reports/roadmap_9997/funnel_712/dll_1k_b0_*.json`。

因此实现了 native HGB student 路径，主要代码：

- `scripts/train_native_student.py`
- `scripts/export_stage2_hgb_json.py`
- `scripts/benchmark_onnx_dll.py`
- `tools/axon_onnx_dll/src/axon_onnx_predict.cpp`

失败与定位过程：

1. 第一版 1520 维 student 的 Python Val F1 约 `0.989899`，但 DLL feature parity 不足。
2. 精简 666 维 student 的 Python Val F1 `0.990099`，DLL 初始只有 F1 `0.926316`，虽然 max 约 `112 ms`。
3. 修复了 C++ HGB selected-feature 索引映射边界错误。
4. 查明真正数据口径：训练 cache 的 stat 和 byte summary 基于固定 `8192` 字节；lightweight 基于最多 `65536` 字节；PE 也是在 `8192` 字节截断 snapshot 上提取。DLL 只读 8KB 时 lightweight/PE runtime profile 不一致。
5. DLL 改为 full-file read 后，质量恢复为 F1 `0.990099`，FP/FN `1/0`，但 max 延迟变成 `2165.1963 ms`，失败。
   - 报告：`reports/roadmap_9997/funnel_712/dll_1k_native_student_fast_val100_fullread_fixed.json`
6. 不含 PE 的纯快特征 student 只有 Val F1 `0.893204`，失败。
   - 报告：`reports/roadmap_9997/funnel_712/native_student_fast_v2_1k_fit.json`
7. 最终采用显式的 `64KB runtime profile`：训练时为每个样本生成 64KB prefix snapshot，并从该 snapshot 提取 PE；stat/byte summary 仍遵守 8192 字节；lightweight 遵守 64KB。这与 DLL runtime 完全同口径。

## 7. 当前 1k 晋级候选

候选名：`native_student_prefix64k`

训练结果：

- Python Val F1 `0.98`
- Precision/Recall `0.98/0.98`
- AUC `0.9952`
- FP/FN `1/1`
- threshold `0.5749999999999997`
- 报告：`reports/roadmap_9997/funnel_712/native_student_prefix64k_1k_fit.json`

raw DLL Val100：

- sample count `100`
- scan errors `0`
- F1 `0.98`
- Precision/Recall `0.98/0.98`
- FP/FN `1/1`
- latency mean/p50/p95/p99/max：`83.5532 / 60.2654 / 198.5009 / 303.9226 / 372.0400 ms`
- 500ms max latency gate：通过
- 报告：`reports/roadmap_9997/funnel_712/dll_1k_native_student_prefix64k_val100.json`

Python/DLL probability parity：

- 100 个 Val 样本 decision mismatch：`0`
- max absolute probability difference：`0.0003723256465`
- mean absolute probability difference：`0.0000037324559`
- 概率不是 bitwise parity，但判决 parity 已闭合；最大误差应继续作为 5k 监控项。

冻结产物：

- `models/funnel_712_fixedv2/1k_native_student_prefix64k/student.pkl`
  - SHA256 `ff7909a00718d772042d31bea5326a1bfb516546139a50a9070d8a7b8c3d49eb`
- `models/funnel_712_fixedv2/1k_native_student_prefix64k/student.json`
  - SHA256 `41a6ed4a7770528aa90d1ef32e04197dfe0bacbc3228549611922cd804a87044`
- `tools/axon_onnx_dll/build-bench/bin/Release/axon_onnx_predict.dll`
  - 本次 receipt 绑定 SHA256 `fd9850c0d955d515f769fa4d9e9248cd743afd79097fd4c59965aea815cef957`
- 晋级 receipt：`reports/roadmap_9997/funnel_712/promotion_1k_prefix64k.json`

结论：1k 只作为学习曲线门，允许进入 5k。不要把 F1 `0.98` 写成最终目标已实现。

## 8. 当前动态状态：5k cache 正在后台构建

用户中断前启动了以下命令，该任务在文档写入时仍在后台运行：

```powershell
& '.\vnev\Scripts\python.exe' -I '.\scripts\build_feature_cache_from_split.py' `
  --split-csv '.\manifests\roadmap_9997\corpus_712_funnel\split_5k.csv' `
  --config '.\config\funnel_712_fixedv2.toml' `
  --cache-dir '.\data\.cache_712_fixedv2' `
  --workers 8 `
  --receipt '.\reports\roadmap_9997\funnel_712\cache_5k_receipt.json' `
  --failures-csv '.\reports\roadmap_9997\funnel_712\cache_5k_failures.csv'
```

22:31-22:35 取证状态：

- 进程树：PowerShell PID `95820` -> Python PID `106424` -> worker/controller PID `91612`；PID 会随重启变化，不应作为持久标识。
- NPZ 数量从 `1462` 持续增长到 `2208`，说明仍在推进；这个数字只是 22:35 的动态快照。
- `manifest_38672ba0.json` 仍是 1k 的 1000 samples，mtime 仍为 20:07:33。
- `cache_5k_receipt.json`、`cache_5k_failures.csv`、`cache_5k_strict_audit.json` 尚不存在。

这不是失败，也不是 5k 完成：构建脚本只在全部 5000 行成功时原子替换 manifest。中间生成的额外 NPZ 是可复用的增量 cache。

接手 Agent 的第一条规则：不要启动第二个 5k cache builder。先检查现有进程和 receipt。

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -like '*build_feature_cache_from_split.py*' } |
  Select-Object ProcessId, ParentProcessId, CreationDate, Name, CommandLine

Test-Path 'reports\roadmap_9997\funnel_712\cache_5k_receipt.json'
(Get-ChildItem 'data\.cache_712_fixedv2' -Filter '*.npz' -File).Count
```

如果进程已退出且 receipt 不存在，可原样重跑构建命令。脚本会命中已有 NPZ，不要删除中间 cache。

## 9. 5k cache 完成后的严格门

只有同时满足以下条件才可训练：

- receipt `rows=5000`
- `success=5000`
- `failures=0`
- manifest samples `5000`
- failures CSV 只有 header

然后执行严格 metadata audit：

```powershell
& '.\vnev\Scripts\python.exe' -I '.\scripts\audit_strict_split_metadata.py' `
  --split-csv '.\manifests\roadmap_9997\corpus_712_funnel\split_5k.csv' `
  --manifest-json '.\data\.cache_712_fixedv2\manifest_38672ba0.json' `
  --strict-unique-source-sha256 `
  --strict `
  --output-json '.\reports\roadmap_9997\funnel_712\cache_5k_strict_audit.json'
```

必须得到 exit 0、`audit_ready=true`、无 row issues、无 duplicate SHA。

## 10. 推荐的 5k 实验顺序

不要先跑 test。推荐只训练/验证 native prefix64k student，因为它是目前唯一同时满足质量学习门、runtime parity 和 500ms 硬门的路线。

### 10.1 准备 5k train/val cache-row CSV

1k 已有：

- `reports/roadmap_9997/funnel_712/stage2_input_1k_train.csv`
- `reports/roadmap_9997/funnel_712/stage2_input_1k_val.csv`

5k 需要按同一逻辑将 strict manifest 的 `cache_path` 映射回 `split_5k.csv`，只输出 train/val 行。不要生成 test prediction input；可复用此前生成 1k stage2 input 的代码路径或写一个最小、可测试的通用脚本。

### 10.2 训练 5k prefix64k student

```powershell
& '.\vnev\Scripts\python.exe' '.\scripts\train_native_student.py' `
  --checkpoint '.\models\funnel_712_fixedv2\1k_b0\best_model.pt' `
  --train-rows '.\reports\roadmap_9997\funnel_712\stage2_input_5k_train.csv' `
  --val-rows '.\reports\roadmap_9997\funnel_712\stage2_input_5k_val.csv' `
  --output-model '.\models\funnel_712_fixedv2\5k_native_student_prefix64k\student.pkl' `
  --output-report '.\reports\roadmap_9997\funnel_712\native_student_prefix64k_5k_fit.json' `
  --runtime-prefix-bytes 65536 `
  --seed 41
```

说明：这里继续用 B0 checkpoint 只为恢复 `AxonExperimentConfig`，student 训练不使用 B0 logits。若后续改变配置，必须重新绑定 checkpoint/config receipt。

### 10.3 导出和验证 JSON

```powershell
& '.\vnev\Scripts\python.exe' '.\scripts\export_stage2_hgb_json.py' `
  --input '.\models\funnel_712_fixedv2\5k_native_student_prefix64k\student.pkl' `
  --output '.\models\funnel_712_fixedv2\5k_native_student_prefix64k\student.json' `
  --verify
```

JSON exporter 的随机矩阵验证必须 `max_abs_diff = 0`。

### 10.4 raw DLL Val500 验收

先确保 DLL 是当前源码重新编译的：

```powershell
cmake --build '.\tools\axon_onnx_dll\build-bench' --config Release --target axon_onnx_predict
```

然后：

```powershell
$env:AXON_NATIVE_STUDENT_ONLY = '1'
$env:AXON_ONNX_TIMING = '1'
& '.\vnev\Scripts\python.exe' '.\scripts\benchmark_onnx_dll.py' `
  --dll '.\tools\axon_onnx_dll\build-bench\bin\Release\axon_onnx_predict.dll' `
  --onnx '.\models\funnel_712_fixedv2\1k_b0\axon_1k_b0.onnx' `
  --stage2 '.\models\funnel_712_fixedv2\5k_native_student_prefix64k\student.json' `
  --split-csv '.\manifests\roadmap_9997\corpus_712_funnel\split_5k.csv' `
  --allowed-root 'G:\私人' `
  --split val `
  --count 500 `
  --warmup 20 `
  --output-json '.\reports\roadmap_9997\funnel_712\dll_5k_native_student_prefix64k_val500.json'
```

晋级最低门：

- scan errors `0`
- raw DLL decision 与 Python decision mismatch `0`
- raw DLL max latency `<500 ms`
- Val F1 相对 1k 的 `0.98` 不退化，并观察错误数是否随数据规模下降

不要因为小 Val 达到 1.0 就宣称最终 99.9% 已实现。

## 11. 后续 10k 和 full

- 5k 通过后，用同样流程增量构建 10k cache、strict audit、prefix64k student、Python Val1000、raw DLL Val1000。
- 10k 仍不得做 test 推理。
- 进入 full 前先向用户确认 full 的平衡口径。
- full train/val 用于最终冻结模型和阈值；冻结后才运行一次 full-test。
- 最终报告必须同时包含 F1、Precision、Recall、FP/FN、FPR/FNR、AUC、scan error、DLL latency p50/p95/p99/max、所有产物 hash。
- 最终 `F1 > 99.9%` 必须来自 `59,758` 行 full-test，且 DLL max latency `<500 ms` 必须在同一冻结 DLL/runtime profile 上证明。

## 12. 已修改或新增的任务文件

本轮核心写集：

- `config/funnel_712_fixedv2.toml`
- `scripts/build_712_corpus_split.py`
- `scripts/build_712_funnel_manifests.py`
- `scripts/build_feature_cache_from_split.py`
- `scripts/benchmark_onnx_dll.py`
- `scripts/train_native_student.py`
- `scripts/export_stage2_hgb_json.py`
- `tools/axon_onnx_dll/CMakeLists.txt`
- `tools/axon_onnx_dll/src/axon_onnx_predict.cpp`
- `tests/test_build_712_funnel_manifests.py`
- `tests/test_build_feature_cache_from_split.py`
- `tests/test_benchmark_onnx_dll.py`
- `tests/test_train_native_student.py`
- `tests/test_native_loop28_parity_source.py`

主要实现点：

- DLL 可通过环境变量配置 ORT 线程和 graph optimization。
- DLL 支持 `AXON_NATIVE_STUDENT_ONLY=1`，绕过慢 ONNX 主模型，直接运行 native HGB student。
- DLL 支持 `AXON_ONNX_TIMING=1` 输出 read/feature/model/total timing。
- HGB JSON 支持 `selected_feature_indices`，并修复了树节点索引到 source feature 的映射。
- native student runtime 读取最多 64KB，同时保存真实 original file size。
- native student 的 byte summary 使用固定 8192 字节 padded sequence，lightweight 和 prefix PE 使用 64KB runtime snapshot。
- `train_native_student.py` 新增 `--runtime-prefix-bytes`，显式生成与 DLL 一致的 prefix PE 特征。

## 13. 已运行验证

最近一次相关测试：

```text
19 passed, 9 skipped
```

命令：

```powershell
& '.\vnev\Scripts\python.exe' -m pytest `
  'tests\test_train_native_student.py' `
  'tests\test_native_loop28_parity_source.py' `
  'tests\test_benchmark_onnx_dll.py' -q
```

跳过的 9 个是 opt-in native integration variants；当前核心 Python/source contract 测试通过。此前漏斗/cache 脚本相关测试为 `5 passed`。

## 14. 接手检查清单

1. 先看 5k cache builder 是否仍在运行；不要启动第二个实例。
2. 等 receipt 出现后核 `5000/5000/0`，再跑 strict audit。
3. 不读取或推理 1k/5k/10k test。
4. 用 prefix64k runtime profile 生成 5k train/val row matrix。
5. 训练、JSON verify、raw DLL Val500、Python/DLL parity、hash receipt。
6. 只有质量与 `<500 ms max` 同时通过才进入 10k。
7. full 前请用户确认平衡 full 口径。
8. full-test 只做一次最终确认；在此之前不得声称 `F1 > 99.9%` 已达成。
