# Axon_V2.6-EXP-Checkpoint-260702 发布说明

发布日期：2026-07-02  
版本定位：实验版 checkpoint，不是最终稳定版 Axon v3

## 一句话总结

Axon_V2.6-EXP-Checkpoint-260702 是 Axon 从 v2.5 的传统 LightGBM 特征工程路线，迈向“字节神经网络 + 稳定 PE 结构特征 + 统计特征 + 二阶段内容特征校正”的关键实验版。它更关注文件本身的内容证据，而不是文件名、目录名、后缀这类在真实部署中很容易变化或被攻击者伪造的线索。

如果把 Axon v2.5 理解成一套经验丰富的“结构化体检表 + 树模型判断器”，那么 Axon v2.6 更像是多了一位能直接阅读文件字节流的分析员：它既看 PE 结构和统计量，也看文件前段字节序列的模式，再用 Loop28 的内容侧 PE metadata 做第二层判断。这让模型在面对加壳、结构异常、导入表异常、overlay 异常、资源结构异常等样本时，有了更丰富的判断依据。

## 本次发布包含什么

本 checkpoint 对应当前仓库中的 Loop28。

### DSRA 是否还在本版本中

在。Axon v2.6 的基础模型仍然包含 DSRA/MHDSRA2 字节序列编码分支。源码和 PyTorch checkpoint 中，`AxonMalwareModel` 会把文件字节送入 `ByteEmbedding -> DSRA 流式编码`，再与 PE 结构特征、统计特征融合后输出二分类结果。只是对第三方交付时，原生 DLL 包不会要求用户安装或调用 Python 版 DSRA 源码，而是把这条 DSRA 基础模型链路导出成 `models/axon_loop28_base.onnx`，由 ONNX Runtime 在 DLL 内部执行。换句话说，DSRA 没有被删除，它被封装进了基础 ONNX 模型里；Loop28 Stage-2 JSON 是叠在这个 DSRA 基础模型输出之上的第二层内容特征校正。

## 和 Axon v2.5-EXP 相比，改进在哪里

| 对比项 | Axon v2.5-EXP | Axon v2.6-EXP Checkpoint-260702 | 对用户的意义 |
|---|---|---|---|
| 主模型路线 | 以 LightGBM、路由门控、Normal/Packed Expert 为主。 | 以 DSRA/MHDSRA2 驱动的 AxonMalwareModel 为主，融合字节序列、PE 结构特征和统计特征，并叠加 Loop28 Stage-2。 | 不再只依赖人工设计的表格特征，能同时学习“文件字节长什么样”和“PE 结构是否异常”。 |
| 输入证据 | 主要是 PE/统计/轻量级哈希等手工特征。 | 三路输入：`byte_seq + pe_features + stat_features`，Loop28 再追加 100 维内容侧 PE metadata。 | 对复杂样本的判断依据更丰富，尤其适合分析结构异常、加壳、overlay、资源和导入行为。 |
| PE schema | v2.5 以固定 PE/统计特征为主，但历史文档和代码里的维度口径并不完全一致，容易给后续复验带来混淆。 | v2.6 主路线使用 `fixed_v2`，PE 维度为 256，统计特征为 49；Loop28 原生 DLL 面向 8192 byte + PE256 + stat49。 | 特征含义更稳定，后续升级和回归验证更容易，不容易因为 section 数量变化导致列语义偏移。 |
| 部署方式 | C++ 扫描内核成熟，但主扫描链路仍以 LightGBM 文本模型和 hardcase manifest 为核心；ONNX 更多是权重转换方向。 | 同时提供 Python-backed DLL 和无 Python 原生 DLL；原生 DLL 直接加载 ONNX、Stage-2 JSON、family classifier JSON。 | 第三方接入成本下降，尤其是原生包不要求客户机器安装 Python。 |
| API 可用边界 | v2.5 的 DLL ABI 已经成熟，但真实扫描主要以路径扫描为主，部分导出接口更偏兼容或预留。 | v2.6 原生包继续沿用 `create/scan/free/destroy` 这类集成习惯，同时把 Loop28、Stage-2、家族分类和嵌套扫描放进同一入口。 | 老用户迁移成本更低，第三方不用重新理解一套完全陌生的调用方式。 |
| 嵌套包扫描 | 主要面向文件或目录扫描。 | `scan_nested` 支持 ZIP、7z、CAB、MSI；任一内层 PE 判为风险对象，外层包返回风险结果。 | 对安装包、压缩包、MSI 投递场景更友好。 |
| 家族分类 | v2.5 已有 HDBSCAN 家族聚类与 family classifier。 | v2.6 把相似组和 family classifier 导出为可部署 JSON，并可在 DLL 推理结果中返回家族信息。 | 不只回答“是不是恶意”，还能给出更接近产品侧可用的家族归属线索。 |
| 评估协议 | 有训练、评估、AutoML、可视化等能力，但历史实验口径更容易混在一起。 | 引入更严格的 split/cache/source_sha256 校验，明确 Val、Test-10k、Full-test 漏斗，并禁止文件名、路径、目录、后缀作为模型证据。 | 指标更可信，减少“看起来很高但其实用了不稳定线索”的风险。 |
| 可解释与调参 | 主要依赖 LightGBM 特征重要性和传统评估图表。 | 支持阈值扫描、FP/FN/FPR/FNR 指标、feature mask、feature importance、错误审计队列。 | 产品决策更容易看清“少误报”和“少漏报”之间的取舍。 |
| 安全加载 | 旧链路更偏传统权重和模型文件管理。 | checkpoint 加载使用受限加载，并要求可信 `.pt/.pth`、必要字段和配置存在。 | 降低随意加载模型文件带来的安全和复现风险。 |

需要特别说明：v2.5 和 v2.6 的模型口径不同，不能把两边 checkpoint、ONNX、PE 特征缓存直接混用。v2.6 的改进重点不是“把 v2.5 的 LightGBM 参数再调一遍”，而是换成更稳定、更适合继续产品化的模型和数据合同。

## 260702 checkpoint 的验证亮点

本次发布的 Loop28 内容侧 PE metadata 是当前 v2.6 实验阶段的重要进展。它使用 PE header、data directory、import/export/resource/TLS/relocation、API 类别比例、overlay、section 权限组合和 entropy 等内容证据。

在当前严格实验漏斗里，Loop28 相比前一轮 Loop26 blend 的方向一致：Val、Test-10k 和 160k full-test 都减少了错误数。这一点很重要，因为它说明提升不只是“在验证集上碰巧更好”，而是在更大的冻结测试上也有同方向收益。

| 阶段 | Loop26 blend 错误数 | Loop28 content PE 错误数 | 变化 |
|---|---:|---:|---:|
| Val | 223 | 162 | -61 |
| Test-10k | 144 | 111 | -33 |
| Full-test | 2571 | 1949 | -622 |

Loop28 content-only 的关键结果如下。

| 阶段 | F1 | 错误数 | FP/FN |
|---|---:|---:|---:|
| Val | 0.9919048571 | 162 | 87/75 |
| Test-10k frozen | 0.9888677164 | 111 | 61/50 |
| Full-test frozen | 0.9878358558 | 1949 / 160000 | 1087/862 |

这个结果值得试用，但也要诚实地说：它仍然不是终点。full-test 还有 1949 个错误，距离 99.9% F1 所需的百级错误预算仍有明显距离。因此 260702 更适合被看作一个可集成、可试用、可反馈的实验 checkpoint，而不是最终的商业稳定版。


## 已知限制

- 这是实验 checkpoint，不是最终稳定版。
- v2.5 的模型、特征缓存和 v2.6 的模型、特征缓存不能直接互换。
- RAR 当前只做识别提示，不作为已完整支持的嵌套提取格式。
- 个别畸形 PE 或严格 PE 解析失败的样本可能无法生成完整特征。
- 当前 full-test 仍有千级错误，不能承诺达到 99.9% F1。
- 文件名、路径、扩展名、目录名、hash、row order 只能用于定位、缓存对齐、审计和人工复核，不能作为模型判断证据。

## 接下来：欢迎试用即将发布的 Axon v3

Axon v3 会把 v2.6-EXP 里已经验证有价值的能力进一步产品化。简单说，v2.6 是“我们证明哪些方向有效”，v3 会更像“把这些方向整理成更稳定、更适合长期使用的正式版本”。

v3 计划重点包括：

- 将 content PE v1 / fixed-v3 这类内容侧 schema 做成稳定产品接口。
- 继续强化纯原生部署，降低第三方接入时对 Python 环境的依赖。
- 延续严格评估漏斗，区分 Val、Test-10k、Full-test，减少指标虚高。
- 继续做 FP/FN 分治，优先降低真实使用中最影响体验的误报与漏报。
- 保留家族分类、嵌套包扫描、阈值扫描、错误审计等对产品集成有直接价值的能力。

如果你现在正在使用 Axon v2.5-EXP，建议先用 v2.6-EXP-Checkpoint-260702 跑一批自己的真实样本，重点观察三件事：第一，是否减少了过去 v2.5 容易误判的安装包、压缩包和加壳样本；第二，误报是否集中在某些固定软件类型；第三，恶意样本的漏报是否集中在 DLL、SYS、overlay 或资源结构异常样本。把这些反馈带到 Axon v3 试用阶段，会直接帮助我们把 v3 调成更贴近真实场景的版本。

欢迎在 Axon v3 发布后第一时间试用。v3 的目标不是做一个参数更多的版本，而是做一个更可信、更容易部署、更方便解释给安全产品用户的版本。
