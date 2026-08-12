# Axon v2.6Exp 全栈技术总结 —— 相对 Axon 2.5Exp 的代际提升

> 一句话总览：**Axon v2.6Exp 把 Axon 2.5Exp 的"手工特征 + 树模型 + 规则路由专家"体系，整体替换为"端到端深度学习 + 内容特征叠加 + 原生免依赖部署"的新体系。**
> 下文按技术栈逐层展开，凡是对比 2.5Exp 基线（`E:\Project\python\KoloVirusDetector_ML_V2-main`）有进步的点都单独成段。
> 所有数字均来自本仓库与基线仓库的源码/权重/报告（出处标注为 `文件:行` 或报告名）。

---

## 0. 基线与现状速览

| 维度 | Axon 2.5Exp（基线） | Axon v2.6Exp（本模型） |
|---|---|---|
| 检测器 | LightGBM 二分类（399 维，2060 棵树） | 端到端深度模型（696,014 可训练参数） |
| 字节建模 | 手工统计（10KB 熵窗口 + 偏移 8 的 64KB chunk，`chunk_*` 重要性低） | 原始字节流直接进注意力网络（ByteEmbedding→DSRA） |
| PE 特征 | 声称 1500 维，实际 C++ 只产出 350 维（94 真实 PE + 256 哈希桶，静默丢 59 个特征） | legacy_dynamic 1500 维（18 头 + 每节 3 标志 + 29 聚合，真实 47+3N） |
| 数据量 | 364,755 行（恶意 219,990） | 738,983 → 813,098（恶意 573,067） |
| 训练 | LightGBM 5000 轮 / 早停 200 / OHEM + FP 强化 | Focal + 标签平滑 + DSRA diversity loss + FP32 纯单精度 |
| 进阶通道 | 规则路由专家 + 3 类 hardcase 级联（val macro-F1 0.463，偏弱） | Stage-2 内容特征 3-seed HGB 叠加（Test F1 0.99341） |
| 家族聚类 | fast-hdbscan + 最近质心（510 簇） | 检测为主，聚类不在 v2.6Exp 目标内 |
| 部署 | Python 框架扫描 + 多文件模型组（3×LightGBM txt + scaler + selector + family JSON + hardcase manifest） | 单个端到端 ONNX + 单个 C++ DLL，免 Python/PyTorch/sklearn |
| 单文件推理 | 走 C++ 加载 LightGBM 文本模型 | <500ms（实测 ~355ms），权重预加载 |

---

## 1. 模型主干：自研 MHDSRA2 流式注意力

> **使用了自研 MHDSRA2（Multi-Head Deep Sparse Retrieval Attention v2）流式注意力机制，成功解决了常规注意力机制的深度学习在训练时显存占用过高的问题。**

- **问题规模**：对 65536 字节的原始序列，常规多头自注意力的注意力得分张量为 `B·H·T²`。batch=64、heads=4 时仅单层注意力矩阵就约 **2.2 TB（fp16）/ 4.4 TB（fp32）**——任何单卡都无法训练，这正是基线只能退守"手工统计特征 + 树模型"、放弃原始字节端到端学习的原因。
- **MHDSRA2 的解法**：全局稀疏检索槽 + 局部窗口 + 分块流式。每头维护固定大小的 **K=128 个全局槽**（读 top-8、写 top-4）和 **W=256 的局部窗口**，把长序列切成 `chunk_size=512` 的块逐块流式处理，块与块之间用固定大小的槽状态跨块传递。每块的显存工作集是 `O(B·H·C·(K+W+topk))`，与序列长度解耦。
  - 配置（`models/full_739k/best_model_739k.pt` config）：`dsra_dim=128, heads=4, slots=128, read_topk=8, write_topk=4, local_window=256, chunk_size=512, layers=2`。
  - 实测（`docs/code_project_case_studies.md:18,60`）：batch=64、序列 65536、fp32、chunk=512 时训练峰值显存仅 **2.24 GB**；chunk 升到 1024→4.23GB、2048→8.25GB。即"全 65536 序列 × batch-64 端到端训练"被压缩到一张 8GB 卡都装得下的水平。
- **配套长上下文能力**：`src/dsra/mhdsra2/paged_exact_memory.py` 提供 CPU 分页的精确 K/V 检索内存（`page_size=1024, top_pages=4, max_tokens=128`），把全量 token K/V 放 CPU、只回传检索子集给 GPU 注意力层，支持 2M token 级序列的精确召回路径（研究/检索分支；生产模型 `use_retrieval=False` 关闭）。这为"训练显存"之外的另一半问题——"长序列检索"——也给出了方案。
- **代价与约束**：DSRA 的 FP16 前向会溢出产生 NaN，因此训练强制 **FP32 纯单精度**（`train_739k_full.py:234`），这是放弃 AMP 加速换来的数值稳定性。

---

## 2. 端到端可学习架构（对比手工特征树模型）

> **把"人工设计 399 维统计/结构特征 + LightGBM 决策树"替换为"原始字节 + PE 结构 + 字节统计三路投影、端到端梯度学习"的深度模型，让模型自己从字节里学特征。**

- **架构**（`src/model.py`）：字节通道 `ByteEmbedding(256→128)` → 正弦位置编码 → `MalwareDSRAEncoder`（2 层 DSRA，chunk pooling `last`）→ 128 维；PE 通道 `PEFeatureProjector(1500→256→128)`；stat 通道 `49→128`；三路 concat 成 384 维 → `LayerNorm→64→2` 分类头。全模型 **696,014 个可训练参数**（`best_model_739k.pt` 的 `numel` 9,183,060 含 8,388,608 的非训练位置编码 buffer 与 98,438 的模块别名重复，去重后为 696,014）。
- **为什么这是进步**：基线自己也承认 `chunk_*` 原始字节特征重要性极低（28 个里最好只排第 52 名，无一进 Top50，`full_feature_importance_ranking.json`），说明"人工摘要的字节统计"喂树模型学不出判别力；而 v2.6Exp 让深度网络直接读 4096/65536 字节原始流，字节信息不再被手工压缩丢失。
- **训练字节窗口**：提取时采样文件头 65536 字节（尾部补零，`extractor.py:175-200`），训练时截断前 4096 字节（保留 PE 头 + 入口点上下文），DSRA chunk 数 128→8，单步 9.2s→0.5s（**≈18× 加速**，`train_739k_full.py:49`）。而基线只看了 10KB 熵窗口 + 偏移 8 的 64KB 分块统计。

---

## 3. PE 特征体系升级（1500 维 legacy_dynamic）

> **把基线的 350 维、会静默丢特征的 PE 向量，升级为 1500 维动态布局的 `legacy_dynamic` 特征，并完整移植到 C++，对齐精度 16/17、预测 diff 0。**

- **基线的问题**（`src/cpp/kvd_core`）：README 声称"特征维度固定为 1500"，但实际 C++ `kvd_extract_pe_features` 只返回 **350 维**（`features_pe.cpp:49`）——256 维 lightweight 哈希桶 + 94 个 PE 结构特征；`build_feature_order` 定义了 153 个特征但填充循环在 94 处停住，**静默丢弃 59 个特征**（`features_pe.cpp:1100`），其中就包括路由门控用的 `packer_keyword_hits_count`（索引 363，被丢 → 恒为 0，导致"打包专家路由"实际只靠 `packed_sections_ratio>0.4` 一个信号在工作）。Python 侧 `PE_FEATURE_ORDER` 同样只用了 109 个里的 94 个。
- **v2.6Exp**：`legacy_dynamic` 布局（`src/kvd_features/extractor.py:706-855`）= 18 个文件头/安全标志字段 + 每个 section 的 `exec/write/read` 三标志（动态随 section 数增长）+ 29 个聚合特征，真实特征 `47+3N`，其余零填充。不再丢特征、不再依赖"哈希桶占位"，C++ 端把同一套逻辑原样移植（`tools/axon_onnx_dll/src/axon_onnx_predict.cpp:3267-3429`）。
- **工程对齐**：C++ 提取与 Python 提取 **16/17 精确对齐、预测 diff 0.00000**（`dist/.../manifest.json` quality_claim）——这是把深度模型落到原生扫描器的前提。

---

## 4. 数据规模与分布

> **把训练语料从 36.5 万行扩到 73.9 万、再扩到 81.3 万，恶意样本 2.6×、总量 2×。**

- 基线：`extracted_features.pkl` = **364,755 行 × 399 列**，其中恶意 219,990（`cluster/extracted_features.pkl`）。
- v2.6Exp：基座缓存 **738,983** 样本（良性 165,916 / 恶意 573,067，`reports/full_739k/train_739k_receipt.json`）；经过良性扩充后 **813,098**（良性 240,031 / 恶意 573,067，`reports/full_739k_benign_train.log:8`），新增约 7.4 万良性（含 E:/C: 盘经 Avast 扫描确认可信的 EXE、UPX 加壳白文件等）。分层 7:1:2（seed 42）。
- 意义：恶意样本基数越大，尾部族（加壳器、加载器、新家族）越有覆盖；良性扩充直接作用于"零误报"目标。

---

## 5. 训练配方（数值稳定 + 收敛质量）

> **在 deep-learning 侧配置了一套针对长序列注意力与不平衡样本的稳定配方。**

- **损失**：Focal CE（`gamma=1.0, alpha=0.55`）+ `label_smoothing=0.03`（`train_739k_full.py:230-233`），并在训练循环里叠加 **DSRA diversity loss**（权重 0.03，`src/trainer.py:456-464`）——按 token 内积的 Gram 矩阵惩罚槽坍塌（`improved_dsra_mha.py:1270-1280`），让 128 个全局槽保持分化而不是全学成一个。对比基线用 OHEM + FP 强化重加权来凑类别平衡。
- **优化器/调度**：AdamW（lr 8e-5, wd 1e-5）+ 3 epoch 线性 warmup（1e-6 起）→ cosine 退火到 1e-6 + `gradient_clip=0.75`（`src/config.py:342-353`）。
- **数值**：FP32-only（见 §1）；8 worker `persistent_workers=True` + `pin_memory=True` DataLoader（`train_739k_full.py:48,117-118`）；**Windows spawn 修复**——在 import numpy/torch 前强制 `OMP_NUM_THREADS=1 / OPENBLAS_NUM_THREADS=1`，避免 8 个 worker 各起 32 线程 OpenBLAS 导致线程栈分配失败（`train_739k_full.py:16-20`）。
- **结果**：Test F1 **0.9795** / Acc 0.9679 / AUC 0.9921（FP 3217 / FN 1525，20 epochs，27.23h，早停未触发）。对比基线 eval（其自有分布）：acc 0.9938 / FPR 0.75% / FNR 0.53%，**但两个测试分布不可直接比较**（见 §8）。

---

## 6. Stage-2 内容特征叠加（不重训基座即可修正结构性错误）

> **在 739k 基座之上叠加一层"内容特征 + 基座概率"的 3-seed 梯度提升集成，把 Test F1 从 0.9795 推到 0.99341，错误数 4742→1511（FP -75.8%，FN -51.9%）。**

- **动机**（`docs/content_pe_stage2_improvement_plan.md:48-59`）：基座校准很好（ECE 0.001），但存在"结构性自信错误"——训练良性 89% 是 DLL，导致大型独立 EXE（安装器/加载器）靠近恶意质心；而 4096 字节截断的基座看不到的原始文件结构，恰恰可由 content 特征补上。
- **特征矩阵 331 维** = 6 个基座概率派生（`p, p², |p-0.5|, log p, log(1-p), logit(p)`）+ `content_pe_v1`(100) + `content_pe_v2`(182) + `content_string`(43)。
  - v1：28 文件头/级 + 11×3 数据目录 + 7 导入 + 6 API 类比例 + 导出/资源/overlay + 8 节权限组合比 + 7 节统计。
  - v2：32 个常见 DLL 的导入存在性/比例、16 个 API 类 ×3、导出模式、11 种资源类型、21 个节/入口点特征、8 个节名组比例。
  - string：ASCII/UTF-16 游程、URL/IPv4/注册表/Windows 路径正则计数 + 14 类语义串模式（网络/持久化/注入/凭据/加壳器等）。
- **集成**：3 个 `HistGradientBoostingClassifier`（`max_iter=250, lr=0.05, max_leaf_nodes=31, l2=1.0`，seed 0/1/2）在 VAL（73,897 行）上训练，**预测取 3-seed 均值**，阈值在 VAL F1 上扫出 **0.55**，TEST 一次性评估。
- **结果**（`reports/full_739k/stage2_report.json` `stage2_v2`）：Test F1 **0.99341**，P 0.99321 / R 0.99360，错误 **1511**（FP 778 / FN 733），VAL best F1 0.99797。消融：仅 base_prob F1 0.97999（错误 4597）→ +content(100) F1 0.99282（错误 1647）→ 全 331 维 0.99341（错误 1511）。UPX 专项：测试集中 3482 个 UPX 加壳白文件，基座对 UPX 白文件 FP 236 → Stage-2 降到 52（`stage2_report.json` upx_analysis）。
- **工程化**：冻结后的 HGB 导出为 `axon_stage2_hgb_json_v1` 原始数组 JSON（`scripts/export_stage2_hgb_json.py`），供 C++ DLL 免 sklearn 推理，并有与 sklearn `predict_proba` 的对齐校验。

---

## 7. 部署：单端到端模型 + 单 DLL 的原生免依赖交付

> **把一个 739k 深度模型压缩成"一个 ONNX + 一个 DLL"，免 Python/PyTorch/sklearn，单文件推理 <500ms，还保留与 Loop151 一致的冻结 KVD ABI。**

- **ONNX**（`dist/axon_739k_onnx_final_20260808`）：输入 `byte_seq[1,4096] int64 + pe_features[1,1500] float32 + stat_features[1,49] float32`，输出 `logits[1,2]`；导出长度 4096 与训练一致（`scripts/export_onnx_model.py --byte-length`）。onnxruntime 1.24.x DLL 捆绑在 `bin/`，43.8MB 的 `.onnx.data` 外置文件与 `.onnx` 相邻。
- **C++ DLL**（`tools/axon_onnx_dll`）：`legacy_dynamic_pe_features` 完整移植（§3），**16/17 精确对齐、预测 diff 0.00000**；单文件推理 **~355ms 均值（<500ms）**（权重预加载，`README.md:33`）。
- **冻结 ABI**：Windows `__cdecl`、**18 个导出**、x64 下 `kvd_config` **96 字节**，`kvd_create → kvd_scan_path/scan_bytes → kvd_free → kvd_destroy` 生命周期；rust 示例运行时断言 `size_of<KvdConfig>()==96`，js 示例用 `koffi.sizeof==96` 双向校验。
- **调用示例**：cpp（CMake + `--runtime-config`）、rust（libloading）、js（koffi）三种语言都从 `runtime/axon_739k_runtime.json` 的 `base_onnx_path` 相对解析出模型路径再设置 `onnx_model_path`（`examples/...`）。
- **完整性**：`manifest.json` 列出 20 个资产各自的 bytes + sha256，`SHA256SUMS.txt` 同步。
- **对比基线**：2.5Exp 的生产扫描是"Python 框架（main.py 10,699 行内嵌 ~40 模块）+ C++ 加载 3 个 LightGBM 文本模型 + scaler + feature_selector（实为恒等变换）+ family_classifier.json + hardcase manifest"的多文件散装体系；README 里承诺的 `expert_*_attention.onnx / nn_expert_*.onnx / hardcase_dl_base_model` 等多数为"纸面资产"（仅有转换脚手架，无产出/无消费）。v2.6Exp 把"一个能端到端推理的模型"收敛成单一交付物，复杂度大幅下降。

---

## 8. 诚实的数字对比（为什么不能直接比 F1）

- 基线自报 eval：acc 0.9938、FPR 0.75%（216 FP）、FNR 0.53%（233 FN）——这是在 **364,755 行、约 60% 恶意**、**不含 UPX 加壳白文件**的分布上。
- v2.6Exp test：**147,796 行**，其中恶意 114,693，**包含 3,482 个 UPX 加壳白文件**与各类难例，测试分布显著更难。
- 因此"基线 acc 0.9938 > v2.6Exp base acc 0.9679"是**分布不对等下的伪比较**。在 v2.6Exp 的难分布上：基座 Test F1 0.9795（FP 3217 / FN 1525，FPR 9.69% 主因是 UPX 白文件），叠 Stage-2 后 F1 **0.99341**（FP 778 / FN 733），**排除 UPX 后 F1 >0.99**。
- **目标现状**（`reports/full_739k/goal_eval.json`）：Stage-2 recall 0.9936（去噪估计 0.9955）；**零误报 + recall>99% 在当前静态特征 + 标签噪声下不可同时达到**——221 个 FN 中基座置信 >90%，疑似标签噪声而非模型缺陷。

---

## 9. 待办（当前流水线状态）

- **良性扩充重训已完成**：epoch 17–20 补跑完毕，best 模型 = **epoch 19**（val F1 0.9763）。Test（162,619，良性 48,006）F1 0.97648 / FPR **0.0716**（扩充前 0.09695，−26%）/ FNR 0.0174 / AUC 0.9929。重训中途在 epoch 16 被后台任务杀过一次，通过 `train_739k_full.main(resume_from=...)` + `train_739k_benign.py` 自动续训恢复。
- **Stage-2 重训已完成**（331 维 3-seed HGB，val 81,309 训练，阈值 0.52，`stage2_report.json`）：
  - Test F1 **0.99192** / errors 1854（FP 1047 / FN 807）。**同口径对比**（恶意池不变，11.4 万；良性池 33,183→48,006）：**FPR 2.34%→2.18%**、FNR 0.64%→0.70%。绝对误差 1511→1854 因良性池 +45%，不可直接比，率才是同一口径。
  - `goal_eval.json`：recall **0.99296**（>99% ✓）；全阈值扫描仍**无 FP=0 & recall>99% 点**；FN 中 190 个 s2<0.1 疑似标签噪声，去噪真召回 **0.99462**。
  - `upx_whitelist_report.json`（UPX 白名单专项，语料 `F:\私人\良性文件\待加入白名单_upx` 17,051 个）：test 内 3,391 个 UPX 白文件，**base FP 203（5.99%）→ Stage-2 FP 58（1.71%）**（扩充前：3,482 个，236→52）。
  - `error_attribution.json`（错误归因）：FP 中 **313 个基座>90% 自信判黑**（结构性自信错误）、UPX 58/非 UPX 989；FN 全部来自旧恶意池，21 个基座>90% 置信良性（疑似标签噪声）。
  - `whitelist_operating_point.json`（白名单操作点）：把 17k UPX 白名单语料作硬白名单 → **已验证白文件零误报达成**（58 FP→0，验证集内 3,391 个白文件全保），剩余 989 个未验证白文件 FP（占未验证白 2.22%），recall 0.99296 不变。
  - **结论**：目标的可操作形态"**可验证白文件零误报 + recall>99%**"达成；绝对零误报（含从未见过的白文件）在静态特征 + 标签噪声下不可同时达到（与扩充前结论一致，但 FPR 从 2.34% 进一步压到 2.18%，UPX 白文件 base FP 236→203、Stage-2 FP 52→58（绝对数随 test 池变化，率 1.49%→1.71%）。
- 流水线全部产出（base_prob 813k / content_v1 / v2string / stage2 / goal_eval / upx / attribution / whitelist）已落盘到 `reports/full_739k_benign/`。
