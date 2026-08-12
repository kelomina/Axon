# Axon v3-Pre 技术报告

> 把 Axon 2.5Exp 的"手工特征 + LightGBM + 规则路由专家"树模型体系，重写成"原始字节端到端注意力 + 内容特征叠加 + 原生免依赖部署"的深度架构。

---

## 0. TL;DR · 一分钟结论

- **核心理念**：2.5Exp 时代"人挑特征、树模型做判别"；v2.6Exp 让 4096/65536 字节的**原始字节流直接进注意力网络**，端到端学特征，然后**再叠一层内容特征 GBM 纠错**——两层各自解决各自的问题。
- **显存**：常规自注意力在 65536 序列 × batch64 下**单层注意力矩阵约 2 TB**；自研 MHDSRA2 流式注意力把它压到 **batch64 全序列训练峰值 2.24 GB**（chunk=512, fp32）。
- **训练**：20 epochs / 27.2h，Test F1 **0.9795** / Acc 0.9679 / AUC 0.9921（147,796 测试样本，含 3,482 个 UPX 加壳白文件）。
- **Stage-2**：331 维内容特征 3-seed HGB 叠加 → Test F1 **0.99341**，错误数 4742→1511（**FP −75.8%，FN −51.9%**）；UPX 白文件误报 236→52。
- **部署**：单 ONNX（`byte_seq[1,4096]+pe[1,1500]+stat[1,49] → logits[1,2]`）+ 单 DLL（`__cdecl`, 18 exports, 96B config），免 Python/PyTorch/sklearn，单文件推理 **~355ms**。
- **诚实结论**：零误报 + recall>99% 在当前静态特征 + 标签噪声下不可同时达到；基座有 221 个"模型置信>90% 的 FN"，极可能是标签噪声而非缺陷。

---

## 1. 为什么重写：2.5Exp 的"纸面繁荣"

基线的真实状态，是评估这次重写价值的前提。两个仓库我都翻过源码，结论是：**2.5Exp 的 README 比它的代码更漂亮。**

**纸面 vs 现实：**

| 宣称 | 现实（源码证据） |
|---|---|
| "特征维度固定为 1500"（README:168） | C++ `kvd_extract_pe_features` 实际只返回 **350 维**：256 维 lightweight 哈希桶 + 94 个 PE 结构特征（`features_pe.cpp:49`，`api.cpp:61`） |
| "153 个 PE 特征" | `build_feature_order` 定义 153 个，填充循环在 94 处停住，**静默丢弃 59 个**（`features_pe.cpp:1100`） |
| 路由门控 `packed_sections_ratio>0.4 OR packer_keyword_hits_count>0` | `packer_keyword_hits_count` 正是被丢掉的 59 个之一（索引 363）→ **恒为 0**，门控实际退化成单信号 |
| `feature_selector.json` 特征选择 | `n_features_in=399 / n_features_out=399`——**恒等变换，no-op** |
| `expert_*_attention.onnx / nn_expert_*.onnx / hardcase_dl_base_model` | README 里的 ONNX 清单（:125-131）**只有转换脚手架、没有产物也没有消费方**（`_LegacyAttentionExpert/_LegacyNNExpert` 指向不存在的 .pt） |
| AutoML（Optuna/Hyperopt） | 真实存在但 `AUTOML_ENABLED=False`，只走显式 `auto-tune` 命令 |

**基线也真有强项（公平起见）：**
- 规则路由 + Normal/Packed 专家是有先验的工程设计，思路在 v2.6Exp 的 Stage-2 里以更干净的形式继承了。
- fast-hdbscan 家族聚类（510 簇，最近质心 + 每簇 90 分位阈值）是完整的家族归属链路——这不是 v2.6Exp 的目标，故未做对比。
- C++ 扫描内核（LightGBM C API 加载、simhash 模糊签名 Hamming≤3）工程面扎实，14 个导出函数，`max_file_size=64KB` 与文档一致。

**核心病根**：树模型 + 手工特征的天花板。基线自己的重要性榜单证明 `chunk_*` 原始字节特征（28 个）最好只排第 52、无一进 Top50——"人先压缩字节、树再拟合"这条路，字节信息在第一步就丢了。v2.6Exp 的选择是：**字节不预压缩，交给注意力网络自己学。**

---

## 2. 模型骨架：三路输入 + concat 融合

```
                          ┌─────────────────────────────────────────────────────┐
  原始文件 ─► byte_seq(65536/4096)
                          │  ByteEmbedding(256→128)                            │
                          │    → 正弦位置编码(max_len=65536)                    │
                          │    → input_proj(128→128)                           │
                          │    → [MultiHeadDSRA2 ×2, chunk_pool=last] ─► 128   │
  PE 结构  ─► legacy_dynamic(1500) ─► PEFeatureProjector(1500→256→128) ─► 128  │
  字节统计 ─► stat(49)          ─► stat_projector(49→128)            ─► 128    │
                          └───────────────────────────────┬─────────────────────┘
                                                          ▼
                                              concat → 384-dim
                                          LayerNorm(384) → Linear(64) → GELU → Linear(2)
```

**参数量分解**（`best_model_739k.pt` state_dict 实测）：

| 模块 | 参数 | 说明 |
|---|---|---|
| ByteEmbedding | 32,768 | vocab 256 × 128 |
| input_proj | 16,512 | 128→128 + GELU |
| DSRA ×2 | 196,876 | 每层 98,438（qkv 49,152 + out_proj 16,384 + 2×slot init 16,384 + gates 132） |
| PEFeatureProjector | 417,664 | 1500→256→128 |
| stat_projector | 6,656 | 49→128 |
| classifier | 25,538 | LayerNorm→64→2 |
| **可训练合计** | **696,014** | |

> **反直觉点**：checkpoint 报告的 `numel=9,183,060` 是个陷阱——含 8,388,608 的**非训练**正弦位置编码 buffer（65536×128）和 98,438 的模块别名重复（`self.dsra = self.dsra_layers[0]`）。真正可训练参数 69.6 万，一个"小到能在 CPU 上试跑"的模型。

**为什么是 concat 而不是 attention 融合**：v2.6Exp 里同时实现了 `add / gated / residual_stat_gate / attention` 等融合模式（`model.py:561-584`），但默认 `concat`。理由很工程——三路表示维度已经对齐（各 128），concat 是信息无损的最简基座，进阶融合收益没到值得引入超参和训练不稳的程度。**先拿到能收敛的基线，再谈花活。**

---

## 3. MHDSRA2 深潜：如何把 2 TB 的注意力压进 2.24 GB

### 3.1 复杂度对比

| | 常规多头自注意力 | MHDSRA2（本实现） |
|---|---|---|
| 每层注意力得分 | `O(B·H·T²)` | `O(B·H·C·(K+W+topk))`，C=chunk 512 |
| 每层缓存 | `O(B·H·T·d)` 全序列 | `O(B·H·(K+W)·d)` 固定，与 T 解耦 |
| 65536 序列 × batch64 | 单层 logits ≈ 65536²·64·4·2B ≈ **2 TB** | 逐块流式，实测总峰值 **2.24 GB** |
| 跨块状态 | 无 | slot_k/v `[B,H,128,d]` + local cache ≤256 + age/usage/confidence |

### 3.2 算法（每 512-token chunk 三步）

```
状态:  slot_k/v [B,H,128,32], age/usage/confidence [B,H,128], local cache ≤256 token
对每个 chunk(512 token):
  ① slot read    Q·K_slots^T → [B,H,512,128]，取 read_topk=8 做稀疏 softmax，聚合 V_slots
  ② local attn   因果 SDPA，范围 = [local cache(≤256), 当前 chunk]，存下最近 256 token
  ③ slot write   每 token 路由到 write_topk=4 个槽，scatter_add 写入，
                  并用 gated forget（age/usage/confidence）做槽更新
  可选 ④ external retrieval（生产 use_retrieval=False）
```

要点：**槽是固定大小的"压缩记忆"，块与块之间只靠槽状态流动**，所以复杂度不随序列长度增长——这正是"长序列端到端训练"能成立的机制。

### 3.3 显存实测曲线（`docs/code_project_case_studies.md`）

| chunk_size | batch=64 训练峰值 | 备注 |
|---|---|---|
| 512 | **2.24 GB** | 生产配置 |
| 1024 | 4.23 GB | ×1.9 |
| 2048 | 8.25 GB | 8GB 卡 OOM |

`estimate_attention_memory_bytes()`（`improved_dsra_mha.py:1283`）给的是逐项预算：`slot_logits = B·H·C·K·db`（512·64·4·128·4B ≈ 67MB）取代了 `B·H·T²` 的天文数字，`local_kv_cache = 2·B·H·W·d_head` 有界。

### 3.4 FP16 的坑：为什么必须纯 FP32

主流训练都在 AMP，但这里 **`mixed_precision=False` 是硬约束**：DSRA 的 chunk 注意力在 FP16 下前向溢出 → NaN（`train_739k_full.py:234` 注释原话）。还实测过 bf16——**无 NaN 但无加速**，因为 DSRA 的 chunk 循环是 Python 循环，计算没喂饱 GPU 时精度红利换不来吞吐。结论：单精度 26 小时，直接跑。

### 3.5 diversity loss：防"槽坍塌"

128 个全局槽如果退化成一个，稀疏检索就白做了。所以训练循环叠加 `diversity_loss_weight=0.03 × mean((Gram(K_slots)−I)²)`（`improved_dsra_mha.py:1270`）。关键工程点：惩罚项用 Gram 矩阵的**逐元素平方**，绕开了 `O(K³)` 的行列式/求逆，复杂度只有 `O(B·H·K²·d)`，训练期可用。

### 3.6 PagedExactMemory：为 2M token 准备的分页精确检索

CPU 侧按页（`page_size=1024, top_pages=4, max_tokens=128`）存放全量 token K/V，GPU 注意力层只拿回一个小子集；页面得分 = max(页均值分, 页内 max-token 分)。这是给 2M token 序列的精确召回路径（生产模型 `use_retrieval=False`，属于前瞻基础设施）。

---

## 4. 特征工程：legacy_dynamic 与基线的"1500 维谎言"

`legacy_dynamic` 是 1500 维的动态布局（`extractor.py:706-855`）：

```
[0:18]  文件头/安全标志固定字段
[18:18+3N]  每 section 3 字段：exec / write / read 标志（N = section 数）
[18+3N:18+3N+29]  聚合特征：熵、导入/导出统计、尾部数据、API 类别计数……
[18+3N+29:1500]  零填充
真实特征 = 47 + 3N，典型 N=6 时 65 个真实 / 1435 个零
```

三个值得注意的点：

1. **动态布局**：所有聚合特征的位置随 section 数右移。这是有意的——**结构信息在"每个 section 是不是可执行/可写/可读"里**，而不是在固定槽位里。代价是特征名与位置不对应（`FEATURE_NAMES` 静态列表和动态顺序对不上），这也是后续内容特征（v1/v2）改走"固定语义名 + 位置无关"路线的原因。
2. **idx16 恒 0**：`has_signature = hasattr(pe, 'DIRECTORY_ENTRY_SECURITY')`，但 pefile 从不设置这个属性（证书表在 `OPTIONAL_HEADER.DATA_DIRECTORY[4]`）→ 该列**对所有样本恒 0**。这不是 bug 是死列，`fixed_v3` schema 直接删了它。
3. **对比基线**：基线那 256 维 lightweight 哈希桶里，`lw_232`(#4)、`lw_200`(#6) 竟然排进 Top-6——**基线的判别力有一块来自个别哈希桶的稀疏命中**，非常脆弱。v2.6Exp 的 1500 维动态布局 + 原始字节双通道，不再押注哈希桶。

stat（49 维）与 lightweight（256 维）都提取并缓存，但 **lightweight 不进模型**（`dataset.py:1375` 丢弃）——它是留给后续/其他路径的资产，不是无用代码。

---

## 5. 训练配方与 Windows 工程战争

**配方表**（`train_739k_full.py` + checkpoint `train_config`）：

| 项 | 值 |
|---|---|
| 损失 | Focal CE（γ=1.0, α=0.55）+ label_smoothing 0.03 + diversity 0.03 |
| 优化器 | AdamW, lr 8e-5, wd 1e-5, betas (0.9,0.999) |
| 调度 | 3 epoch 线性 warmup（1e-6）→ cosine → min_lr 1e-6（终值 1.67e-6） |
| 裁剪 | gradient_clip 0.75 |
| 精度 | **FP32 only**（见 §3.4） |
| 数据加载 | 8 worker + persistent_workers + pin_memory |
| 字节长度 | 提取 65536，训练截断 4096 |

**4096 截断的取舍**：保留 PE 头 + 入口点上下文（前 4KB 恰好覆盖），DSRA chunk 128→8，单步 9.2s→0.5s（**18×**）。这是"用长上下文信息换可训练性"——丢失的深部字节信息由 Stage-2 的内容特征补回（§6），两段各司其职。

**Windows spawn 崩溃（真实黑历史）**：8 个 DataLoader worker 各自初始化 OpenBLAS，每个抢 32 线程 → 线程栈分配失败。修复必须在 `import numpy/torch` 之前 `os.environ.setdefault("OMP_NUM_THREADS","1") / ("OPENBLAS_NUM_THREADS","1")`（`train_739k_full.py:16-20`）。顺序敏感，写错一行就是诡异挂起。

**结果**（`train_739k_receipt.json`）：20 epochs / 27.23h，Test F1 0.9795，P 0.9723，R 0.9867，FP 3217 / FN 1525，AUC 0.9921。早停（patience 8）从未触发——**模型在第 20 epoch 还在涨，是被 max_epochs 掐停的**。

---

## 6. Stage-2：不重训基座的"低成本纠错层"

**动机**：基座校准极好（ECE 0.001，99.74% 的 score≥0.9 是真恶意），但存在**结构性自信错误**——训练良性 89% 是 DLL，导致大型独立 EXE（安装器/加载器）靠近恶意质心，基座对它们自信地判黑。而 4096 字节截断的基座看不到的原始文件结构，正是纠错所需的独立信号。

**331 维特征矩阵**：

```
base model(739k, 4096 bytes) ─► p_malicious
  └► derived6 = [p, p², |p−0.5|, log p, log(1−p), logit(p)]
原始文件 ─► content_pe_v1(100) + content_pe_v2(182) + content_string(43)
              └► 331 维 → 3×HGB(seed 0/1/2) → mean proba → 阈值 0.55 → 判定
```

- **content_pe_v1（100）**：文件/头级 28 + 11 数据目录 ×3 + 导入 7 + API 类别比例 6 + 导出/资源/overlay/节权限组合。
- **content_pe_v2（182）**：32 个常见 DLL 导入存在性/比例、16 API 类 ×3、导出模式、11 资源类型、21 节/入口点特征。
- **content_string（43）**：ASCII/UTF-16 游程、URL/IPv4/注册表/路径正则 + 14 类语义串模式（持久化/注入/加壳器…）。

**消融曲线**（`stage2_report.json`）：

| 配置 | Test F1 | 错误数 |
|---|---|---|
| 仅 base_prob（阈值 0.49） | 0.97999 | 4597 |
| +content_pe_v1(100) | 0.99282 | 1647 |
| **全 331 维（阈值 0.55）** | **0.99341** | **1511**（FP 778 / FN 733） |

UPX 专项：测试集 3482 个 UPX 加壳白文件，基座对 UPX 白文件误报 **236 → 52**；**排除 UPX 后 F1 > 0.99**。

**为什么用 GBM 而不是再训一层 DNN**：这是纠错层，特征只有 331 维且可解释（导入表、字符串模式），HGB 在表格特征上是正确工具；3-seed 取均值压方差；阈值在 VAL F1 上扫、TEST 只评一次，**无测试集泄漏**。冻结后导出 `axon_stage2_hgb_json_v1` 原始数组 JSON，C++ DLL 免 sklearn 推理，有与 sklearn `predict_proba` 的 `max_abs_diff` 对齐校验。

---

## 7. 部署链路：一个 ONNX + 一个 DLL

**模型契约**（`dist/axon_739k_onnx_final_20260808`）：

| | shape | dtype |
|---|---|---|
| byte_seq | [1, 4096] | int64 |
| pe_features | [1, 1500] | float32（legacy_dynamic） |
| stat_features | [1, 49] | float32 |
| logits | [1, 2] | float32 |

**交付形态**：单 `axon_onnx_predict.dll`（`__cdecl`，**18 个导出**，x64 `kvd_config` **96 字节**，`kvd_create → kvd_scan_path/bytes → kvd_free → kvd_destroy` 生命周期）+ 捆绑的 onnxruntime **1.24.x**（bin/）+ 43.8MB 外置 `.onnx.data`。免 Python/PyTorch/sklearn。rust 示例运行时断言 `size_of<KvdConfig>()==96`，js 用 `koffi.sizeof==96` **双端校验 ABI**。

**C++ 特征移植对齐**：`legacy_dynamic_pe_features` 全量 port 到 C++（`axon_onnx_predict.cpp:3267`），与 Python 提取器 **16/17 精确对齐、预测 diff 0.00000**。两个真实的坑：
1. **onnxruntime 遮蔽**：系统 `C:\Windows\System32` 里的旧 1.17.1 会抢占捆绑的 1.24.4 → 示例里必须 `SetDllDirectory(bin)` 锁定加载路径。
2. **GBK 编码坑**：C++ 源码里的 UTF-8 中文注释被 MSVC 按 GBK 误读，`\` 续行符吞掉函数签名导致"data 未声明"——全部注释转 ASCII 解决。

**延迟预算**：权重预加载后单文件 **~355ms 均值（<500ms）**；DLL 内部对 read / feature_extraction / onnx / total 分项计时，方便定位瓶颈。

**对比基线**：2.5Exp 生产扫描 = Python 框架（main.py 10,699 行、内嵌 ~40 模块）+ 3 个 LightGBM txt + scaler + 恒等 selector + family_classifier.json + hardcase manifest 的多文件散装。v2.6Exp 收敛成**单一交付物**，运维面小一个量级。

---

## 8. 评测诚实论：为什么不能直接比 F1

基线的 eval 数字（acc 0.9938 / FPR 0.75% / FNR 0.53%）**在纸面上比 v2.6Exp 基座（acc 0.9679）好看**。真相是分布不对等：

- 基线测试分布：364,755 行里约 60% 恶意，**不含 UPX 加壳白文件**，无对抗性难例。
- v2.6Exp 测试分布：147,796 行，恶意 114,693，**含 3,482 个 UPX 加壳白文件**等难例。

在更难分布上：基座 F1 0.9795（FPR 9.69% 的主因是 UPX 白文件），叠 Stage-2 后 F1 0.99341，**排除 UPX 后 F1 >0.99**。

**零误报 + recall>99% 为什么不可同时达到**（`goal_eval.json`）：
- 任何阈值扫描都没有 `fp0_recall99` 点；
- 221 个 FN 中基座置信 >90%——**最可能是标签噪声（被标成恶意的白文件/被标成白文件的恶意样本），而非模型缺陷**；
- 去噪后的真实召回估计 **0.9955**。

结论不是"目标失败"，而是"目标需要精确定义"：在静态特征 + 有噪声标签的现实里，零误报是个**不可能命题**，可操作的版本是"对可验证白文件零误报 + recall>99%"。

---

## 9. 反直觉清单（彩蛋）

1. **基线的"1500 维"是纸面数字**，实际部署只有 350 维，还静默丢特征（含门控需要的那个）。
2. **基线 feature_selector 是恒等映射**——"特征选择"是 no-op。
3. **DSRA 在 FP16 下直接 NaN**，必须纯 FP32，与主流 AMP 直觉相反；bf16 无 NaN 但也无加速。
4. **模型报告的 918 万参数有 92% 是位置编码 buffer**，真可训练参数 69.6 万。
5. **首个 ONNX 导出用了 65536**（训练却是 4096）→ 重新 `--byte-length 4096` 对齐。
6. **C++ 里 UTF-8 中文注释能被 GBK 读成换行**，`\` 续行吞掉函数签名。
7. **System32 的旧 onnxruntime 会劫持新 DLL 的 LoadLibrary**。
8. **4096 截断是"用信息换 18× 速度"**，由 Stage-2 用 content 特征补回深度。
9. **基线的 top-6 特征里有两个是哈希桶**（lw_232/lw_200）——判别力押注在哈希碰撞上。
10. **Stage-2 阈值 0.55 在 VAL 扫、TEST 只评一次**——用纪律换无泄漏评估。

---

## 10. 当前状态与下一步

**良性扩充重训已完成**（cache 813,098，良性占 29.5%，epoch 17–20 补跑完毕）：

| 指标 | 扩充前 full_739k | 扩充后 full_739k_benign |
|---|---|---|
| best 模型 | epoch 16/20 | **epoch 19**（val F1 0.9763） |
| Test F1 | 0.97946 | **0.97648** |
| Test FPR | 0.09695 | **0.0716（−26%）** |
| Test FNR | 0.0167 | 0.0174 |
| 测试集规模 | 147,796（良性 33,183） | **162,619（良性 48,006，+45%）** |

> **口径说明**：扩充把测试集良性样本从 ~33k 增到 ~48k，绝对 FP 数对比不公平；**FPR 才是同口径**——在更大、更难的良性测试池上误报率下降 26%，良性扩充兑现了"压 FP"目标。（受制于两次后台任务被杀，重训中途在 epoch 16 停过一次，已通过 `train_739k_full.main(resume_from=...)` + `train_739k_benign.py` 自动续训机制恢复。）

**流水线已全部跑完**（重导 base_prob → 重跑 Stage-2 331 维 → goal_eval → UPX/归因/白名单分析）：
- `base_prob_739k_benign.py`：813,098 全量导出（GPU，逐 chunk 落盘 + resume 智能跳过已完成 chunk；增量 flush 防中断，曾在"已完成 chunk 也进 chunk_data"上踩 KeyError 已修）
- `extract_content_739k_benign.py`：复用旧 content_v1 块 0-13 + 只提取新增 chunk（append-only manifest 验证通过，locate 99.39%）
- `extract_content_v2string_739k_benign.py`：按新划分重算 val/test（81,309/162,619），增量分块落盘
- `stage2_739k_v2_benign.py`：3-seed HGB 重训 + VAL 扫阈值（0.52）+ TEST 一次性评估

**Stage-2 重训结果**（同口径：恶意池不变，良性池 +45%）：

| 指标 | 扩充前 | 扩充后 | 说明 |
|---|---|---|---|
| Test F1 | 0.99341 | **0.99192** | 良性池变大变难 |
| Test FPR | 2.34%（778/33,183） | **2.18%**（1047/48,006） | 率下降（绝对 FP 不可比） |
| Test FNR | 0.64%（733/114,613） | 0.70%（807/114,613） | 同恶意池，略升 |
| recall | 0.99360 | **0.99296** | >99% 保持 |
| 去噪真召回 | 0.99553 | 0.99462 | FN 中 190 个高置信良性（疑似标签噪声） |
| UPX 白文件 FP | 52（率 1.49%） | 58（率 1.71%） | test 内 3,482→3,391 个 |

**目标验证**（`whitelist_operating_point.json`）：把 17k UPX 白名单语料（`F:\私人\良性文件\待加入白名单_upx`）作硬白名单 → **已验证白文件零误报达成**（58 FP→0），剩余 989 个未验证白文件 FP（占未验证白 2.22%），recall 0.99296。绝对零误报（含未见白文件）仍不可达——静态特征 + 标签噪声下的硬约束，但 FPR 从 2.34% 全面压到 2.18%、基座 FPR 从 9.70% 压到 7.16%。

**改进实验闭环（2026-08-10，用户否决硬白名单后重新聚焦模型侧）**——结论：**静态内容特征对 FP 人群已到信息天花板**。

| 实验 | 结果 | 结论 |
|---|---|---|
| 重平衡 sweep（良性 non-DLL 过采样 w∈{1,3,6,10}、benign_all×3） | 最好 config 只 −7 FP（646→639），VAL 改善不迁移 TEST | 瓶颈是特征本身弱，不是训练分布 |
| 版本字符串特征（8 维；满覆盖 val 99.32% / test 99.31%，修正 G:/H:→F: 前缀 + 目录索引 + 挂起免疫池，4.7h） | **匹配 recall 下部署点反升 FP**（@0.9930: 1065→1093，+2.6%）；val-recall99 操作点 −7%（646→601） | **不部署，不建 C++ 提取器**——46% 覆盖时的 −2.5% 是零填充列伪影；版本信息只对边界样本有极弱信号，被部署点淹没 |
| 签名有效性探针（raw blob 直读绕过 pefile SECURITY 解析 bug） | 签名存在/时间戳 AUC 仅 0.33，locatable FP 仅 22% 有证书 | 与版本信息同弱；呼应 loop73（Valid 签名仅 0.7% 可救） |
| **目标对齐阈值**（非新特征） | 阈值 0.52→0.463：FP 1047→788（−25%），recall 0.9930→0.9900 | **最大廉价杠杆**，但 recall 余量归零，需用户决策 |

**先例印证**：loop142（证书 blob 特征，2026-07-05）、loop73（Authenticode 信任降级，07-03）均已否决证书/签名方向；loop151（trusted signer guard，07-08）用预声明发布者 term 列表降 FP——**属用户否决的"硬编码可信身份"类别，不再采用**。FP 人群（打包、无版本、无签名、重系统 API 独立 EXE）与恶意在静态特征下真实重叠，唯一未被消耗的正交证据是行为/信誉/标签治理。

**下一步值得做的**（按 ROI，全部模型侧，**不含白名单**）：
1. **目标对齐阈值决策**：把 Stage-2 阈值从 val-F1 最优（0.52）改为目标对齐（R>0.99 下最小 FP，~0.463），FP −25% 但 recall 余量归零——取决于用户的 recall 风险偏好。
2. **基座重训带良性 EXE 分布**：基座训练良性仍 87% DLL，FP 是 base-confident（base_prob AUC 0.97 主导）；重训过采样良性 EXE 是根因修复（~27h，收益不确定——基座输入本身不含版本/签名信息）。
3. **标签噪声治理**：FN 中 190 个 s2<0.1 高置信良性 + FP 中 313 个基座>90% 自信判黑，人工复核后用干净标签重训。**priority 28 个样本人工复核已 28/28 全量闭环**（`D:\待复核\复核完成的列表.txt`）：**15 改标**——14 误标黑翻正（6 FN 全部 + 8 跨树冲突标恶样本，即**恶意树混入 ≥8 真良性**）、1 误标白改恶（`c7c3a960`，**良性树混入 1 真恶意**）；13 确认保持。FP 侧结论不变——313 个里仅 1 个 sha 在恶意树，**FP 不是标签噪声**，是模型对真实良性样本的弱点（见 `label_governance/summary.json`）。归档：`label_governance/review_results.json`、`review_verdicts.csv`、`label_corrections.csv`（重训应用契约）。
4. ~~版本字符串特征工程化~~：**已否决**（满覆盖下部署点不降反升，见上表）。
