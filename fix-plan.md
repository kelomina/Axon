# Axon v2.6 待修复问题

> 更新日期: 2026-05-28
> 所有严重/中严重度问题及低严重度问题已全部修复
> 已修复问题详见 [fixed.md](fixed.md)

---

## 一、观察项（无需代码修改）

### H10-F3. 旧缓存文件未自动清理

- **文件**: `src/dataset.py`
- **严重程度**: 🟢 低
- **触发条件**: 存在旧格式缓存文件时
- **状态**: 观察项，手动清理即可：`rm -rf cache_dir/*.npz`

### H10-F4. `_extract_fallback` 不含新增API类别特征

- **文件**: `src/kvd_features/extractor.py:533-552`
- **严重程度**: 🟢 低（预期行为）
- **触发条件**: PE解析失败时
- **状态**: 预期行为——fallback用于非PE文件，零填充是合理的降级策略。建议监控fallback调用频率。

---

## 二、配置优化建议

> 以下建议针对恶意软件检测场景，基于代码审查中发现的架构瓶颈和参数不合理之处。

---

### 2.1 数据层配置

#### 2.1.1 MalwareDataset

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `max_byte_length` | 65536 | **32768** | 64KB过长导致DSRA计算量巨大。PE文件关键信息集中在头部32KB内 |
| `pe_feature_dim` | 1500 | **1500** | 保持不变 |
| `use_cache` | True | **True** | 保持不变，H9修复后缓存已生效 |

#### 2.1.2 NPZDataLoader

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `num_workers` | 4 | **0 (Windows) / 4 (Linux)** | Windows下多进程pickle问题频发 |

---

### 2.2 特征提取层配置

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `max_file_size` | 65536 | **32768** | 与max_byte_length对齐 |
| `entropy_high_threshold` | 0.8 | **0.75** | 恶意软件加壳后熵值通常在0.75-0.9之间 |
| `section_entropy_min_size` | 256 | **128** | 某些恶意样本的section很小 |

---

### 2.3 模型层配置

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `byte_embedding_dim` | 128 | **64** | 字节值只有256种，128维嵌入过于冗余 |
| `pos_encoding_mode` | sinusoidal | **sinusoidal** | 已修复(M5) |
| PEFeatureProjector `hidden_dim` | 256 | **512** | 1500→256压缩比6:1过大 |
| PEFeatureProjector `dropout` | 0.1 | **0.2** | PE特征维度高但样本量可能有限 |
| `fusion_type` | concat | **attention** | attention融合让模型自动学习交互权重 |
| `dropout` | 0.1 | **0.15** | 略增dropout防过拟合 |

---

### 2.4 DSRA引擎配置

#### 容量配置

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `dim` | 128 | **256** | 恶意软件字节模式复杂度高，128维不足 |
| `slots` | 128 | **256** | 更多slot存储不同模式，防collapse |
| `local_window` | 256 | **512** | 恶意软件局部模式通常跨200-500字节 |

#### 读取配置

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `read_topk` | 8 | **16** | 恶意软件检测需综合多个记忆模式 |
| `tau_init` | 8.0 | **4.0** | tau=8.0使softmax接近均匀分布，过于分散 |
| `conf_read_bias` | 0.50 | **0.30** | 降低置信度偏置权重 |
| `age_read_penalty` | 0.005 | **0.003** | 降低年龄惩罚，保留早期关键模式 |

#### 写入配置

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `write_topk` | 4 | **6** | 更多写入目标减少信息稀释 |
| `tau_write_init` | 4.0 | **6.0** | 更锐利的写入路由 |
| `eta` | 0.25 | **0.15** | 更保守的写入饱和速度 |
| `max_update` | 0.50 | **0.35** | 限制单次写入强度 |
| `write_gate_min` | 0.2 | **0.05** | 大幅降低下限 |
| `novelty_threshold` | 0.0 | **0.1** | 过滤低新颖度token |
| `write_protection` | 0 | **16** | 写入后保护slot不被覆盖 |

#### 遗忘配置

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `forget_base` | 0.001 | **0.0005** | 保留稀有恶意特征 |
| `forget_conflict` | 0.20 | **0.10** | 避免新benign特征覆盖旧恶意特征 |
| `forget_age` | 0.0002 | **0.0001** | 保留早期关键模式 |
| `conflict_protection_coef` | 0.3 | **0.5** | 更强的冲突保护 |

#### 状态衰减配置

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `usage_decay` | 0.995 | **0.998** | 保持slot活跃度信息更持久 |
| `confidence` 初始值 | 0.01 | **0.01** | 已修复(M14) |

---

### 2.5 训练配置

#### 优化器配置

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `learning_rate` | 1e-4 | **3e-4** | DSRA的tau参数需要较大学习率 |
| `weight_decay` | 1e-5 | **1e-4** | 防止slot_k/slot_v_init过拟合 |
| `betas` | (0.9, 0.999) | **(0.9, 0.98)** | 让优化器对梯度变化更敏感 |
| `eps` | 1e-8 | **1e-6** | FP16下数值稳定性 |

#### 学习率调度

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `warmup_epochs` | 3 | **5** | DSRA的slot需要更长时间预热 |
| `warmup_start_lr` | 1e-6 | **1e-5** | 避免初期完全无梯度更新 |
| `min_lr` | 1e-6 | **1e-5** | 避免训练末期完全停滞 |

#### 梯度与精度

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `gradient_clip` | 1.0 | **0.5** | DSRA梯度不稳定，更保守的裁剪更安全 |
| `label_smoothing` | 0.0 | **0.05** | 防止过度自信 |

#### 损失函数

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `focal_gamma` | 0.0 | **2.0** | 处理类别不平衡（H7已修复） |
| `diversity_loss_weight` | 0.05 | **0.05** | 已启用，防slot collapse |

#### 训练策略

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `max_epochs` | 50 | **100** | DSRA收敛较慢 |
| `early_stopping_patience` | 5 | **10** | DSRA训练波动大 |
| `batch_size` | 16 | **32** | 增大batch提高训练稳定性 |
| `eval_interval` | 1 | **2** | 减少评估频率 |

---

### 2.6 配置优化汇总表

| 层级 | 组件 | 参数 | 当前值 | 建议值 |
|------|------|------|--------|--------|
| 数据 | MalwareDataset | `max_byte_length` | 65536 | **32768** |
| 数据 | NPZDataLoader | `num_workers` | 4 | **0 (Win) / 4 (Linux)** |
| 特征 | ExtractionConfig | `max_file_size` | 65536 | **32768** |
| 特征 | ExtractionConfig | `entropy_high_threshold` | 0.8 | **0.75** |
| 特征 | ExtractionConfig | `section_entropy_min_size` | 256 | **128** |
| 模型 | ByteEmbedding | `byte_embedding_dim` | 128 | **64** |
| 模型 | PEFeatureProjector | `hidden_dim` | 256 | **512** |
| 模型 | PEFeatureProjector | `dropout` | 0.1 | **0.2** |
| 模型 | AxonMalwareModel | `fusion_type` | concat | **attention** |
| 模型 | AxonMalwareModel | `dropout` | 0.1 | **0.15** |
| DSRA | 容量 | `dim` | 128 | **256** |
| DSRA | 容量 | `slots` | 128 | **256** |
| DSRA | 容量 | `local_window` | 256 | **512** |
| DSRA | 读取 | `read_topk` | 8 | **16** |
| DSRA | 读取 | `tau_init` | 8.0 | **4.0** |
| DSRA | 读取 | `conf_read_bias` | 0.50 | **0.30** |
| DSRA | 读取 | `age_read_penalty` | 0.005 | **0.003** |
| DSRA | 写入 | `write_topk` | 4 | **6** |
| DSRA | 写入 | `tau_write_init` | 4.0 | **6.0** |
| DSRA | 写入 | `eta` | 0.25 | **0.15** |
| DSRA | 写入 | `max_update` | 0.50 | **0.35** |
| DSRA | 写入 | `write_gate_min` | 0.2 | **0.05** |
| DSRA | 写入 | `novelty_threshold` | 0.0 | **0.1** |
| DSRA | 写入 | `write_protection` | 0 | **16** |
| DSRA | 遗忘 | `forget_base` | 0.001 | **0.0005** |
| DSRA | 遗忘 | `forget_conflict` | 0.20 | **0.10** |
| DSRA | 遗忘 | `forget_age` | 0.0002 | **0.0001** |
| DSRA | 遗忘 | `conflict_protection_coef` | 0.3 | **0.5** |
| DSRA | 状态 | `usage_decay` | 0.995 | **0.998** |
| 训练 | 优化器 | `learning_rate` | 1e-4 | **3e-4** |
| 训练 | 优化器 | `weight_decay` | 1e-5 | **1e-4** |
| 训练 | 优化器 | `betas` | (0.9, 0.999) | **(0.9, 0.98)** |
| 训练 | 优化器 | `eps` | 1e-8 | **1e-6** |
| 训练 | 调度器 | `warmup_epochs` | 3 | **5** |
| 训练 | 调度器 | `warmup_start_lr` | 1e-6 | **1e-5** |
| 训练 | 调度器 | `min_lr` | 1e-6 | **1e-5** |
| 训练 | 梯度 | `gradient_clip` | 1.0 | **0.5** |
| 训练 | 梯度 | `label_smoothing` | 0.0 | **0.05** |
| 训练 | 损失 | `focal_gamma` | 0.0 | **2.0** |
| 训练 | 策略 | `max_epochs` | 50 | **100** |
| 训练 | 策略 | `early_stopping_patience` | 5 | **10** |
| 训练 | 策略 | `batch_size` | 16 | **32** |
| 训练 | 策略 | `eval_interval` | 1 | **2** |

---

### 2.7 配置变更影响预估

#### 参数量对比

| 配置 | 当前参数量 | 优化后参数量 | 变化 |
|------|-----------|-------------|------|
| ByteEmbedding | 32,768 | 16,384 | -50% |
| PositionalEncoding | 0 (sinusoidal) | 0 | 不变 |
| DSRA (dim=128→256, slots=128→256) | ~500K | ~2M | +300% |
| PEFeatureProjector | ~420K | ~900K | +114% |
| 分类器 | ~33K | ~65K | +97% |
| **总计** | **~1.0M** | **~3.0M** | **+200%** |

---

## 附录：问题统计

| 严重程度 | 已修复 | 待修复 | 合计 |
|---------|--------|--------|------|
| 🔴 高 | 14 | 0 | 14 |
| 🟡 中 | 17 | 0 | 17 |
| 🟢 低 | 15 | 0 | 15 |
| **合计** | **46** | **0** | **46** |

> 已修复问题详见 [fixed.md](fixed.md)
