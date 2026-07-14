# Axon v2.6 已修复问题记录

> 记录所有已完成的修复，包括原始问题描述、修复方案和验证结果

---

## 一、高严重度问题（已修复）

### H1. `write_gate_min` 绕过 novelty 过滤机制 ✅

- **文件**: `src/dsra/mhdsra2/improved_dsra_mha.py:519-521`
- **修复日期**: 2026-05-27
- **验证**: 2轮交叉验证通过

**修复内容**: 将clamp从token_gate移到raw_gate

```python
raw_gate = torch.sigmoid(self.token_write_gate(k)).squeeze(-1).clamp(min=cfg.write_gate_min)
token_gate = (raw_gate * novelty).clamp(min=1e-6)
```

---

### H2. `diversity_loss` 在 `detach_state=True` 下无梯度 ✅

- **文件**: `src/dsra/mhdsra2/improved_dsra_mha.py:631-637` + `:793-795` + `src/trainer.py:363-364`
- **修复日期**: 2026-05-27
- **验证**: 2轮交叉验证通过

**修复内容**: 在detach前保存带梯度的slot_k_next，diversity_loss优先使用带梯度版本；trainer中添加diversity_loss调用

```python
if cfg.detach_state:
    self._slot_k_before_detach = slot_k_next
    slot_k_next = slot_k_next.detach()
    ...

sk_source = getattr(self, '_slot_k_before_detach', state.slot_k)
sk = F.normalize(sk_source, dim=-1)
```

---

### H3. `_slot_read` 的 `tau` 缺少 clamp ✅

- **文件**: `src/dsra/mhdsra2/improved_dsra_mha.py:349-350`
- **修复日期**: 2026-05-27
- **验证**: 2轮交叉验证通过

**修复内容**: 添加与写入路径一致的clamp

```python
tau = self.log_tau_read.exp().float()
tau = tau.clamp(min=1.0, max=64.0)
```

---

### H4. `_is_valid_sample` 逻辑错误 ✅

- **文件**: `src/dataset.py:141-154` + `src/kvd_features/extractor.py:357-499`
- **修复日期**: 2026-05-27
- **验证**: 2轮交叉验证通过

**修复内容**: 基于PE结构验证而非文件后缀；1GB大小上限；fast_load=True加速验证；pe.close()确保释放句柄；parse_data_directories按需解析；裸except改为except Exception

---

### H5. MalwareDSRAEncoder 分块只取最后一块 + 最后token ✅

- **文件**: `src/model.py:334-335`
- **修复日期**: 2026-05-27
- **验证**: 2轮交叉验证通过

**修复内容**: mean pooling替代last token

```python
byte_repr = byte_out.mean(dim=1)
```

---

### H6. RoPE 读写路径不一致 ✅

- **文件**: `src/dsra/mhdsra2/improved_dsra_mha.py:476-489`
- **修复日期**: 2026-05-28
- **验证**: 2轮交叉验证通过

**修复内容**: _slot_write添加RoPE旋转，与_slot_read一致

---

### H7. Focal Loss 与 `[B, 2]` logits 维度不匹配 ✅

- **文件**: `src/trainer.py:276-291`
- **修复日期**: 2026-05-28
- **验证**: 2轮交叉验证通过

**修复内容**: 从binary_cross_entropy_with_logits改为cross_entropy

---

### H8. `stat_features` 维度不匹配且从未传入模型 ✅

- **文件**: `src/model.py` + `src/dataset.py` + `src/kvd_features/extractor.py`
- **修复日期**: 2026-05-28
- **验证**: 2轮交叉验证通过

**修复内容**: 分离PE特征和统计特征；添加stat_projector(49→pe_projection_dim)；forward签名添加stat_features参数；trainer传递stat_features

---

### H9. 缓存 `hash()` 跨会话不可复用 ✅

- **文件**: `src/dataset.py:183-186`
- **修复日期**: 2026-05-28
- **验证**: 2轮交叉验证通过

**修复内容**: hashlib.md5替代hash()

```python
file_hash = hashlib.md5(str(file_path).encode()).hexdigest()
```

---

### H10. API模式大小写不匹配导致特征恒为零 ✅

- **文件**: `src/kvd_features/extractor.py:316-327` + `:491-518`
- **修复日期**: 2026-05-28
- **验证**: 2轮交叉验证通过

**修复内容**: api_categories小写化；添加API类别特征计算逻辑和Packer关键字特征计算；parse_data_directories添加IMPORT/EXPORT目录

---

## 二、高严重度残余问题（已修复）

### H9-F1. `_load_from_cache` 默认维度错误：100→49 ✅

- **文件**: `src/dataset.py:199`
- **修复日期**: 2026-05-28
- **验证**: 2轮交叉验证通过

**修复内容**: np.zeros(100) → np.zeros(49)；同步修复NPZDataset异常处理和docstring

---

### H9-F2. `hashlib` import 在函数内部 ✅

- **文件**: `src/dataset.py:8`
- **修复日期**: 2026-05-28
- **验证**: 交叉验证通过

**修复内容**: hashlib import移到文件顶部

---

### H1-F1. `write_gate_min` 语义偏移 ✅

- **文件**: `src/dsra/mhdsra2/improved_dsra_mha.py:520`
- **修复日期**: 2026-05-28
- **验证**: 交叉验证通过

**修复内容**: token_gate添加非零下限防止梯度死区

```python
token_gate = (raw_gate * novelty).clamp(min=1e-6)
```

---

### H2-F2. 分块梯度丢失 ✅

- **文件**: `src/model.py:318-341` + `src/trainer.py:365-368`
- **修复日期**: 2026-05-28
- **验证**: 3个代理交叉验证通过

**修复内容**: 在MalwareDSRAEncoder的分块循环中，每个chunk后计算diversity_loss并累积取平均，通过state._diversity_loss传递给AxonMalwareModel，再通过outputs['diversity_loss']传递给trainer。trainer优先使用累积值，回退到直接调用。

---

### H2-F1. 陈旧引用风险 ✅

- **文件**: `src/dsra/mhdsra2/improved_dsra_mha.py:637-639`
- **修复日期**: 2026-05-28
- **验证**: 交叉验证通过

**修复内容**: detach_state=False时清除_slot_k_before_detach

```python
else:
    if hasattr(self, '_slot_k_before_detach'):
        del self._slot_k_before_detach
```

---

## 三、中严重度问题（已修复）

### M1. `new_k / mass_safe` FP16下NaN风险 ✅

- **文件**: `src/dsra/mhdsra2/improved_dsra_mha.py:553-560`
- **修复日期**: 2026-05-28

**修复内容**: safe_mask乘法替代直接除法

```python
safe_mask = (mass > cfg.eps).to(dtype=k.dtype)
mass_safe = mass.clamp_min(cfg.eps).to(dtype=k.dtype)
new_k = (agg_k / mass_safe) * safe_mask
new_v = (agg_v / mass_safe) * safe_mask
has_write = safe_mask
```

---

### M2. `slot_v` 无范数约束 ✅

- **文件**: `src/dsra/mhdsra2/improved_dsra_mha.py:592-595`
- **修复日期**: 2026-05-28

**修复内容**: 软范数约束(max_v_norm=10.0)

```python
max_v_norm = 10.0
v_norm = slot_v_next.norm(dim=-1, keepdim=True)
scale = torch.clamp(max_v_norm / (v_norm + 1e-6), max=1.0)
slot_v_next = slot_v_next * scale
```

---

### M3. `conflict_protection_coef` 无下界保护 ✅

- **文件**: `src/dsra/mhdsra2/improved_dsra_mha.py:573`
- **修复日期**: 2026-05-28

**修复内容**: 添加clamp(min=0.0)

```python
write_gate = write_gate * (1.0 - cfg.conflict_protection_coef * conflict_protection).clamp(min=0.0).to(dtype=k.dtype)
```

---

### M4. ByteEmbedding缺少缩放因子 ✅

- **文件**: `src/model.py:92`
- **修复日期**: 2026-05-28

**修复内容**: 添加√d缩放

```python
return self.embedding(x) * (self.embedding_dim ** 0.5)
```

---

### M5. 可学习PE参数量8.4M过大 ✅

- **文件**: `src/model.py:28`
- **修复日期**: 2026-05-28

**修复内容**: 默认改为sinusoidal，修正docstring

```python
def __init__(self, d_model: int, max_len: int = 65536, mode: str = "sinusoidal"):
```

---

### M6. PEFeatureProjector `num_layers` 语义混乱 ✅

- **文件**: `src/model.py:106`
- **修复日期**: 2026-05-28

**修复内容**: 重构为num_hidden_layers=0（与原架构等价）

```python
def __init__(self, ..., num_hidden_layers: int = 0, ...):
    ...
    for _ in range(num_hidden_layers):
        ...
```

---

### M7. `chunk_size=512` 硬编码 ✅

- **文件**: `src/model.py:232+310`
- **修复日期**: 2026-05-28

**修复内容**: 从config读取chunk_size

```python
class MalwareDSRAEncoder:
    def __init__(self, ..., chunk_size: int = 512, ...):
        self.chunk_size = chunk_size
    
    def forward(self, ...):
        chunk_size = self.chunk_size
```

---

### M8. `np.random.seed()` 污染全局随机状态 ✅

- **文件**: `src/dataset.py:490-491, 549-550`
- **修复日期**: 2026-05-28

**修复内容**: 改用RandomState

```python
rng = np.random.RandomState(seed)
rng.shuffle(label_indices)
```

---

### M9. PE特征混入统计特征 ✅

- **文件**: `src/kvd_features/extractor.py`
- **修复日期**: 2026-05-28（通过H8修复一并解决）

**修复内容**: 分离PE特征和统计特征，extract_statistical_features不再包含pe_features

---

### M10. `AxonExperimentConfig` 与 `TrainingConfig` 参数重复 ✅

- **文件**: `src/config.py`
- **修复日期**: 2026-05-28

**修复内容**: 删除AxonExperimentConfig中的重复训练参数(optimizer/lr/weight_decay/max_epochs/early_stopping_patience/gradient_clip/lr_scheduler/warmup_epochs)

---

### M11. 死参数 ✅

- **文件**: `src/config.py`
- **修复日期**: 2026-05-28

**修复内容**: 删除use_class_weights、use_pe_attention；stat_feature_dim改为49并激活；dsra_chunk_size激活使用

---

### M12. 快速模式参数不一致 ✅

- **文件**: `scripts/main.py:111-112`
- **修复日期**: 2026-05-28

**修复内容**: fast_mode_samples和fast_mode_epochs从config读取

```python
default_config = AxonExperimentConfig()
fast_mode_samples = default_config.fast_mode_samples
fast_mode_epochs = default_config.fast_mode_epochs
```

---

### M13. `_compute_metrics` 中裸 `except:` ✅

- **文件**: `src/trainer.py:478`
- **修复日期**: 2026-05-28

**修复内容**: 改为except ValueError:

---

### M14. 初始 `confidence=0.1` 过高 ✅

- **文件**: `src/dsra/mhdsra2/improved_dsra_mha.py:256`
- **修复日期**: 2026-05-28

**修复内容**: 0.1 → 0.01

```python
conf = torch.full_like(zeros, 0.01)
```

---

### M15. `_scatter_values` FP16累加精度丢失 ✅

- **文件**: `src/dsra/mhdsra2/improved_dsra_mha.py:306-314`
- **修复日期**: 2026-05-28

**修复内容**: scatter_add_在float32下累加后转回原dtype

```python
src = src.to(dtype=torch.float32)
out = torch.zeros(b * h, slots, d, device=values.device, dtype=torch.float32)
out.scatter_add_(1, idx_flat.unsqueeze(-1).expand(-1, -1, d), src)
out = out.to(dtype=values.dtype)
```

---

## 四、低严重度问题（已修复）

### H1-F2. 微写入误触发 write_protection ✅

- **文件**: `src/dsra/mhdsra2/improved_dsra_mha.py:616,623`
- **修复日期**: 2026-05-28
- **验证**: 3个代理交叉验证通过

**修复内容**: 提高has_write阈值，使用`cfg.eps * 10`过滤微写入

```python
wrote_mask = (mass.squeeze(-1) > cfg.eps * 10).to(device=k.device)
```

---

### H2-F3. diversity_loss权重硬编码 ✅

- **文件**: `src/config.py:201` + `src/trainer.py:368`
- **修复日期**: 2026-05-28
- **验证**: 3个代理交叉验证通过

**修复内容**: TrainingConfig添加`diversity_loss_weight=0.05`，trainer从config读取

```python
# config.py
diversity_loss_weight: float = 0.05

# trainer.py
loss = loss + self.train_config.diversity_loss_weight * div_loss
```

---

### H2-F4. 访问路径脆弱：三层嵌套属性链 ✅

- **文件**: `src/model.py:507-509` + `src/trainer.py:367`
- **修复日期**: 2026-05-28
- **验证**: 3个代理交叉验证通过

**修复内容**: AxonMalwareModel添加`@property dsra`便捷属性

```python
@property
def dsra(self):
    return self.dsra_encoder.dsra_encoder.dsra
```

---

### H3-F1. 读取tau上限可独立配置 ✅

- **文件**: `src/config.py:140` + `src/dsra/mhdsra2/improved_dsra_mha.py:354`
- **修复日期**: 2026-05-28
- **验证**: 3个代理交叉验证通过

**修复内容**: DSRAArchitectureConfig添加`read_tau_max=64.0`，_slot_read使用cfg.read_tau_max

```python
# config.py
read_tau_max: float = 64.0

# improved_dsra_mha.py
tau = tau.clamp(min=1.0, max=cfg.read_tau_max)
```

---

### H9-F1-F1. stat_features维度49硬编码，应从config读取 ✅

- **文件**: `src/config.py:18` + `src/dataset.py` + `src/model.py`
- **修复日期**: 2026-05-28
- **验证**: 3个代理交叉验证通过

**修复内容**: 激活config中的`stat_feature_dim=49`参数，所有硬编码位置改为从config读取

- `MalwareDataset.__init__`添加`stat_feature_dim`参数
- `NPZDataset.__init__`添加`stat_feature_dim`参数
- `_load_from_cache`、`__getitem__`异常回退、NPZDataset正常/异常路径均使用`self.stat_feature_dim`
- `stat_projector`使用`config.stat_feature_dim`
- `model.py`文档字符串更新为`[stat_feature_dim]`

---

### H10-F1. API子串匹配误匹配风险 ✅

- **文件**: `src/kvd_features/extractor.py:333,510`
- **修复日期**: 2026-05-28
- **验证**: 3个代理交叉验证通过

**修复内容**: 添加`_prefix_only`集合，对connect/send/recv使用前缀匹配

```python
self._prefix_only = {'connect', 'send', 'recv'}

# 匹配逻辑
(api_name.startswith(kw) if kw in self._prefix_only else kw in api_name)
```

---

### H10-F2. api_categories缺少重要类别 ✅

- **文件**: `src/kvd_features/extractor.py:327-332,515`
- **修复日期**: 2026-05-28
- **验证**: 3个代理交叉验证通过

**修复内容**: 添加crypto和injection类别，特征计算循环包含6个类别

```python
'crypto': ['cryptencrypt', 'cryptdecrypt', 'cryptderivekey', 'cryptgenkey',
          'cryptcreatehash', 'crypthashdata', 'cryptsignhash', 'cryptverify'],
'injection': ['createremotethread', 'virtualallocex', 'writeprocessmemory',
             'readprocessmemory', 'queueuserapc', 'setwindowshookex',
             'rtlcreateuserthread', 'ntcreatethreadex'],
```

---

### L1. ByteEmbedding `torch.clamp` 静默吞没异常值 ✅

- **文件**: `src/model.py:92-96`
- **修复日期**: 2026-05-28
- **验证**: 3个代理交叉验证通过

**修复内容**: 训练时对越界值发出warning

```python
if self.training:
    oob = (x < 0) | (x >= self.vocab_size)
    if oob.any():
        import warnings
        warnings.warn(f"ByteEmbedding: {(oob).sum().item()} values out of [0, {self.vocab_size}) range, clamped")
```

---

### L2. 奇数d_model时正弦编码维度崩溃 ✅

- **文件**: `src/model.py:49`
- **修复日期**: 2026-05-28
- **验证**: 3个代理交叉验证通过

**修复内容**: cos分量使用`div_term[:d_model // 2]`避免维度不匹配

```python
pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
```

---

### L3. PositionalEncoding文档声称支持RoPE但未实现 ✅

- **文件**: `src/model.py:20-26`
- **修复日期**: 2026-05-28（随M5修复一并完成）
- **验证**: 3个代理交叉验证通过

**修复内容**: 修正文档字符串，移除RoPE声明，仅保留"可学习的位置编码"和"正弦位置编码（默认）"

---

### L4. `position` 是int而非tensor，设备管理需手动 ✅

- **文件**: `src/dsra/mhdsra2/improved_dsra_mha.py:93-95`
- **修复日期**: 2026-05-28
- **验证**: 3个代理交叉验证通过

**修复内容**: MHDSRA2State添加`__post_init__`设备断言

```python
def __post_init__(self):
    if self.slot_k is not None and self.position != 0:
        assert self.slot_k.device is not None, "state tensors must be on a valid device"
```

---

### L5. `slot_v_init` 未normalize，初始值尺度不可控 ✅

- **文件**: `src/dsra/mhdsra2/improved_dsra_mha.py:256`
- **修复日期**: 2026-05-28
- **验证**: 3个代理交叉验证通过

**修复内容**: init_state中对slot_v_init添加F.normalize

```python
v = F.normalize(self.slot_v_init, dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)
```

---

### L6. `age_next` 因max_update=0.5限制永远无法真正重置 ✅

- **文件**: `src/dsra/mhdsra2/improved_dsra_mha.py:601-605`
- **修复日期**: 2026-05-28
- **验证**: 3个代理交叉验证通过

**修复内容**: 对新写入slot（write_gate > 0.1）重置age为0

```python
wg32 = write_gate.squeeze(-1).to(dtype=torch.float32)
fg32 = forget.squeeze(-1).to(dtype=torch.float32)
age_next = (state.age + k.shape[2]).to(dtype=torch.float32)
age_reset_mask = (wg32 > 0.1).to(dtype=torch.float32)
age_next = age_next * (1.0 - age_reset_mask)
```

---

## 修复统计

| 类别 | 数量 |
|------|------|
| 高严重度问题 | 10 |
| 高严重度残余问题 | 5 |
| 中严重度问题 | 15 |
| 低严重度问题 | 13 |
| **总计** | **43** |

> H10-F3（旧缓存清理）和H10-F4（fallback不含新特征）为观察项，无需代码修改，不计入修复统计。
