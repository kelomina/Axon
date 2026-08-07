# Loop179 Phase 0 完成报告 + Phase A 提案 + 极端 Lineage 候选

**日期**: 2026-07-20
**授权状态**: A1_scoped_change（Phase 0 代码与测试完成）
**请求**: A2_heavy_compute 授权以启动 Phase A 资源门验证
**前置决策**: Loop175 postdecision `close_loop175_allow_loop179_resource_proposal_only`

---

## 1. Phase 0 完成报告

### 1.1 已实现文件（7/7 白名单）

| 文件 | SHA256 (前16) | 职责 |
|------|---------------|------|
| `src/loop179/__init__.py` | `25148a6ee9b1f53d` | 导出接口 |
| `src/loop179/contracts.py` | `21d66528c74ca2c4` | 冻结常量、ABI、资源门、四臂门 |
| `src/loop179/hgconv.py` | `5321ead559a718da` | HGConv 核心数学（bind/conv/unbind/GLU） |
| `src/loop179/model.py` | `b848b25f80525ca9` | HGConvRegionNet 完整模型 |
| `src/loop179/source_closure.py` | `e49ca164ce680845` | 源码闭包验证器（白名单/SHA/禁导入） |
| `src/loop179/data_adapter.py` | `3d8a09c5038db7f2` | 数据适配器骨架（不访问真实数据） |
| `src/loop179/resource_cell.py` | `f7b925eada70942f` | 资源门框架（GPU/RSS/wall/完整性） |

### 1.2 测试结果

```
54 passed in 2.60s
- tests/test_loop179_contracts.py: 10 passed
- tests/test_loop179_source_closure.py: 6 passed
- tests/test_loop179_data_adapter.py: 14 passed
- tests/test_loop179_resource_cell.py: 13 passed
- tests/test_loop179_hgconv.py: 6 passed
- tests/test_loop179_model.py: 5 passed
```

### 1.3 Phase 0 验收门

| 门 | 状态 | 证据 |
|----|------|------|
| 契约自检 | ✅ | `assert_contract_invariants()` 通过 |
| 预算自检 | ✅ | `assert_budget_invariants()` 通过 |
| 源码闭包 | ✅ | 7 文件全在白名单，无禁止导入，无 SHA drift |
| 数学正确性 | ✅ | circular convolution = explicit modulo sum (float64 rtol 1e-12) |
| mask 隔离 | ✅ | trailing/interior/empty/all-masked 均通过 |
| 梯度有限性 | ✅ | float64 gradcheck + float32 backward finite |
| 确定性 | ✅ | eval 模式固定 seed bitwise 一致 |
| ABI 形状 | ✅ | [B,16,8192] → region_features [B,192] / fusion_logits [B,2] |
| 数据隔离 | ✅ | `_load_region_cache_rows` 抛 NotImplementedError |

### 1.4 Receipt 位置

`reports/roadmap_9997/loop175/loop179_phase0_receipt.json`

---

## 2. Loop179 Phase A 提案（资源门验证）

### 2.1 目标与非目标

**目标**: 验证 HGConv-Region 模型在 Loop175 region cache 上的资源可行性（GPU/RSS/wall/完整性），**不产生任何质量或 promotion 声明**。

**非目标**:
- 不产生 OOF F1、error count、override precision 等质量指标
- 不晋级 Phase B
- 不触碰 Val/Test-10k/full-test
- 不训练 fold0 模型
- 不选阈值

### 2.2 数据口径

| 项 | 值 | 依据 |
|----|-----|------|
| fit folds | 2, 3, 4 | Loop175 seed41 OOF 合同 |
| selection fold | 1 | Loop175 seed41 OOF 合同 |
| forbidden fold | 0 | Phase A 不触碰 fold0 |
| fit rows | 12,000 | 3 folds × 4,000 |
| selection rows | 4,000 | 1 fold × 4,000 |
| total Train rows | 16,000 | Loop175 region cache 全量 |

### 2.3 训练超参（已冻结）

| 参数 | 值 |
|------|-----|
| max_epochs | 12 |
| microbatch | 2 |
| accumulation | 16 |
| effective_batch | 32 |
| learning_rate | 3e-4 |
| weight_decay | 1e-2 |
| warmup_steps | 1 |
| grad_clip | 1.0 |
| ema_decay | 0.999 |
| autocast | bfloat16 |
| master/FFT dtype | float32 |

### 2.4 资源门阈值

| 资源 | Loop179 Phase A 上限 | Loop175 seed41 上限 | 安全边际 |
|------|----------------------|---------------------|----------|
| GPU allocated | 6.5 GiB | 6.98 GiB | 7% |
| RSS | 11 GiB | 11.81 GiB | 7% |
| wall | 21,600 s (6h) | 21,600 s | 0% (同上限) |

### 2.5 完整性门

| 门 | 要求 |
|----|------|
| silent_drop_rows | = 0 |
| all_rows_accounted | True (16000/16000) |
| OOM | False |
| timeout | False |
| nonfinite | False |
| bitwise_deterministic_eval | True (固定输入两次 forward bitwise 一致) |

### 2.6 Phase A 成功判据

**资源门通过**（不等于质量通过）:
1. GPU/RSS/wall 全部在预算内
2. 完整性门全部通过
3. 12 epochs 完成
4. 16,000 行全部 accounted
5. 固定输入 eval logits bitwise 一致

**Phase A 不产生**:
- F1 / error count / override precision
- Phase B 晋级决策
- 任何 heldout 访问

### 2.7 需要的 A2 授权

- [ ] fresh resource guard JSON
- [ ] machine authorization JSON
- [ ] 用户显式授权启动 Phase A 训练
- [ ] Loop175 region cache 只读访问授权（folds 1/2/3/4）

---

## 3. Phase B 提案（决定性对照实验，Phase A 通过后启动）

### 3.1 四臂设计

| 臂 | 描述 |
|----|------|
| A | frozen 571-value B0 HGB control（Loop175 baseline） |
| H | HGConv-Region only（无 B0 fusion） |
| J | B0 + HGConv-Region early fusion |
| K | J + partition-local zero-fixed-point whole-region ownership shuffle（反事实） |

### 3.2 J 臂晋级门（全部必须通过）

| 门 | 阈值 |
|----|------|
| J net fewer errors vs A | ≥ 30 |
| J repairs vs A | ≥ 50 |
| J override precision | ≥ 0.80 |
| J net positive folds | ≥ 4/5 |
| J bootstrap LCB vs A | > 0 (one-sided 95%) |
| FP relative worsening | ≤ 5% |
| FN relative worsening | ≤ 5% |

### 3.3 K 臂反事实门

| 门 | 阈值 |
|----|------|
| K more errors vs J | ≥ 30 |
| K bootstrap LCB vs J | > 0 (one-sided 95%) |

### 3.4 失败判据

任一 J 臂门失败 → 关闭 HGConv-Region 路线，不再追加变体。
K 臂门失败 → 收益不来自区域归属，J 即使通过也不晋级。

---

## 4. 极端 Lineage 候选（Loop180-184）

用户要求"激进推进，包括更改模型架构、极端模型权重等极端方式"。以下 5 个候选是为 A2 授权后的并行探索准备的新 lineage 提案。

### 4.1 Loop180: 极端难例加权 + HGConv Fusion

**假设**: Loop175 E 臂 8x 权重在 RegionNet 上失败，但在 HGConv fusion head 上可能不同，因为 HGConv 提供了不同的表示空间。

**实验**:
- 在 Loop179 HGConvRegionNet 的 fusion_head 上施加 16x/32x 难例权重
- 难例 = Loop151 FP ∪ FN 样本
- 对照: Loop179 J 臂（1x 权重）
- 三档: 16x, 32x, 64x

**门**:
- 相对 J 臂净减 ≥ 30 errors
- override precision ≥ 0.80
- FP/FN 相对恶化 ≤ 5%
- 若 64x 仍无收益，永久关闭极端权重路线

**风险**: Loop175 E 臂已证明 8x 无效，16x/32x/64x 可能只是线性放大失败。

### 4.2 Loop181: Focal Loss with Extreme Gamma

**假设**: 标准 cross-entropy 对难例关注不足。Focal loss with gamma=4/5/6 可能更有效地聚焦难例。

**实验**:
- 在 HGConvRegionNet 的 fusion_head 上用 focal loss 替代 CE
- 三档: gamma=4, gamma=5, gamma=6
- 对照: J 臂（CE）
- alpha=0.25（标准值）

**门**:
- 相对 J 臂净减 ≥ 30 errors
- override precision ≥ 0.80
- 若 gamma=6 仍无收益，关闭 focal loss 路线

**风险**: focal loss 在极端 gamma 下可能过拟合难例，导致 FP 上升。

### 4.3 Loop182: Contrastive Learning on Hard Pairs

**假设**: Loop151 的 FP/FN 样本可能在 region 表示空间中与 TN/TP 样本过于接近。Contrastive learning 可以拉开它们。

**实验**:
- 在 HGConv region features 上添加 contrastive head
- 正样本对: 同类（malware-malware, clean-clean）
- 负样本对: 异类（malware-clean）
- 难负样本: Loop151 FP 样本与对应 TP 样本
- 温度参数: 0.05/0.1/0.2
- 损失: contrastive + CE（权重 0.5/0.5）

**门**:
- 相对 J 臂净减 ≥ 30 errors
- override precision ≥ 0.80
- embedding space separation（FP/TP 在 t-SNE 上的可分性）

**风险**: contrastive learning 需要大量样本对，16000 行可能不足。

### 4.4 Loop183: Deep HGConv (6 blocks) + MoE Region Router

**假设**: 单层 HGConv 不足以捕获跨区域长程依赖。6 blocks + MoE router 可能更好。

**实验**:
- HGConv blocks: 1 → 6
- MoE router: 4 experts, top-2 routing
- expert: region-type-specific HGConv block
- 对照: J 臂（1 block, no MoE）
- 资源门: GPU 6.5 GiB, wall 6h（同 Loop179）

**门**:
- 相对 J 臂净减 ≥ 30 errors
- override precision ≥ 0.80
- MoE load balance（每 expert 处理 ≥ 15% tokens）

**风险**: 6 blocks + MoE 可能超 GPU 6.5 GiB 预算；若超限则降级到 4 blocks。

### 4.5 Loop184: Extreme Architecture - Sparse MoE + HGConv + Hard Example Mining + Curriculum

**假设**: 单一技术不足以突破 0.9997。组合 Sparse MoE + HGConv + 难例挖掘 + 课程学习。

**实验**:
- 阶段1 (epoch 1-4): 标准 CE，1x 权重
- 阶段2 (epoch 5-8): focal loss gamma=4，难例 8x 权重
- 阶段3 (epoch 9-12): focal loss gamma=6，难例 32x 权重
- MoE: 4 experts, top-2 routing, region-type-specific
- HGConv: 4 blocks

**门**:
- 相对 J 臂净减 ≥ 50 errors（更高门槛，因为复杂度更高）
- override precision ≥ 0.85
- 若失败，关闭组合极端路线

**风险**: 复杂度最高，调试最难；若 MoE load imbalance，收益可能为负。

---

## 5. 推荐路线

### 5.1 立即行动（A1 范围已完成）

✅ Loop179 Phase 0 完整化（7 文件, 54 测试）

### 5.2 需要 A2 授权的下一步

**优先级 1**: Loop179 Phase A 资源门验证
- 若通过 → 进入 Phase B 决定性对照
- 若失败 → HGConv-Region 路线关闭

**优先级 2**（Phase A 通过后并行）:
- Loop180 极端权重（最激进，成本最低）
- Loop181 Focal loss（中等激进，成本中等）
- Loop183 Deep HGConv + MoE（最激进架构，成本最高）

**优先级 3**（Phase B 通过后）:
- Loop182 Contrastive learning（需要 Phase B 的 J 臂作为基础）
- Loop184 组合极端（需要 Loop180/181/183 的证据）

### 5.3 失败路线

若 Loop179 Phase A 资源门失败:
- 降低 max_epochs 到 8
- 降低 microbatch 到 1
- 若仍失败，关闭 HGConv-Region 路线

若 Loop179 Phase B J 臂失败:
- 关闭 HGConv-Region 路线
- 转向 Loop183 Deep HGConv + MoE（新架构）
- 或 Loop182 Contrastive learning（新损失）

---

## 6. 授权请求总结

**本轮已完成（A1）**:
- Loop179 Phase 0 代码与测试完整化
- 54/54 测试通过
- 源码闭包、契约自检、预算自检全部通过
- Phase 0 receipt 生成

**请求 A2 授权**:
- Loop179 Phase A 资源门验证
- 资源边界: GPU ≤ 6.5 GiB, RSS ≤ 11 GiB, wall ≤ 6h
- 数据边界: Train folds 2/3/4 fit + fold 1 selection, fold 0 forbidden
- 不触碰: Val/Test-10k/full-test
- 不产生: 质量声明、Phase B 晋级

**后续 A2 请求（Phase A 通过后）**:
- Loop179 Phase B 四臂五折 OOF
- Loop180-184 极端 lineage（各自独立 proposal）
