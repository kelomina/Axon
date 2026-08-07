# Loop151 实战决策链消融 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建统一 raw-file 输入的 Loop28/Loop151 A/B/C/D/E 消融运行器，准确定位实战 TPR 回退最早出现在哪个决策阶段。

**Architecture:** 在现有 `Loop151Runtime` 中公开一次特征提取后产生全部冻结阶段结果的接口，并使用 Loop28 冻结 metadata 验证并加载其 Stage-2 模型。批量运行器逐文件只提取一次特征，输出五臂预测、各阶段分数、入口语义、耗时和失败状态；不修改任何冻结模型、阈值、原生 DLL 或生产配置。

**Tech Stack:** Python 3、NumPy、PyTorch、现有 sklearn pickle scorer、pytest、JSONL/JSON。

---

### Task 1: 冻结消融阶段合同

**Files:**
- Modify: `src/loop151_runtime/raw_runtime.py`
- Test: `tests/test_loop151_field_ablation.py`

- [ ] **Step 1: 写失败测试**

验证阶段映射严格为：A=Loop28 Stage-2、B=Loop151 primary、C=Loop130 content rules、D=Loop136 selector、E=Loop151 signer guard；验证关闭 Authenticode 时 E 与 D 一致。

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_loop151_field_ablation.py -q`

Expected: FAIL，因为阶段结果接口尚不存在。

- [ ] **Step 3: 实现最小阶段接口**

增加冻结 dataclass，包含五臂预测、各概率、selector 分数、R5 flip、signer downgrade 和单阶段耗时。将 `predict_path` 的决策计算提取为可复用内部方法，保持现有返回值完全兼容。

- [ ] **Step 4: 运行测试并确认通过**

Run: `python -m pytest tests/test_loop151_field_ablation.py tests/test_loop151_trusted_signer_runtime.py -q`

Expected: PASS。

### Task 2: 加载并校验 Loop28 冻结 Stage-2

**Files:**
- Modify: `src/loop151_runtime/raw_runtime.py`
- Test: `tests/test_loop151_field_ablation.py`

- [ ] **Step 1: 写失败测试**

验证 Loop28 metadata schema、checkpoint SHA、模型 SHA、阈值 `0.5`、特征维数 `1520` 和 feature config；篡改任一值必须 fail closed。

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_loop151_field_ablation.py -q`

Expected: FAIL，因为 Loop28 冻结资产尚未接入消融路径。

- [ ] **Step 3: 实现 Loop28 只读评分**

复用同一次 raw feature 与 base probability 构造 Loop28 1520 维向量，使用 metadata 中冻结 SHA 校验模型后评分，不允许请求覆盖阈值。

- [ ] **Step 4: 运行测试并确认通过**

Run: `python -m pytest tests/test_loop151_field_ablation.py -q`

Expected: PASS。

### Task 3: 实现批量实战消融运行器

**Files:**
- Create: `scripts/run_loop151_field_ablation.py`
- Test: `tests/test_loop151_field_ablation.py`

- [ ] **Step 1: 写失败测试**

覆盖输入 manifest 解析、SHA 校验、重复样本拒绝、每样本 JSONL、汇总 JSON、异常计数、统一 `path` 入口和不可覆盖阈值。

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_loop151_field_ablation.py -q`

Expected: FAIL，因为 CLI 尚不存在。

- [ ] **Step 3: 实现 CLI**

CLI 接收包含 `sample_id,path,sha256,label` 的 CSV，仅支持 path 入口；逐文件输出 A/B/C/D/E、阶段分数、耗时和错误。汇总报告每臂 TP/FN/FP/TN、TPR、FPR、accuracy、平均/P50/P95 延迟，并输出相邻阶段 repairs/breaks。

- [ ] **Step 4: 运行测试并确认通过**

Run: `python -m pytest tests/test_loop151_field_ablation.py -q`

Expected: PASS。

### Task 4: 回归与产物一致性验证

**Files:**
- Test: `tests/test_loop151_field_ablation.py`
- Test: `tests/test_replay_loop151_raw.py`
- Test: `tests/test_loop151_trusted_signer_runtime.py`

- [ ] **Step 1: 运行目标测试组**

Run: `python -m pytest tests/test_loop151_field_ablation.py tests/test_replay_loop151_raw.py tests/test_loop151_trusted_signer_runtime.py -q`

Expected: PASS。

- [ ] **Step 2: 运行语法与 CLI 检查**

Run: `python -m py_compile src/loop151_runtime/raw_runtime.py scripts/run_loop151_field_ablation.py`

Expected: exit code 0。

- [ ] **Step 3: 核验范围**

Run: `git diff -- src/loop151_runtime/raw_runtime.py scripts/run_loop151_field_ablation.py tests/test_loop151_field_ablation.py`

Expected: 仅包含消融接口、运行器和测试；冻结资产、阈值、DLL、生产配置均未修改。

### 验收门槛

- 每个样本五臂使用同一次文件快照、特征提取与基础概率。
- A/B/C/D/E 阶段语义固定，阈值不可由 CLI 覆盖。
- SHA 不符、重复 identity、缺失标签、非有限分数和扫描失败均显式记录或 fail closed，不得静默丢样本。
- 完整 E 臂与现有 `Loop151Runtime.predict_path` 决策逐样本一致。
- 当前旧离线集合不用于候选排序；真实 3 万样本只有在能提供逐样本 manifest 时才执行正式消融。
