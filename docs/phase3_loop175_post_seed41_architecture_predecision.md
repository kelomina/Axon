# Loop175 seed41 后续架构预决策

更新时间：2026-07-17

## 1. 目的与边界

本文件在 Loop175 seed41 任何 pilot、OOF 或 aggregate receipt 出现前冻结后续分支，避免依据
结果事后选择阈值、权重、架构或 seed。它不修改 Loop175 Phase-B source closure，不授权新训练，
也不打开 Val、Test-10k 或 legacy full-test。

当前唯一 research champion 仍是 Loop151。`F1 >= 0.9997` 的完成证据必须来自冻结 pipeline 的
完整分母评估；Train-only OOF、resource smoke 或单个 seed 都不能替代。

## 2. 当前冻结事实

- Loop175 使用 20,000 行 Train、16,949 个 content components、5 个 outer folds；
- Region cache 覆盖 `19,984/20,000 = 0.9992`，16 行 read failure 留在完整分母，silent drop 为 0；
- seed41 先执行 B/C/D/E epoch pilot，再执行 A-E 五折；
- hard decision 固定为 `p > 0.5`；
- 首次 CUDA OOM 只允许同 arm-fold 以 microbatch 1、accumulation 32 从初始状态重启；
- 第二次 OOM、nonfinite、timeout、资源越界、完整性失败或非 OOM worker failure 关闭 recipe；
- E 只检验极端残差权重，不能救援失败的 C。

## 3. seed41 receipt 分类

### 3.1 receipt 尚未产生

继续等待。没有 receipt 既不是通过也不是失败。只有外部中断且 closure、argv、输出目录和
append-only ledger 全部一致时，才允许走 controller 已实现的恢复路径；不得删除或重建已有产物。

### 3.2 工程或资源失败

若失败来自 OOM、nonfinite、timeout、GPU/RSS/disk、cache/commitment 漂移、OOF 缺失或重复：

1. 决策固定为关闭 Loop175 seed41；
2. 禁止 seeds42/43、Val、Test-10k 和 full-test；
3. 先生成不可变 failure attestation，区分工程失败与模型质量失败；
4. 下一候选固定为 Loop179 HGConv-Region resource cell，不把失败解释成 RegionNet 无质量信号。

### 3.3 科学门失败但区域归属有因果信号

满足以下任一模式时，允许规划 Loop179，但仍关闭 Loop175：

- C 相对 D 的净优势至少 30 且 component LCB `>0`，但 C 相对 A 的减错、repairs、precision、
  fold stability 或 FP/FN guard 未过；
- B 单独满足与 C 相同的 A-relative 质量门，而 C 未过，说明 early fusion 可能压制区域专家。

这类结果只说明区域内容值得换编码器或融合方式，不允许调整 Loop175 的 epoch、阈值、权重、
dropout、seed 或现有 checkpoint。

### 3.4 科学门失败且区域归属被否证

若 C 相对 D 的净优势小于 30 或 component LCB `<=0`，停止 region/section ownership 同类路线。
不运行 HGConv-Region，不重启已关闭的 Loop166 BPE/MLM，也不继续 CFG/capa 的已关闭 host recipe。
下一候选转为全新、隔离的行为/能力/时间可信证据专家；其前提是 disposable sandbox、全分母
coverage/fallback 和新的 source-closed authorization。

### 3.5 seed41 通过

先检查 seed41 是否同时满足 proposal 已冻结的三 seed 单 seed门：C 相对 A errors 减少至少
20%。若不满足，则三 seed 全过在数学上已经不可能，立即关闭，不浪费 seeds42/43。

若满足，才为 seeds42/43 新建 authorization/source closure。两者复用 seed41 冻结 epoch，三个
seed 必须逐个满足 20% 减错、4/5 folds 正向、component LCB `>0`、FP/FN 恶化不超过 5%、
C 相对 D 优势至少 30。任一失败都关闭，不能用均值、最佳 seed 或 2/3 多数票救援。

## 4. Loop179 HGConv-Region 新架构计划

Loop179 是全新 lineage，不修改 `src/loop175/`，也不宣称复现论文的全文件 HGConv。

### Phase 0：语义与实现冻结

1. 以 HGConv 官方 JAX/Flax 实现和论文为 reference，写独立 PyTorch bind -> FFT circular
   convolution -> unbind 模块；
2. 输入输出保持 `[batch * 16, 512 patches, 192]`，patch size 16、region bytes 8192；
3. 保留 Loop175 byte embedding、region type/offset/length embeddings、region Transformer 和
   571 维 B0 projector；只替换六层 dilated-GLU patch backbone；
4. 冻结 1 个 HGConv block、model dim 192、相同 dropout、优化器、阈值和 component folds；
5. 通过 mask、padding、FFT shape、有限值、确定性、gradient 和 source-closure tests。

### Phase A：单 cell 资源门

只使用 Train outer-fit folds 2/3/4，fold 1 作 inner selection，outer fold 0 完全不读取。最多
12 epochs，复用 Loop175 BF16、AdamW、EMA 和 effective batch 32 合同。

资源门：GPU allocated `<=6.5 GiB`、RSS `<=11 GiB`、wall `<=6h`、无 nonfinite、无 silent
drop、同输入重复运行 commitment 一致。resource cell 不产生质量或 promotion 声明。

### Phase B：seed41 五折决定性对照

固定四臂：

- A：Loop175 同定义的 571-value B0 HGB；
- H：HGConv-Region-only；
- J：B0 + HGConv-Region early fusion；
- K：J 同容量、partition-local 零固定点 whole-region ownership shuffle。

所有行恰好一次 OOF，threshold 固定 `0.5`。J 必须同时满足：相对 A 净减至少 30 errors、
repairs 至少 50、override precision 至少 0.80、4/5 folds 净正向、component LCB `>0`、
FP/FN 任一侧相对恶化不超过 5%、K errors - J errors 至少 30 且 LCB `>0`。任一失败关闭
HGConv-Region，不增加 block、sequence、epoch 或 seed 续命。

### Phase C：后续上限

只有 Phase B 全门通过才运行 seeds42/43。三 seed 全过后，才允许新建 full-file HGConv
16K -> 32K prefix resource ladder；每一级都是新 cache 和新 source closure，region cache 不能
冒充连续文件前缀。131K 不在 8 GiB GPU 上直接起跑。

## 5. Loop176 通过分支

若 Loop175 三 seed 全过，先冻结唯一 final candidate 生成规则、全 Train final checkpoint 训练规则、
三 seed 集成规则和 Val FP/FN business guard，缺一不得打开 Val。Val-A 与 Val-B 必须分别相对
Loop151 净减至少 50 errors，才允许建设 Loop176 same-partition nested-OOF router。

Loop176 只消费专家 OOF logit、entropy、margin、missingness 和冻结 trust evidence；path、filename、
extension、SHA、row、fold、component、source/family/time identity 不得成为模型输入。legacy
Test-10k/full-test 继续只作整个 pipeline 冻结后的一次 confirmation。

## 6. 已关闭路线不得伪装恢复

- Loop166 `BPE-1024 + MLM` 已因 nonfinite 永久关闭；修改 scaler、精度、optimizer、LR 或 schedule
  只能属于新的 lineage；
- Loop170 CFG 当前 coverage `0.13671875`，不能进入 OOF；
- Loop171 host capa coverage attempt 已因 fast-fail/timeout containment 风险关闭；只有 disposable
  sandbox 和新 authorization 才能重新提出；
- full-test 错误、标签、review verdict、路径或 hash 不得成为训练、路由、阈值或权重输入。

## 7. 当前决策

`pre_registered_wait_for_loop175_seed41_receipt_no_new_execution_authorized`
