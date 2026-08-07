# Phase 3 Loop175 Section/Region-MoE Proposal

更新时间：2026-07-17

## 真实目标

Loop175 从冻结的 Loop151 research champion 继续推进，目标保持为完整测试口径
`F1 >= 0.9997`，不以 smoke、局部 parity、样本过滤或 abstain 删除替代。

当前 Loop151 legacy development full-test 为：

| F1 | Errors | FP | FN | Rows |
|---:|---:|---:|---:|---:|
| `0.9908541911` | `1466` | `879` | `587` | `160000` |

平衡 `80000/80000` 标签下，目标的精确整数条件是
`10003 * FN + 9997 * FP <= 480000`。总错误 `<=47` 必过；恰好 `48`
时仅 `FN<=24` 可过。Loop151 至少还要净消除 `1418` 个错误，即当前残差的
`96.7258%`。因此 Loop175 不继续扫描 Loop151 阈值、signer terms 或同源 selector，
而是验证新的区域级字节归纳偏置。

Legacy Val、Test-10k 和 full-test 已被多次观察，均只作 development evidence。
Loop175 的最终成功仍要求新的 temporal/family/source/near-duplicate 隔离窗口和
component-bootstrap 下界；legacy full-test 不参与 Loop175 的模型、权重或阈值选择。

## 已关闭路线

- Loop164 whole-file GCG standalone OOF 为 `748` 错，盲切 override precision 仅
  `0.112`，不得换名复跑。
- Loop166 BPE/MLM 当前 recipe 发生 nonfinite，不能只改 scaler、LR 或 optimizer
  后宣称新路线。
- Loop167 EMBER-v3 novel delta 为 `245` 错，对照 B0 为 `232` 错，net `-13`，
  component LCB `-29`，不得补 threshold/features/seeds 续命。
- Loop168 focal/contrastive、Loop169 broad Authenticode calibrator、Loop170 CFG
  低覆盖、Loop171 CAPA/guest readiness 均未形成可晋级专家。
- Speakeasy timeout-as-benign 已在 confirmation subset 明显新增 FN；动态证据只能在
  后续尾部级联中以完整 trace、显式 missingness 和静态 fallback 重新验证。

## 核心假设

Loop151 主要读取固定前缀和聚合结构特征；Loop164 则把全文件压成单个 global-max
表示。两者都可能丢失“哪个 PE 语义区域包含什么字节模式”的条件关系。Loop175 的
假设是：显式分离 header、entrypoint、可执行节、高熵/资源区和 overlay，再在区域
token 间做轻量注意力，可以稳定修复高置信 FP/FN，并且其收益必须显著超过区域归属
置乱对照。

## 冻结区域协议

每个文件最多 `16` 个区域，每区最多 `8192` bytes，总 byte budget 不超过
`131072`。所有 offsets 来自文件内容和 PE 结构，不使用 path、filename、extension、
hash、sample index、split 或 row order 作为模型输入。

优先级固定如下：

1. DOS/PE header window：从 offset `0` 开始，最多 `8192` bytes；
2. entrypoint-centered window：entrypoint file offset 前后各最多 `4096` bytes；
3. executable sections：按 section table 顺序取最多 `4` 个 section，每个取 head/tail，
   共最多 `8` 个区域；
4. semantic tail：从未覆盖 section 中按 entropy、resource 属性和 raw size 的冻结排序，
   取最多 `3` 个 head window；
5. overlay head/tail：存在 overlay 时最多 `2` 个区域；
6. 剩余槽使用显式 missing token，不复制已有区域。

重叠候选按 `(region_type, start, length)` 去重。越界窗口裁剪但不得静默丢弃；
PE parse failure、empty、oversize、read failure 和 unsupported 必须成为显式 reason 并
留在分母。padding token 固定为 `256`，真实 bytes 为 `0..255`。

## 冻结模型架构

### RegionNet

- byte vocabulary `257`，embedding dim `64`；
- `Conv1d` patchify：kernel/stride `16`，输出 dim `192`；
- `6` 个 depthwise-separable GLU residual blocks，kernel `7`，dilation
  `1/2/4/8/16/32`；
- 每区使用 attentive pooling 与 max pooling 的 gated projection，输出 `192`；
- 加入 region type、relative offset bucket、length bucket embedding；
- `2` 层 region Transformer，dim `192`、`6` heads、FFN `768`；
- CLS 输出 `192`，RegionNet-only head 为 `192 -> 128 -> 2`。

### 结构臂与融合

- B0 使用冻结、去重的 `571` 维 productized structural allowlist；
- B0 MLP 为 `571 -> 256 -> 128`，GELU + LayerNorm + dropout `0.10`；
- early fusion 为 `[Region CLS 192; B0 128] -> 128 -> 2`；
- 总参数目标 `5-8M`，参数超过 `10M` 直接失败，不靠隐藏增宽追分。

## 五臂因果实验

所有臂共享同一个 content-component 五折 partition、相同 rows、相同 seeds 和固定
threshold `0.5`：

| Arm | 定义 | 目的 |
|---|---|---|
| A | B0 frozen HGB | 当前可执行静态对照 |
| B | RegionNet only | 区域字节证据的 standalone 上界 |
| C | B0 + RegionNet early fusion | 主候选 |
| D | B0 + label-free shuffled region ownership | 排除 missing/status/边际分布伪收益 |
| E | C + inner-OOF residual weights | 极端难例权重消融 |

E 臂权重只能由每个 outer-fit 内部的 inner-OOF 产生：Loop151/B0 inner-OOF error
固定 `8x`，near-boundary 固定 `3x`，其余 `1x`；按类别归一并 cap `8x`。只有 E8
相对 C 通过全部门，才允许另立 E16 proposal；本轮禁止权重 sweep。

## 训练协议

- 第一 seed `41`；通过杀停门后才运行 `42/43`；
- 五折 content-component Train-only OOF；每个 eligible row 每 seed 恰好一次 holdout；
- BF16 autocast，FP32 optimizer、normalization 和 loss；
- AdamW `lr=3e-4`、`weight_decay=1e-2`；
- microbatch `2`，gradient accumulation `16`，effective batch `32`；
- 最多 `12` epochs，warmup `1` epoch + cosine；
- gradient clip `1.0`，EMA `0.999`；
- ordinary CE 为 C 主损失；focal/contrastive 不进入本轮；
- outer holdout 不参与 early stopping、权重构造或 threshold selection；
- 每 fold 只保留一个 best inner-selection checkpoint。

## Phase 0：实现与合成验证

在任何真实样本读取前完成：

1. region extractor 的 synthetic PE tests；
2. window priority、dedup、tail coverage、overlay 和 parse-failure tests；
3. 模型 shape、padding invariance、finite forward/backward tests；
4. shuffled ownership 的 label-free determinism test；
5. resource guard 与 no-silent-drop accounting test。

Phase 0 不训练、不访问 raw/cache/checkpoint/Val/Test/full-test。

## Phase A：256 行 Train-only 覆盖探针

从已冻结 Train role 取 label-balanced `256` 行，采样 seed 固定 `175`。标签只用于平衡
采样和最终 aggregate coverage，不传给 extractor。输出只保留 aggregate counts、
reason counts、bytes read、region counts、wall、RSS/GPU 和 SHA commitments。

杀停条件：

- supported coverage `<0.995`；
- silent drop `>0`；
- 任一类 coverage 差异 `>0.02`；
- peak RSS `>11 GiB` 或 extractor cache estimate `>30 GiB`；
- p95 extraction wall per file `>2s`；
- raw read/region budget 超约；
- nonfinite 或不可复现输出。

## Phase B：seed41 五折

seed41 任一条件失败即关闭当前 recipe：

- C 相对 A net error reduction `<30`；
- repairs `<50`；
- override precision `<0.80`；
- 少于 `4/5` folds net positive；
- content-component bootstrap 单侧 95% LCB `<=0`；
- FP 或 FN 相对恶化 `>5%`；
- C 相对 D 的 net advantage `<30`；
- coverage `<0.995`、silent drop、OOM、timeout 或 nonfinite；
- GPU allocated `>6.5 GiB`、RSS `>11 GiB`、wall `>6h/fold`。

E 的权重收益单独归因；E 若不优于 C 不影响 C 的结论，也不得改权重续命。

## 三 seed 与 Val 晋级门

seed41 通过后才运行 `42/43`。三个 seed 必须全部满足：

- C 相对 A errors 减少至少 `20%`；
- 至少 `4/5` folds net positive；
- component LCB `>0`；
- FP/FN 任一侧相对恶化不超过 `5%`；
- C 相对 D 保持至少 `30` 的语义优势。

之后冻结一次进入 Val-A/Val-B。两者相对 Loop151 都必须净减至少 `50` 错，且不突破
冻结 FP/FN business guard，才值得支付 Loop176 decision-aligned OOF router 成本。
Legacy Test-10k/full-test 只允许在整个 pipeline 冻结后做一次 confirmation，不参与选择。

## 后续路线

- Loop176：同 partition nested OOF router，只消费专家 OOF logit、entropy、margin、
  missingness 和冻结 trust evidence；禁止 identity/group/time 字段作为特征。
- Loop177：只路由静态分歧、高风险 packed/parser-fail 尾部的隔离行为专家；timeout、
  invalid/no-trace 全部 fallback 且计入分母。
- Loop178：冻结 connected cascade，在全覆盖 legacy confirmation 与两个新的 sealed
  temporal windows 上验收。
- Loop174：双独立 reviewer + adjudication 并行推进；确认坏标签时生成新数据版本并全链
  重训，review verdict 永不成为 runtime feature。

## 资源合同

- RTX 4070 Laptop 8 GiB；单任务 GPU allocated `<=6.5 GiB`；
- system RAM `<=11 GiB`；workers `1`、prefetch `1`；
- 单 fold wall `<=6h`，seed41 总 wall `<=30h`；
- Loop175 新增磁盘 `<=30 GiB`，每 fold 只保留 best checkpoint；
- OOM 时只允许降低 microbatch 并保持 effective batch，不得缩区域、模型或数据后冒充
  同一实验；第二次 OOM 关闭当前 recipe。

## 当前授权边界

当前允许：写 proposal/manifest、实现 Phase 0、运行 synthetic tests，并在 source closure
完成后执行一次 Phase A 256-row Train-only coverage probe。当前不允许：Val、Test-10k、
legacy full-test、阈值搜索、权重搜索、silent filtering、自动改标签或 promotion。

当前 research champion 保持 Loop151，目标未达成。
