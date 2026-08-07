# Loop175 Phase-B Section/Region-MoE Execution Plan

更新时间：2026-07-17

## 1. 任务与证据边界

Phase-B 只在冻结的 20,000 行 Train 上进行 content-component 隔离 OOF，验证新的
PE 语义区域归纳偏置是否能稳定优于结构基线。它不是 Val、Test-10k、legacy full-test、
promotion 或 `F1 >= 0.9997` 的完成证据。

Phase-A 已证明区域提取器在 256 行 Train-only 探针上达到 `256/256` supported、零静默
丢弃和可接受资源开销。Phase-B 在 source closure、全量 region cache gate 和 fresh resource
guard 完成前不得启动模型训练。

## 2. 冻结输入

- Train authority：`reports/roadmap_9997/loop164/local_train_diagnostic_folds.jsonl`，
  SHA-256 `00a31a1bd86d7b887447f3e86e5e753ebcaaee45be74311199332e073a3880a5`；
- 行数 `20,000`，标签 `10,000/10,000`，五折各 `4,000`；
- `16,949` 个 content components，任一 component 只能属于一个 outer fold；
- B0：只使用 Loop167 v12 sealed cache 中的 `b0_values float32[20000,571]`，cache
  SHA-256 `7826abfc76e04f93ea4b6ee4bc31cf25e651dab4355fa989f29e5488c7fda18b`；
- B0 的 6 个 missing indicators 不进入 Loop175 A/C/D/E。历史 Loop167 `232 errors`
  来自 577 维控制，不得冒充 Loop175 A 的预期结果；
- path、filename、extension、SHA、row index、fold 和 component ID 只用于读取、对齐、
  分区与审计，进入模型前必须从 fit payload 移除。

## 3. 冻结 region cache

全量 Train 每个源只读取一次，先流式验证 size 与 `source_sha256`，再运行已经通过
Phase-A 的 region extractor。cache 使用一个 `ZIP_STORED`、无身份字段的 ragged archive：

- `token_values.npy`：所有非 padding bytes 连续存储为 `uint8`；
- 其余 `.npy` members：每个 canonical ordinal 的 row/region/token offsets、file size、
  region type、region start、offset bucket 与 length bucket；
- `metadata.json`：只保存 schema、shape、dtype、公式和 numeric payload commitment；
- cache 内不保存 path、filename、SHA、label、fold 或 component；
- metadata 绑定 proposal、Phase-A receipt、fold manifest、B0 cache、byte blob 和 array
  payload SHA；
- unsupported/read-failure/parse-failure/oversize 行保留 16 个 missing regions，不得删除；
- production fit payload 强制每行恰好 16 个 region slots。

桶公式固定为：

```text
offset_bucket = floor(63 * start / max(file_size - 1, 1))
length_bucket = ceil(63 * length / 8192)
```

两者范围必须为 `0..63`；模型入口对 tokens、lengths、types、buckets 和 B0 非有限值或
越界全部 fail closed，不允许 `clamp` 或 `nan_to_num` 静默修复。

全量 cache gate：coverage `>=0.995`、class coverage gap `<=0.02`、silent drop `0`、
20,000 行全部有明确状态、所有模型输入 finite、cache `<30 GiB`、RSS `<=11 GiB`。

## 4. 五臂与外层 OOF

每个 arm 在同一五折上产生恰好 20,000 个 OOF scores，每行恰好一次 outer holdout：

| Arm | 冻结定义 |
|---|---|
| A | 571 维 B0 values-only HGB，每 fold 只 fit 其余 16,000 行 |
| B | 独立训练的 RegionNet-only |
| C | 独立训练的 B0 + RegionNet early fusion，主候选 |
| D | 与 C 同容量，但在 fit/holdout 分区内分别置乱整条 region record ownership |
| E | 与 C 同架构，使用 outer-fit 内 B0 inner-OOF 生成的固定残差权重 |

A 的 HGB 参数固定：`loss=log_loss`、`learning_rate=0.06`、`max_iter=260`、
`max_leaf_nodes=31`、`min_samples_leaf=20`、`l2_regularization=0`、
`max_bins=255`、`early_stopping=false`、`random_state=41`。所有 arms 的 hard decision
固定为 `probability > 0.5`，恰好 `0.5` 判 benign，不允许 threshold search。

D 使用由 protocol SHA、seed、outer fold 和 role 派生的非零循环位移；fit 与 holdout
分别形成 bijection，固定点必须为零。receiver 的 B0、label 与 ordinal 不变，bytes、type、
offset bucket、length bucket 和 length 必须作为一个整体一起移动。

## 5. E 臂极端权重

本轮没有合法的 Loop151 Train OOF score，因此 E 的 residual source 明确冻结为
`B0-only inner OOF`，不得读取 Loop151 full-fit、Val、Test 或 full-test scores。

对每个 outer fold，在其 16,000 行 outer-fit 内使用剩余四个既有 component folds做
四折 inner OOF。每行恰好一次 inner holdout：

```text
inner B0 prediction error              -> raw weight 8
inner B0 correct and 0.35 <= p <= 0.65 -> raw weight 3
otherwise                              -> raw weight 1
```

错误规则优先于 near-boundary。每个 class 的 raw weights 分别除以该 class 均值，因此
每类均值固定为 1；最后 cap 到 8。权重不得搜索，E 相对 C 失败只关闭极端权重结论，
不否定 C，也不得改成 16x 续命。

## 6. Epoch 选择与训练

为同时满足 outer holdout 隔离和 seed41 总 wall `<=30h`，epoch 只选择一次：

1. seed41、outer fold 0 仅作 epoch pilot；outer fold 0 完全不读取；
2. 在 outer-fit folds `1/2/3/4` 中固定 fold 1 为 inner selection，folds `2/3/4` 为
   pilot fit；
3. B/C/D/E 分别独立训练最多 12 epochs，以 inner-selection unweighted CE 最小选择 epoch，
   tie 取最早 epoch；pilot checkpoint 随即丢弃；
4. 所有五个 outer folds 从相同初始化规则重新训练完整 16,000 行，训练到该 arm 已冻结
   epoch，使用 final EMA 预测 4,000 行 outer holdout；
5. seed42/43 若获准运行，沿用 seed41 的冻结 epochs，不重新选择。

RegionNet 训练保持 proposal：BF16 autocast，FP32 optimizer/norm/loss，AdamW
`3e-4/1e-2`，microbatch 2、accumulation 16、effective batch 32，warmup 1 epoch + cosine，
gradient clip 1.0，EMA 0.999。B/C/D/E 必须是独立模型，不能把 C 的共享 backbone 输出
冒充 B。

第一次 CUDA OOM 只允许完整重启当前 arm-fold 为 microbatch 1、accumulation 32；第二次
OOM 或 nonfinite、timeout、GPU `>6.5 GiB`、RSS `>11 GiB`、disk `>30 GiB` 立即关闭
当前 recipe，不得缩 region、模型或数据。

## 7. seed41 评价门

先验证 A-E 各自 20,000 finite OOF scores、每折 4,000、缺失/重复为零，再以 component
为重采样单位做固定 200,000 次 PCG64 bootstrap，报告单侧 95% lower bound。

C 必须同时满足：

- 相对 A net error reduction `>=30`；
- repairs `>=50`；
- override precision `>=0.80`；
- 至少 `4/5` folds 净正向；
- A-C component-bootstrap LCB `>0`；
- FP 和 FN 任一侧相对 A 恶化 `<=5%`；
- D errors - C errors `>=30`，且 D-C component-bootstrap LCB `>0`；
- 全量 coverage、完整性与资源门全部通过。

任一门失败，decision 固定为 `closed_seed41_gate`，禁止 seeds42/43、Val、Test-10k 和
legacy full-test。全部通过只允许 `seed41_pass_allow_seed42_43`，仍不允许直接进入 Val。

## 8. 执行顺序

1. 实现并测试 Phase-B data/cache/model/training/evaluation/controller；
2. 生成 source closure、fresh resource guard、run authorization 和一次性 lease；
3. 构建并验证全量 Train ragged region cache；
4. 执行 seed41 epoch pilot；
5. 依次执行 A-E 五折，arm-fold 独立子进程、append-only ledger、可断点续跑；
6. 聚合 seed41 receipt 并执行杀停门；
7. 只有 gate 通过才新建 seeds42/43 execution authorization。
