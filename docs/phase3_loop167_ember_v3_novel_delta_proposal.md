# Loop167 EMBER-v3 Novel-Delta Structural Control

## 目标

Loop167 不直接把 EMBER2024 的 `2568` 维全部叠到现有模型，而是先回答一个更严格的问题：在扣除 Axon 已有 PE/stat/content/string/certificate 语义后，EMBER v3 真正新增的结构列能否稳定修复错误？这是一条低成本、可否证的 fallback，用来区分“仍缺结构语义”与“必须转 CFG、behavior 或 label-quality”的两种未来。

Loop151 继续是唯一 research champion；Loop166 当前 BPE/MLM recipe 已关闭。`F1 >= 0.9997` 目标仍未达成。

## 已查证来源

- [EMBER2024](https://github.com/FutureComputing4AI/EMBER2024) 固定 commit `0ef753e81d98bf209f71b03cd331dfc190b5b54d`。
- 官方 `src/thrember/features.py` SHA-256 为 `58a085e9ad307aa2c52e165985ff80db8fd5b763891c0cba2d1758a4825f7273`；`src/thrember/pefile_warnings.txt` SHA-256 为 `a23a9d0a7a938b19390a75fe0eb024dbc9bad7a134bb1511a2913f365a52e5fb`。
- [EMBER2024 论文](https://arxiv.org/abs/2506.05074) 说明数据使用 52 周训练、后续 12 周测试。这个时间设计是未来协议参考，不会被当前本地 content-component Train OOF 冒充。
- 官方 v3 维数由 `7 + 256 + 256 + 177 + 224 + 1282 + 129 + 74 + 34 + 33 + 8 + 88 = 2568` 自证。

官方代码仅作为固定语义来源；实验不得在线拉取或执行外部代码，必须使用项目原生、source-bound 的实现。

## Phase A：Semantic Delta Freeze

Phase A 是 static-only：raw opens、checkpoint opens、prediction-row opens、training、fitting 和 dependency install 全部为 `0`。

每个官方列必须唯一归入：

1. `exact_overlap`；
2. `partial_overlap`；
3. `genuinely_novel`；
4. `forbidden_or_unstable`。

分类总数必须精确守恒为 `2568`。novel set 至少跨三个语义组，不能全部是 hash 列；feature order、reference vectors、有限值与 missing semantics 必须由 synthetic tests 冻结。Authenticode 与 data directories 强制进入 overlap controls，不能重复包装成 novel。Rich Header、PE warnings 与尚未覆盖的 DOS/header 列只是候选，最终归类仍由逐列 mapping 决定。

任一维数、语义或合成测试失败，都在 raw access 前关闭 Loop167。

## Phase B：Train-only 五臂对照

Phase B 当前尚未授权执行。未来 source closure 完成后，唯一输入是 SHA-bound Train-only `20,000` 行五折：`reports/roadmap_9997/loop164/local_train_diagnostic_folds.jsonl`，SHA-256 `00a31a1bd86d7b887447f3e86e5e753ebcaaee45be74311199332e073a3880a5`。它只证明 content-component isolation，不具备 time/family/source causal 边界。

固定 seeds `41/42/43`、五折、threshold `0.5`、同一个不调参的 HGB：

- `B0`：Phase A 去重后冻结的 Axon productized structural allowlist；
- `B1`：EMBER exact/partial overlap-only；
- `M`：`B0 + genuinely novel delta`；
- `A`：novel delta-only；
- `CF`：`B0 +` 在 fit/holdout 内分别做固定无标签置乱的 novel delta。

全部 missing 留在 `20,000` 分母；`M/CF` 的 novel missing 行精确回退 `B0` score/decision。禁止使用 filename/path/extension/hash/sample-index/fold/row-order、label-derived extractor input、family/source/time/reviewer verdict、历史模型逐行 score，以及任何 Loop166 tokenizer/checkpoint/code-token artifact。

## 决定性门槛

每个 seed 都必须同时满足：

- `M` 相对 `B0/B1` 中错误更少的 control 净减错 `>= max(30, ceil(10% * control_errors))`；
- repairs `>=50`、override precision `>=0.80`；
- 至少 `4/5` folds 净改善；
- content-component cluster bootstrap 单侧 95% LCB 的净减错 `>0`，固定 `200,000` replicates；
- FP、FN 任一侧相对恶化不超过 `5%`；
- `A` 与 `B0` error overlap `<=0.80`；
- `M` 相对 `CF` 多净减至少 `30`，且 `CF` 不得通过 LCB 门。

三个 seed 全部通过，才能记录 `local_train_novel_delta_signal_observed_not_promotion`。任一门失败就执行 `close_ember_v3_novel_delta_control`，不补 threshold、features、models 或 seeds。

## 资源与权限

未来 Phase B 只允许一个 Train raw pass：最多 `20,000` opens、`25 GiB`；提取 `<=2h`、cache `<=1 GiB`、RSS `<=4 GiB`；HGB 最多 `75` fits、训练 `<=6h`、RSS `<=8 GiB`、GPU `0`；总 wall `<=8h`。OOM、nonfinite、timeout、cache 不可恢复、第二次 raw pass 或 source/scope drift 都 fail closed，失败 lease 不重试。

当前 authorization 只授予 Phase A 静态 mapping/实现/合成测试。Phase B 必须等 extractor、controller、allowlist、tests、runtime、argv、resource guard 全部 source-bound 后，另发新的 one-shot run authorization 并在首次 raw open 前原子消费 lease。用户已经给予本地继续推进授权，因此 source closure 完成后不需要再次询问小步骤；但这不会降低机器闸门。

本地 Train-only Phase A/B 不需要 public key，也不授予 A2/A3、Val、Test-10k、legacy full-test、promotion 或 certification。
