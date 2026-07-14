# Loop166 Code-Section Foundation Expert Proposal

## 目标

Loop166 的真实目标不是再调一个现有分类器，而是验证一种与 Loop151 足够正交的新表示：只读取 PE 可执行代码段，学习多字节代码模式，再与结构专家在未来共享 OOF router 中互补。主实验固定为 **MalwarePT-inspired scaled code-section BPE/MLM diagnostic**；它不是 MalwarePT 复现，也不继承论文的模型质量声明。

Loop151 继续是唯一 research champion。Loop164 当前 recipe 保持 parked。`F1 >= 0.9997` 目标不变且未达成。

## 已查证依据

- [MalwarePT](https://arxiv.org/abs/2605.16455) v1 发布于 2026-05-15，使用 PE code-section bytes、BPE、MLM 和 ModernBERT-style encoder；论文报告 `1024` BPE vocabulary 是最强整体折衷，并观察到 code model 与 Ember 结构模型互补。
- 论文完整模型约 `86M` 参数，预训练 `150k` steps、每个 vocabulary 配置约 `81 GPU-hours` on `8 x L40S`；代码和权重尚未公开。本机是 8 GB RTX 4070 Laptop，因此只能做项目原生缩放诊断。
- [EMBER2024](https://github.com/FutureComputing4AI/EMBER2024) 固定 commit `0ef753e81d98bf209f71b03cd331dfc190b5b54d`。官方 v3 是 `2568` 维，但 Axon 已覆盖大量 PE/stat/byte/string/cert 语义；它只能作为 novel-delta structural control，不能冒充完全新专家。
- [TESSERACT](https://github.com/s2labres/tesseract-ml-release) 固定 commit `05132c83eccaa0be24b5403bfe9a73c22199ddef`，用于未来 temporal evaluation contract，不是本轮模型。
- [Nebula](https://github.com/dtrizna/nebula) 固定 commit `3d3b97e5b079d64371224db3dcfbdf175975e90d`，保留为 packed/parser-fail/high-uncertainty 尾部专家；当前 trace coverage 不足，不能先于静态 foundation gate。

ProjectAnalysis CLI 当前因 `/mnt/c/MCP/ProjectAnalysis/dist/index.js:14704` JavaScript syntax error 不可用；本轮架构结论由当前源码、artifacts、外部固定来源和三个并行子个体交叉核验。

## 独立证据边界

主专家只允许以下模型输入：

- `IMAGE_SCN_MEM_EXECUTE` sections 的 raw bytes；
- reserved PAD/CLS/SEP/MASK tokens；
- 显式 code-extraction missingness 仅供系统 fallback/router，不进入 code encoder 伪造内容。

禁止输入：PE/stat/content sidecar、Loop151/Loop69/Loop164 score、label-derived weight、filename、path、extension、SHA、sample index、fold、source/family/time、signer 或 reviewer verdict。身份字段只能做加载、完整性、对齐和 component partition 审计。

BPE vocabulary、normalizer、MLM、classifier 和 aggregator 都属于 fitted preprocessing/model。每个 outer holdout 的 bytes 即使没有标签，也不得进入 outer-fit tokenizer 或 MLM。当前五折只有 content-component isolation，缺 authoritative time/family/source，因此成功也只能叫 `local_train_foundation_signal_observed`。

## Phase A: 256-row extractor gate

复用 SHA-bound balanced Train-only bundle：

- `reports/roadmap_9997/loop164/local_probe_bundle.jsonl`
- bundle SHA-256 `90961bfed0460787e261965a3180e1b0569df0f9d275f9693daad1ccf53dc233`
- `128` benign + `128` malicious；文件大小固定在 `64 KiB..8 MiB`

提取语义冻结为：读取所有 executable sections；raw spans 必须完整落在文件内；任一 executable span 越界则整行 missing；有效 spans 按文件 offset 排序并合并重叠，避免重复字节；不把 section name 当特征。输出只记录 aggregate counts、coverage、span/code-byte 分布、资源和 extraction commitment，不落盘 raw code bytes。

工程门：

- `success + missing = 256`，silent drop `0`；
- source size/SHA/fingerprint mismatch `0`；
- extractor coverage `>=0.85`；未来系统全分母 coverage 固定 `1.0`，code missing 必须走结构 fallback；
- wall time `<=180s`、peak RSS `<1 GiB`、output raw-code artifact `0 bytes`；
- 任一非有限计数、重复 identity、heldout role 或输入 binding 漂移立即失败。

Phase A 不训练 tokenizer/model，不计算 F1/accuracy，不使用 public key，也不产生 promotion evidence。

## Phase B: one-outer tiny MLM resource cell

只有 Phase A 通过后才实现。冻结缩放配方：

- byte-bijective BPE vocabulary `1024 + 5 special`；tokenizer corpus 使用 outer-fit code bytes 的 non-overlapping `512-byte` windows；
- sequence length `512`、最多 `8` deterministic code chunks/file；
- 6 layers、hidden `384`、6 heads、FFN `1536`、local attention window `128`、global CLS；约 `10-15M` 参数；
- MLM mask ratio `0.25`，AMP + activation checkpointing；
- tokenizer/MLM/classifier 全部 outer-fit-only；固定一个 real outer cell 校准资源，不允许减少 inner/outer 语义后冒充 OOF。

资源杀停：单 fold wall time `>8h`、吞吐 `<2000 original-byte-equivalent tokens/s`、batch 1 + checkpoint 仍 OOM、VRAM `>8GB`、nonfinite 或 checkpoint/cache 不可恢复时停止，不扩五折。

## Phase C: one-seed five-fold Train OOF

完整分母固定 `20,000`；复用 Loop164 content-component folds，仅作为 local diagnostic。三组同折同预算对照：

1. BPE-1024 + MLM 主配方；
2. 同 BPE/architecture supervised-from-scratch，不做 MLM；
3. byte-256 + MLM；
4. probability-free structural HGB 作为外部强 control，不输入 code model。

所有 missing 进入分母并使用预注册 structural fallback。hard threshold 固定 `0.5`，禁止 outer-result threshold sweep。每行输出 score、hard decision、uncertainty、missing reason、SHA/component/fold 对齐字段；身份字段不进入模型。

首 seed 继续门：

- outer-holdout normalized MLM bits/original-byte 相对 byte control 改善 `>=5%`；
- 相对最强 control errors 至少减少 `max(30, 10%)`；
- 至少 `4/5` folds 净改善，component-cluster paired one-sided 95% LCB `>0`；
- FP 或 FN 任一侧恶化不得超过 `5%`；
- 相对 structural control residual overlap `<0.80`，至少在 `4/5` folds 累计修复 `>=50` 个 structural errors。

未过任一门则关闭当前 BPE/MLM recipe，转 EMBER-v3 novel-delta control 或更晚的 CFG/behavior route，不靠增大模型、补阈值或查看 heldout 续命。

## Phase D: three-seed confirmation

只在首 seed 全门通过后执行。三 seed 必须方向一致、paired component CI 不覆盖 `0`。成功仍不能打开 Val/Test/full；下一步是补 time/family/source contract，再进入独立 selection windows。

## Public Key

本地 Phase A-D 不需要 public key。public key 只属于未来外部 A2/A3 custody/certification trust anchor，不能被用户或 agent 临时生成来伪造独立认证。
