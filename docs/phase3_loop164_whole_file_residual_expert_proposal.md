# Phase 3 Loop164 Whole-File Residual Expert Proposal

更新时间：2026-07-12

## 真实目标

Loop164 不是为了再拿到几个样本级改善，而是验证一个能覆盖 Loop151 大面积残差的新静态专家：使用 MalConv2 风格的低内存 whole-file byte CNN 与 Global Channel Gating，读取当前 8192-byte MHDSRA 前缀之外的内容，再通过严格 Train-OOF 残差融合尝试同时减少 FP 和 FN。

当前 Loop151 仍是唯一 research champion：

| Split | F1 | Errors | FP | FN |
|---|---:|---:|---:|---:|
| Val | `0.9919193935` | `162` | `105` | `57` |
| Test-10k | `0.9921921922` | `78` | `49` | `29` |
| Legacy full-test | `0.9908541911` | `1466` | `879` | `587` |

`F1 >= 0.9997` 在 legacy 160k 平衡开发参考的精确门为 `10003*FN + 9997*FP <= 480000`：总错误 `<=47` 必过；恰好 `48` 时只允许 `FN<=24`；`>=49` 必败。Loop151 仍需净消除至少 `1418` 个错误。这个点估计边界不是认证余量；Loop164 即使成功也只是静态专家阶段，不能替代未来两个 sealed temporal full-test 和其 group-bootstrap 下界。

## 已查证事实

1. Loop149 的同架构 32768-byte MHDSRA 在 Val 修复 `59` 个 Loop136 错误，却破坏 `1414` 个正确样本；继续调同类长前缀模型已经关闭。
2. Loop163 只有 `9` 个 Val R11 disagreements，其中 `8` fix、`1` break；probability/R11 selector 不具备继续训练的支撑度。
3. Loop151 的 Authenticode trusted-signer evidence 在 Val/Test-10k/legacy full-test 分别净降 `17/5/78` 个错误，证明独立证据有价值，但实际触发 support 仍小，继续扫 threshold 或 signer term 已关闭。
4. Loop157 已导出全部 `162` 个 Loop151 Val 错误的安全外部标注包，Loop158 当前 returned annotations 为 `0`。它是并行数据治理，不是部署时的新模型证据。
5. 现有 Stage-2 OOF 使用随机 `StratifiedKFold`；历史 group diagnostics 只覆盖 `40,000` 行，且报告 `338` 个 cross-split groups 与 `273` 个 leakage groups，不能充当 Loop164 的 full-pool group/time contract。
6. 当前 fixed-v2 `256` 维只定义 `143` 维，content PE v2/string/cert sidecars 虽各有 Train/Val `40,000` 行缓存，但尚未形成 fixed-v3、temporal/family/source isolated 的统一契约。

ProjectAnalysis 通过 Windows Node `25.2.1` 完成 doctor，SQLite、security filter 和 parser probes 均通过；但本次扫描 preview 在 `5000` 文件后停止并返回 `0` supported source files，因此不把它作为架构结论证据，模块判断来自当前代码和聚合 artifacts。

## 前沿查证

- [MalConv2](https://github.com/FutureComputing4AI/MalConv2) 官方仓库 HEAD `b5a865420e984b22a2474834bd1748318e84b553`，对应 [AAAI 2021 论文](https://arxiv.org/abs/2012.09390)。其 `LowMemConv` 用分块和固定内存池化处理超长序列，GCG 用于跨远距离区域交互。上游明确标注为 research-quality，并指出 non-negative 训练选项有错误，因此只能作为算法参考，不能未经审计直接复制。
- [EMBER2024](https://github.com/FutureComputing4AI/EMBER2024) 官方仓库 HEAD `0ef753e81d98bf209f71b03cd331dfc190b5b54d`，对应 [KDD 2025 论文](https://arxiv.org/abs/2506.05074)。它使用前 `52` 周训练、后 `12` 周测试，并提供 EMBER v3、Authenticode、parser warnings 与 evasive challenge，支持 Axon 的 P1 temporal reset 和后续 tabular expert。
- [TESSERACT](https://github.com/s2labres/tesseract-ml-release) 官方仓库 HEAD `05132c83eccaa0be24b5403bfe9a73c22199ddef`，明确指出 random malware split 会产生时空偏差，支持 time-aware evaluation 与 AUT 指标。
- [Nebula](https://github.com/dtrizna/nebula) 官方仓库 HEAD `3d3b97e5b079d64371224db3dcfbdf175975e90d`，对应 [IEEE TIFS 2024 论文](https://arxiv.org/abs/2310.10664)。它建模行为事件序列，不等于 Axon 已失败的 timeout-as-benign 规则；但当前 trace coverage、failure semantics 和成本合同都未就绪，所以不是近期第一候选。

完整机器记录见 `manifests/roadmap_9997/loop164_whole_file_residual_expert/frontier_sources.json`。

## 子智能体结论

- 候选设计角色推荐 whole-file MalConv2-style expert：它能在本地运行、覆盖广、同时可能影响 FP/FN，且当前 lineage 尚未做三 seed group-aware OOF。
- schema/root-cause 角色推荐 time-causal trust-as-teacher router：方向有价值，但 Train Authenticode as-of cache、SPKI/publisher/source/family/time groups 与 Loop151-equivalent Train OOF 都缺失，近期不可执行。
- 治理反方认为 external verdicts 和 P1/P2 契约必须并行，否则任何 legacy 高分仍只是 development leaderboard。
- 落地结论是先预注册 whole-file expert，external annotations 并行等待；trust/reputation 和 Nebula-style behavior 分别进入后续 connected expert，而不是塞进同一轮。

## 方向排序

| 方向 | 上限 | 当前可执行性 | 主要问题 | 决策 |
|---|---|---|---|---|
| Whole-file GCG byte expert + Train-OOF residual fusion | 高 | 中 | 缺实现、完整 group/time contract、Loop151 Train OOF 和 A2 | `selected_next_proposal` |
| Independent Loop157 annotations | 数据真值必需 | 低 | 返回标注仍为 `0`，且不是 runtime feature | `parallel_prerequisite` |
| Time-causal certificate/reputation expert | 高 | 低 | 缺 as-of provider、group manifest 与完整链证据 | `next_connected_static_expert` |
| Nebula-style behavior expert | 很高 | 低 | 当前仅有失败的 timeout 规则，trace coverage/cost 未就绪 | `defer_to_dynamic_cascade` |
| R11/probability/signer threshold 微调 | 低 | 高 | support 太小且已发生跨 split 反转 | `closed` |

## 预注册设计

Loop164 固定采用 seeds `41/42/43` 共用的一份五折 purged-forward group Train OOF partition。最早时间窗必须显式标为 `train_anchor` / `warmup_not_meta_eligible`，只为第一折提供历史 fit；每个剩余 `eligible` train record 恰好一次 outer holdout。component time 固定为 `max(first_seen_time_utc)`，embargo 至少 `30` 天；每个累计 fit window / outer holdout 每类至少 `1000 / 100` 条 locked rows。whole-file expert 不允许静默截断；oversize、timeout、unsupported、read failure 都必须以 missing reason 进入分母。融合器的 runtime matrix 只能使用预注册的六列 OOF score / uncertainty / missingness，不能以 alias 形式带入 path、filename、hash、sample index、split、row order、signer/source/family/time group。

### Full-pool isolation gate

本轮新增 `scripts/validate_loop164_isolation_contract.py`。它的普通 CLI 只能接受 A2 metadata v3 authority：repo 外 trust anchor 的固定 key、attestation、精确 canonical argv、当前 Python、validator 与 `pre_run_resource_leak_guard` source closure、fresh guard、canonical output、contract/rows SHA、custodian metadata root 必须同时绑定，且 `authority_scope` 精确限定为 `{tier:A2, operation:metadata_isolation_only, protected_input_scope:metadata_only, grants:[]}`；存在 output、路径/symlink、runtime/source/binding drift 或任何 scope 扩权时不会消费 lease。通过后以 stable issuer+lease-id consumption id 和 `O_EXCL + fsync` 写不可回滚 marker，随后才允许从同一受控 descriptor 打开 future custodian 生成的 contract JSON 与私有 JSONL metadata inventory，并在解析前后重验 rows SHA/文件指纹；它不打开 raw、cache、prediction、checkpoint 或模型。validator 对 exact、near-duplicate、带 evidence version 的 family/campaign、以及 `source_id:source_group_id` 求传递闭包，要求每个稳定 component 只落在一个 role 和一个 OOF outer/inner fold。`source_group_key` 之类的路径日期 review bucket、PE timestamp、filesystem mtime、unknown-like group/time 和未覆盖的大桶一律 fail closed。

contract 必须绑定 proposal / inventory / rows artifact / grouping 参数的 SHA，要求 custody full pool 最少 `200000` active rows 与所有预注册 roles、`first_seen_time_utc` 驱动 component 时间、至少 `30` 天 embargo、`denominator_status=included`，并把 residual fusion runtime matrix 固定为六列 allowlist。feature contract v2 的唯一实现字段是 `implementation_binding_phase=deferred_to_a2_training_authority`；它不得携带 `implementation_manifest_sha256` 或任何 placeholder，原因是 metadata receipt 尚未锁定生产源码。输出只含 aggregate counts、训练角色 label support、sealed-role 行数、时间边界、binding fingerprints、fold fingerprint、不可暴露 path/row identity 的 authorization provenance 与 blocker codes；缺失或伪造 provenance 的 receipt 会被 fold-scope、training-authority 和 nested-OOF verifier 拒绝。即使 `decision=pass` 也只代表 partition integrity，`loop164_train_oof_data_boundary` 仍保持 false，直到独立 nested OOF execution receipt 验证完成，绝不自动授权 A2 或任何评估。

### Custodian A2 metadata-only request checklist

申请人只能请求一次 metadata-only A2，不得同时请求训练、实现、输入 bundle、Val、Test-10k 或 full-test 权限。保管人请求材料必须同时包含：

1. future full-pool contract v2，含 proposal/inventory/rows/grouping SHA、六列 feature allowlist、`feature_matrix_receipt_required=true` 与 `implementation_binding_phase=deferred_to_a2_training_authority`，且不存在任何实现 manifest SHA；
2. repo 外 verified trust anchor、固定 key fingerprint 与 verification receipt，fresh `pre_run_resource_leak_guard`，当前 Python、validator/resource-guard source closure、custodian metadata root、空的 canonical output path；
3. 唯一 canonical argv 的逐字列表，以及将其、contract SHA、rows SHA、output path、stable issuer+lease-id 和精确 metadata-only/empty-grants scope 写入 A2 metadata v3 authorization 的外部签名；
4. 确认 output、lease marker、train authority、train-only input bundle、production implementation manifest、checkpoint 和所有评估 artifacts 均不存在；
5. 由保管人运行受控 CLI，保存仅含 aggregate counts/provenance 的 isolation receipt v4；receipt provenance 必须重现该 metadata-only/empty-grants scope，training 和 nested OOF gate 会精确拒绝其它 scope。若任何 binding、same-FD rows fingerprint、guard freshness 或 output precondition 失败，保留 blocker，不重试或改写 authority。

该请求的成功回执只冻结 partition 与 feature semantics。下一阶段才是独立 static implementation review；再之后才可请求训练 A2 authority，由它绑定真实 manifest、resource guard、scope validation 和 train-only bundle。

### Non-authorizing request contract

`scripts/validate_loop164_a2_request.py` 与两份 `templates/a2_*_request.template.json` 将申请材料固定为 `custodian_request_not_authorization`。metadata request 只能是 `draft`，training request 固定为 `blocked_pending_metadata_and_static_review`；两者都必须 `authorization_granted=false`。模板只能列出由保管人未来补齐的 binding 名称和 canonical target paths，不能携带 `decision`、签发时窗、runtime binding、lease 或任何 protected-input grant，并且 request 文件不得占用 canonical authorization 路径。metadata template 精确声明 metadata-only / empty-grants scope；training template 只允许 `train_anchor/train_oof`、固定 `15` 次 outer runs，且逐项列出 isolation/scope/implementation/input/controller 前置。模板通过仅表示请求结构正确，不表示任何 A2 已签发、任何 rows 可打开或任何训练可开始。

### Final certification preregistration

认证协议已在 proposal 中作为 A1 静态字段冻结，但它不是 A3 授权或任何 sealed 结果。W1 `certification` 与 W2 `later replication` 必须时间有序、各自独立 A3 authorization/lease，固定 `evaluation_generation=1`，禁止把两窗池化、替换失败窗口或在 W1 后重选 bundle、阈值或 calibration。每窗都要同时满足 point F1 和单侧 `97.5%` LCB `>=0.9997`；这以 family-wise `alpha=0.05` 保护“两窗都通过”的最终声明，同时仍报告单窗 95% LCB。

LCB 使用预注册的 relationship-component bootstrap：exact/near-duplicate/family/campaign/source 的 union component 加入固定 calendar block 后作为不可拆分重采样单元，`200,000` 次、确定性 SHA-derived seed、固定分位数及 conservative guard 都必须绑定到 statistics runner。abstain、timeout、missing feature、unsupported、parser/runtime failure 需要在 A3 前固定映射并进入分母；未知分组、空类别、跨窗 component、统计失败或 silent drop 均为 `insufficient_evidence`。在任何 A3 request 前，还需有 hash-bound aggregate-only 功效模拟：至少 `50,000` 次，联合功效 `>=0.90`，并给出 blinded strict point floor、最小每类/独立 component/时间层支持和最大 component 集中度。产品负责人仍需另行冻结 FPR/FNR、FP/M、FN/1000、coverage、P95 latency、成本和关键 slice 阈值。

### Nested OOF execution gate

在 nested receipt 之前，`scripts/validate_loop164_fold_scope_plan.py` 会只读取 proposal、isolation contract/pass receipt 与 custodian 的 aggregate scope plan，验证五个外折的固定顺序、30 天 embargo、outer/inner component commitment、inner OOF `eligible + warmup + purged` 账目守恒、父 outer-fit 绑定和每个 inner fold 的 union coverage。它输出的 canonical `fold_scope_plan_validation.json` 是后续训练前 controller 必须绑定的独立证据；其 `pass` 只代表 scope frozen，不是 A2 training authority，不能消费 training lease 或打开任何数据/评估 split。

`scripts/validate_loop164_nested_oof_execution_receipt.py` 只验证未来的 aggregate-only JSON，不读取 raw、cache、预测行、checkpoint 或 split 行。它要求 custodian-attested `fold_scope_plan` 固定所有 seed 的同一五外折分区，并逐一核对 `41/42/43` 的 `15` 个 outer run、每个 outer 的 `5` 个 inner run、`30` 天 purge、fit/holdout commitment、每个 eligible row 每 seed 恰好一次 outer holdout、Loop151/whole-file/fusion 的全流程 OOF provenance、A2 training authorization、content-addressed final lease、implementation review、Loop151 Train OOF manifest 与 resource guard。

### Training authority gate

`scripts/validate_loop164_training_authority.py` 是未来 controller 的启动前门，而不是事后 receipt parser。它只读取 aggregate JSON，并要求 A2 v2 authorization 与 repo 外 trust anchor 的固定 key 相符，且精确重验当前 Python、controller、canonical argv、scope-plan validation receipt、train-only input bundle、resource guard、proposal/contract/isolation/implementation/Loop151 lineage 的 SHA。它拒绝 project-root 外路径、symbolic-link binding、过期 authority、现有 output、argv/runtime drift 和未 pinned 的信任根。

通过检查后，未来 controller 必须在同一进程、任何 `torch`/dataset/raw/cache import 或受保护输入打开前，以 `O_EXCL + fsync` 先写 content-addressed marker，再写 canonical final lease。授权 JSON 永远保持 `state=ready`；marker 一旦创建即不回滚，最终 lease 写失败也会烧掉本次 authority，避免重放。当前 A1 只提供合成测试；尚未创建 A2 authorization、external trust anchor、input bundle、marker、final lease 或 controller。升级后的 nested receipt verifier 会后验复核 v2 authority、input bundle、marker/consumption id 和 final lease，但它仍不能反向授予执行权限。

融合矩阵只能逐字使用六列 OOF score / uncertainty / missingness；label 只能作为 outer-fit 内 inner-OOF 训练目标，不能成为矩阵列。任何 identity alias、family/source/time/fold 特征、outer holdout 在 fit 期的 feature/label/metric 读取、fold drift、in-sample score substitution、重复/未匹配/静默丢弃输出，或 Val/Test-10k/full-test 访问，都会 fail closed。即使 execution receipt `pass`，结果也只证明 Train-OOF data boundary；`a2_training_authorization`、Val、Test-10k 与 full-test 的 `ready_for` 都保持 false。

进入 Test-10k 前不仅要满足 Loop161 的 `Val/Test-10k 各 -3` 底线，还要通过更强的 program gate：

- Val errors `<=152`，FP `<=105`，FN `<=57`；三 seed 全部同向。
- Val disagreements `>=100`，至少修复 `30` 个 Loop151 错误，accepted override precision `>=0.80`。
- 冻结后 Test-10k errors `<=73`，FP `<=49`，FN `<=29`。
- legacy full-test 只允许一次 frozen confirmation，绝不用于 term、feature、router、threshold 或模型选择。

## 当前 Preflight

机器 preflight 决策为：

```text
static_preflight_ready_execution_blocked_missing_prerequisites
```

执行阻塞项：

- `a2_training_authorization_missing`
- `a2_isolation_validation_authorization_missing`
- `whole_file_implementation_manifest_missing`
- `loop151_train_oof_manifest_missing`
- `loop164_train_oof_execution_receipt_missing`
- `fold_scope_plan_missing`
- `fold_scope_plan_validation_missing`
- `train_oof_input_bundle_missing`
- `training_final_lease_missing`
- `full_pool_group_manifest_missing`
- `full_pool_isolation_validation_missing`
- `resource_guard_missing`

晋级阻塞项：

- `val_a_manifest_missing`
- `val_b_manifest_missing`

`champion_registry_missing` 已在本轮消除。新增生成器从 Loop151 truth manifest、Loop28 smoke/parity receipts、Loop28 pause record 和 recommendation ledger 生成三槽注册表：research=`Loop151`、native_offline=`Loop28`、connected_system=`none`。它明确 Loop28 是 parity-blocked native reference，不是 quality champion。

因此本轮只完成 proposal、source pin、A1 authorization、aggregate-only preflight、champion registry、full-pool isolation validator、fold scope plan validator、training authority validator、nested OOF receipt validator 和治理更新。没有打开 raw、checkpoint、split row、prediction row、cache row 或 private map；没有安装依赖、训练、拟合、阈值选择、Val/Test-10k/full-test 评估或 F1 重算。

## A1 v2 Static Implementation Contract

三路只读审查确认当前 `AxonMalwareModel`、`MalwareDataset`、`scripts/main.py` 和 `AxonTrainer` 都不适合作为 Loop164 的 runtime：它们分别依赖固定长度 byte tensor、缓存归一化截断、授权前导入训练栈，或暴露任意 data/split 与 Val/Test 逻辑。未来必须使用独立 controller、流式 loader、whole-file model、OOF protocol 与 fusion 模块；当前没有创建这些生产实现。

新增 `scripts/validate_loop164_whole_file_implementation.py` 仅验证未来 v2 manifest 的静态 source closure。它固定 controller/model/loader/OOF/fusion/authority/receipt/test 的非链接相对路径和 SHA，拒绝动态 import、`exec/eval`、`read_bytes`、零参数 `read`、`mmap`、`np.fromfile`、路径逃逸和 model/fusion 内的身份字段。输入必须使用 `257` token 或等价显式长度语义；supported file 必须分块读取全部字节，oversize 不得回退到 prefix，而要作为 explicit missingness。

v2 contract 还固定 exact independent-region pooling、two-pass context、negative-infinity winner 初始化、尾段覆盖、每文件 timeout 的中性 score/确定性 uncertainty/missingness 三元组，以及 batch/worker/OOM/未知错误的 abort-without-receipt。五个 missingness reason 的总数必须等于 whole-file missing rows，success + missing 必须等于分母；Loop151 score 不得作为 missing fallback。

`validate_loop164_training_authority.py` 现在实际解析该 v2 contract，而非只绑定 manifest 字节；fresh resource guard 还须绑定 manifest、source closure 和 memory contract SHA。nested receipt 则回链 source/config/runtime/input/missingness SHA，并要求每个 whole-file fit artifact 的 code/config/input 等值绑定，防止用另一份 loader、配置或输入 bundle 替换训练路径。

这只是 A1 的合成静态验证。真实 `reports/roadmap_9997/loop164/whole_file_expert_implementation_manifest.json` 仍不存在，`whole_file_implementation_manifest_missing` 继续阻断执行；它不能被 A1 测试、临时 fixture 或本节文字解除。

### Pure-synthetic exactness oracle

新增 `tests/test_loop164_whole_file_gcg.py` 是 in-memory 数学 oracle，不是 `src/loop164/whole_file_gcg.py` 的替代品。它以独立 dense `conv1d + global max` 为 reference，对 output-position 分块的 independent-region pooling 比较 forward、input/kernel/bias gradient、winner 全局位置和同分数时的首位置语义；还覆盖尾段唯一 winner、所有 activation 小于 `-1`、raw `0x00` 与 PAD/EOF 分离、determinism、missingness 分母，以及把不连续 winner 区域拼接后人为制造更大 activation 的反例。

一个纯合成的 two-pass GCG toy 同时检查 context pass 不 detach 时的数值和梯度等价。它的结论严格限于该小型张量协议；没有读取 raw、cache、checkpoint、split 或 prediction，没有创建生产 model/loader，不能推出吞吐、真实文件行为、F1 或 production exact。preflight 现在 source-bind 该 oracle；未来真实 manifest 仍须把它作为 source closure 的测试项，并在实现闭包冻结后重新运行。

## 决策

Loop164 是 Loop151 F1 主线的下一候选 proposal，不是新 champion，也未获执行授权。下一次重计算预算应先让 custodian full-pool group/time manifest 在 one-shot A2 metadata v3 lease（repo 外 pinned trust anchor、source closure、canonical argv 与 metadata-only/empty-grants scope）下通过 provenance-bound isolation receipt，冻结 attested fold scope plan 并取得它的 canonical validation receipt，再完成资源 probe、项目原生 whole-file implementation review、Loop151-equivalent Train OOF、repo 外 pinned trust anchor、train-only input bundle、独立 A2 v2 training authorization/final lease 和 upgraded nested OOF execution receipt；这些条件与单独 A2 authorization 缺一不可。
