# Axon Frontier 99.97 长期执行合同

## 0. 合同定位

本项目的长期目标是把 Axon 从“在已反复观察的随机 20 万开发集上迭代的静态分类器”，演进为可审计、可复验、可原生交付的恶意软件检测系统，并在全量样本口径上达到：

```text
Full-test F1 >= 0.9997
```

`0.9997` 是系统能力目标，不是允许通过换口径、筛样本、反复查看 test、删除拒判样本或事后改规则得到的榜单数字。可以分阶段逼近，但不得降低、偷换或伪造终局。

每轮只完成一个预注册、可终止、可复验的闭环。事实优先级固定为：

```text
当前代码与测试 > machine-readable artifacts > experiment journal > summary docs > 对话结论
```

低层证据不支持时，高层文档不得授权或宣称能力。

## 1. 当前事实基线

以下是 `2026-07-11` 的事实快照；数值必须由 P0 manifest 重新解析和冻结，不得长期手填维护。

### 1.1 三种冠军必须分开

| 口径 | 当前状态 | 证据边界 |
|---|---|---|
| `research_champion` | Loop151，legacy full-test F1 `0.9908541911`，错误 `1466`，FP/FN `879/587` | Loop136 预测 CSV + Authenticode cache + 冻结 signer guard 的离线评估 |
| `native_offline` | Loop28 | ONNX 基础模型 + Stage-2 HGB + native DLL；不包含 Loop151 signer guard |
| `connected_system` | 尚不存在 | 未来的静态、信誉、动态行为和复核级联 |

不得把研究 CSV 评估称为已原生部署，也不得因 native bundle 可运行就把它写成当前研究冠军。

### 1.2 当前停止闸门

- Loop163 的有效结论是 `reject_low_support_no_selector_training`。
- R11/probability-only selector 路线只有 `9` 个 Val disagreements，必须停止。
- 阈值、低支持规则和同质 selector 微调不可能消除剩余 `96%+` 错误。
- 当前 Test-10k 是 legacy full-test 的前 10k；两者均被多轮观察，只能作为 `development leaderboard`，不能用于最终认证。

### 1.3 每轮开工前必读

- `docs/ml_improvement_recommendations.md`
- `reports/model_review/final_model_selection/ml_recommendation_status.json`
- `reports/hard_family_finetune/experiment_journal.md`
- Loop151、Loop161、Loop163 的报告与机器 JSON
- 当前 `git status`、本轮 proposal、authorization 和 preflight

## 2. 目标数学与认证等级

### 2.1 Legacy 160k 的精确误差预算

对 legacy 平衡开发参考集 `P=N=80,000`，令目标精确分数为 `9997/10000`：

```text
F1 = 2(P-FN) / (2P - FN + FP)
10003 * FN + 9997 * FP <= 480000
```

这不是“任意 48 个错误都可以”的近似预算：总错误 `<=47` 时任意 FP/FN 分配均通过；总错误恰为 `48` 时仅 `FN<=24` 通过，`24 FP + 24 FN` 恰好为 `0.9997`；`0 FP + 48 FN` 已低于线；总错误 `>=49` 必败。必须报告原始 TP/TN/FP/FN 和未四舍五入 F1。该整数边界只服务 legacy development point reference，绝不能代替后续 sealed 认证下界。

Loop151 有 `1466` 个错误。即使按宽松的 `48` 错预算，也必须净消除至少 `1418` 个，即 `96.73%` 的当前残差。

### 2.2 三种结果等级

- `development_observed`：允许反复分析，只能说明开发榜结果。
- `confirmation_passed`：冻结候选在独立 confirmation split 上一次通过，不能反向改候选。
- `certified_99_97`：同时通过下列全部条件，且在第二个更晚时间窗复验。

### 2.3 最终认证条件

1. 两个 temporal + family/campaign + source + near-duplicate 隔离的 sealed full-test 均有 `point_f1 >= 0.9997`。
2. 两窗联合声明采用 family-wise `alpha=0.05`：每个窗口按 family/campaign/source/near-duplicate 连通 component 与预注册时间块计算单侧 `97.5%` 下界，均须达到 `f1_lcb_97_5 >= 0.9997`；同时报告单窗 `95%` 下界。不得把两个单窗 `95%` 下界误称为联合 `95%` 认证。
3. 同时通过预注册的 FPR、FNR、FP per million benign、FN per 1,000 malicious、coverage、P95 latency 和成本门槛。
4. 关键月份、家族、来源、packer、签名状态、parser failure、archive、evasive/adversarial slice 不得单独耗尽总误差预算。
5. 所有 abstain、timeout、missing feature、unsupported 和 dynamic failure 都进入全系统分母。

若样本量不足以证明统计下界，结论只能是 point target observed，不能称 `certified_99_97`。

### 2.4 认证统计预注册

在任一 A3 sealed window 解封前，必须冻结两窗认证协议：W1 认证与 W2 更晚时间复验不得池化、替换或复用；`evaluation_generation=1`，每窗各有独立 A3 authorization、lease、exclusive output 与 content-addressed manifest。冻结 bundle 必须绑定 checkpoint、模型源码、config、calibration、threshold/abstain policy、runtime、SBOM 与 statistics runner SHA；W1 后不得重新选择候选、阈值或校准。

主统计量是全系统 TP/TN/FP/FN 的未四舍五入 F1。exact/near-duplicate/family/campaign/source 的 union component 是不可拆分重采样单元，component time 按 `max(first_seen_time_utc)` 进入预注册 calendar block；bootstrap 固定 `200,000` 次、确定性 seed/分位数规则与保守 guard。空类别、未知 grouping、跨窗 component、统计失败或静默丢行一律为 `insufficient_evidence`。认证前还必须用 aggregate-only group bootstrap simulation 完成至少 `50,000` 次功效模拟，证明在严格高于目标的 blinded point floor 下联合通过概率 `>=0.90`；样本、独立 component、时间层或产品业务阈值不足则不得申请 A3。

### 2.4 现实 prevalence

每轮必须在产品负责人预注册的 `pi_low / pi_expected / pi_high` 下报告：

```text
precision(pi) = pi * TPR / (pi * TPR + (1 - pi) * FPR)
```

平衡 F1 不能替代生产误报成本。产品 FPR/FNR 硬上限优先于 aggregate F1。

## 3. 授权状态机

用户授权决定能否执行重操作；机器 gate 决定某个 split 是否在科学协议上允许执行。两者缺一不可。

| 等级 | 默认权限 | 额外要求 |
|---|---|---|
| `A0_readonly` | 读代码/报告、搜索、git status、artifact existence/SHA、静态分析 | 不改文件、不运行重计算 |
| `A1_scoped_change` | 约定范围内的 docs/code/tests、小型只读报告 | 用户一次范围授权；范围内不重复询问 |
| `A2_heavy_compute` | 训练、cache build、批量特征/预测、大规模评估、依赖安装 | 用户显式授权 + fresh resource guard + machine authorization JSON |
| `A3_heldout` | sentinel、confirmation、certification 分阶段评估 | 每个 split 独立授权；每个 frozen bundle 仅一次 |
| `A4_data_release` | split/cache mutation、删除、部署、commit、push | 每项单独显式授权和回滚方案 |

当前长期任务只自动授权 `A0` 与为 P0 合同/工具所需的 `A1`。不得据此启动训练、批量提取、Test-10k、full-test 或数据修改。

重操作的 canonical argv 使用项目 Windows 虚拟环境；只读诊断可在 WSL 执行。run manifest 必须记录真实 shell、解释器和 argv，不得只保留人工改写的命令文本。

## 4. 数据与 sealed holdout 合同

### 4.1 数据角色不可混用

每份数据只能有一个角色：

- `development`：允许反复分析。
- `selection`：只用于模型、阈值、规则和校准选择。
- `confirmation`：只验证预冻结候选。
- `certification`：一次性封存验收。

当前 random 20w、legacy Test-10k 和 legacy full-test 全部降级为 development。新 sentinel、confirmation 和 certification 必须彼此无样本重叠。

### 4.2 新 full-test-v2

每行必须有 content-addressed manifest，至少包含：

```text
source_sha256, acquisition_time, first_seen_time, source,
locked_label, label_provenance, label_evidence_version,
family_id, campaign_id, near_duplicate_cluster,
parser_status, schema_version, split_role
```

硬约束：

- train/Val-A/Val-B/calibration/sentinel/full/external challenge 按时间因果构建。
- exact、near-duplicate、family/variant、campaign 和 source leakage 均为 `0`。
- normalization、imputation、feature selection、selector、calibrator 全部 train/selection-only fit。
- 不得因 cache missing、parse failure、动态失败或困难样本而过滤出分母。
- 标签在预测前冻结；冲突标签双人独立盲审并仲裁，同时随机审计非错误样本。
- sealing 后发现坏标签时，整版 holdout 作废并重新版本化；不得只删改让候选过线。
- custodian 保存逐行身份与标签；评估方默认只获得签名聚合报告和 prediction artifact hash。

## 5. 长期系统骨架

目标形态不是单一更大的 Axon，而是异构、时间因果、可缺失感知的级联系统：

1. `native static core`：现有 MHDSRA2 control + fixed-v3 PE/stat + whole-file byte expert。
2. `byte foundation expert`：在合法未标注 PE code sections/whole-file bytes 上做 BPE/span/MLM 自监督预训练；MalwarePT 仅作前沿参考，必须自行复现验证。
3. `tabular diversity expert`：EMBER v3/HGB 与生产级 byte n-gram，提供不同归纳偏置。
4. `PE graph expert`：CFG/FCG 表示与时间域适配；packed/unparseable 是显式 missingness，不是丢样本理由。
5. `time-causal trust expert`：证书 leaf/SPKI/chain、EKU、timestamp、revocation-as-of-time、publisher identity；禁止用未来信誉回填历史样本。
6. `dynamic behavior expert`：Nebula 风格行为事件编码；Speakeasy/CAPE 只处理静态冲突、高风险、packed 或不确定样本。
7. `nested OOF router`：融合专家 OOF 分数、missingness、packer/parser 状态和校准风险；禁止输入路径、文件名、hash、row id 或 split。
8. `review/fallback layer`：拒判和动态失败进入真实复核容量与最终 confusion matrix，不假设人工是 oracle。

已有 `scripts/train_stage2_oof_stacker.py` 可作为 OOF 协议参考，但任何新融合必须重新做 temporal/group isolation 审计。

## 6. P0-P7 超长期路线

### P0 Truth Freeze

- 冻结 Loop151 三 split 指标、输入/模型/规则/报告 SHA-256、git/worktree 状态和证据存在性。
- 建立 machine-readable artifact manifest 与 champion registry 骨架。
- 从 raw file 到 Loop151 report 做精确重放；当前仅有 CSV evaluator 不算完成。
- 成功门：TP/TN/FP/FN/F1 与冻结报告逐项一致，所有必需 artifact 可追溯，无 silent missing。

### P1 Evaluation Reset

- 构建 Val-A、Val-B、disjoint sentinel、future full-test-v2 和 external challenge。
- exact/near-dup/family/source/temporal leakage 为 `0`，label provenance 覆盖 `100%`。
- 用新协议重测 Loop151，旧 160k 只保留 regression 作用。

### P2 Evidence Backbone

- 固化 fixed-v3 schema、whole-file byte、证书链和统一动态日志 schema。
- 所有 failure/missing 有显式 token 与原因码，silent missing 为 `0`。
- Python/native 特征 parity 为 `100%`，任何 decision mismatch 阻断晋级。

### P3 Static Expert Frontier

- 并行验证 DSRA control、MalConv2 类 whole-file、byte foundation、HGB/n-gram、CFG/FCG 专家。
- 每个专家必须有 control、3 seeds、时间窗与家族 slice。
- 阶段门：两个 selection 窗口相对新 baseline 错误至少减少 `50%`，方向一致，paired cluster CI 下界大于 `0`；目标 F1 至少 `0.997`。

### P4 Nested OOF Fusion

- 只使用各专家严格 OOF 输出训练 meta-router。
- missingness 和不确定性可入模，身份字段不得入模。
- 阶段门：两个窗口均净降错，FP/FN 和高价值 slice 不回退，目标 F1 至少 `0.9990`。

### P5 Dynamic Cascade

- 静态分支冲突、packed、parser failure 和高不确定样本进入动态专家。
- 初期 dynamic route 不超过 `10%`，成熟目标不超过 `3%`；有效 trace 率、timeout、OS/build 域偏移均必须报告。
- 阶段门：selection system F1 至少 `0.9995`，路由净纠错 precision 至少 `95%`，失败有保守 fallback。

### P6 Robustness And Continual Learning

- 引入 evasive/adversarial、archive、new-family、publisher drift、sandbox drift 和 parser-failure challenge。
- 只从后到前滚动时间窗更新，旧 sealed 集不得成为训练反馈。
- 建立 drift detector、shadow evaluation、rollback 和 lineage invalidation。

### P7 Certification And Productization

- 一个预注册 frozen bundle 在两个 sealed future full-test 上通过 point、lower bound、业务成本、coverage 和 robustness 门。
- 对至少 `100k` raw files 做 Python/native/connected end-to-end parity；决策 `0` mismatch，概率容差 `<=1e-6`。
- bundle 包含模型、schema、calibration、policy、certificate evidence version、provider ABI、签名、SBOM 和 rollback。

## 7. 每轮预注册 Proposal

没有以下完整块，不得进入实现或实验：

```text
loop_id / parent_reference / hypothesis / failure_observation
new_independent_evidence / runtime_availability
allowed_splits / forbidden_inputs / frozen_thresholds
minimum_support / sample_count / seeds / compute_budget
success_gate / business_gate / slice_gate / stop_condition
expected_artifacts / authorization_level / native_path
```

每轮固定流程：

1. 读取事实台账和 dirty tree。
2. 启动愿景、前沿、架构、实验、反方、落地角色；中大型任务使用真实并行子智能体。
3. 搜索本地代码与可信外部来源，区分已查证事实、合理推断、未确认假设。
4. 从失败原因推出一个正交假设，不做想法堆砌。
5. 写 proposal、authorization 和 preflight。
6. 同时检查人类授权与 machine gate。
7. 实现、定向测试、资源检查。
8. 只运行 allowed split；threshold sweep 只能在 selection split。
9. 由机器判定 promotion gate。
10. 写 artifacts、recommendation status、journal 和 failure analysis。
11. 本轮结束；不得自动启动下一轮重实验。

## 8. 晋级漏斗

新协议固定为：

```text
Train OOF -> Val-A -> Val-B -> disjoint sentinel -> confirmation -> certification -> later-window replication
```

硬门槛：

- `integrity_gate`：split、label、cache、SHA、near-dup、family/time isolation 全通过。
- `development_gate`：至少 3 seeds，收益方向一致，无单侧 FP/FN 崩坏。
- `selection_gate`：模型、阈值、signer terms、abstain/cascade policy 全部冻结。
- `confirmation_gate`：paired family-cluster bootstrap 单侧下界 `delta_f1_lower_95 > 0`，所有业务硬指标不退化。
- `certification_gate`：point、lower bound、prevalence、coverage、latency、robustness 同时通过。
- `replication_gate`：第二个更晚时间窗独立复验后，才允许 shadow/canary promotion。

当 incumbent errors 仍高于 `200` 时，新路线的最小有效改善默认为：

```text
max(30, ceil(0.05 * (incumbent_errors - 48)))
```

低于该幅度只能记为诊断，不消耗 confirmation/certification holdout。selector/rule 的 Val disagreements 默认至少 `100`；低于此值直接熔断，除非 proposal 用统计功效证明更低支持仍可靠。

## 9. 不可作弊约束

- 禁止 test 用于训练、阈值、term、feature、rule、model、replacement 或 abstain policy 选择。
- 禁止 full-test posthoc rows 反向生成候选；逐行结果被查看后，该 split 对该 lineage 即不再独立。
- 禁止 filename、path、directory、extension、hash、sample index、split、row order 或未来 AV verdict 成为模型证据。
- identity 字段只用于 alignment、cache lookup、dedup 和审计，并必须与训练特征隔离。
- external trust 必须使用 scan-time 可得的 as-of evidence；标签来源与模型 evidence 不得形成循环。
- 禁止过滤困难、missing、parse-failed、unsupported、timeout 或 abstain 样本。
- 禁止只报告最有利阈值；test 只应用冻结阈值。
- 禁止 direct relabel、自填 reviewer verdict 或只审计模型错误；坏标签只能触发独立版本化流程。
- full-test 只运行 frozen bundle；任何重跑产生新 `evaluation_generation`，不得再称首次独立确认。
- 任一 leakage 会使整条 lineage invalidated。

## 10. Artifact 与 champion 规范

每轮固定目录：

```text
reports/roadmap_9997/loopNNN/
  proposal.json
  authorization.json
  preflight.json
  run_manifest.json
  metrics.json
  paired_delta.json
  slice_metrics.json
  leakage_audit.json
  resource_report.json
  decision.json
  failure_analysis.md
  artifact_manifest.json
```

大体积运行证据保留在 `reports/roadmap_9997/loopNNN/`；可版本控制的不可变索引同时写入 `manifests/roadmap_9997/loopNNN/`。后者已从全局 JSON ignore 规则中显式放行。

`run_manifest.json` 至少记录：

```text
git HEAD, branch, dirty status, tracked patch SHA,
OS, shell, Python, Torch, CUDA, exact argv,
seed, start/end time, authorization receipt,
config/data/split/label/checkpoint/schema/input/output SHA,
allowed split, threshold source, evaluation generation
```

预测、checkpoint、rule、calibrator、report 和 native bundle 均记录 SHA-256。ledger 和 champion registry 只能解析机器 artifacts 生成，不得手填指标。

`reports/model_review/final_model_selection/champion_registry.json` 必须分别记录 research/native/connected champion、父 lineage、协议版本、证据和 deployment parity。

## 11. 失败复盘与杀停

每个失败实验必须记录：原始假设、实际失败指标、失败原因、证据强度、排除的路线、允许重启的前提。下一步必须来自“失败观察 -> 原因 -> 新证据”的因果链，禁止只写“再调参”。

出现任一条件立即停止当前路线：

- 标签噪声或其 95% 上界达到总误差预算量级。
- confirmation/certification 与 selection 重叠，或 test 被用于选择。
- 改善只存在于 random split，在 temporal/family holdout 消失。
- 多 seed 或时间窗 CI 覆盖 incumbent。
- F1 提升但 FPR、FNR、FP/M、FN/M、coverage 或高价值 slice 越过硬上限。
- dynamic timeout、unsupported、P95 latency、VM 成本或人工队列超过预注册容量。
- evidence 在 native/connected runtime 不可得，或 native parity 出现任一 decision mismatch。
- 同一路线连续两轮未达到预注册最小效应，或 residual overlap 持续超过 `80%`。
- 为继续实验必须查看 sealed identities、标签或逐行错误。

杀停只说明当前路线失败，不得降低长期目标。后续只能换正交证据源、修复标签/评测合同或等待新时间窗。

## 12. 外部技术锚点

优先复核并持续更新以下来源，不把论文数字直接当作 Axon 结论：

- EMBER2024：大规模、多格式、evasive challenge 与 EMBER v3 特征。
- TESSERACT：时间与空间偏差、AUT 评估。
- MalConv2：whole-file byte 建模基线。
- MalwarePT：code-section byte foundation model 前沿参考；当前属于预印本，不能假设代码/权重可用。
- NDSS 2025 Windows malware drift：CFG 与时间域适配。
- Nebula / Quo Vadis：动态行为序列与级联证据。
- Conformal Risk Control：coverage-risk 约束，但不得用 abstain 删除分母。

任何涉及 SOTA、框架、政策、价格或运行时可用性的决策都必须重新查证来源和日期。

## 13. 唯一合法的近期动作

当前优先级已从 Loop28 部署支线纠正回 Loop151 F1 主线。阶段可以并行设计，但证据门和授权级别不得越级：

1. Loop28 native decode-compat 保持暂停。已有 synthetic/native 证据保留为 P2/P7 交付债务，但不得继续创建 preflight authorization、lease、raw run 或 runtime package，除非未来有独立部署里程碑重新提案并重绑完整 authority chain。
2. Loop151 继续作为唯一 research champion。当前 A0/A1 只允许 aggregate evidence audit、Loop157/158 外部标注治理、P1/P2 合同设计，以及 Loop164 whole-file residual expert 的 proposal、source pin、静态 preflight、代码评审和测试。
3. Loop164 是当前推荐的下一科研候选：MalConv2-style low-memory whole-file GCG byte expert + 严格 Train-OOF residual fusion。它必须证明相对 8192-byte Loop151 lineage 的新 byte-region evidence，而不是复跑 32768 MHDSRA、byte n-gram 微融合、R11/probability selector 或 signer threshold sweep。
4. 在打开 custodian metadata inventory 前，必须补齐至少 `200000` active rows、所有预注册 roles 的 full-pool exact/near-duplicate/family/campaign/custodian-source/time group contract，并由 `scripts/validate_loop164_isolation_contract.py` 在 repo 外固定 key 的 A2 metadata v3 trust anchor、精确 canonical argv、validator/resource-guard source closure、canonical output 预检、custodian metadata root 与稳定 issuer+lease-id 消费键均通过后，以 `O_EXCL + fsync` 原子烧掉一次性 lease，才可打开 JSONL；其 `authority_scope` 必须精确为 `{tier:A2, operation:metadata_isolation_only, protected_input_scope:metadata_only, grants:[]}`。同一文件描述符解析前后必须重验 rows SHA/文件指纹，输出 receipt 只含 aggregate 结果和 authorization provenance。metadata contract v2 只冻结六列 residual feature 语义，并强制 `implementation_binding_phase=deferred_to_a2_training_authority`；不得带 placeholder implementation SHA，也不得暗示 source 已锁定。OOF 固定为三 seed 共用的五折 purged-forward partition、component `max(first_seen)`、至少 `30` 天 embargo、fit/holdout 每类至少 `1000/100` 行。custodian-attested `fold_scope_plan` 必须先由 `scripts/validate_loop164_fold_scope_plan.py` 生成绑定 contract/isolation receipt 的 canonical aggregate-only validation receipt，且该 pass 只冻结 scope，绝不授权训练或消费训练 lease。通过 isolation/scope 后，才可在独立静态审查中冻结真实 implementation manifest；未来 controller 必须先用 `scripts/validate_loop164_training_authority.py` 在同一进程内验证 repo 外 trust anchor 的固定 key、A2 v2 authorization、当前 Python/controller/argv、fresh resource guard、scope validation、真实 implementation manifest 和 train-only input bundle；随后以 `O_EXCL + fsync` 消费 content-addressed final lease，才可打开受保护训练输入。它不能接受任意 data path，auth 文件也不得从 `ready` 改写为 `consumed`。升级后的 `scripts/validate_loop164_nested_oof_execution_receipt.py` 必须后验复核 metadata provenance、该 v2 authority、input bundle、content-addressed marker/final lease，以及 `3 × 5` outer、每 outer `5` inner、Loop151/whole-file/fusion 全流程 OOF、六列 allowlist、零 identity/heldout access、零 silent drop。随后仍须补齐 Loop151-equivalent Train OOF、whole-file 资源预算与 fail-closed loader、项目原生 implementation review，并取得独立 A2 human authorization 和 machine authorization；该回执 pass 只开放 Train-OOF data boundary，绝不自动打开 Val、Test-10k 或 full-test。
   在请求阶段，`scripts/validate_loop164_a2_request.py` 只接受 `custodian_request_not_authorization` 模板：metadata request 只能保持 `draft`，training request 必须保持 `blocked_pending_metadata_and_static_review`。两者都必须 `authorization_granted=false`，不得携带 `decision`、签发时窗、runtime binding 或 lease；模板本身也不得放到 canonical authorization 路径。保管人只有在逐项补齐受控 binding 后，才能在 repo 外另行签发真正的 A2 authorization。
   最终双 sealed-window 证据必须再通过 `scripts/validate_loop164_certification_evidence.py`：它只读取 aggregate JSON，复核 power、同 bundle、A3 authorization/lease SHA、时间顺序、全分母 confusion matrix、200,000 次 component bootstrap、97.5% LCB、业务 gates 和不可 pooling/replacement 的 replication chain；A1 合成通过绝不构成 A3 结果。
5. 当前不得打开 raw、checkpoint、split row、prediction row、cache row 或 private map，不得安装依赖、构建 cache、训练、拟合、选阈值、运行 Val/Test-10k/full-test，亦不得生成 A2 execution lease、checkpoint 或 metrics artifact。
6. Loop157 独立 annotations 继续作为并行数据真值入口；返回仍为 `0` 时，不得自动 verdict、relabel、redraw 或把 reviewer 信息作为 runtime feature。time-causal trust/reputation 与 Nebula-style behavior 保留为后续 connected experts，不塞入 Loop164。
7. Loop164 若未来获批，先做三 seed、五折 group-aware Train OOF；只有 Val errors `<=152` 且 FP/FN 不高于 `105/57`，冻结后 Test-10k errors `<=73` 且 FP/FN 不高于 `49/29`，才允许另行申请一次 legacy full-test confirmation。旧 full-test 永不反向选择候选。
8. 同步生成 machine-readable preflight、recommendation status 和 experiment journal；所有未满足项必须显式显示为 blocker，不能因为 proposal 完整就宣称模型可执行或 F1 已提升。

P0 的 raw/native replay 仍未完成，但它不再垄断科研优先级；P0 部署工作与 P1-P3 的 A1 设计可以并行，重计算和晋级仍严格受 A2/A3 门禁约束。没有新 sealed 数据、独立标签、算力授权和 machine gate 时，任何 legacy 改善只能称 `development_observed`。

## 14. 对外表述

只有全部认证合同通过后，才能写：

> `Axon <version>` 在 `<sealed split version>` 上达到 `F1 point / one-sided 95% lower bound = ... / ...`，同时满足预注册 prevalence、FPR/FNR、coverage、temporal/family 和 adversarial 门槛，并已在第二时间窗口独立复验。

否则只能写 `development_observed` 或 `confirmation_passed`，禁止写“已达到 99.97%”“deployable best”或“生产级”。

## 15. Loop164 v2 implementation contract

在 full-pool isolation receipt 通过、但在训练 A2 authority 发出之前，Loop164 只能在独立静态审查中冻结生产 whole-file manifest；metadata contract 不得提前填充或声称该 SHA。`scripts/validate_loop164_whole_file_implementation.py` 的 future v2 manifest 必须固定 dedicated controller/model/streaming loader/OOF/fusion 的 source closure、config/runtime lock、257-token 或显式长度语义、all-bytes chunk coverage、exact independent-region pooling、bounded two-pass memory contract、五类 missingness、zero identity feature 和 required equivalence tests。它必须被 training authority、fresh resource guard 与 nested receipt 分别重新验证；每个 whole-file fit artifact 的 code/config/input 必须回链，且 `success + missing = denominator`、reason total = missing。A1 合成测试或 placeholder 不得生成、替代或解除真实 implementation-manifest blocker，也不得打开任何受保护输入或改变 F1 状态。

## 16. 本地负责人研发授权

用户已明确授权总负责人继续本地研发。该授权允许一个单独标记为 `user_directed_local_custody` 的工程可行性 probe：只能使用 canonical train role 内部形成的 SHA-bound 小型 bundle，必须不读取 Val、Test-10k、legacy full-test、sentinel、confirmation 或 certification；不得写 checkpoint、不得计算 F1/accuracy、不得选择阈值、不得复用 probe 权重。首个 probe 上限为每类 `128` 行、单 epoch、最多 `256` optimizer steps、单进程、`batch=1`、`accumulate=8`、30 分钟硬超时，并须完整记录 success/missing、两次全文件扫描、NaN/OOM、峰值资源和 source/config SHA。

该本地授权不等同于独立 custody：它可以解除“必须先有 public key 才能开始本地工程验证”的误解，但绝不能产生 `external_certification_eligible`、`certified_99_97`、Val/Test 结果或任何晋级资格。真正的 Train-OOF、选择窗、heldout confirmation 和双 sealed-window 认证仍分别受其原有 A2/A3 合同约束。

当前 A1 还允许 `tests/test_loop164_whole_file_gcg.py` 的 pure in-memory oracle：它只能证明预注册的小型 dense/chunk/GCG-toy 数学关系，不能替代未来源码闭包冻结后的真实 implementation oracle，不能访问文件句柄或受保护输入，也不能构成训练、真实吞吐、F1 或生产能力证据。

## 17. Loop164 本地工程 probe 实况

本地负责人授权下的首个 probe 已完成。canonical train-only bundle 为每类 `128` 行；全部 `256` 行成功，missing `0`，两遍全文件扫描和独立 SHA 验证均为 `512/512`，读取 `382001358` raw bytes，完成 `256` 个 backward microbatch 和 `32` 个 optimizer step。全程 timeout/OOM/nonfinite 均为 `0`；耗时 `14.092734s`，峰值 RSS `1745612800` bytes，CUDA allocated/reserved 峰值 `104785920/125829120` bytes。未写 checkpoint、模型状态或预测，未计算 F1/accuracy/概率，未执行阈值操作。

工程 probe receipt 是 `reports/roadmap_9997/loop164/local_feasibility_probe_receipt.json`，SHA-256 为 `cfb299be80c7c1b535d4bf1d61f86ddf76ddd78c94fd747d8c87158a0a8f15e1`。该结果只解除本机 whole-file streaming、exact pooling 和资源可行性疑问，不解除 A2/A3、Train-OOF、Val/Test 或认证门禁；Loop151 仍是唯一 research champion，`>=0.9997` 目标仍未达成。

## 18. Loop164 本地 Train-only OOF 诊断实况

用户随后明确授权总负责人在本地 custody 内继续推进。机器授权固定 `public_key_required=false`，并显式禁止 A2 training authority、Val/Test/full access、candidate promotion、checkpoint/model-state write 和 threshold selection。它只允许 canonical Train 前 `20000` 行的 one-seed、five-fold、one-epoch content-group diagnostic；本地授权不是外部 trust anchor，也不能替代生产 OOF 或 A3 sealed-window lease。

完整运行结果为：supported `19540/20000`、coverage `0.977`、固定阈值 `0.5` F1 `0.9620420176595961`、errors `748`、FP/FN `488/260`。将 `460` 个 missing 全部按错误计入时，保守 F1 为 `0.9400971932956461`、errors `1208`。五折 F1 范围为 `0.9496402877697842..0.9709694142042509`；posthoc descriptive ROC AUC 为 `0.9894106708508038`，但 `448` 个错误已经是高置信反向错误，不能把阈值微调当成主解。

工程合同完整通过：`195400` 次 source SHA pass、`78160` 个 backward microbatch、`9772` 个 optimizer step、OOM/nonfinite `0/0`，耗时 `2662.7030187s`。没有读取 heldout、写 checkpoint 或执行 threshold sweep。OOF report SHA 为 `da55531d39b628a2a02ec008451b7ad0455f6876cabd91dcb8c56f7e18c3e07f`；predictions SHA 为 `4f706788d812987714ebd9f717b77f75b10997309dbe7991c083b9928ad3d4df`；analysis SHA 为 `a6c0098231e9e358278061bb682410f6065cda9cb42aef80c951f100238e3c10`。

Decision: `stop_current_standalone_scale_preserve_oof_for_future_complementarity_audit_only`。禁止继续增加 standalone seed、epoch、threshold search 或 heldout run。先重建 decision-aligned Loop151 Train OOF，再用冻结的 score/uncertainty/missingness 做一次 cross-fitted complementarity gate；若不能以足够 precision 修复 Loop151 错误，关闭 Loop164。Loop151 仍是唯一 research champion，`>=0.9997` 目标仍未达成。

## 19. Loop165 代理互补性熔断与当前路线

Loop69 x Loop164 的一次 Train-only、无训练、无阈值搜索、无 heldout 代理审计已完成。两个快照各 `20,000` 行，仅有 `19,996` 个共同 SHA；索引 `1..4` 是真实替换，禁止按 index 强拼。共同样本 supported/missing 为 `19,540/456`。

代理 hard-decision overlap 为 repairs/breaks `75/595`、decision changes `670`、blind-switch precision `0.1119402985`、net error reduction `-520`；五个 Loop164 diagnostic folds 全部净负。成本熔断失败，所以不再为当前 Loop164 recipe 重建昂贵的 Loop151 exact OOF，也不增加 standalone seed、epoch、threshold 或 heldout run。

证据边界必须保留：Loop69 是旧 Loop61-style lineage，其 random folds 与 Loop164 content-component folds 不一致；`356/393` 个 non-singleton components 跨 Loop69 folds。因此正式 Loop151 complementarity gate 的状态是 `not_run / blocked_wrong_base_lineage_and_fold_scope`，不能写成正式失败或 lineage certification。当前决策是 `park_current_loop164_recipe_surrogate_negative_exact_loop151_gate_not_run`，而不是把未来所有 whole-file 方法永久关闭。

近期唯一主线改为：Loop151 保持唯一 research champion；并行建立独立 label-noise upper bound、time/family/source 数据合同、MalwarePT-style byte foundation、EMBER-v3 structural expert 与 DSRA control，再为 packed/parser-fail/high-uncertainty 尾部设计 time-causal trust 和 Nebula/Speakeasy-style behavior cascade。只有至少两个更强且正交的新专家在同一 outer partition 上显示稳定 Train-OOF 纠错后，才支付共享 decision-aligned OOF router 成本。`public key` 只属于未来 A2/A3 独立 custody/certification，不阻塞本地研发。

机器证据：`reports/roadmap_9997/loop165/loop69_loop164_surrogate_complementarity.json`，SHA-256 `d0aa06074f0123ba5a9ad89a31e3912dfde261eca71d97ed0f2df7d73b5c92ec`。长期 `F1 >= 0.9997` 目标保持不变且尚未达成。

## 20. Loop166 code-section foundation 启动

Loop166 主实验冻结为 MalwarePT-inspired scaled code-section BPE/MLM expert，EMBER-v3 novel-delta HGB 作为廉价结构 control。主模型只读 executable-section raw bytes；BPE tokenizer、MLM、classifier 和 aggregator 全部 outer-fit-only，任何 identity、PE/stat sidecar、旧模型 score、signer 或 heldout bytes 均不得入模。它不是对 86M MalwarePT 的复现，本机配方固定为约 `10-15M` 参数的资源受限诊断。

Phase A 已在 SHA-bound balanced Train-only `256` 行完成：success/missing `251/5`、coverage `0.98046875`、silent drop `0`；验证 `191000679` raw bytes，观察但不持久化 `104869232` code bytes，raw-code artifact `0`；耗时 `1.2943352s`、峰值 RSS `48603136` bytes。所有 missing 都是显式 `no_executable_section`，没有 parse、invalid-span 或 zero-raw failure。没有训练、模型、F1、threshold 或 heldout access，`public_key_required=false`。

当前允许的唯一重计算是一个 real outer tiny-MLM resource cell：BPE `1024+5`、sequence `512`、最多 `8` chunks/file、6-layer/384-hidden。单折 `>8h`、吞吐 `<2000 original-byte-equivalent tokens/s`、OOM/nonfinite、8GB VRAM 不足或不可恢复即停止，不扩五折。只有该资源门通过，才允许 one-seed five-fold Train OOF；Loop151 仍是唯一 champion，Loop166 只是 `phase_a_extractor_pass_tiny_mlm_pending`，`F1 >=0.9997` 仍未达成。
