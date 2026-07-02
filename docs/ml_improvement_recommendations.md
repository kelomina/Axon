# Axon 机器学习改进建议

更新时间：2026-07-02

## 2026-07-03 补充：Loop61 override-only classifier 未通过 Test-10k

Loop61 把 Loop57 的 FN gate 改成更窄的 override-only classifier：只在 locked base 判白、overlay-aware candidate 判黑的 possible override 行上训练 allow/block 分类器，仍然只允许 `0 -> 1`，不允许把 base 判黑改成白。该实验继续使用严格 OOF train 分数，Val 选择 classifier 和 allow threshold，身份字段只用于加载、对齐和审计，不作为模型证据。

Val 上 Loop61 确实超过 Loop57：从 Loop57 的 F1 `0.9926635724`、`147` errors、FP/FN `92/55`，提升到 F1 `0.9930139721`、`140` errors、FP/FN `90/50`。但冻结 Test-10k 没有超过当前 best：Loop61 为 F1 `0.9897816069`、`102` errors、FP/FN `62/40`，与 Loop57 的 `102` errors 持平，只是把 FP `-3` 换成 FN `+3`。因此按漏斗规则拒绝 full-test，Loop57 仍是当前 best full-test reference。

这个结果说明 override-only 分类方向能削减一部分新增 FP，但 possible override 训练样本太少：本轮 train possible override 只有 `160` 行，其中真修复 `54`、新 FP `106`；Val possible override 只有 `130` 行，其中真修复 `42`、新 FP `88`。继续在同一批 score/overlay gate 上薄调很容易过拟合 Val。下一步应转向更强的内容证据或回到噪声/源标签审计，而不是把 Test-10k tie 强行推进 full-test。相关记录见 `docs/phase3_loop61_override_classifier.md`。

随后补做了两件只读审计。第一，Loop61 exchange audit 解释了 Val/Test-10k 不一致：Val 上 Loop61 相对 Loop57 是 `13` 个修复对 `6` 个新增错，Test-10k 变成 `6` 对 `6`，净收益归零。第二，Loop62 尝试把匿名 content matrix 加入 override-only classifier，但 Val 退到 `148` errors、FP/FN `87/61`，说明高维内容特征在百级 possible override 行上更像过拟合或过度保守，不进 Test-10k。

Loop63 已把路线切回噪声/数据复核：基于当前 best Loop57 full-test 生成持久错误队列，`160000` test 行中仍有 `1868` 个错误，其中 `1760` 个是 Loop28 与 Loop57 都错的持久错误，`108` 个是 Loop57 新增错误；更关键的是 `643` 个 current-best 错误与 Loop39 高置信冲突队列重合。这个量级本身已经远超 `F1 >= 99.9%` 的容错空间，因此下一阶段优先级应是人工/外部证据复核这些持久高置信冲突，而不是继续薄调同一类 sparse gate。Loop63 只用于目标可行性和复核排序，不允许作为训练、阈值选择或自动改标依据；若确认坏样本，仍必须 fresh same-label redraw，保持严格 `200000`。

Loop63 A-lane `643` 条又复用了 Loop50 content/cache health audit：cache/source SHA、NPZ shape、active split、manifest、strict PE parse 均未发现客观可自动替换的问题，`objective_issue_row_count=0`，仅有 `5` 条重复 SHA 组需要按内容组复核。因此不能自动删除或改标这些样本；它们只能进入人工/外部证据判定，或者被记录为当前模型盲区。

## 2026-07-02 补充：命名不是证据，content PE v1 已产品化

最新硬规则已经固定：文件名、路径、扩展名、目录名、`source_sha256`、`cache_path`、`sample_index`、`split` 和行顺序只能用于加载、缓存对齐、覆盖审计、去重、人工复核、以及生成一次性的人工标签清单，不能作为模型特征、二阶段融合特征、阈值捷径、自动改标证据或上线推理依据。原因是实战文件命名和训练集命名完全不是同一个分布，且攻击者改名几乎没有成本；训练集目录只能说明人工当时把样本放进哪个标签桶，不能说明文件本身因名字而恶意或良性。

当前 `docs/identity_feature_policy.md` 已把这条写成策略文档，`scripts/identity_feature_guard.py` 已接入 Stage-2 cache matrix 和 OOF stacker。后续如果新增类似 filename/path/extension/hash/split/row-id 的派生特征，训练脚本应直接失败，而不是让这类身份线索进入模型。

Loop52 已把 Loop28 的 100 维 content PE metadata 从 Stage-2 临时脚本产品化为 `src/kvd_features/content_pe_v1.py`，并让 Stage-2 训练矩阵和 sidecar cache builder 都引用同一份稳定 schema。这个 schema 只从文件字节和 PE 结构中提取 header、data directory、import/export/resource、overlay、section 权限/熵等内容信号；路径参数只用于打开文件，不编码 filename、path、extension、directory、hash、split 或行号。新增测试要求同一内容在不同文件名下提取结果完全一致，并要求全部 feature names 通过 identity guard。32 条限量 smoke 必须显式使用 `--smoke --limit`，只验证 extractor/cache writer 链路，`feature_dim=100`、`created=32`、`zero_features=0`，不作为 Val 指标，也不触碰 Test-10k；正式 sidecar cache 报告必须满足 `"limit": null` 且 `"unique_rows" == "deduplicated_rows_before_limit"`。相关记录见 `docs/phase3_loop52_content_pe_v1_productization.md`。

Loop53 已完成产品化回归复验。新的 cache builder 会打开已有 `.npz` 检查 100 维 shape 和 finite 数值，正式 Train/Val audit 为 `40000/40000`、`limit=null`、`smoke=false`、`refreshed_invalid=0`、`zero_features=0`。随后用产品化后的 `content_pe_v1` 复跑 Loop28 Val-only Stage-2，仍选中 `hgb_lr0.06_leaf31_l2_0__noise_none`，Val F1 `0.9919048571`、`162` errors、FP/FN `87/75`，`test=null`。这说明抽成稳定模块没有改变模型语义，也没有引入命名/路径/hash 泄漏。相关记录见 `docs/phase3_loop53_content_pe_v1_replay.md`。

Loop54 复测了严格 OOF residual gate，并启用 `--gate-content-features` 让 gate 看到内容矩阵。结果没有超过 Loop28：最佳 `extra_trees_300_leaf1 + gate_logreg_balanced_c0.25` 只有 Val F1 `0.9917676994`、`165` errors、FP/FN `104/61`，比 Loop28 locked reference 的 `162` errors 还差 `3`，更远未达到浅 gate/blend 候选进入 Test-10k 所需的约 `<=152` errors。因此拒绝 Test-10k。结论是：继续围绕同一批 score/gate 参数做薄 margin 调参价值不高；下一步应转向 security directory 与真实 overlay payload 边界、DLL/export/exception/TLS 组合、signed/overlay 恶意 FN 与复杂正常 FP 的内容结构区分。相关记录见 `docs/phase3_loop54_content_aware_oof_gate.md`。

Loop55 针对 security directory 与真实 overlay payload 边界做了窄内容特征探针。它正确处理 PE Security Directory 的文件偏移语义，从 overlay 中扣除证书 blob 后计算真实 payload segment、payload 熵、payload 是否在证书后、与最后 section 的间隙等 32 维特征。Train/Val sidecar cache 完整 `40000/40000`，`zero_features=43`；测试确认同一内容不同文件名输出一致，feature names 通过 identity guard。完整 Val-only 结果未过门槛：最佳 `hgb_lr0.06_leaf31_l2_0__noise_none` 为 F1 `0.9913208300`、`174` errors、FP/FN `111/63`，比 Loop28 多 `12` 个错误。它降低 FN 但明显增加 FP，因此拒绝 Test-10k。结论是：overlay/security boundary 信号合法但直接拼接会变成 FP/FN 交换；后续若复用，只适合作为残差分层或极保守 gate 的辅助信号。相关记录见 `docs/phase3_loop55_overlay_boundary_valonly.md`。

Loop56 对 Loop55 做了只读错误交换审计，不训练、不调阈值、不触碰 Test。完整 Val `20000` 行中，Loop28 和 Loop55 都正确 `19795`，Loop28 错而 Loop55 对 `31`，Loop28 对而 Loop55 错 `43`，两者都错 `131`；净增 `12` 个错误，正好解释 Loop55 的 `174` errors 对 Loop28 的 `162` errors。迁移上，Loop55 修复 `20` 个 FN 和 `11` 个 FP，但新增 `35` 个 FP 和 `8` 个 FN。修复组的 `overlay_boundary_payload_log_size`、`overlay_boundary_overlay_log_size`、`overlay_boundary_gap_last_section_to_security_log` 明显更高，说明 overlay/security boundary 是真实内容信号；但它同时会误伤复杂正常 PE，因此不能 standalone 进入 Test-10k。后续若复用，只能做极保守 FN-specific residual gate 或残差分层，且身份字段仍只允许用于对齐和 cache lookup。相关记录见 `docs/phase3_loop56_loop55_overlay_exchange_audit.md`。

Loop57 把 Loop56 的结论转成严格 FN-specific overlay gate：只允许 `0 -> 1`，即只在 Loop28 判白时尝试修复漏报，没有路径把 base 判黑改成白。它使用 5-fold OOF 训练 base/candidate/gate，Val 选择 `extra_trees_300_leaf1 + gate_logreg_balanced_c0.25`，candidate threshold `0.515`、gate threshold `0.88`。Val 从 Loop28 的 `162` errors 降到 `147`，Test-10k 从 `111` 降到 `102`，因此进入 16 万 full-test；full-test 从 Loop28 的 F1 `0.9878358558`、`1949` errors、FP/FN `1087/862` 改进到 F1 `0.9883629658`、`1868` errors、FP/FN `1195/673`，净少 `81` 个错误。结论：Loop57 是当前 best full-test reference，适合作为低漏报/安全偏置候选；但它通过增加 `108` 个 FP 换来减少 `189` 个 FN，仍远未达到 `F1 >= 99.9%`。下一步应重点削减 Loop57 新增 FP，同时保留大部分 FN repair。相关记录见 `docs/phase3_loop57_fn_overlay_gate.md`。

Loop58 对 Loop57 做了只读 full-test exchange audit。完整 `160000` 行中，Loop57 修复 Loop28 错误 `189` 行，新增错误 `108` 行，两者全部来自 FN gate 的 `0 -> 1` 覆盖；没有隐藏的 `1 -> 0` 或其它预测变化。内容归因显示，新增 FP 的 `overlay_boundary_payload_log_size` 和 payload entropy 更高，但 `overlay_boundary_payload_after_cert_log_size` 与 `payload_after_security` 基本为零；修复 FN 更常带有 security/cert 后 payload 结构。结论是：Loop57 的问题不是 gate 方向错，而是缺少 FP guard；下一轮应回到 Train/Val 验证“高熵大 payload 且无 after-cert/after-security 证据”的二级过滤，不能直接从 full-test 审计里拿阈值。相关记录见 `docs/phase3_loop58_loop57_full_exchange_audit.md`。

Loop59 按 Loop58 的方向做了 Val-only FP guard 快速探针。手工测试了 payload size、overlay size、payload entropy、security size、after-cert、after-security、last-section gap 及若干组合规则，只使用 content-derived overlay boundary features，不触碰 Test。最佳规则仅把 Loop57 Val 从 `147` errors 降到 `146`，只拒绝 `1` 个覆盖；其它规则要么不动，要么损伤 FN 修复。该 margin 太薄，按 Loop37 经验不允许进入 Test-10k。结论：不要继续手工拧 overlay 阈值；若要削减 Loop57 新增 FP，应训练 OOF 二级模型，并加入 import/resource/section 等正常软件结构信号。相关记录见 `docs/phase3_loop59_fp_guard_val_probe.md`。

Loop60 将 Loop57 的 gate 扩展为 content-aware gate，新增 `--gate-content-features`，让二级 gate 除 score + overlay 外还能看到匿名化 content/cache 矩阵。默认 Loop57 兼容路径保持不变，新增 content aliases 通过 identity guard 检查。Val-only 结果没有超过 Loop57：最佳仍是 `147` errors，但 FP/FN 从 Loop57 的 `92/55` 变成 `95/52`，更偏低漏报但误报更多。因此拒绝 Test-10k。结论：泛化 content 矩阵直接喂给 gate 没有解决新增 FP；下一步应更明确地建模 benign-like import/resource/section 结构，或做 override-only classifier。相关记录见 `docs/phase3_loop60_content_aware_fn_gate.md`。

Loop48 对 fresh seed43 current-split checkpoint 做了 Val-only 复验。它使用 `config/random_20w_8192_seed43.toml`，保持当前 `loop27_corrected_split.csv`、fixed-v2、8192-byte、PE 256、stat 49 的同一口径；split/cache 复审仍为 `200000/200000` 覆盖、missing `0`。1 epoch smoke Val F1 为 `0.8158`；完整训练保留的 `models/random_20w_8192_seed43/best_model.pt` 在 epoch `17` 的 best Val F1 只有 `0.9500494559841741`，且没有 final checkpoint。该结果远低于 Loop28 content PE metadata 的 Val F1 `0.9919048571`，因此 seed43 未进入 Test-10k，也不导出给 Stage-2 stacker 使用。相关记录见 `docs/phase3_loop48_fresh_seed43_valonly.md`。

这个反例很关键：Loop47 说明现有 checkpoint 池不能安全直接 stack；Loop48 又说明“只换 seed 重训同款 8192 fixed-v2 神经底座”不是高价值路线。下一阶段不应继续盲目堆同款 seed，而应把 Loop28 content PE metadata 产品化为稳定 schema，同时训练真正多样化的 current-split base：不同输入长度或区域视角、不同模型结构、不同内容特征族，全部用 OOF 协议和 Val gate 验证。

噪声问题仍然要作为主线处理。Loop38/Loop39 的高置信冲突队列不能靠命名猜标签，也不能自动改标；一旦人工确认样本 `feature_broken`、`out_of_scope` 或 `label_wrong`，必须从同标签候选池 fresh re-draw 替换，并重新生成完整 `20000 train / 20000 val / 160000 test` split 与 cache readiness。坏样本不补齐，坏样本只触发重新抽样。

Loop49 又补了一次产品化前置审计：当前 fixed-v2 PE 主 schema 配置维度是 `256`，实际使用 `143` 维，保留位 `113` 维；Loop28 的 100 维 content PE 特征中只有 `20` 个能和 fixed-v2 形成已有/部分覆盖，仍有 `80` 个产品化缺口。高价值缺口集中在 data directory size/ratio、header flags、section permission combo、layout ratio、import shape、overlay 和 resource shape。结论是：Loop28 的收益不是命名泄漏，而是主 schema 尚未稳定吸收的内容侧 PE 结构信号。下一步应把这些信号迁入明确版本的稳定 schema，例如 `fixed_v3` 或 `content_pe_v1`，再进入 Train/Val 漏斗；不能悄悄复用 reserved 位让旧 checkpoint 语义变得模糊。相关记录见 `docs/phase3_loop49_content_pe_productization_audit.md`。

Loop50 对 Loop39 的 `649` 条高置信冲突做了只读内容/缓存卫生审计：全部仍在 active `loop27_corrected_split.csv` 和当前 manifest 中，cache NPZ shape/dtype/finite、cache label/source SHA、源文件 SHA、strict PE parse 均未发现可直接支持 `feature_broken/out_of_scope` 的客观问题；`objective_issue_row_count=0`。仅有 `5` 行落入重复 SHA 组，需要人工按内容组复核。结论是：这些冲突不能自动改标或自动替换，必须继续依赖人工/外部证据；若未来确认坏样本，仍然按 fresh same-label re-draw 保持严格 `200000`。相关记录见 `docs/phase3_loop50_conflict_content_audit.md`。

Loop51 已把“语义区域字节交给神经模型”推进到可训练状态：基于 PE/content-derived offsets 生成了 region-view byte cache，只覆盖 train/val，严格不生成 Test cache、不跑 Test-10k。结果为 `40000/40000` train/val rows，train `20000`、val `20000`、黑白各 `20000`，issue_counts `{}`。512/512 的 1 epoch neural smoke 能完整通过 Trainer，证明链路可用；随后完整 train/val 训练到 epoch `3` 后 best Val F1 只有 `0.9448912132`，明显低于 Loop28 的 `0.9919048571`，因此提前停止且不进入 Test-10k。结论是：region-view neural 不能作为 standalone 替代主线；后续若复用，只能作为真正多样化 OOF 辅助视角，并且必须先证明残差互补。相关记录见 `docs/phase3_loop51_region_view_cache_and_smoke.md`。

## 2026-07-01 补充：20w 严格漏斗实验后的最新判断

在 20 万完整 split 上，当前最强可复现实验是 Loop28 的 Stage-2 content PE metadata 模型。它严格保持 `20000 train / 20000 val / 160000 test`，每个 split 内黑白样本平衡不变，fixed-v2 uncompressed cache 覆盖 `200000/200000`。Val 只用于选模型和阈值；Test-10k 只做确认；full-test 只做一次冻结评估。Loop28 新增的 100 维 PE metadata 只来自文件内容和 PE 结构，不包含 filename、extension、目录名或路径文本。

补充硬规则：文件名、路径、扩展名、目录名、`source_sha256`、`sample_index`、`split` 和行顺序只能用于装载、对齐、审计、去重、人工复核或生成一次性的标签清单，不能作为模型特征或调参依据。原因很简单：实战文件名和训练集命名通常完全不同，而且攻击者可以随时改名。训练集目录最多表示“人工把这批样本放在哪个标签桶里”，不是“这个文件因为叫某个名字所以恶意/良性”的证据。本轮已新增 `scripts/identity_feature_guard.py`，并接入 Stage-2 cache matrix 与 OOF stacker，后续训练矩阵若出现路径/命名/扩展名/hash/split/row-id 派生特征会直接失败。

最新结果如下：

| 方案 | Val F1 | Test-10k F1 | Full-test F1 | Full-test errors | FP / FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Loop24 blend | 0.98820 | 0.98423 | 0.98323 | 2685 | 1379 / 1306 |
| Loop26 blend | 0.98886 | 0.98558 | 0.98397 | 2571 | 1493 / 1078 |
| Loop27 blend | 0.98891 | 未进入 | 未进入 | 未进入 | 未进入 |
| Loop28 content PE | 0.99190 | 0.98887 | 0.98784 | 1949 | 1087 / 862 |

Loop26 是有效改进：full-test 错误从 `2685` 降到 `2571`，少了 `114` 个错误。Loop27 只替换了 2 个高置信 Val 噪声样本，Val 只少 1 个错误，F1 提升约 `0.005` 个百分点，未达到“明显提升”门槛，因此没有进入 Test-10k。Loop28 证明“补内容侧 PE metadata”是当前最有效方向：full-test 错误从 Loop26 的 `2571` 降到 `1949`，少了 `622` 个错误；Test-10k 也从 Loop26 blend 的 `144` errors 降到 `111` errors。Loop28 同时复验了 content PE + OOF kNN，但 Val 最优为 `165` errors，低于 content-only 的 `162` errors，因此不采用 kNN 叠加。

要达到 `F1 >= 99.9%`，16 万 full-test 上大致只能容忍百级错误；当前 Loop28 仍有 `1949` 个错误，量级差距仍然很大。这个差距不能靠阈值微调解决，因为 Loop28 full-test AUC 已达 `0.99898`，但 FP/FN 仍然都有明显存量。换句话说，模型不是“分数刻度差一点”，而是在一批文件上真的看错了。

错误集中形态也很清楚：

| 维度 | Loop26 full-test 错误集中点 |
| --- | --- |
| 当前数据中表现为无扩展名的白样本 | `<none>` 扩展错误 `1261`，其中 FP `1234` |
| 恶意 exe | `.exe` 错误 `987`，其中 FN `729` |
| 恶意 DLL | `.dll` 错误 `306`，其中 FN `305` |
| 月份热点 | `2026-03`、`2020-11`、`2021-09`、`2026-02` 等恶意 FN 热点明显 |

Loop28 已经压低这些热点，但没有消灭：full-test 剩余错误中 `<none>` 仍有 `887` 个错误（FP `849`），`.exe` 有 `831` 个错误（FN `594`），`.dll` 有 `218` 个错误（FN `217`）。这些切片只能用于错误归因和采样审计，不能作为生产模型输入，也不能拿来做阈值或融合权重的捷径。

Loop29/Loop30/Loop31 复验说明几条近路暂时不成立：Loop28 + Loop27 的三路浅融合在 Val 上从 `162` errors 降到 `147`，但冻结 Test-10k 为 `112` errors，未超过 Loop28 content-only 的 `111` errors；宽泛二进制字符串/关键词特征在 Val 最好 `167` errors；Authenticode certificate blob 特征在 Val 最好 `168` errors，也弱于 Loop28。因此下一步不应继续堆浅融合、粗粒度字符串关键词或浅证书 blob 指标，而应围绕 Loop28 残差做更有针对性的内容 schema。

2026-07-01 又补充了 Loop32/Loop33。Loop32 先新增 `scripts/analyze_stage2_residual_content.py`，只读取冻结预测和 content sidecar cache 做残差归因，不训练、不扫阈值、不使用 filename/path/extension 作为模型输入。归因显示：Loop28 的 FN 在 signed/security directory、overlay、export、DLL、exception/debug、非 32-bit/large-address-aware 结构上更集中；FP 在 system DLL ratio、import count、section entropy、RW section 和大文件上更集中。随后新增 `182` 维 content PE v2 sidecar，覆盖更细的 import DLL/API、export、resource tree、section/entrypoint 结构。Train/Val v2 cache 覆盖 `40000/40000`，零特征 `0`。但 Val 结果没有通过门槛：Loop32 `content PE v1 + v2` 最好 `170` errors，Loop33 `v2 only` 最好 `192` errors，均弱于 Loop28 的 `162` errors，因此二者都没有进入 Test-10k。结论是：残差信号真实存在，但“宽撒式继续堆 PE 细特征”当前会引入冗余和噪声；下一轮应做更窄的特征组选择、OOF stacking 或更高质量 PE/证书解析，而不是简单增加维度。

同时做了后向兼容复验：新增 v2 代码后，旧 Loop28 冻结模型在锁定 Test-10k 上仍为 F1 `0.9888677164`、`111` errors、FP/FN `61/50`，说明旧模型 replay 没有被新代码破坏。相关文档见 `docs/phase3_loop32_33_residual_and_content_pe_v2.md`。

随后 Loop34/Loop35 对 v2 做了子组选择，新增 `--content-pe-v2-groups`，分别测试 `import_dll`、`api`、`imports`、`export`、`resource`、`section` 以及若干组合。第一轮 9 个组合全部使用完整 `20000 train / 20000 val`，但只限制候选模型范围；最好的 `imports`、`export`、`export,section` 都是 `164` Val errors。第二轮对这三个近门槛组合跑完整默认候选矩阵和全部现有 noise modes，最佳仍然停在 `164` errors，未打过 Loop28 的 `162`。因此 Loop34/35 也没有进入 Test-10k。这个结果进一步收窄结论：不是 v2 太宽才失败，当前 v2 子组本身也没有形成可验证收益。相关文档见 `docs/phase3_loop34_35_content_pe_v2_group_selection.md`。

2026-07-02 又补了 Loop36 严格 OOF stacker。这个实验用 5 折 train 内 OOF 预测训练 meta model，并启用 `--drop-base-prob-features`，去掉 Stage-2 矩阵前 6 个非 OOF 的导出概率特征，避免 train 内概率泄漏。base learners 是 3 个 HGB 变体，meta candidates 包含 logistic 和小 HGB。最佳 Val 为 F1 `0.9917594766`、`165` errors、FP/FN `94/71`，仍弱于 Loop28 的 `162` errors，因此没有进入 Test-10k。结论是：OOF stacking 协议本身正确，但只堆同一矩阵上的几个相似 HGB base learner 不够；下一次 stacking 必须引入真正多样的 base 预测，比如不同神经 checkpoint/seed、不同字节长度或独立特征族。相关文档见 `docs/phase3_loop36_oof_stacker.md`。

同日又做了 Loop37 byte n-gram 融合。byte n-gram 是内容侧弱模型，单独 Val 为 `1250` errors，但和 Loop28 错误重合低。Val-only 融合 `0.8 * Loop28 + 0.2 * byte_ngram`、阈值 `0.48` 从 `162` errors 降到 `159` errors；冻结 Test-10k 也从 `111` errors 降到 `110` errors。但 16 万 full-test 反转为 `1960` errors，差于 Loop28 的 `1949` errors，因此拒绝。这个实验的价值是证明：1-3 个样本级别的 Val/Test-10k 改善太薄，很容易是抽样波动；后续候选必须争取更宽的 Val/Test-10k margin，或者全量通过后才算真正改进。相关文档见 `docs/phase3_loop37_byte_ngram_blend.md`。

随后 Loop38 对 Loop28 full-test 残差做了噪声和多模型重合审计，不训练、不调阈值。Loop28 的 `1949` 个 full-test 错误中，`910` 个落入高置信冲突或近阈值可疑桶，`649` 个属于 severe/high confidence conflict。另一方面，`921` 个 Loop28 错误至少被 Loop37、byte-ngram 或 Loop26 blend 中的一个纠正，其中 FP `385`、FN `536`。这说明可学习残差和噪声/边界上限同时存在：有一批错误可以被其它视角修复，但还有大量高置信冲突不是当前候选能解决的。相关文档见 `docs/phase3_loop38_residual_noise_strata.md`。

Loop39 已把这 `649` 个 severe/high confidence conflict 转成手工复核队列：FP `416`、FN `233`；其中 `501` 条没有被任何对比模型修正，`148` 条至少被一个候选模型修正。队列中的人工结论和推荐动作字段全部留空，不做自动改标。队列还显式标出 `2` 个重复 SHA 内容组、`2` 条额外重复行，复核时必须按内容组处理。若人工确认 `feature_broken`、`out_of_scope` 或 `label_wrong`，不能用这些样本“补齐”，只能从同标签候选池重新抽取新样本替换，并保持总量严格 `200000`。相关文档见 `docs/phase3_loop39_conflict_adjudication_queue.md`。

Loop40 为 Loop39 队列补了只读 replacement preflight。真实队列复验显示当前 split 仍严格满足 `200000 = 20000/20000/160000` 和各 split 黑白平衡，但 `649/649` 条 manual verdict/action 仍为空，因此 `preflight_status=blocked_no_verdicts`、`preflight_ok=false`。这正是期望行为：没有人工证据时不能重抽、不能替换、不能构建 corrected split；一旦人工确认坏样本，必须先用 fresh same-label candidate pool 通过预检，再进入 corrected split 和 cache readiness。相关文档见 `docs/phase3_loop40_loop39_replacement_preflight.md`。

Loop41 继续做了不触碰 Test 的 stronger byte n-gram Val-only 复验。更大 hash 空间、更密 stride、更多 epoch 和多 alpha 后，byte n-gram 独立 Val 错误从 Loop37 的 `1250` 降到 `944`，说明这条内容弱模型确实能变强；但和 Loop28 做细粒度 Val-only 融合时最好仍是 `159` errors，只比 Loop28 的 `161/162` errors 少 `2-3` 个，margin 太薄。基于 Loop37 已经证明小样本改善会在 full-test 反转，本轮不进入 Test-10k。相关文档见 `docs/phase3_loop41_stronger_byte_ngram_valonly.md`。

Loop42 按 Loop38 的建议实现了严格 OOF residual gate：base/candidate 在 train 内产生 OOF 分数，gate 只用 train OOF 信号训练，Val 只选 gate model 和阈值，Test-10k/full-test 都没有使用。最佳结果是 `extra_trees_300_leaf1 + gate_logreg_balanced_c0.25`，Val `160` errors、FP/FN `98/62`，比 Loop42 内部 base 的 `180` errors 少 `20`，说明受控覆盖方向确有信号；但相比官方 Loop28 锁定 Val `162` errors 只少 `2` 个，未达到浅融合/覆盖类候选进入 Test-10k 的 `<=152` errors 门槛，因此拒绝 Test-10k。相关文档见 `docs/phase3_loop42_oof_residual_gate.md`。

Loop43 继续验证内容侧更窄交叉特征，而不是再宽泛堆 v2。它基于已有 content PE v1/v2 cache 追加 `66` 个内容派生交叉特征，覆盖 DLL/driver、security/overlay、section/entropy、import/API、resource/export 等残差主题。fast Val probe 最好 `176` errors；完整候选矩阵和 noise modes 后最好 `172` errors、FP/FN `107/65`，仍比 Loop28 多 `10` 个错误，因此拒绝 Test-10k。结论是：手工乘法式内容交叉没有形成收益，下一步应转向真实 Authenticode/签名覆盖解析、regionized byte n-gram 或 parser-quality 改进。相关文档见 `docs/phase3_loop43_content_cross.md`。

Loop44 已验证 regionized byte n-gram。它不再只看缓存前缀，而是从 PE 内容中定位 head/tail、entrypoint、resource/import/export/security directory、first exec section、last section、max-entropy section 和排除证书 blob 后的 overlay payload，再对这些区域做带 region salt 的 hashed n-gram。路径、文件名、扩展名、目录、hash、sample index、split 和行顺序仍只用于打开文件、对齐和审计，不作为模型特征。完整 `20000 train / 20000 val` Val-only 结果显示：standalone Val `596` errors、F1 `0.97031`，比 Loop41 stronger prefix byte n-gram 的 `944` errors 明显更强；但和 Loop28 做 Val-only 融合时最佳仍只有 `161` errors，和 Loop28 在同一扫描里的 `161` errors 持平，未形成 margin。因此 Loop44 拒绝 Test-10k。结论是：语义区域字节确有互补信息，但浅 SGD 弱模型不足以修复 Loop28；若复用这条信号，应优先做严格 OOF residual gate 或更高质量 parser 特征，而不是直接扩大 hash 空间。相关文档见 `docs/phase3_loop44_region_byte_ngram_valonly.md`。

Loop45 已把 Loop44 的 regionized byte n-gram 放进严格 OOF residual gate：base 与 region candidate 的 train 分数都使用 5 折 OOF，gate 只在 train OOF 上学习“candidate 何时能纠正 base”，Val 只选择 gate 模型和阈值。完整 `20000 train / 20000 val`、cache miss `0`。最佳 `region_byte_ngram_sgd + gate_logreg_balanced_c0.25` 的 Val 为 F1 `0.9905113863`、`190` errors、FP/FN `107/83`，比 gate 内部 base 少 `3` 个错误，但仍比 Loop28 锁定参考多 `28` 个错误；train OOF 中 beneficial overrides 只有 `92`，harmful overrides 有 `676`，说明这条弱模型信号不够干净。Loop45 因此拒绝 Test-10k，立即的 region n-gram OOF gate 分支也应暂停。相关文档见 `docs/phase3_loop45_oof_region_gate.md`。

Loop46 按策略重审建议转向新信息源：真实一点的 Authenticode/ASN.1 结构解析。它只读取 PE Security Directory 的 WIN_CERTIFICATE/PKCS#7 内容，新增 ASN.1 parse 成功/异常、sequence/set/context/OID/time/string 聚合、标准 OID presence 等 `63` 维结构特征，不使用 filename/path/extension/hash/id/split。完整 `20000 train / 20000 val`、cache miss `0`，签名结构覆盖 train `6815`、val `6936`。最佳 `hgb_lr0.08_leaf31_l2_1e-3__noise_none` 为 Val F1 `0.9909891870`、`180` errors、FP/FN `78/102`，比 Loop28 多 `18` 个错误，也弱于 Loop31 浅证书 blob 的 `168` errors，因此拒绝 Test-10k。结论是：证书结构解析本身没有解决 signed-file residual，除非接入真实信任链/吊销/时间戳验证或人工证据，否则不建议继续做小幅证书字段变体。相关文档见 `docs/phase3_loop46_cert_structure_valonly.md`。

Loop41-46 已触发停滞熔断和策略重审：连续多轮没有候选达到 `<=152` Val errors 的 Test-10k 门槛，最好也只是 Loop41 的 `159` errors，只有 `2-3` 个错误级别的改善。应暂停继续围绕 prefix/region byte n-gram、浅融合、同款 OOF gate、手工内容交叉、浅/结构证书追加做微调。后续必须换成真正不同的信息源、真实多 checkpoint/多 seed OOF、多字节长度神经模型，或带人工证据的数据清洗，而不是再扩大 hash 空间、调 stride/alpha/epoch 或重复堆 gate。

Loop47 对现有 `models/` 做了 checkpoint provenance 审计，用来判断是否能直接进入“多 checkpoint / 多 seed OOF stacking”。结论是否定的：共扫描 `177` 个 `.pt` checkpoint，只有 `1` 个能明确归属当前 `loop27_corrected_split.csv`、fixed-v2、8192-byte random 20w 口径，即 `models/random_20w_8192/best_model.pt`；其余为 `provenance_mismatch=2`、`incompatible=40`、`unknown=134`。因此，当前仓库没有可安全直接 stack 的多 checkpoint 池。若要做真正多样化 OOF stacking，必须重新训练当前 split 的新 seed / 新 byte length checkpoint，不能拿旧 group-isolated、comparison-cache、hard replay 或未知来源 checkpoint 混入当前 Val/Test 漏斗。相关文档见 `docs/phase3_loop47_checkpoint_provenance_audit.md`。

因此，下一阶段 P1 不应继续把主要时间花在“再替换少量 Val 噪声样本”上，而应转为三个方向：

1. **把 Loop28 content PE metadata 正式产品化，但不要继续在当前 v2 上排列组合。** 当前大量白样本在数据集中表现为 SHA 文件名或无扩展名，但实战文件名可被任意改写，所以 filename/extension 只能作为错误分析切片，不能作为生产模型输入。Loop28 已证明 PE 内容侧信号有效；Loop32-35 又证明“直接追加一大包细特征”以及“把这包细特征拆子组”都不够稳。下一步应把 Loop28 的 100 维内容特征并入稳定 schema，同时转向 OOF stacking 或更高质量解析，而不是继续消耗轮次在 v2 group permutation 上。
2. **补 DLL/sys 恶意召回特征，但必须窄口径验证。** DLL 和 sys 的恶意 FN 表明当前 PE/stat/8192 字节头部特征对库文件、驱动类样本仍不够敏感。建议继续研究 exports、subsystem、service/driver hints、section 权限组合、TLS、relocation、import category 和 driver/service 相关 API 特征，但每次只引入一个明确子组，先过 Val 的 `162` errors 门槛，再进入 Test-10k。
3. **从 Stage-2 过渡到真正多样化的 OOF stacking。** Loop36 已经证明“协议正确但 base learner 太相似”的 OOF stacker不够。下一步若继续 stacking，应训练多 seed/多视角 base 模型，使用 out-of-fold 预测训练校准器/stacker，避免单模型盲区被 Stage-2 继承；不要再只堆同一 cache matrix 上的 HGB 小变体。
4. **建立高置信冲突人工复核/替换队列。** Loop38 显示 severe/high confidence conflict 足够多，已经影响 `99.9%` 目标的可行性。这里不能自动改标签，也不能用文件名猜标签；只能生成人工判定队列。若确认为坏标签或坏文件，必须按同标签候选池重新抽样替换，保持总量严格 `200000`。

当前科学判断：`99.9%` 不是短期阈值、浅融合或小清洗能达到的指标。Loop28 把 full-test F1 推到 `0.98784`，说明内容特征方向正确，但距离 `0.999` 仍有约 `1900` 个错误的缺口。Loop38 又显示 `649` 个错误是 severe/high confidence conflict，不能假设全都可由模型无监督学掉。合理阶段目标应先定为 `99.0%+` full-test F1，并把 FP/FN 分业务成本分别设门槛。继续挑战 `99.9%` 可以保留为长期目标，但需要把“数据标签可信度”“PE/DLL/sys 内容覆盖”和“真正多样化 OOF stacking”作为前置工程。

## 先给结论

Axon 现在最值得优先改进的地方，不是马上把模型做得更复杂，而是把“模型到底好不好”的判断流程固定下来。当前项目已经有很多有价值的能力：DSRA 字节分支、PE 结构特征、stat 统计特征、相似族群隔离切分、小族群加权、阈值扫描、hard-example 微调、GA 特征筛选、错误分析、概率校准脚本。问题在于这些能力目前像一套工具箱，工具很多，但每次评估时使用的样本口径、阈值、checkpoint、hard holdout 和报告格式并不完全统一。对产品决策来说，这会带来一个风险：某个实验看起来 F1 更高，但可能只是换了更容易的测试口径，或者它减少了漏报，却悄悄增加了很多误报。

用业务语言说，下一阶段的核心目标应该是：让每个候选模型都经过同一套“考试”。这套考试要同时看总体测试集、相似族群隔离后的泛化能力、hard false negative 样本、hard false positive 样本、阈值选择，以及不同恶意来源目录或家族代理上的表现。只有这样，我们才知道一个模型是真的更稳，还是只是对某一批样本背得更熟。

我建议把改进优先级分成三层：

1. **P1：先把评估链路和产品闸门固定住。** GA 特征掩码的定位已经明确为“高安全模式候选，不默认启用”；hard-example replay 当前配方已经严格复验证明不实用。近期 P1 不应继续重复这些重实验，而应优先解决大样本评估吞吐、阈值/校准流程固化、以及 cache 覆盖审计自动化。
2. **P2：再考虑更重的模型和数据升级。** 包括更长字节序列、SpeakeasyX 动态行为特征、家族级分类器、主动学习式样本采集。

已完成并从待办建议中移除：原 P0“统一模型评审闸门”。本轮已新增 `scripts/build_model_review_report.py`、`tests/test_build_model_review_report.py`，并生成 `reports/model_review/final_model_selection/model_review_summary.md` 与 `model_review_report.json`；真实 artifact 评审状态为 `usable`，核心 gate 均 PASS。当前统一报告已经覆盖最终模型选择、val-selected threshold、错误分析、group evaluation、概率校准正式全量 test 结果、完整 hard-FN/hard-error/high-value benign 结果，以及 GA 特征掩码的 20k 阈值、来源目录 trade-off、完整 hard-holdout 和高价值白样本证据。

## 本轮执行状态台账

| 建议项 | 当前状态 | 处理方式 |
| --- | --- | --- |
| 统一模型评审闸门 | 已完成 | 已从待办建议中移除，只保留完成记录和复用入口 |
| fixed-v2 20w 未压缩 cache 覆盖 | 已完成 | 已按授权清空 `data\.cache` 并重建未压缩 cache；覆盖 `199870/200000`，剩余 `130` 条为严格 PE 提取失败 |
| 概率校准 | 已确认实用且已彻底完成 | 已从待办建议中移除；严格全量 test、hard-FN、hard-error 和高价值白样本都已复验 |
| RL 主线扩大 | 实验确认当前不实用 | 显眼保留为“不建议近期主推”，除非奖励设计有新证据 |
| SWA / EMA / all combined | 实验确认当前不实用 | 显眼保留为“不建议一次性叠加训练技巧” |
| GA 特征掩码 | 已确认低漏报方向实用，但不适合默认启用 | 保留为高安全模式候选；现有 20k、完整 hard-holdout 和高价值白样本证据都已补齐，白样本 FP 成本仍高 |
| byte noise / near-threshold weighting | ⚠️ 实验验证：不实用 | 显眼保留为负面记录；三 seed cache-covered group-isolated 复验中 test F1 均值低于 baseline，FN 均值更高，不建议默认启用 |
| gated / residual fusion | 实验确认当前不适合作为近期 P1 | 显眼保留为“不建议继续同路线投入”；除非有新约束设计，否则不再优先训练 |
| hard-example replay | ⚠️ 实验验证：不实用 | 严格 source-group 隔离复验显示，FN holdout 从 39→15 正确（阈值 0.63），净增益仅 +4/423；不同阈值下 FP/FN 交换，无单一阈值同时改善两类错误 |
| SpeakeasyX 动态行为特征 | 已做边界验证，当前不适合主线合并 | 保留为 P2 二阶段复核信号；不作为直接替代/覆盖主分类器 |

机器可读状态审计已生成到 `reports/model_review/final_model_selection/ml_recommendation_status.json`。当前审计结论是：`3` 项已完成并从待办移除，`5` 项已实验确认不实用并显眼保留，`2` 项仍开放。这里的“仍开放”不是指 A/B 还没跑完，而是指 GA 特征掩码仍只适合做高安全模式候选、SpeakeasyX 仍只适合做 P2 二阶段复核研究。所有列入审计的证据路径当前都存在。2026-06-29 完成了 D 组 hard-example replay 的严格 source-group 隔离复验，确认当前配方不实用，已更新为负面记录。

cache 覆盖审计已生成到 `reports/model_review/final_model_selection/cache_coverage_audit.json`。2026-06-29 的 A/B 严格复验把 official test、hard-FN、hard-error、高价值白样本、GA hard-holdout 的 full/mask 导出都纳入同一张审计表，`10/10` 个检查面均为 `missing=0`。专项报告在 `reports/model_review/final_model_selection/ab_strict_reverification_report.md` 和 `ab_strict_reverification_report.json`，后续判断 A/B 状态应以这两份报告和当前 cache audit 为准。

2026-06-28 的不重建、不删除 cache 恢复计划仍保留为历史施工记录：`reports/model_review/final_model_selection/cache_recovery_plan.md` 和 `cache_recovery_plan.json`。它当时把 cache 缺口拆成三个恢复目标：official test 缺 `15,624` 行，hard-FN 缺 `162` 行，hard-error 缺 `162` 行。2026-06-29 已按 fixed-v2 cache 口径补齐并复验；这份计划现在不再是阻塞项，只用于说明当时的恢复边界和“禁止清空 `data/.cache`、不要把 64/8192 脚本当作 fixed-v2 修复入口”的护栏。

同日又补充了高价值白样本清单：`reports/model_review/final_model_selection/high_value_benign_manifest.csv` 和 `high_value_benign_manifest_summary.json`。这份清单把 official test 缺失 cache、hard-error 缺失 cache、以及当前 hard-error 可评估子集里的白名单良性样本合并去重，共 `8,127` 条，全部来自 `data\待加入白名单`。其中 `84` 条当前 cache 可读，`7,962` 条来自 official test 缺失 cache，`81` 条来自 hard-error 缺失 cache。它已经被用来直接评估概率校准和 GA 特征掩码的高价值白样本误报成本。

授权前完整性检查也已经生成到 `reports/model_review/final_model_selection/ml_authorization_preflight.json`。它只读授权计划和状态，不做任何重操作；A/B 完成后，preflight 只把未完成包当作待授权对象。A/B 本身的完成状态看 `ab_strict_reverification_report.*`。

剩余开放项的执行计划仍写在 `docs/ml_experiment_authorization_plan.md`，机器可读清单在 `reports/model_review/final_model_selection/ml_experiment_authorization_plan.json`。A/B 两个重操作包已经完成严格复验并移出待办；这份计划现在主要覆盖仍开放的实验项和负面记录。

## 2026-06-30 补充：fixed-v2 20w 未压缩 cache 已重建完成

这次按授权清空了 `data\.cache`，然后用 `reports/random_20w_split/random_20w_split.csv` 和 `models/random_20w_8192/best_model.pt` 对应配置，重建了 random 20w、8192 字节、fixed-v2 PE256 的未压缩特征 cache。这里的“未压缩”可以理解成把每个样本的特征文件存成更直接可读的格式，少做一层现场解压；它主要解决的是读取时的存储格式疑虑，不等于模型本身会立刻变快。

重建报告是 `reports/random_20w_split/random_20w_8192_uncompressed_cache_rebuild_full_current_split.json`。结果是：输入 `200000` 行，已有命中 `2998` 行，新提取 `196872` 行，严格 PE 特征提取失败 `130` 行，最终 manifest 为 `data/.cache/manifest_38672ba0.json`，样本数 `199870`，`cache_storage_format=uncompressed`。抽查 NPZ 文件的 zip 压缩类型为 `ZIP_STORED`，确认不是压缩 NPZ。

覆盖审计报告是 `reports/random_20w_split/random_20w_8192_uncompressed_cache_coverage_audit.json`，缺失明细在 `reports/random_20w_split/random_20w_8192_uncompressed_missing_cache.csv`。审计结论：`199870/200000 = 99.935%` 已覆盖，剩余 `130` 行不是路径匹配失败，也不是 cache 物流没补齐，而是严格 PE 特征提取失败。缺失标签分布为白样本 `125`、黑样本 `5`；split 分布为 train `12`、val `19`、test `99`。

这说明 fixed-v2 cache 覆盖问题已经基本收口。下一步不要继续靠“再清一次 cache”解决评估问题；真正瓶颈已经转移到模型评估路径本身。

### 评估吞吐结论

未压缩 cache 重建后，最初的 `scripts/evaluate_split_from_cache.py` 评估仍然很慢：`10000` 行评估在 30 分钟内没有完成，`1000` 行评估在 10 分钟内也没有完成。随后做了一个小 profile，发现最大慢点不是 NPZ 读取，也不是单批模型 forward，而是 manifest lookup 构建时对大量路径做了偏重的解析；这个步骤一度需要约 `75.7` 秒。

已修正 `scripts/evaluate_split_from_cache.py` 的路径索引逻辑，改成不触碰文件系统的字符串归一化。修正后，同一个 manifest lookup 从约 `75.7` 秒降到约 `4.4` 秒；`1000` 行 test 评估可以在约半分钟内完成，`10000` 行 test 评估可以在约 `3` 分 `42` 秒内完成。

新的 10k test 评估报告是 `reports/random_20w_split/random_20w_8192_uncompressed_test10k_after_lookup_opt_eval.json`。口径是 checkpoint `models/random_20w_8192/best_model.pt`，split `reports/random_20w_split/random_20w_split.csv`，manifest `data/.cache/manifest_38672ba0.json`，device `cuda`，batch size `64`，未使用概率校准器，未使用 GA feature mask。`10000` 行 raw test 中 `9994` 行成功预测，`6` 行 missing cache；missing 原因与全量覆盖审计一致，来自那 `130` 条严格 PE 提取失败。

10k 默认阈值 `0.50` 结果：Accuracy `0.9315`，Precision `0.9441`，Recall `0.9166`，F1 `0.9302`，AUC `0.9782`，FP `270`，FN `415`，FPR 约 `5.38%`，FNR 约 `8.34%`。阈值扫描显示 `0.45` 的 F1 略高，为 `0.9320`，FP/FN 为 `330/346`；阈值升到 `0.65` 时 FP 降到 `146`，但 FN 升到 `746`。这再次说明阈值只是 FP/FN 取舍，不是接近 `99.9%` F1 的根本解法。

因此，random 20w 模型在当前证据下没有达到 `F1 >= 99.9%`。这不是因为 cache 覆盖缺口太大：10k 口径只缺 `6/10000`，全量覆盖也有 `99.935%`。更关键的是，这个 10k slice 来自固定 test split；在已扫阈值里错误最少的是 threshold `0.45`，也已经有 `676` 个错误。即使假设剩余约 150k test 样本全部预测正确，完整 test 在这个阈值下的理论 F1 上界也只有约 `0.9958`，达不到 `0.999`。主要瓶颈是模型能力和当前训练/特征路线本身还不够，表现为 FP 和 FN 都仍然明显存在，其中 10k 默认阈值下 FN `415` 高于 FP `270`。如果未来要得到完整 160k test 精确指标，可以继续用修正后的评估脚本跑全量，但按 10k 耗时线性估算仍可能接近一小时；效率更高的路线仍是进一步做批量 tensor 或 memmap 评估。不要用 test set 反复挑阈值，阈值选择仍应只使用 val，test 只做最终确认。

## 当前机器学习链路的真实现状

当前主配置已经不是早期文档里的 PE1500 + 65536 字节大输入路线，而是 `config/default_config.toml` 里的 fixed_v2 PE256 主线：`max_byte_length=512`、`pe_feature_dim=256`、`dsra_dim=160`、`dsra_slots=160`、`fusion_type="concat"`、`decision_threshold=0.50`。这说明项目已经从“尽量塞更多原始信息”转向“用更短、更稳定的固定特征协议做可复现实验”。

模型本体也已经不是简单的二路拼接。`src/model.py` 中 `AxonMalwareModel` 会把三类信息合起来：第一类是文件开头字节序列，经 ByteEmbedding 和 DSRA 编码；第二类是 PE 结构特征，经 PE projector 投影；第三类是 stat 统计特征，经 stat projector 投影。融合方式目前支持 `concat`、`add`、`attention`、`gated`、`residual_stat_gate`、`residual_channel_gate`。这意味着改进空间已经从“有没有某个分支”升级为“什么时候应该相信哪个分支，以及如何防止某个分支在特定样本上误导模型”。

训练器也已经具备比较完整的机器学习实验能力。`src/trainer.py` 支持 label smoothing、focal loss、DSRA diversity loss、小族群样本加权、near-threshold weighting、SWA、EMA、阈值扫描、FP/FN/FPR/FNR 指标。这里的重点不是缺少训练技巧，而是要避免同时打开太多技巧导致无法判断到底是谁产生了收益。

数据层已经支持更严格的泛化检查。`src/dataset.py` 里有 `create_split_from_file()`，可以使用 `reports/raw_group_diagnostics/group_isolated_split.csv` 这类外部 CSV，把同一个相似族群固定放进 train、val、test 的其中一个集合，避免近亲样本同时出现在训练和测试里。这个能力非常重要，因为恶意软件样本经常有大量相似变种；如果近亲样本同时出现在训练集和测试集，模型分数会看起来很好，但上线遇到真正新家族时可能掉得很快。

## 关键证据

### 1. DSRA 分支不是摆设

`reports/hard_family_finetune/experiment_journal.md` 里的 DSRA 消融诊断显示，在 balanced test subset 上，完整模型 F1 为 `0.9565`；去掉 DSRA 后，F1 降到 `0.8476`，FP 从 `20` 增到 `88`，并出现 `74` 个预测翻转。hard-holdout 上完整模型是 `37/37`，去掉 DSRA 后变成 `30/37`。

这说明 byte/DSRA 分支确实在抓一些 PE/stat 分支抓不到的信息。后续不建议为了简化而删除 DSRA；如果要优化，应优先考虑让 DSRA、PE、stat 更好地融合。

### 2. stat 分支不是纯噪声，但会影响误报

同一份 journal 里的 stat full ablation 显示，直接在推理时关闭 stat 分支会减少 FP，但会明显增加 FN。例如 threshold `0.55` 下，full 模型 FP 为 `821`、FN 为 `726`、F1 为 `0.9513`；no_stat 模式 FP 降到 `698`，但 FN 升到 `1096`，F1 降到 `0.9427`。

这说明 stat 分支像一个“敏感但有点吵的报警器”。它会制造一些误报，但也能帮模型抓住更多恶意样本。早期判断是训练 gated 或 residual gate 来动态调节 stat 权重；后续实验已经证明当前这些 gate 设计没有形成可用收益，因此近期不再优先沿同一门控路线继续投入。

### 3. hard-example 微调有效，但会带来新的取舍

journal 记录显示，hard-example fine-tune 曾把 hard-holdout 37 条恶意样本从 baseline 的 `7/37` 提升到 `37/37`，但 limited overall eval 中 FP 从 `377` 增加到 `471`。后续最终模型选择文件 `reports/hard_family_finetune/model_selection_final/final_model_selection_summary.md` 又显示，`hard_fn_candidate@0.63` 在 original hard-family test 上比 previous candidate@0.55 更好：FP 从 `821` 降到 `680`，FN 从 `725` 降到 `720`，F1 从 `0.9514` 升到 `0.9558`。

但 `final_model_selection_report.json` 也显示，这个候选在 hard_error_holdout 上仍然很弱，且 hard_fn_holdout 也不是零错误。这说明 hard-example 微调是有效药，但不是万能药。下一步必须把 hard-FN 和 hard-FP 同时放进模型选择闸门，避免修好一类错误时扩大另一类错误。

### 4. 阈值选择本身就是一个产品策略

项目已经有阈值扫描能力，且结果说明阈值不是小事。比如 previous candidate 在 original hard-family test 上，threshold `0.55` 时 FP/FN 是 `821/725`；提高到 `0.63` 会变成 `648/919`，误报减少但漏报增加。hard_fn_candidate 在 threshold `0.63` 时达到当前最终推荐，但它在不同 holdout 上的表现并不完全一致。

通俗地说，阈值就是“报警器灵敏度旋钮”。旋得低，恶意更容易被抓到，但白文件更容易被误报；旋得高，白文件更安全，但恶意漏掉的风险上升。这个旋钮不应该只靠一次 test 集最大 F1 决定，而应该由产品策略决定：我们更怕误报打扰用户，还是更怕漏报放过恶意文件。

### 5. GA 特征掩码有价值，但不能直接默认上线

`docs/feature_subset_ga_report.md` 显示，`config/feature_masks/ga_recall_guard_2000.json` 从 192 个有效 PE/stat 特征中保留了 125 个。在 20,000 样本评估中，GA 掩码 @ `0.525` 的总错误数为 `1210`，比完整特征 @ `0.50` 的 `1340` 少 `130` 个；FN 从 `958` 降到 `670`，但 FP 从 `382` 升到 `540`。

这说明 GA 掩码更像“减少漏报”的候选方案，不是无成本提升。它值得继续验证，尤其是在 group-isolated split、时间切分或真实家族标签切分上验证，但不建议没有闸门就变成默认推理开关。

### 6. 一些训练技巧已经有负面证据，不能盲目叠加

`models/comparison_experiments_from_cache/summary.md` 显示，在 20,000 样本 cache 抽样版实验中，baseline F1 为 `0.9287`，byte noise 为 `0.9263`，SWA 为 `0.8882`，EMA 为 `0.9132`，near-threshold 为 `0.9255`，all combined 为 `0.8549`。这说明“把所有增强都打开”反而明显变差。

但 `models/generalization_group_isolated/summary.md` 又显示，在单 seed group-isolated 泛化对比里，baseline test F1 为 `0.8869`，byte noise 为 `0.8893`，near-threshold 为 `0.8899`。这曾经提示 byte noise 和 near-threshold 可能有小幅价值，但 2026-06-29 的三 seed 复验证明这个方向不稳定：`models/generalization_group_isolated_seed_confirm/summary.md` 中 baseline test F1 均值为 `0.9444`，byte noise 为 `0.9260`，near-threshold 为 `0.9200`。更关键的是，byte noise 的 test FN 均值从 baseline 的 `623.7` 升到 `851.7`，near-threshold 升到 `958.7`。这说明单次小涨更像随机波动或种子敏感，不足以改默认配置。

### 7. RL 路线目前不是主线优先项

`reports/pro_runs/fixed_pe256_2k_summary.json` 显示，3 个 seed 下 CE baseline 的平均 F1 为 `0.8910`，RL 的平均 F1 为 `0.8427`，并且 `continue_to_larger_scale=false`。除非后续有新的奖励设计和同协议反证，否则不建议把 Pro RL 当成近期主线。

### 8. 512 字节输入目前有经验优势，但仍需统一协议复验

`models/length_tests/results.json` 显示，在那轮输入长度实验中，`max_byte_length=512` 的 F1 为 `0.9588`，高于 1024、2048、256、128、64。这个结果支持继续把 512 作为主线输入长度，但因为历史实验口径不一定和最新 group-isolated 主线完全一致，不能据此断言“512 永远最优”。更稳妥的做法是把 512 作为默认，把 1024/2048 作为 P2 复验，不要现在盲目加长。

## 已完成：把阈值和概率校准正式产品化

**实验状态：已确认实用且已彻底完成。**

历史正式全量报告 `reports/hard_family_finetune/clean_hyperparam_search/train_calibrator_no_metadata_test_confirmation_scripted.json` 显示，在完整 `31,816` 条 test 样本上，校准器相对同 CSV baseline：

- F1：`0.9514 -> 0.9720`，提升 `+0.0206`
- 错误数：`1549 -> 898`，减少 `651`
- FP：`872 -> 645`，减少 `227`
- FN：`677 -> 253`，减少 `424`

历史上曾先跑过 `scripts/evaluate_probability_calibrator.py --allow-missing-cache` 的诊断子集，随后又补齐严格模式的完整评审；这些步骤现在都只是过程记录。最终已经确认，校准器在完整 test、hard-FN、hard-error 和高价值白样本四个口径上都有效。

2026-06-29 又做了一次 A/B 专项严格复验，报告位于 `reports/model_review/final_model_selection/ab_strict_reverification_report.md`。这次复验确认校准器训练协议是 train split 训练、val split 选模型和阈值、test 不参与训练；四个严格评估口径全部 `kept=total` 且 `skipped_missing_cache=0`。

### 完成内容

把阈值选择从“训练配置里的固定 `0.50`”升级成“每个模型都有自己的推荐阈值和业务阈值区间”。同时把 `scripts/train_probability_calibrator.py` 和 `scripts/evaluate_probability_calibrator.py` 纳入标准流程，验证 Logistic Regression calibrator 或 temperature scaling 是否能在不重训主模型的情况下减少错误。

这里的“概率校准”可以理解成给模型的分数表重新校准刻度。模型说 `0.8` 不一定真的代表 80% 风险，校准器的作用是用验证集把分数刻度调得更可靠。它不是重新训练主模型，更像是给温度计重新标定。

### 结果总结

项目已经有阈值扫描和校准脚本，严格复验后，校准器把完整 test 错误从 `1549` 降到 `898`，hard-FN 从 `20` 降到 `6`，hard-error 从 `300` 降到 `132`，高价值白样本 FP 从 `604` 降到 `406`。这说明“只调分数刻度”是实用的低成本改进。

### 结论

校准器已经完成，保留为标准流程的一部分，不再占用 P1 待办。

## P1 建议：gated/residual fusion 暂停作为近期主线

**实验状态：当前路线已验证不实用，保留记录，不从文档移除。**

已经验证过 `gated`、`residual_stat_gate`、`residual_channel_gate` 三类融合方式。原始想法是对的：不要直接删除 stat 分支，而是让模型学会什么时候少信 stat、什么时候多信 DSRA/PE。但现有实验结果说明，当前这些 gate 设计没有把“音量旋钮”学好，反而带来了更差的整体结果。

### 关键证据

- `reports/hard_family_finetune/gated_full_threshold_sweep.json`：gated fusion 最佳 F1 约 `0.9411`，FP/FN 为 `1254/648`，总错误 `1902`。当前 concat 主线推荐候选在 original hard-family test 上 F1 为 `0.9558`，FP/FN 为 `680/720`，总错误 `1400`。
- `reports/hard_family_finetune/residual_stat_gate_full_threshold_sweep.json`：residual_stat_gate 最佳 F1 约 `0.9395`，FP/FN 为 `911/1002`，总错误 `1913`。
- `reports/hard_family_finetune/clean_hyperparam_search/f1_probe_residual_channel_gate_val_sweep.json`：residual_channel_gate 小验证 probe 最佳 F1 约 `0.9247`，只比同预算 concat baseline 略高，但低于已知 scheduler-free probe；journal 已记录“不 seed-confirm 或 full-train”。

### 当前判断

这条路线不是“理论上永远不行”，而是“当前实现和当前数据下不值得继续作为近期 P1”。如果未来要重启，必须先有新的约束设计，例如 gate 只能在很小范围内调节 stat，或者只在特定误报风险区间启用；否则不要再重复训练同类 gated/residual 模型。

### 仍然保留的原因

stat 分支的双刃剑问题仍然存在，只是当前 gate 不是好解法。这个记录要保留，避免后续重复投入同一条已经失败的架构路线。

## ⚠️ 实验验证：不实用 - hard-example balanced replay 当前配方不建议投入

**实验状态：⚠️ 实验验证：不实用。当前配方（4 epoch、1e-5 LR、FP/FN weight 4x）在严格 source-group 隔离下未实现 balance improvement，保留为负面记录，不移除。**

### 要做什么

继续做 hard-example fine-tune，但不要只喂当前漏报样本。每轮 hard-example 包应该同时包含：

- hard FN：模型漏掉的恶意样本。
- hard FP：模型误报的白样本。
- clean replay：普通 train/val 中的稳定样本，用来防止模型忘掉原有能力。
- family/group 限制：避免某一个来源目录或相似族群占比过高。

### 为什么值得做

现有 hard-example fine-tune 能显著修复目标 hard-FN，但也出现过误报增加和另一个 holdout 表现变差的问题。这类似给员工突击训练某一类题，如果只练这一类题，他可能在另一类题上退步。反向回放就是每次专项训练时保留一部分旧题，防止模型“偏科”。

已有正向证据很强：`reports/hard_family_finetune/hard_error_finetune_threshold055/hard_error_finetuned_full_threshold_sweep.json` 显示 hard-error fine-tune 在该 split 的 threshold `0.55` 下达到 F1 `0.9776`，FP/FN 为 `329/355`；同一 hard-error holdout 上，旧模型错误从 `309` 降到 `219`。但后续 hard-FN targeted candidate 虽然提高 original hard-family test F1 到 `0.9558`，hard-error holdout errors 仍为 `229`，比上一轮 hard-error holdout `219` 更差。这说明 hard-example 是有效药，但“只追一种错题”会让另一类错题回潮，反向回放还没有完成。

2026-06-28 已完成一个不训练的准备步骤：使用当前推荐候选 `hard_fn_candidate@0.63` 的错题分析，生成 `reports/hard_family_finetune/balanced_replay_from_current_candidate_threshold063/`。这个包同时包含当前候选在 original hard-family test 上的 `680` 个 FP 和 `720` 个 FN，并按 `source_group_id` 做 group 级 train/val/holdout 分配：hard train `859` 行、hard val `291` 行、hard holdout `415` 行，其中 `840` 个 hard train 样本带显式权重。它还保留 base train/val/test 行作为 clean replay 背景。注意，这一步只是把“错题本”整理好，没有启动微调训练，也没有证明 replay 配方有效。

### 建议实验

每轮 hard-example 包都要写清楚：

- 样本来自哪些错误类型。
- 每类样本数量。
- 是否按 group 去重。
- 是否包含 clean replay。
- 与上一轮模型相比，FP/FN 各自变化多少。

只有当 original test、hard-FN holdout、hard-error holdout 三者综合改善，才进入下一轮。

但 2026-06-29 的严格 source-group 隔离复验显示，当前 replay 配方不能满足这个标准。

#### 复验结果

在 strict source-group 隔离的 replay 包（`reports/hard_family_finetune/balanced_replay_strict_source_group_threshold063/`）上训练后，同一 hard-error holdout 的 baseline 对比：

| 角色 | Baseline correct（阈值 0.63） | Replay correct（阈值 0.63） | 变化 |
| --- | ---: | ---: | ---: |
| hard-error FN holdout（n=144） | 39（27.1%） | 15（10.4%） | **-24** |
| hard-error FP holdout（n=136） | 41（30.1%） | 79（58.1%） | **+38** |
| context holdout（n=143） | 143（100%） | 133（93.0%） | -10 |
| **合计（423）** | **223（52.7%）** | **227（53.7%）** | **+4** |

在阈值 0.50 时，交换反转：FN holdout 从 51→78 正确，FP holdout 从 16→7 正确。这说明 replay 训练使模型整体更“激进”（得分更高），但牺牲了 FN vs FP 的平衡。当前业务优先级（误报比漏报更严重）下，FN 退化不可接受。

#### 为什么不建议继续

- 净值太小：423 条 holdout 上仅多修复 4 条（+0.9%）。
- 方向不平衡：同一配方下，FP 改善以 FN 退化为代价。
- 复验是严格 source-group 隔离的，不是“保留近亲样本”的宽松口径；这个结果比历史宽松实验更有参考价值。

建议后续如果尝试不同 replay 配方（如更长训练、更小 LR、不同权重），必须先用本包的严格隔离口径预验，不能跳过。但以当前证据，不推荐将此列为 P1。



## P1 建议：GA 特征掩码继续验证，但先不要默认启用

**实验状态：已确认“减少漏报”方向实用，但高价值白样本误报仍偏高，因此继续保留在 P1，不能默认启用。**

### 要做什么

保留 `config/feature_masks/ga_recall_guard_2000.json` 作为候选特征子集，在统一评审闸门下复验。本轮已经把 20,000 样本阈值评估、来源目录 trade-off、完整 hard-holdout，以及高价值白样本评估都补齐了；现在的重点不是继续证明“它能不能减少漏报”，而是把它的业务定位说清楚。

### 为什么值得做

GA 掩码确实展示出减少总错误和减少 FN 的潜力。20,000 样本评估中，GA mask @ `0.525` 相对完整特征 baseline @ `0.50`，F1 从 `0.9310` 到 `0.9391`，总错误从 `1340` 到 `1210`，FN 从 `958` 降到 `670`。来源目录里三个恶意来源的 FN 也都下降，所以它不是无效想法。

### 补齐后的复验结果

`ga_feature_mask_full_holdout_summary.json` 里，完整 hard-FN 上 full baseline 从 `19` 个错误到 mask 的 `18` 个错误，完整 hard-error 上从 `288` 个错误到 `286` 个错误，确实继续减少了漏报。

但高价值白样本的结果更关键：`high_value_benign_baseline_analysis/prediction_error_summary.json` 显示 full baseline FP 为 `604`，`high_value_benign_ga_mask_analysis/prediction_error_summary.json` 显示 GA mask FP 为 `638`，多了 `34` 个误报。也就是说，它对误报最贵的正常文件不划算。

2026-06-29 的专项复验还确认，GA 相关导出在 fixed-v2 cache 上没有缺口：20k 评估、hard-FN/hard-error full 与 mask、高价值白样本 full 与 mask 都已经纳入 `cache_coverage_audit.json`，相关检查面全部 `missing=0`。

### 结论

GA 掩码保留为高安全模式候选，不默认启用，也不从文档中移除。它的价值在于“更少漏报”，代价是“更多白样本误报”，业务上要明确把它放到更保守的场景里。

## ⚠️ 实验验证：不实用 - byte noise 和 near-threshold weighting 不建议默认启用

### 做了什么

在 group-isolated split 下，已经完成 byte noise 和 near-threshold weighting 的三 seed 小规模复验。实验没有和 SWA、EMA、GA 掩码、门控融合一次性全叠加，而是只比较 baseline、byte_noise、near_threshold 三个变量，避免结果解释不清。

2026-06-29 先修正了 `scripts/run_generalization_group_split.py`：新增 `--cache-manifest`，并补了 manifest 规格校验，防止用 64-byte 或 8192-byte cache 去跑 512-byte 主线配置；同时把转换后的 split 主键改回原始文件路径，保证训练时 `FeatureCacheDataset` 能正确匹配。最终复验固定使用 `data/.cache/manifest_ee122d6c.json`，也就是 512-byte fixed-v2 cache。这个 cache 覆盖 `20,000/40,000` 条 raw group-isolated rows，因此本轮结论的准确名字是“cache-covered group-isolated subset 多 seed 复验”，不是完整 40k raw split。

### 结果

`models/generalization_group_isolated_seed_confirm/summary.md` 的三 seed 汇总如下：

- baseline：test F1 mean `0.9444`，std `0.0047`，FP mean `271.0`，FN mean `623.7`。
- byte noise：test F1 mean `0.9260`，std `0.0555`，相对 baseline `-0.0184`；FP mean `293.3`，FN mean `851.7`。
- near-threshold：test F1 mean `0.9200`，std `0.0515`，相对 baseline `-0.0244`；FP mean `275.3`，FN mean `958.7`。

最有解释力的是 seed43：byte noise 的 test F1 比同 seed baseline 低 `0.0793`，FN 多 `1105`；near-threshold 的 test F1 低 `0.0808`，FN 多 `1164`。如果上线场景重视漏报风险，这种不稳定性不能接受。

### 不建议做什么

不建议直接启用 all combined，也不建议近期继续把 SWA 或 EMA 当作主线训练技巧。已有 cache 抽样实验里 all combined 明显低于 baseline，SWA 和 EMA 也明显退化，说明训练技巧不是越多越好。现在 byte noise 和 near-threshold 也已经有多 seed 负面证据，不应进入默认配置。

### 结论

⚠️ 实验验证：不实用。byte noise 和 near-threshold weighting 保留为负面记录，不从文档中删除，避免以后因为单 seed 小涨再次重复投入。除非未来更换数据规模、训练预算或设计出新的约束方式，否则不要把它们作为默认训练技巧。

## P2 建议：输入长度暂时保持 512，长序列只做受控实验

### 要做什么

主线继续保持 `max_byte_length=512`。如果要验证 1024 或 2048，必须使用相同 split、相同训练预算、相同模型选择闸门。

### 为什么

历史输入长度实验里 512 表现最好，而且训练更快。对恶意软件检测来说，文件开头通常包含 PE header、section table、import 等很多高价值线索；更长输入不一定更好，因为它会增加训练成本，也可能引入更多噪声。

### 什么时候值得加长

只有在错误分析发现大量 FN 的关键恶意行为线索出现在 512 字节之后，才值得把长序列作为主线候选。

## P2 建议：SpeakeasyX 动态行为特征只作为二阶段复核候选

### 要做什么

SpeakeasyX 这类动态行为特征不要直接大规模加入训练，也不要直接替代当前概率校准器。现有验证显示，它更适合做“二阶段复核”：当主模型已经判定某个样本可疑时，用 SpeakeasyX 的运行状态、timeout、unsupported、是否产生 trace 等信号辅助判断这个告警是不是可能误报。

### 为什么

动态行为特征像是在沙盒里观察程序运行动作，理论上很有价值，但成本高、失败率高、环境敏感。现有 `reports/hard_family_finetune/clean_hyperparam_search/speakeasy_feasibility_report.md` 已经做了边界验证：SpeakeasyX 对校准器残差样本确实有强信号，尤其能识别一批 false positive；但固定 timeout filter 在 test confirmation 子集上虽然把 FP 从 `122` 降到 `0`，也把 FN 从 `120` 增到 `168`，新增 `48` 个漏报。对安全产品来说，这个代价不能直接并入主分类器。

Ordered API sequence 方向也还没有证明“API 顺序本身”能提升 F1。严格 API-order-only DSRA 控制里，sequence-only 结果退化到 F1 `0.5000`；真正有用的信号更多来自“是否快速产生 trace / timeout / unsupported”这类执行可达性状态。换句话说，当前 SpeakeasyX 更像一个人工复核用的体检指标，不是已经成熟的主模型新器官。

### 建议闸门

只有当动态行为特征在固定 holdout 上同时减少误报、且新增漏报受控，才允许进入 P1 或主线融合。下一步如果继续做，优先验证更保守的 FP 复核规则，例如降低 downgrade 的触发范围、把 `.NET unsupported` 明确设为“不要降级”的恶意侧风险信号，或者训练一个只用于复核的 held-out 小校准器。不要先做全量 Speakeasy 抽取，也不要直接把 timeout 规则接进生产推理。

## P2 建议：家族分类器用于解释和分流，不要替代二分类主模型

### 要做什么

继续保留 `scripts/export_family_classifier.py` 这类 family classifier 能力，但定位应是：

- 帮助解释“这个恶意样本像哪个已知相似族群”。
- 辅助错误分析和报表。
- 作为二阶段风险分流信号。

不要把它当成替代二分类模型的主检测器。

### 为什么

家族分类依赖已有相似族群中心点。它擅长认“见过的家族附近的样本”，但遇到新家族时容易没有合适归属。二分类模型负责先判断恶意风险，家族分类器负责进一步解释和归类，这两个角色不要混淆。

## 暂不建议优先投入的方向

1. **【实验确认当前不实用，保留记录】不建议近期主推 RL。** 当前 3 seed 结果显示 RL 明显落后 CE baseline，除非奖励函数有新设计，否则继续扩大训练成本不划算。
2. **不建议盲目加长字节输入。** 512 在已有实验中表现最好，长输入会增加成本和噪声。
3. **不建议直接删除 stat 分支。** 它能减少漏报，问题应通过门控解决。
4. **【实验确认当前不实用，保留记录】不建议继续同路线投入 gated/residual fusion。** full gated、residual_stat_gate 和 residual_channel_gate probe 都没有形成可用收益；除非先设计新约束，否则不要重复训练。
5. **不建议直接默认启用 GA 掩码。** 它减少 FN，但增加 FP，需要按产品策略决定。
6. **【实验确认当前不实用，保留记录】不建议一次性叠加所有训练技巧，也不建议近期主推 SWA/EMA、byte noise 或 near-threshold weighting。** 现有 all combined 结果已经提示这种做法可能显著退化；cache 抽样实验中 SWA、EMA 和 all combined 均明显低于 baseline。byte noise 和 near-threshold 虽然在单 seed group-isolated 口径有过小幅 test F1 收益，但三 seed 复验 test F1 均值低于 baseline，FN 均值更高，因此保留为负面记录。

## 失败实验复盘闸门

下面这一节专门回答“为什么失败、失败排除了什么、下一步从哪里来”。它不是为了否定所有新想法，而是避免把已经被证据排除的旧配方再跑一遍。对非技术视角来说，可以把它理解成实验复盘表：不是只说这条路没通，而是写清楚它撞到的是哪堵墙。

### RL 主线扩大

- 失败观察：3 个 seed 下，CE baseline 平均 F1 为 `0.8910`，RL 平均 F1 为 `0.8427`；RL 平均 reward、accuracy、precision、recall、F1 都低于 CE，`continue_to_larger_scale=false`。
- 推理出的可能原因：当前 RL 分支把二分类模型包装成一阶 bandit 奖励环境，但奖励设计没有比交叉熵训练提供更稳定的学习信号。它有时会改变 FP/FN 取舍，但不是稳定提升识别能力。
- 证据强度：强证据。证据来自 `reports/pro_runs/fixed_pe256_2k_summary.json` 的 3 seed 对照。
- 因此不建议继续：不建议扩大 RL 训练规模，也不建议把 Pro RL 当成主分类器训练路线。
- 因此建议下一步：只有在奖励函数有明确新设计，并且先通过小规模多 seed 同协议对照后，才允许重启。
- 最小验证实验：固定同一 split、同一模型容量、同一阈值选择协议，比较新 reward RL vs CE baseline，至少 3 seed。
- 成功标准：RL 的平均 F1、FN、FP 不能只改善一个方向，必须整体不低于 CE，且方差不明显更大。
- 失败后如何停止：若 3 seed 平均 F1 仍低于 CE，或只是以大幅增加 FP/FN 之一换取另一项改善，继续保留负面记录。

### SWA / EMA / all combined

- 失败观察：20k cache 抽样实验中，baseline F1 `0.9287`；SWA `0.8882`，EMA `0.9132`，all combined `0.8549`，都明显低于 baseline。
- 推理出的可能原因：这些技巧本身不是坏技术，但当前数据和训练预算下同时或直接套用，会改变模型收敛轨迹，反而削弱已经有效的 DSRA + PE + stat 表示。all combined 尤其像一次把多个旋钮都拧了，结果无法分辨哪个旋钮造成退化。
- 证据强度：中到强证据。SWA/EMA/all combined 在 20k cache 抽样对照里退化明显；但它们没有必要再做 P1 full-scale，因为退化幅度已经足以否决近期默认路线。
- 因此不建议继续：不建议近期主推 SWA、EMA 或“所有训练技巧全开”。
- 因此建议下一步：如果未来重启，只能一个变量一个变量做小规模诊断，不能组合上车。
- 最小验证实验：单独开启一个技巧，固定 split、seed 组和阈值协议，先跑小样本多 seed。
- 成功标准：多 seed 平均 F1 高于 baseline，同时 FP/FN 没有单侧显著恶化。
- 失败后如何停止：只要收益来自单 seed，或平均 FN/FP 明显恶化，就不进入全量训练。

### byte noise / near-threshold weighting

- 失败观察：单 seed group-isolated 实验曾有小幅 test F1 提升，但三 seed cache-covered group-isolated 复验反转。baseline test F1 mean `0.9444`；byte noise `0.9260`，near-threshold `0.9200`。byte noise 的 FN mean 从 `623.7` 升到 `851.7`，near-threshold 升到 `958.7`。
- 推理出的可能原因：这是典型的训练不稳定和 seed 敏感。byte noise 原本想让模型别死记字节位置，near-threshold 原本想让模型多学边界样本；但实际结果像是在给已经脆弱的边界加噪声，恶意漏报变多。
- 证据强度：强证据。证据来自 `models/generalization_group_isolated_seed_confirm/summary.md` 和 `reports/model_review/final_model_selection/training_trick_summary.json` 的多 seed 复验。
- 因此不建议继续：不建议默认启用 byte noise 或 near-threshold weighting，也不建议把单 seed 小涨当成产品依据。
- 因此建议下一步：除非换成更保守的增强策略，否则不要再做同配方训练；如果要重启，先做最小诊断，观察 FN 是否受控。
- 最小验证实验：只改变扰动强度或 near-threshold 权重之一，3 seed，小样本，固定 group-isolated split。
- 成功标准：平均 test F1 高于 baseline，且 FN mean 不能高于 baseline。
- 失败后如何停止：只要 FN mean 上升或 std 明显扩大，立即停止，不进默认配置。

### gated / residual fusion

- 失败观察：gated fusion 最佳 F1 约 `0.9411`，总错误 `1902`；residual_stat_gate 最佳 F1 约 `0.9395`，总错误 `1913`；residual_channel_gate 小验证 probe 也未超过已知更强 baseline。
- 推理出的可能原因：原始假设是对的：stat 分支像一个有用但有噪声的报警器，模型应该学会什么时候少信它。但当前 gate/residual 设计没有学到可靠的“谁该被信任”，反而增加了结构复杂度和训练难度。
- 证据强度：强证据。已有 full sweep 和 probe 证据，且都没有形成可用收益。
- 因此不建议继续：不建议重复训练同类 gated/residual 结构。
- 因此建议下一步：如果未来重启，必须先提出新的约束机制，例如 gate 只能小幅调节、只在高误报风险区间生效，或先做可解释 gate 分布诊断。
- 最小验证实验：先在 val 上验证 gate 权重是否与已知 FP/FN 风险相关，而不是直接 full train。
- 成功标准：gate 权重有可解释分布，并在 val/hard holdout 上同时减少错误。
- 失败后如何停止：如果 gate 权重不可解释，或只是转移 FP/FN，就停止架构路线。

### hard-example replay

- 失败观察：严格 source-group 隔离 replay 后，阈值 `0.63` 下 hard-error FN holdout 正确数从 `39` 降到 `15`，FP holdout 从 `41` 升到 `79`，合计只净增 `+4/423`。阈值 `0.50` 时取舍反向变化，没有单一阈值同时改善 FP/FN。
- 推理出的可能原因：当前 replay 配方把模型整体推得更激进或更保守，而不是学到了更稳的恶意/良性边界。它修一类错题时会伤另一类错题，是 hard-example 过拟合和阈值 trade-off 的混合问题。
- 证据强度：强证据。证据来自严格 source-group 隔离包、baseline vs replay holdout 对比和阈值对照。
- 因此不建议继续：不建议重复当前 `4 epoch + 1e-5 LR + FP/FN 4x` 配方。
- 因此建议下一步：如果要继续 hard-example，只能先改协议：更严格的 source-group 隔离、clean replay 比例、双向 FP/FN holdout、val-only 阈值选择必须同时存在。
- 最小验证实验：在当前严格隔离 replay 包上做小学习率或短训练诊断，并只看 val/hard holdout，不碰 test 调参。
- 成功标准：hard-FN、hard-FP、context holdout 三者同时不退化，净改善不能只靠牺牲其中一类。
- 失败后如何停止：如果任一 hard holdout 明显退化，或净收益小于人工复核价值，就停止，不再进入 P1。

### GA 特征掩码

- 失败观察：它不是失败实验，而是“有收益但有业务代价”的候选。20k 评估中总错误 `1340 -> 1210`，FN `958 -> 670`，但 FP `382 -> 540`；高价值白样本 FP `604 -> 638`。
- 推理出的可能原因：GA 掩码减少了一些会造成漏报的噪声特征或保留了更偏召回的信号，但也削弱了识别白样本的证据，所以误报增加。
- 证据强度：强证据。证据来自 20k、hard-holdout、高价值白样本和 cache 覆盖审计。
- 因此不建议继续：不建议默认启用 GA mask，也不建议把它包装成无成本 F1 提升。
- 因此建议下一步：把它定义为高安全模式候选，只在“少漏报比少误报更重要”的业务场景使用。
- 最小验证实验：如需上线前再验，只在 val 上选阈值，然后固定到高价值白样本和 hard holdout 做确认。
- 成功标准：高安全模式下 FN 明显下降，同时 FP 增量在产品可接受范围内。
- 失败后如何停止：如果高价值白样本 FP 超过产品容忍线，就不进入自动模式，只保留为分析工具。

### SpeakeasyX 动态行为特征

- 失败观察：SpeakeasyX 在残差样本上有强信号，但固定 timeout filter 在 test confirmation 子集上把 FP 从 `122` 降到 `0` 的同时，把 FN 从 `120` 增到 `168`。
- 推理出的可能原因：动态行为信号能识别一批误报，但 timeout/unsupported 不是纯良性信号，一些真实恶意也会 timeout 或不受支持。直接自动降级会把恶意样本误放过去。
- 证据强度：中到强证据。残差和 test confirmation 都支持“有信号但有代价”；但它还不是全量动态特征主线实验。
- 因此不建议继续：不建议直接合入主分类器，也不建议用 timeout 规则自动覆盖概率校准器。
- 因此建议下一步：只作为二阶段 FP 复核或人工调查信号，先做更保守的 val-first 小实验。
- 最小验证实验：只对主模型已报恶意且置信度不极高的样本做 FP triage，`.NET unsupported` 不得简单降级。
- 成功标准：减少 FP 的同时，新增 FN 接近 0，并且规则完全由 val 选出。
- 失败后如何停止：只要新增 FN 超过产品可接受线，就不能进入自动降级，只能作为解释特征。

### byte n-gram 融合

- 失败观察：Loop37 的 byte n-gram SGD 与 Loop28 错误重合很低，Val 从 `162` 错降到 `159` 错，Test-10k 从 `111` 错降到 `110` 错；但 16 万全量测试反转为 `1960` 错，差于 Loop28 的 `1949` 错。
- 推理出的可能原因：byte n-gram 确实带来一部分不同视角的内容信号，但独立模型太弱，融合收益只有几个样本，容易被抽样噪声吞掉。Test-10k 是漏斗，不是最终证明；当改善只有 1 个样本时，证据强度不足。
- 证据强度：强证据。证据来自 Loop37 的 Val、冻结 Test-10k 和冻结 16 万 full-test 全流程，且预测表做了 `source_sha256`/label 对齐审计。
- 因此不建议继续：不建议继续用同一个 byte n-gram SGD 与 Loop28 做小权重线性融合。
- 因此建议下一步：如果重启 byte n-gram 路线，应先把它变成更强的独立基模型，或作为 OOF 多模型栈的一员，而不是直接做边际线性混合。
- 最小验证实验：只在 Val 上验证更强 byte n-gram 配方是否把独立错误降到明显低于当前 `1250`，且融合至少减少 10 个以上 Val 错误后再进 Test-10k。
- 成功标准：Val 和 Test-10k 都有足够宽的错误数改善，而不是 1-3 个样本级别的波动。
- 失败后如何停止：若 Test-10k 只改善 1-2 个错误，必须视为边际候选，只有全量通过才保留；全量反转则停止该配方。

### 禁止重复投入清单

- 不重复跑当前 RL 奖励设计的大规模训练；除非先有新 reward 并通过 3 seed 小实验。
- 不重复跑当前 SWA、EMA、all combined 默认配方；除非单变量、多 seed、小样本先过关。
- 不重复跑当前 byte noise / near-threshold 配方；除非能证明 FN mean 不再上升。
- 不重复跑当前 gated / residual fusion 设计；除非先提出并验证新的 gate 约束机制。
- 不重复跑当前 hard-example replay 配方；除非改成更严格的 source-group、clean replay、双向 holdout 协议。
- 不把 GA mask 默认启用；除非产品明确接受更高白样本误报。
- 不把 SpeakeasyX timeout filter 直接接进生产自动降级；除非新增 FN 在 val 和固定 holdout 上都可控。
- 不重复跑当前 byte n-gram SGD 小权重线性融合；除非先显著增强 byte n-gram 独立模型或改为严格 OOF 栈。

## 推荐的下一阶段路线图

### 第一步：优化大样本评估流水线

目标：先把 random 20w / 160k test 这种大口径评估跑顺，再谈新模型。当前 cache 覆盖已经足够，继续重建 cache 的收益很低；真正该优化的是推理评估路径。

建议顺序：

1. 先做评估 profile，分清楚时间花在 NPZ 打开、CPU 到 GPU 搬运、DSRA forward，还是指标汇总。
2. 把已覆盖 cache 预导出为批量 tensor 或 memmap，避免每次评估逐样本打开 NPZ。
3. 在 val split 上选择阈值和 batch 配置，再用 test split 做一次最终确认。

### 第二步：固化产品模式

GA 掩码和概率校准不需要继续重跑来证明方向。当前应该把它们固化成产品策略：

- 普通模式：默认使用完整特征 + 概率校准。
- 高安全模式候选：允许使用 GA 掩码，但必须明确它会增加高价值白样本误报。
- hard-example replay：当前配方作为负面记录保留，不进入默认训练。

### 第三步：决定产品策略阈值

这一步需要产品负责人参与，因为它不是纯技术问题。我们要明确：

- 如果误报一个白文件，产品成本有多大？
- 如果漏报一个恶意文件，安全成本有多大？
- 是否需要“普通模式”和“高安全模式”两个阈值？
- 是否允许高风险样本进入二阶段复核，而不是直接拦截？

## 建议的复现命令模板

下面这些命令不要求现在立刻运行，它们是后续验证时的标准入口。

### 使用虚拟环境运行单个评估

```powershell
cd "E:\Project\python\Axon_v2.6Exp"; & "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\main.py eval --checkpoint "models\group_isolated_rare_weighted_ft_rebuilt_cache\best_model.pt" --data-dir "data" --split-file "reports\raw_group_diagnostics\group_isolated_split.csv" --split test --batch-size 32 --device cuda --sweep-thresholds "0.50,0.53,0.55,0.60,0.63,0.65" --output "reports\model_review\baseline_eval.json"
```

### 评估 GA 特征掩码

```powershell
cd "E:\Project\python\Axon_v2.6Exp"; & "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\evaluate_feature_mask.py --checkpoint "models\best_model.pt" --data-dir "data" --feature-mask "config\feature_masks\ga_recall_guard_2000.json" --samples-per-class 10000 --batch-size 256 --device cuda --thresholds "0.45,0.50,0.525,0.55,0.60,0.65" --baseline-threshold 0.50 --output-json "reports\model_review\ga_feature_mask_eval.json"
```

### 生成错误分析

```powershell
cd "E:\Project\python\Axon_v2.6Exp"; & "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\analyze_prediction_errors.py --predictions "reports\hard_family_finetune\finetuned_test_predictions_threshold055.csv" --threshold 0.55 --output-dir "reports\model_review\error_analysis"
```

## 最终建议

统一模型评审闸门已经落地，下一步不要再回到零散对比。后续每一次训练、阈值、特征掩码和 hard-example 微调，都应进入 `scripts/build_model_review_report.py` 生成的同一类报告，再判断它到底是在减少误报、减少漏报，还是只是换了口径后看起来更好。已经失败的 gated/residual 融合路线不要重复投入；如果未来有新约束设计，也必须作为全新候选重新进入统一报告。
