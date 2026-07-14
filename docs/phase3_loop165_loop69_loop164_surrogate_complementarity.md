# Loop165 Loop69 x Loop164 Train-only 代理互补性审计

## 结论

当前 Loop164 recipe 进入 `parked_no_further_standalone_or_exact_oof_spend`。Loop151 仍是唯一 research champion，legacy development full-test F1 仍为 `0.9908541911012403`；`F1 >= 0.9997` 未达成，也没有产生 promotion 或 certification evidence。

这不是正式的 Loop151 x Loop164 complementarity gate。Loop69 是旧 Loop61-style Train-only nested OOF surrogate，且随机 folds 与 Loop164 content-component folds 不共享 outer partition。因此正式状态是 `not_run / blocked_wrong_base_lineage_and_fold_scope`；本轮只用负向代理证据决定不为当前 Loop164 recipe 支付昂贵的 Loop151 exact OOF 重建成本。

## 协议

- 只读取冻结的 Train-only aggregate/report/prediction artifacts；不训练、不拟合、不扫阈值。
- 不读取 Val、Test-10k、legacy full-test、sentinel 或 sealed window。
- 跨快照只按 `source_sha256` 连接；`sample_index` 仅用于漂移审计，绝不作为 fallback join key。
- Loop164 missing 行保留 null decision，并回退 Loop69；不得把 neutral score `0.5` 阈值化为恶意判断。
- repairs/breaks 只统计相对 Loop69 hard decision 的真实 decision changes，support 恒等于 `repairs + breaks`。

输入 SHA-256：

- Loop69 predictions: `9f942d68d523a0663ff1f5e9e03e6a0e47feffea34050cdfde96728d1e524a9a`
- Loop69 report: `0eb98433fc756e568e7bc0e66fd1e43c23788fe37bfd026168007972afc4ebb5`
- Loop69 readiness: `78f1452dd8157bdab6a2a2cf58bb1d4566a222324eba3f8edd45cafcaeb76ebd`
- Loop164 predictions: `4f706788d812987714ebd9f717b77f75b10997309dbe7991c083b9928ad3d4df`
- Loop164 report: `da55531d39b628a2a02ec008451b7ad0455f6876cabd91dcb8c56f7e18c3e07f`
- Loop164 folds: `00a31a1bd86d7b887447f3e86e5e753ebcaaee45be74311199332e073a3880a5`

## 对齐回执

两个文件各有 `20,000` 个唯一 SHA，共同样本 `19,996`。索引 `1..4` 是四个同索引替换，不是排序漂移；这四个 Loop164 current-only 样本均为 `read_failure`。共同样本中 supported/missing 为 `19,540/456`，Loop164 完整分母 missing 为 `460`。

fold scope 不兼容：Loop69 是 seed `69` 的随机分层五折，Loop164 是 seed `164` 的 content-component 五折。fold id 归一化后只有 `4,076/19,996` 行编号相同；Loop164 的 `393` 个 non-singleton components 中有 `356` 个跨越 Loop69 folds，涉及 `3,368` 行。因此 Loop69 只能是成本熔断 surrogate，不能替代 decision-aligned Loop151 OOF。

## 代理结果

在 `19,540` 个 supported common rows 上：

| 指标 | 结果 |
|---|---:|
| decision changes | 670 |
| Loop69 error repairs | 75 |
| Loop69 correct breaks | 595 |
| blind-switch precision | 0.1119402985 |
| net error reduction | -520 |
| both wrong | 153 |

五个 Loop164 diagnostic folds 的 repairs/breaks 分别为 `13/130`、`11/101`、`15/116`、`13/84`、`23/164`，全部净负。预注册代理成本熔断要求 support `>=100`、repairs `>=30`、precision `>=0.80`、net reduction `>=1`；前两项通过，后两项失败。

共同分母上的 Loop69 errors 为 `252`。盲切到 Loop164 并在 missing 上保留 base 后 errors 变成 `772`。即使用真值选择两专家中正确者的不可部署 oracle，仍有 `177` 个共同分母错误；把四个 unmatched current rows 全部保守计错后为 `181`。这些数字不是 Full-test 指标，也不能外推为 `0.9997` 证据。

## 为什么不启动 Loop151 exact OOF

现有 Loop151 Train artifacts 不能直接复用：Loop127/134 meta 对同一 Train stack fit 后再预测，content-cross 和 Loop136 selector 也不是 Train OOF；Train signer cache 缺失，Loop134 noise weights 还依赖同一 Train full-fit neural probabilities。历史 hard decision 与保留 score 也不完全 decision-aligned。

不改变历史 recipe 的严格重建至少需要 `3 seeds x 5 outer x (5 inner + 1 final) = 90` 个链级 scope。同 scope 共享 primary/conservative 后仍约 `11,175` 次 downstream estimator fit，串行估计约 `32.3` 小时，计 I/O/receipt 约 `35-36` 小时，并与当前 `24h` A2 TTL 冲突。当前 surrogate 又在所有五折强烈净负，因此不能为 Loop164 单独支付该成本。

## 后续路线

1. Loop151 保持唯一 research champion；当前 Loop164 recipe 不再增加 seed、epoch、threshold、heldout 或 exact OOF 支出。
2. 先建设独立 label-quality 上界、time/family/source data contract，以及更正交的静态 foundation/EMBER-v3 structural/DSRA control 专家。
3. 对 packed、parser-fail 和高不确定尾部，预研 Nebula/Speakeasy-style behavior cascade；timeout 必须进入显式 missing/fallback，不能视为 benign。
4. 只有至少两个新专家各自在相同 outer partition 上显示强、稳定、独立的 Train-OOF 纠错后，才建设共享的 decision-aligned OOF router。届时可以预注册 anchor-teacher 变体降低 neural nested 成本，但它属于新 recipe，不得冒充历史 Loop151 bitwise replay。
5. `public key` 仍不属于本地开发前置条件；它只服务未来独立 A2/A3 custody/certification trust anchor。

## Artifacts

- `scripts/audit_loop165_loop69_loop164_surrogate_complementarity.py`
- `tests/test_audit_loop165_loop69_loop164_surrogate_complementarity.py`
- `reports/roadmap_9997/loop165/loop69_loop164_surrogate_complementarity.json`
- `manifests/roadmap_9997/loop165_surrogate_complementarity/decision.json`

报告 SHA-256 为 `d0aa06074f0123ba5a9ad89a31e3912dfde261eca71d97ed0f2df7d73b5c92ec`。该 artifact 的 evidence role 是 `surrogate_negative_supporting_evidence_not_formal_loop151_gate`。
