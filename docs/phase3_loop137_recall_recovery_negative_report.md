# Phase 3 Loop137 Recall Recovery Report

更新时间：2026-07-05

## 目标

Loop136 已经是当前 strict best，但它相比 R5 的代价是 FN 增加。Loop137 的目标是只在 Train/Val 证据允许时做召回恢复：优先减少 FN，同时不把 Test-10k 或 full-test 结果用于阈值、规则或模型选择。

本轮仍遵守身份字段禁令：`source_path`、文件名、目录名、后缀、`source_sha256`、`sample_index`、`cache_path`、split 和 row order 只用于对齐、cache lookup 与审计，不作为模型证据。

## 门禁补强

先补强 `scripts/train_loop135_pairwise_selector.py` 的 Val 选择逻辑，新增：

- `--require-val-improvement`
- `--min-val-error-reduction`
- `--min-val-f1-delta`

这样 selector 不能再选中“只是满足 FN 约束但没有真正优于 baseline”的候选。测试：`tests/test_loop135_pairwise_selector.py`，结果 `7 passed`。

## 候选 A：Loop136 vs fixed-v2 string selector

在 Loop136 baseline 上继续尝试接纳 fixed-v2/string 候选，只允许 Val 上 FN 不增加，并且要求 Val 总错误和 F1 真改善。

| Candidate | Val F1 | Val Errors | FP | FN | Test-10k Errors | Decision |
|---|---:|---:|---:|---:|---:|---|
| fixed-v2 + string | `0.9912280702` | `176` | `120` | `56` | `84` | Test-10k 弱于 Loop136 `83`，拒绝 |
| fixed-v2 no-logreg | `0.9911777900` | `177` | `120` | `57` | `83` | Test-10k 只打平且没有接管，拒绝 |
| fixed-v2 with-logreg | `0.9911292734` | `178` | `122` | `56` | `83` | Test-10k 只打平且没有接管，拒绝 |

关键 artifact：

- `reports/phase3_loop137/loop136_vs_fixedv2_string_fn0_valonly/loop135_pairwise_selector_report.json`
- `reports/phase3_loop137/loop136_vs_fixedv2_string_fn0_test10k_eval.json`
- `reports/phase3_loop137/loop136_vs_fixedv2_nologreg_fn0_test10k_eval.json`
- `reports/phase3_loop137/loop136_vs_fixedv2_logreg_fn0_test10k_eval.json`

## 候选 B：Loop136 FN recovery R11

复用预声明的 Loop132 FN recovery 规则，但把 base 从 R5 换成 Loop136。Val 选择仍只在 Val 上完成，最终选择 R11。

| Split | Model | F1 | Errors | FP | FN | Decision |
|---|---|---:|---:|---:|---:|---|
| Val | Loop136 baseline | `0.9910789933` | `179` | `122` | `57` | baseline |
| Val | Loop137 R11 | `0.9913329348` | `174` | `125` | `49` | enter Test-10k |
| Test-10k | Loop136 baseline | `0.9916958479` | `83` | `54` | `29` | baseline |
| Test-10k | Loop137 R11 | `0.9917024893` | `83` | `58` | `25` | F1/recall improve, enter full-test |
| Full-test | Loop136 baseline | `0.9903723842` | `1544` | `958` | `586` | current best |
| Full-test | Loop137 R11 | `0.9901392220` | `1583` | `1059` | `524` | reject |

R11 在 full-test 上确实修复了 `62` 个 FN，但新增了 `101` 个 FP，净错误 `+39`。这说明该 FN recovery slice 在 Val 和 Test-10k 上看起来合理，但跨到 16 万 full-test 后 FP 外溢明显，不能采用。

Full-test overlap：

- `reports/phase3_loop137/loop137_r11_full_overlap_vs_loop136.json`
- fixed Loop136 errors：`62`
- new Loop137 errors：`101`
- shared errors：`1482`

## Decision

Loop137 全部拒绝，不替代 Loop136。当前 strict best 仍是 Loop136：

- full-test F1：`0.9903723842`
- errors：`1544 / 160000`
- FP/FN：`958 / 586`

本轮最重要的科学结论是：当前召回恢复规则在 full-test 上的 FP 风险高于 Val/Test-10k 所显示的风险。下一步不应继续扩写同类小规则，而应做更强的噪声/邻域审计，或者引入更正交的行为/动态特征；否则距离 F1 `99.9%` 仍然过远。

