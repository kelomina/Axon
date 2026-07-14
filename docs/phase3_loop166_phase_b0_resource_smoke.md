# Loop166 Phase B0 Zero-Drop Resource Smoke

## 结论

Loop166 Phase B0 最终通过 `phase_b0_zero_drop_resource_gate_pass_allow_one_b1_full_outer_resource_cell`。这只证明本机能够运行冻结的 11.24M tiny MLM，并且 byte-BPE 序列准备已经做到原始字节零丢失；它不是分类质量、OOF、F1、promotion 或认证证据。

Loop151 仍是唯一 research champion。B1 只允许执行一次 outer-fold-0 留出、Train-only 的完整 fit 资源单元；五折、Val、Test-10k、full-test 和阈值操作仍全部关闭。本地 Phase B0/B1 不需要 public key。

## 两次硬化

首次使用 PyTorch 默认 GradScaler scale 的尝试在第一个 optimizer boundary 失败：loss 与 logits 有限，但 `unscale_` 后 gradient norm 非有限，optimizer step 为 `0`。该失败回执曾被后续同路径稳定重跑覆盖，因此只保留在治理 observation 中，不能冒充有独立 artifact 的复验。最终合同固定 `initial_scale=128`、`growth_interval=1000`，稳定运行的 nonfinite 为 `0`。

第一版稳定 smoke 暴露了更重要的方法学问题：`510` 个 512-byte windows 中有 `11` 个 BPE 后超过 510 个 content tokens，旧实现把整窗排除。这会系统性低估高熵或 packed code，不能放大到完整 outer。该版本虽然资源门通过，仍被裁决为 `full_outer_blocked_pending_zero_drop_remediation`。

最终实现把每个 window 的 BPE ids 按最多 `510` tokens 无损拆段，每段加 CLS/SEP。每段独立解码并逐字节复原原窗口；特殊 token、unknown id、空 chunk 或 byte commitment 漂移都会 fail closed。

## 最终结果

| 项目 | 结果 |
|---|---:|
| fit / outer-holdout raw opens | `64 / 0` |
| selected windows | `510` |
| prepared sequences | `521` |
| split windows / sequence expansion | `11 / 11` |
| original / prepared bytes | `260628 / 260628` |
| overlength windows excluded | `0` |
| parameters | `11241093` |
| optimizer steps / microbatches | `8 / 16` |
| training throughput | `10461.1251 original bytes/s` |
| peak CUDA allocated / reserved | `265512960 / 293601280 bytes` |
| peak process RSS | `1995390976 bytes` |
| total elapsed | `11.5645 s` |
| OOM / nonfinite | `0 / 0` |
| atomic tokenizer / checkpoint | `pass / pass` |
| weights-only exact-logit recovery | `pass` |

训练没有使用 label、path、SHA、fold、Loop score、PE/stat/content sidecar 或 signer 作为模型输入。没有计算 loss curve、perplexity、accuracy、precision、recall、F1 或 AUC，也没有生成 hard decision 或扫描阈值。

## B1 边界

B1 的真实口径是 outer fold 0 留出后的 `16000` 条 Train metadata，而不是只把 `15649` 条旧 8 MiB-supported rows 称为 full outer。fold 0 的 `4000` 条只允许 metadata scope audit，raw opens 必须为 `0`。fit 侧必须尝试 `15988` 条已知大小文件并重试 `12` 条旧 source-unavailable rows；旧 `339` 条 oversize 不能因 Loop164 的 8 MiB cap 被静默排除。

B1 仍只做资源和恢复裁决：fresh outer-fit tokenizer、one exact corpus epoch、compact uint16 token storage、4096-step atomic checkpoint 和真实 fresh-process resume。任何字节丢失、identity/label 入模、OOM/nonfinite、资源门失败、heldout raw access 或质量指标计算都会关闭当前 cell，不扩五折。

## Artifacts

- `manifests/roadmap_9997/loop166_code_section_foundation/phase_b0_pre_zero_drop_observation.json`
- `manifests/roadmap_9997/loop166_code_section_foundation/phase_b0_resource_smoke.json`
- `manifests/roadmap_9997/loop166_code_section_foundation/phase_b0_decision.json`
- `reports/roadmap_9997/loop166/phase_b0_resource_smoke.json`
- `reports/roadmap_9997/loop166/phase_b0_tokenizer.json`
- `models/roadmap_9997/loop166/phase_b0_tiny_mlm.pt`
- `src/loop166/byte_bpe.py`
- `src/loop166/mlm_model.py`
- `scripts/run_loop166_phase_b0_resource_smoke.py`
- `tests/test_loop166_phase_b0.py`
