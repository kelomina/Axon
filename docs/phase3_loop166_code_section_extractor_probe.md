# Loop166 Code-Section Extractor Phase A Report

## 结论

Phase A 工程门通过：`phase_a_extractor_gate_pass`。在 SHA-bound、balanced Train-only `256` 行上，提取成功 `251`、显式 missing `5`、silent drop `0`，coverage `0.98046875`。这只证明 code-section 数据入口、完整性和资源可行，不是 tokenizer/MLM/model quality、OOF、promotion 或 `F1 >= 0.9997` 证据。

Loop151 仍是唯一 research champion；Loop166 现在只允许进入一个 real outer tiny-MLM resource cell，五折、Val、Test-10k、full-test 和 promotion 仍全部关闭。

## 输入与协议

- bundle: `reports/roadmap_9997/loop164/local_probe_bundle.jsonl`
- bundle SHA-256: `90961bfed0460787e261965a3180e1b0569df0f9d275f9693daad1ccf53dc233`
- rows: `128 benign + 128 malicious`
- role: canonical Train-only local runtime probe
- source size: `64 KiB..8 MiB`
- public key: not required

每个文件先验证 size、SHA-256 和打开前后 fingerprint，再用 `pefile` 读取所有 `IMAGE_SCN_MEM_EXECUTE` sections。任一 executable span 越界会使整行显式 missing；有效 spans 按 raw offset 排序并合并重叠，section name 不进入模型。代码字节只在内存中用于统计与 aggregate commitment，不写 raw-code artifact。

## 结果

| 指标 | 结果 |
|---|---:|
| denominator | 256 |
| success | 251 |
| missing | 5 |
| no executable section | 5 |
| parse / invalid-span / zero-raw failure | 0 / 0 / 0 |
| coverage | 0.98046875 |
| raw bytes verified | 191,000,679 |
| code bytes observed, not persisted | 104,869,232 |
| raw-code artifact bytes | 0 |
| elapsed | 1.2943352 s |
| peak RSS | 48,603,136 bytes |

五个 missing 全部来自 benign rows 的 `no_executable_section`。这只是 extractor coverage slice，不是分类结论，不能把 missing 当 benign；未来系统必须在完整 `20,000` 分母上走预注册 structural fallback。

成功样本 code bytes 分布：median `172,544`、p95 `1,516,032`、p99 `3,049,482`、max `5,615,616`。extraction commitment SHA-256 为 `a3939a1ade92e1b379c57529252565b797f148ff11812a1ad8db940f8290e83e`。

所有门均通过：denominator conserved、coverage `>=0.85`、wall `<=180s`、peak RSS `<1GiB`、silent drop `0`、raw-code output `0`。

## 下一门

下一步只实现 proposal 中冻结的 one-outer tiny-MLM resource cell：BPE `1024+5`、sequence `512`、最多 `8` chunks/file、6-layer/384-hidden scaled encoder。tokenizer、MLM 和 classifier 必须 outer-fit-only。单折超过 `8h`、低于 `2000 original-byte-equivalent tokens/s`、OOM/nonfinite 或不可恢复时立即停止，不扩五折。

该 resource cell 仍不选阈值、不访问 heldout、不产生新 champion。只有资源门通过，才允许 one-seed five-fold Train OOF 与 byte/scratch/structural controls。

## Artifacts

- `src/loop166/code_sections.py`
- `scripts/run_loop166_code_section_extractor_probe.py`
- `tests/test_loop166_code_sections.py`
- `reports/roadmap_9997/loop166/code_section_extractor_probe.json`
- `manifests/roadmap_9997/loop166_code_section_foundation/proposal.json`
