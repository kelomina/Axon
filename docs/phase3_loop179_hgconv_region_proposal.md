# Loop179 HGConv-Region Proposal

Updated: 2026-07-20

## 1. Status and Claim Boundary

This document freezes a candidate architecture and experiment contract before any Loop179
implementation, resource cell, fit, OOF score, or held-out access. It does not authorize execution.

Loop179 is a new lineage. It does not modify `src/loop175/`, does not rescue Loop175 checkpoints,
and does not claim to reproduce a full-file HGConv system. Its first question is narrower:

> Does an HGConv patch backbone extract causal PE-region evidence that improves the same frozen
> B0 control under the same component-isolated Train OOF protocol?

The research champion remains Loop151 until a later frozen pipeline passes Val, Test-10k, and one
confirmation-only full-test with `F1 >= 0.9997`. Train-only OOF cannot establish that target.

## 2. Entry Condition

Loop179 may create a new proposal/source closure only after the immutable Loop175 postdecision
selects one of these branches:

- `close_loop175_allow_loop179_resource_proposal_only`; or
- `close_loop175_allow_loop179_hgconv_region_proposal_only`.

The current missing-exit E4 replay may provide scientific OOF information, but it is not promotion
authority. No Loop179 real-input access or training may begin from this document alone.

## 3. External Reference Closure

The implementation reference set is frozen to:

- paper: *Holographic Global Convolutional Networks for Long-Range Prediction Tasks in Malware
  Detection*, arXiv `2403.17978v1`, 2024-03-23, AISTATS 2024;
- HGConv repository: `FutureComputing4AI/HGConv`, commit
  `95bea530460afec8be967d3766d185ff4475b8ec`, Apache-2.0;
- original HGConv code parent: commit
  `7f9a6b00b9f1e2805b204e2aeae4010f8e7023bc`;
- HRR bind/unbind reference: `MahmudulAlam/Holographic-Reduced-Representations`, commit
  `d2e34b073b39936adcc314dbe9efd5f6ed8353b0`, MIT.

Before implementation closure, the exact referenced files, license files, and golden vectors must
be downloaded into an evidence-only reference bundle with SHA-256 commitments. A URL alone is not
a source closure. A new PyTorch implementation is project code, not an official port.

## 4. Frozen HGConv Variant

The first variant follows the official malware-task shape rather than selecting among inconsistent
LRA variants after observing results:

| Field | Frozen value |
|---|---:|
| input patches | `[N, 512, 192]`, `N = batch * 16` |
| HGConv blocks | `1` |
| sequence length `T` | `512` |
| hidden width `H` | `192` |
| convolution filter length `K` | `32` |
| bind filter shape | `[192]` |
| sequence filter shape | `[1, 32, 192]`, zero-padded to `T` |
| unbind filter shape | `[192]` |
| bias/gate weight shape | `[1, 1, 192]` |
| dropout | `0.1` |
| patch size | `16` bytes |
| regions per file | `16` |
| bytes per region | `8192` |

The mathematical axes are frozen:

1. bind is real circular convolution along feature axis `H`;
2. global convolution is real circular convolution along sequence axis `T`;
3. unbind uses the official HRR approximate inverse, `flip` then `roll(+1)` along `H`, followed by
   circular binding; it must not be silently replaced by reciprocal-spectrum exact inversion;
4. the malware reference kernel preconditioning is frozen and tested separately from the paper
   equation: FFT to length `T`, inverse FFT with `norm="ortho"`, then circular binding;
5. all FFT operations execute in FP32 or FP64 even when the public training path uses BF16;
6. complex tensors remain internal and outputs are real, finite, and shape preserving.

The official repositories use different normalization in malware and LRA tasks. Loop179 therefore
records the exact normalization mode in its protocol and does not search it.

## 5. Model Boundary

Loop179 uses an independent `src/loop179/` package. It does not import private Loop175 model or
engine classes.

The public model ABI remains comparable with Loop175:

- `region_tokens`: `[B, 16, 8192]`;
- `region_lengths`, `region_types`, `offset_buckets`, `length_buckets`: `[B, 16]`;
- optional B0 values: `[B, 571]`;
- output `region_features`: `[B, 192]`;
- output `region_logits`: `[B, 2]`;
- output `fusion_logits`: `[B, 2]` when B0 is present.

Only the six-layer dilated-GLU patch backbone is replaced. Byte embedding, patch projection,
region metadata embeddings, gated attentive/max pooling, two-layer region Transformer, B0
projector, and classification heads remain shape-equivalent and separately committed.

The minimum Phase0/resource write set is:

- `src/loop179/contracts.py`;
- `src/loop179/hgconv.py`;
- `src/loop179/model.py`;
- `src/loop179/data_adapter.py`;
- `src/loop179/resource_cell.py`;
- `src/loop179/source_closure.py`;
- Phase0/source-closure/resource scripts and tests.

Phase-B evaluation, worker, controller, and receipt modules are deferred until the resource cell
passes. Loop175 arm names, seed domains, receipt schemas, or checkpoint schemas must not be reused.

## 6. Mask and Failure Semantics

Patch masks apply before bind and after every residual block. Masked patches are exactly zero and
cannot wrap through circular convolution into valid positions. Changing padding bytes outside the
declared length must not change a valid output. An all-masked region returns the frozen zero/null
representation and remains finite.

The implementation must fail closed on:

- wrong rank or any shape other than the frozen ABI;
- non-finite input or output;
- invalid token, length, type, bucket, or padding semantics;
- complex leakage at a public boundary;
- unexpected FFT axis, sign, normalization, or shift orientation;
- silent clamp, `nan_to_num`, gradient repair, row filtering, or denominator reduction.

The official `nan_to_num` gradient helper is not copied because it would conceal a non-finite
failure that the Axon resource contract requires to close.

## 7. Phase0: Mathematical and Static Closure

Phase0 uses synthetic tensors only and opens no raw PE, region cache, Train row, Val, Test-10k, or
full-test data.

Required tests:

1. circular bind equals an explicit modulo-index sum along `H`;
2. length-`K` sequence convolution, zero-padded to `T`, equals an explicit modulo-index sum along
   `T`, including non-power-of-two synthetic sizes and impulse orientation cases;
3. approximate inverse equals the committed HRR `flip + roll(+1)` reference;
4. a golden JAX forward and first-order gradient vector matches the PyTorch implementation under
   the frozen malware normalization;
5. padding/mask isolation covers trailing, interior, empty, and all-masked cases;
6. CPU float64 gradcheck and ordinary FP32 backward produce finite gradients for every parameter;
7. fixed seed creates the same state-dict commitment and deterministic eval output;
8. full synthetic `[2, 16, 8192]` forward preserves the expected model ABI;
9. source closure rejects missing, extra, duplicate, path-escape, symlink, SHA, reference, or
   transitive-import drift.

Phase0 output is a canonical receipt with `raw_rows_opened=0`, `training_runs=0`, and
`execution_authorized=false`. Passing Phase0 is not resource or quality evidence.

## 8. Cache and Data-Control Semantics

Loop179 may bind the existing sealed Loop175 region cache and 571-value B0/fold authorities by
exact SHA, but it requires a new adapter and new source closure.

The existing loader validates and reads the complete ZIP_STORED archive. Therefore the resource
cell may claim only:

- `fold0_model_rows_materialized = 0`;
- `fold0_fit_rows = 0`;
- `fold0_selection_rows = 0`;
- `fold0_prediction_rows = 0`.

It must not claim physical zero-byte access to fold0 during whole-artifact hash/schema validation.
If byte-level isolation becomes mandatory, a separately authorized fold-sharded cache is required.

Path, filename, extension, SHA, row, fold, component, family, source, and reviewer identity remain
control-plane fields and never enter model tensors.

## 9. PhaseA: Maximum-Shape Resource Cell

After Phase0, a fresh resource source closure, guard, short-lived authorization, and atomically
consumed lease are required. The cell runs only the maximum J path:

- fit rows: component folds `2/3/4`, exactly `12,000`;
- selection rows: component fold `1`, exactly `4,000`;
- fold0 model rows: zero;
- maximum epochs: `12`, earliest minimum unweighted CE;
- microbatch `2`, accumulation `16`, effective batch `32`;
- BF16 autocast with FP32 optimizer, norm, loss, and FFT;
- AdamW `3e-4`, weight decay `1e-2`, warmup `1`, cosine schedule, clip `1.0`, EMA `0.999`.

Resource gates:

- GPU allocated `<= 6,979,321,856` bytes;
- RSS `<= 11,811,160,064` bytes;
- wall `<= 21,600` seconds;
- silent drops `0` and all `16,000` rows accounted;
- OOM, timeout, and non-finite are all false;
- fixed-input eval logits are bitwise deterministic;
- output disk remains inside a separately frozen budget.

Any failure closes this architecture variant. No block, width, filter, epoch, LR, precision, or
sequence-length change is allowed as rescue.

## 10. PhaseB: Train-Only Causal OOF

Only a passing resource cell may authorize the four-arm, five-fold seed41 experiment:

- A: frozen 571-value B0 HGB control;
- H: HGConv-Region only;
- J: B0 + HGConv-Region early fusion;
- K: J with partition-local, zero-fixed-point whole-region ownership shuffle.

K uses a new Loop179 permutation domain and moves bytes, type, start, offset bucket, length bucket,
and length together. J and K must have identical parameter sets and capacity.

Every arm produces exactly 20,000 finite OOF probabilities, one score per row, with five folds of
4,000 and fixed decision `p > 0.5`. J must satisfy all gates:

- at least `30` fewer errors than A;
- at least `50` repairs;
- override precision at least `0.80`;
- at least `4/5` net-positive folds;
- A-to-J component-bootstrap one-sided 95% LCB `>0`;
- FP and FN relative worsening each `<=5%`;
- K has at least `30` more errors than J;
- K-to-J component-bootstrap one-sided 95% LCB `>0`;
- all integrity and resource gates pass.

Any gate failure closes HGConv-Region. It does not authorize posthoc threshold, seed, architecture,
weight, filter, or normalization search.

## 11. Reproduction Artifacts

Each executable phase requires immutable bindings for code revision, protocol, external reference
bundle, input authorities, cache, fold manifest, exact argv, Python/Torch/CUDA versions, seed domain,
resource guard, authorization, lease, logs, checkpoints, OOF numeric commitments, worker receipts,
aggregate receipt, and failure/interruption receipts.

The current decision remains:

`draft_wait_for_loop175_postdecision_no_loop179_execution_authorized`
