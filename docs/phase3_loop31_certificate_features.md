# Loop31 Certificate Feature Follow-Up

Date: 2026-07-01

## Objective

Loop31 tested whether Authenticode certificate blob features can improve the
Loop28 content PE metadata model. This was motivated by Loop28 full-test error
diagnostics:

- True benign rows are often signed.
- Loop28 false negatives also contain many signed malicious files.
- Therefore, the binary signal "has signature" is not enough; signer/OID/blob
  details might have helped.

## Feature Boundary

The new features are content-derived only. They read the PE Security Directory
blob and do not encode filename, extension, directory name, or path text.

Feature groups:

- WIN_CERTIFICATE header fields
- certificate blob size and entropy
- printable/UTF-16 string run statistics
- PKCS#7/code-signing/timestamp/hash/signature OID presence
- common CA/vendor text hits, such as Microsoft, DigiCert, Sectigo, GlobalSign,
  VeriSign/Symantec/Thawte, Entrust, Google, Adobe, Intel, NVIDIA, Oracle,
  Mozilla, Kaspersky, Avast

## Train/Val Cache

- Input rows: `40000`
- Unique rows: `40000`
- Feature dim: `55`
- Created: `40000`
- zero_features: `26249`

`zero_features` mostly means the file has no Authenticode Security Directory,
not that extraction failed.

## Val Result

Loop31 used Loop28 content PE metadata plus certificate blob features.

Best Val candidate:

- Model: `hgb_lr0.06_leaf31_l2_0__noise_trim_extreme_conflict`
- Threshold: `0.465`
- F1: `0.9916100679`
- Errors: `168`
- FP/FN: `96/72`

Comparison:

| Candidate | Val F1 | Val errors |
| --- | ---: | ---: |
| Loop28 content PE | `0.9919048571` | `162` |
| Loop31 content PE + cert | `0.9916100679` | `168` |

## Decision

Reject Loop31 for Test-10k promotion. It did not beat Loop28 on Val, so it
fails the funnel gate.

Interpretation: the residual signed-file problem is real, but shallow
certificate blob indicators are not enough. If this direction is revisited, it
should use a real PKCS#7/certificate parser or Windows trust-chain validation,
then validate strictly on Val before any Test-10k run.
