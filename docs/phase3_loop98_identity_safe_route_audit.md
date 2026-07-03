# Phase 3 Loop98 Identity-Safe Route Audit

Loop98 adds a read-only route gate after the full-error evidence queue and
Speakeasy triage work. It does not train, tune thresholds, load checkpoints,
open NPZ arrays, or mutate the split/cache.

## Decision

Current decision: `await_independent_blinded_verdicts`.

The current best remains Loop57 full-test:

- rows: `160000`
- F1: `0.9883629658239992`
- errors: `1868`
- FP/FN: `1195/673`

The approximate `F1 >= 0.999` error budget is about `160` errors, so the
remaining gap is roughly `1708` fewer errors. Existing automatic routes do not
cover that gap.

## Route Status

Fixed-v2 cache and redraw state is healthy:

- 130 bad-feature slots were replaced by a fresh full redraw, not by filling
  from the bad rows.
- replacements: `130/130`
- self replacements: `0`
- current split/cache: `200000/200000`, missing `0`
- split shape remains `20000/20000/160000`

Closed or blocked automatic routes:

- Probability calibrator: useful versus the 8192 baseline, but worse than
  Loop57 on full test, so not a final candidate.
- Current Loop57/calibrator fusion: Loop83 score-delta and Loop84 content
  selector both failed Val-only evidence.
- Speakeasy timeout/dynamic triage: useful as manual context only; confirmation
  introduced false negatives, so no automatic merge, threshold override,
  training, or Test-10k.
- Full queue review: `1868/1868` rows are packaged and blinded, but actionable
  verdicts are still `0`; no replacement, training, or Test-10k is authorized.

## Identity Boundary

User concern is correct: real deployment names and paths do not match training
names, and attackers can rename files freely. Therefore filename, path,
directory, extension, hash, `source_sha256`, `sample_index`, split, row order,
and model scores are forbidden as model, verdict, replacement-sampling,
threshold, fusion, or production inference evidence.

Those fields are allowed only for loading, alignment, cache audit, duplicate
detection, and private review indexing. Labels for the 20w protocol come from
the locked split/manifest. If a bad row is confirmed, it must be quarantined
and replaced by a fresh valid sample from the same locked-manifest original
label pool.

## Next Step

The only open route toward the target is independent blinded content/external
verdicts through Loop96 -> Loop87. If verdicts confirm `label_wrong`,
`feature_broken`, or `out_of_scope`, the next step is non-destructive fresh
same-original-label redraw preflight, then full cache readiness, then Val-first
training or evaluation. No current evidence authorizes direct Test-10k or
full-test experimentation.
