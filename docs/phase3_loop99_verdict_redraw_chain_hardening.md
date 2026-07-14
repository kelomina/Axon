# Phase 3 Loop99 Verdict-Redraw Chain Hardening

Loop99 tightens the data-governance path after the full-error blinded review.
It does not train, evaluate, tune thresholds, load checkpoints, open NPZ arrays,
or mutate split/cache files.

## Change

`scripts/build_loop76_redraw_readiness.py` now blocks any adjustment plan row
that is not `exclude_and_replace` in the full-error redraw workflow.

This closes an important bypass: the generic manual adjustment planner can still
express `label_wrong + corrected_label` as `relabel` for older train/val
workflows, but the 20w full-error governance path must not use held-out verdicts
as training policy or direct relabel instructions. Confirmed `label_wrong`,
`feature_broken`, or `out_of_scope` rows must be quarantined and replaced by a
fresh valid sample from the same locked-manifest original-label pool.

Loop76 also now accepts Loop87 full-queue verdict-import JSON schema. Loop87
uses `rows/expected_rows/duplicate_sample_index_rows` rather than the older
Loop74/75 `review_rows/input_alignment` fields. Loop76 maps those fields into
the common readiness summary while still requiring the adjustment plan to prove
the exact `200000 = 20000/20000/160000` split shape.

`scripts/build_corrected_split_from_plan.py` is also hardened. It now enforces
the strict 20w shape by default and rejects direct `relabel` plan rows unless
the caller explicitly passes `--allow-relabel-legacy`. The Loop87/Loop96
full-error redraw path must not pass that legacy flag. This makes the low-level
split materializer match the higher-level Loop76 policy: confirmed bad rows are
fresh redraw requests, not direct training-policy relabels.

## Verified Current State

Real no-op replay:

- strict import: `reports/random_20w_split/loop96_full_queue_verdict_import.json`
- adjustment plan: `reports/random_20w_split/loop75_empty_external_adjustment_plan.json`
- output: `reports/random_20w_split/loop99_noop_redraw_readiness.json`

Result:

- decision: `await_external_verdicts`
- strict failures: `[]`
- review rows: `1868`
- sample-index matches: `1868`
- replacement required: `0`
- training policy rows: `0`
- Train/Val allowed: `false`
- Test-10k allowed: `false`
- full-test allowed: `false`

## Red Lines

- No direct relabel from Loop96/87 full-error review.
- No low-level corrected split materialization from a `relabel` row unless an
  older non-full-error workflow explicitly opts into `--allow-relabel-legacy`.
- No corrected split build outside the strict 20w shape unless a non-20w test
  fixture explicitly opts out with `--no-strict-20w`.
- No held-out test verdict may become training policy.
- No identity fields or model scores may be verdict evidence.
- No self-fill or count supplementation from bad rows.
- No Train/Val, Test-10k, or full-test until fresh redraw, replacement
  integrity, cache readiness, and Val-first selection gates pass.
