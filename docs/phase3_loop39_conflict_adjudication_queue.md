# Loop39 High-Confidence Conflict Adjudication Queue

Date: 2026-07-02

## Objective

Loop39 converts the Loop38 high-confidence conflict strata into a manual
adjudication queue. It does not train a model, does not relabel automatically,
and does not alter the `20000 / 20000 / 160000` split.

The queue exists because `F1 >= 99.9%` cannot be treated as a pure modeling
problem while hundreds of high-confidence conflicts remain unresolved.

## Hard Rules

- Filename, path, extension, directory, source hash, sample id, split, and row
  order are not model features.
- `source_path` and `source_sha256` are audit and identity fields only.
- Manual verdict fields are intentionally blank.
- A bad row must never be used to fill itself.
- If a row is confirmed `feature_broken`, `out_of_scope`, or `label_wrong`, the
  replacement action is to re-sample one fresh valid candidate from the same
  label pool while preserving the exact `200000` total row count.
- Duplicate content must be handled as a content group, not as unrelated rows.

## Inputs

- Loop38 residual strata details:
  `reports/random_20w_split/loop38_loop28_residual_strata/loop28_residual_strata_details.csv`

Selected buckets:

- `severe_fn_conflict_prob_le_0.01`
- `high_fn_conflict_prob_le_0.05`
- `severe_fp_conflict_prob_ge_0.99`
- `high_fp_conflict_prob_ge_0.95`

## Output

- Queue CSV:
  `reports/random_20w_split/loop39_loop28_conflict_adjudication/loop28_conflict_adjudication_queue.csv`
- Summary JSON:
  `reports/random_20w_split/loop39_loop28_conflict_adjudication/loop28_conflict_adjudication_summary.json`

## Result

Queue size:

- total rows: `649`
- FP: `416`
- FN: `233`

Lane counts:

| Lane | Count |
| --- | ---: |
| `A_unfixed_severe_conflict` | `256` |
| `B_corrected_severe_conflict` | `42` |
| `C_unfixed_high_conflict` | `245` |
| `D_corrected_high_conflict` | `106` |

Conflict buckets:

| Bucket | Count |
| --- | ---: |
| `severe_fp_conflict_prob_ge_0.99` | `200` |
| `high_fp_conflict_prob_ge_0.95` | `216` |
| `severe_fn_conflict_prob_le_0.01` | `98` |
| `high_fn_conflict_prob_le_0.05` | `135` |

Model-agreement split:

- not corrected by any compared model: `501`
- corrected by at least one compared model: `148`

Manual fields:

- blank `manual_label_verdict`: `649/649`
- blank `recommended_action`: `649/649`

Duplicate content:

- duplicate SHA groups in the queue: `2`
- duplicate extra rows in the queue: `2`

The duplicate SHA groups are malicious FN rows from the `2026-03-01` batch that
appear twice with the same content hash but different paths. They should be
reviewed as content groups. If removal or replacement is approved, each removed
row still requires a fresh valid same-label replacement to keep the total split
size unchanged.

## Interpretation

Loop39 turns the data/noise problem into a controlled work queue. It does not
claim these `649` rows are all bad labels. It claims they are the rows most
likely to determine whether `99.9%` is feasible.

The highest-priority group is `A_unfixed_severe_conflict`: `256` rows that
Loop28 predicts with extreme confidence in the wrong direction and no compared
candidate fixes. These should be inspected before spending more time on broad
feature additions.

## Verification

Commands run:

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\build_loop28_conflict_adjudication_queue.py
.\vnev\Scripts\python.exe -m pytest tests\test_build_loop28_conflict_adjudication_queue.py
```
