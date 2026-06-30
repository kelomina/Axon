# Phase 2 Loop 2: kNN Stage2 Error and Noise Audit

## Scope

This loop analyzes the frozen kNN Stage2 full-test result. It does not change labels, remove samples, tune thresholds, or train a new model.

Inputs:

- Full-test evaluation: `reports/random_20w_split/stage2_knn_extended_frozen_full_test_eval.json`
- Full-test predictions: `reports/random_20w_split/stage2_knn_extended_frozen_full_test_predictions.csv`
- Error queue: `reports/random_20w_split/stage2_knn_full_test_error_review_queue.csv`
- Frozen kNN Stage2 model: `reports/random_20w_split/stage2_knn_extended_valonly/stage2_selected_model.pkl`

Full-test status:

- Rows: `160000 / 160000`
- F1: `0.9828037453`
- Errors: `2753`
- FP/FN: `1423 / 1330`

This confirms that the kNN candidate is useful but still far below the `F1 >= 99.9%` target.

## P0/P1 Error Shape

P0/P1 queue summary:

- Report: `reports/random_20w_split/stage2_knn_full_test_p0_p1_error_summary.json`
- CSV: `reports/random_20w_split/stage2_knn_full_test_p0_p1_error_summary.csv`
- Selected rows: `1262`
- FP/FN: `728 / 534`
- P0/P1 reasons:
  - severe FP (`prob >= 0.95`): `445`
  - high-confidence FP (`prob >= 0.85`): `283`
  - severe FN (`prob <= 0.05`): `251`
  - high-confidence FN (`prob <= 0.15`): `283`

Path and file-shape signals:

- All P0/P1 FP rows come from `待加入白名单`.
- All P0/P1 FN rows come from `待拉黑`.
- P0/P1 FP rows are mostly extensionless PE files: extension `<none>` has `584` FP and only `4` FN.
- P0/P1 FN rows are mostly `.exe`/`.dll`: `.exe` has `372` FN, `.dll` has `143` FN.
- FN errors cluster by month. The largest cluster is `2026-03` with `89` high-priority FN rows, followed by `2020-11` with `66`.

Interpretation: the high-confidence errors are not randomly scattered. They are structured by source bucket and file family/time bucket, which is exactly the kind of pattern expected from source-label ambiguity, family drift, or clustered blind spots.

## PE Metadata Audit

PE audit:

- Report: `reports/random_20w_split/stage2_knn_full_test_p0_p1_pe_metadata.json`
- CSV: `reports/random_20w_split/stage2_knn_full_test_p0_p1_pe_metadata.csv`
- Selected rows: `1262`
- Parseable PE rows: `1262`
- Parse errors: `0`
- High-entropy section samples: `322`
- Writable+executable section samples: `168`
- Overlay > 1MB: `68`

Interpretation: these are not cache-corrupt or non-PE garbage samples. They are valid PE files. Some have suspicious PE properties, especially among high-confidence FP rows, where writable+executable sections and high entropy are more common. That makes automatic deletion unsafe: a high-confidence FP can be either a false alarm on a suspicious benign utility or a mislabeled risky binary in the white bucket.

## kNN Neighbor Audit

Frozen kNN neighbor audit:

- Report: `reports/random_20w_split/stage2_knn_full_test_p0_p1_neighbor_audit.json`
- CSV: `reports/random_20w_split/stage2_knn_full_test_p0_p1_neighbor_audit.csv`
- Selected rows: `1262`
- Train memory rows: `20000`
- Neighbor top-k: `25`

Support buckets:

- Neighbors support model prediction: `843`
- Neighbors mixed: `352`
- Neighbors support dataset label: `67`

This is the strongest signal in this loop. For `843 / 1262` high-priority errors, the nearest training examples mostly agree with the model's prediction rather than the dataset label. That does not prove every such row is mislabeled, but it strongly suggests that source-label conflict and semantic ambiguity are major blockers.

## Scientific Feasibility Update

The current full-test model still makes `2753` errors. The P0/P1 subset alone contains `1262` high-confidence errors, and `843` of those have train-neighbor evidence supporting the model's prediction. Even if some are real model mistakes, the amount of high-confidence conflict is much larger than the error budget implied by `F1 >= 99.9%`.

Therefore, the current evidence does not support continuing with blind model tuning as the main path to 99.9%. The next limiting factor is data adjudication quality:

1. Confirm whether high-confidence white-bucket FP rows are actually safe benign files.
2. Confirm whether high-confidence black-bucket FN rows are truly malicious and in-scope.
3. Identify repeated families/months where test rows are near-identical to oppositely labeled train rows.

Until that is done, the realistic model-only target remains around the current `98.x%` band, not `99.9%`.

## Next Actions

Data-Agent:

- Build a manual review package from the top P0 rows where neighbors support model prediction.
- Split it into FP-white and FN-black packages so business adjudication can be done separately.
- Do not auto-remove or relabel samples from these reports alone.

Error-Agent:

- For the `2026-03` FN cluster, compare section names and nearest-neighbor paths to determine whether it is a new family blind spot or a source-label conflict.
- For high-confidence FP extensionless PE rows, inspect whether they are benign security tools, packed installers, or mislabeled malware-like files.

Model-Agent:

- Do not start another full-test candidate until a new Val candidate beats `0.9833349965`.
- Next model experiments should focus on features that can separate suspicious benign PE files from true malware, not just stronger tabular fitting.

Eval-Agent:

- Keep Test-10k and full-test frozen for confirmation only.
- If manual adjudication produces a revised train/val policy, rerun Val from scratch before any new Test-10k.
