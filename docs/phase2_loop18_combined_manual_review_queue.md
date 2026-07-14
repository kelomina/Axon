# Phase 2 Loop 18: Combined Manual Review Queue

## Scope

Loop 18 consolidates the two Loop 17 P0/P1 manual review packages into one
deduplicated queue for human reviewers.

This loop does not train, relabel, replace samples, edit the split, edit the
cache, tune thresholds, tune blend weights, tune feature masks, run Test-10k,
or touch the full-test split. It only reduces human-review coordination risk:
one reviewer queue is easier to assign and less likely to produce conflicting
manual decisions across separate CSV files.

## New Tool

Script:

- `scripts/build_combined_manual_review_queue.py`

Test:

- `tests/test_build_combined_manual_review_queue.py`

Safety behavior:

- Combines multiple manual-review CSVs into one output CSV/JSON.
- Deduplicates by `source_sha256`; if SHA is missing, falls back to normalized
  `source_path`.
- Adds `combined_rank`, `review_sources`, `review_source_count`, `dedup_key`,
  and `dedup_method`.
- Keeps model evidence columns unchanged.
- Clears output manual fields by design:
  `manual_label_verdict`, `manual_verdict_note`, `recommended_action`.
- Rejects inputs with pre-filled manual fields by default, so it cannot
  silently overwrite or merge human verdicts after review has started.

## Inputs

- Model-supported P0/P1:
  `reports/random_20w_split/stage2_corrected_best_val_model_supported_p0_p1_manual_review.csv`
- Mixed P0/P1:
  `reports/random_20w_split/stage2_corrected_best_val_mixed_p0_p1_manual_review.csv`

Both source packages are corrected-Val-only artifacts from Loop 16/17.

## Combined Queue

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_combined_manual_review_queue.py `
  --input model_supported=reports\random_20w_split\stage2_corrected_best_val_model_supported_p0_p1_manual_review.csv `
  --input mixed=reports\random_20w_split\stage2_corrected_best_val_mixed_p0_p1_manual_review.csv `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_manual_review.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_manual_review.json
```

Result:

- Input rows: `137`
  - `model_supported`: `93`
  - `mixed`: `44`
- Output rows: `137`
- Deduplicated rows: `0`
- Dedup policy: `source_sha256_then_normalized_source_path`
- Filled manual rows in inputs: `0`
- Output manual fields blank: `true`
- FP/FN: `90 / 47`
- Labels: `0:90`, `1:47`
- Support buckets:
  - `neighbors_support_model_prediction`: `93`
  - `neighbors_mixed`: `44`
- Priority P0/P1: `93 / 44`
- Review source count: every row appears in exactly one source package

Interpretation: Data-Agent's independent read-only check also found no overlap
between the two source packages by `source_sha256` or normalized `source_path`.
The combined queue is therefore a union, not a conflict resolution step.

## Combined Readiness

Command:

```powershell
.\vnev\Scripts\python.exe scripts\audit_manual_review_package_readiness.py `
  --review-csv reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_manual_review.csv `
  --manifest-json data\.cache\manifest_38672ba0.json `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_readiness.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_readiness.json
```

Result:

- Rows: `137`
- Ready rows: `137`
- Not-ready rows: `0`
- Review queue ready: `true`
- Manual review ready: `true`
- Verdict package ready: `false`
- Blocking issues: `manual_verdict_empty`, `recommended_action_empty`
- Manual verdict blank count: `137`
- Recommended action blank count: `137`
- Source exists: `137 / 137`
- Source SHA-256 OK: `137 / 137`
- Cache exists and NPZ loads: `137 / 137`
- NPZ label/source SHA/shape OK: `137 / 137`
- PE rows: `137 / 137`
- Top-5 neighbor evidence OK: `137 / 137`
- Top-5 neighbor cache/path evidence: `685 / 685`
- Duplicate source SHA-256 count: `0`

Interpretation: the combined queue is ready for humans to inspect, but still
blocked from any automated action. This is intentional.

## Combined Adjudication Guide

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_manual_review_adjudication_guide.py `
  --readiness-csv reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_readiness.csv `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_adjudication_guide.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_adjudication_guide.json `
  --output-md reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_adjudication_guide.md `
  --markdown-rows 40
```

Result:

- Rows: `137`
- Manual-review ready rows: `137`
- Suspicion levels:
  - `critical_label_conflict`: `3`
  - `strong_label_conflict`: `6`
  - `moderate_label_conflict`: `111`
  - `review_required`: `17`

The guide remains read-only. It does not contain manual verdict/action fields.

## Combined Source Summary

Command:

```powershell
.\vnev\Scripts\python.exe scripts\summarize_manual_review_sources.py `
  --review-csv reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_manual_review.csv `
  --output-csv reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_source_summary.csv `
  --output-json reports\random_20w_split\stage2_corrected_best_val_combined_p0_p1_source_summary.json `
  --prefix-depth 3 `
  --example-limit 8
```

Result:

- Rows: `137`
- FP/FN: `90 / 47`
- P0/P1: `93 / 44`
- Data directories:
  - `待加入白名单`: `90`
  - `待拉黑`: `47`
- High-similarity conflicts, nearest similarity `>= 0.90`: `9`
- Critical conflicts, nearest similarity `>= 0.95`: `4`
- Largest source prefix: `待加入白名单/<flat>` with `90` rows, all FP
- For that largest prefix:
  - Average malicious probability: `0.963428`
  - Average opposite-label neighbor ratio: `0.839556`
  - Maximum opposite-label neighbor ratio: `1.000000`

Interpretation: the human workload is still dominated by high-confidence FP
rows from the flat white-list directory. That does not prove labels are wrong.
It means this directory is the first source that needs business provenance,
allow-list, threat-intel, or sandbox-backed review.

## Validation

Command:

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_build_combined_manual_review_queue.py -q
```

Result:

- `3 passed`

Covered behavior:

- combining blank manual-review rows;
- SHA-first deduplication, including same SHA with different paths;
- default rejection when input manual fields are already filled.

## Safety Decision

Do not enter Test-10k from Loop 18.

Reasoning:

1. Loop 18 added only a review-queue consolidation tool.
2. It produced no new model, threshold, blend, feature mask, or Val metric.
3. The combined queue is blocked from action until humans fill verdict/action
   fields.
4. The full-test split remains frozen and unused.
5. Any later `feature_broken` or `out_of_scope` verdict must still trigger
   fresh same-label replacement and strict cache readiness before training.

The 20w invariant remains unchanged:

```text
200000 = 20000 train + 20000 val + 160000 test
```

## Agent Review

Data-Agent independently checked the two input packages before the combined
queue was generated:

- No overlap was found between the two packages by `source_sha256`.
- No overlap was found by normalized `source_path`.
- Each source package had unique non-empty SHA and path values.
- Recommended dedup policy was SHA-first with normalized path fallback, which
  is now the script's default policy.

Eval-Agent reviewed the Loop 18 flow:

- The new script only reads manual-review CSVs and writes a combined CSV/JSON.
- No Test-10k or full-test artifact is used.
- No training, evaluation, threshold sweep, blend search, feature-mask tuning,
  or metric promotion path exists in the script.
- Inputs with pre-filled manual fields are rejected by default, protecting
  human verdicts from being overwritten.
- The combined queue is ready for human review but not verdict-ready, so the
  Test-10k gate stays closed.

## Next Procedure

1. Use the combined CSV as the single human-facing queue:
   `reports/random_20w_split/stage2_corrected_best_val_combined_p0_p1_manual_review.csv`
2. Humans fill `manual_label_verdict`, `manual_verdict_note`, and
   `recommended_action` in that combined CSV.
3. Rerun readiness with `--strict`.
4. Convert filled verdicts into a non-destructive adjustment plan.
5. For every excluded row, redraw exactly one fresh same-label valid candidate.
6. Build a corrected split, run strict cache readiness, and then rerun
   corrected Train/Val evaluation.
7. Only a clear Val-only gain can reopen the Test-10k gate.
