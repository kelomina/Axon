# Phase 3 Loop87 Review Evidence Verdict Import

## Purpose

Loop87 is the strict ingress gate for manual or external verdicts attached to
the Loop86 review-evidence package.

It validates verdict syntax, note quality, duplicate row identity, and redraw
planning intent. It does not train, tune thresholds, relabel automatically,
mutate the split, mutate cache, build a corrected split, or authorize
Train/Val/Test evaluation.

## Why This Gate Exists

Loop86 produced content facts for `62` high-priority review rows. Those facts
are useful for adjudication, but they are not verdicts by themselves. Loop87
prevents the next common failure mode: treating filenames, paths, hashes, row
ids, review ranks, or model probabilities as if they were ground truth.

Actionable verdicts must cite content or external evidence, such as PE/header
facts, import/resource/overlay structure, NPZ/cache mismatch, sandbox result,
vendor/multi-engine evidence, signature evidence, or other auditable content
evidence.

## Identity Policy

These fields are allowed only for loading, alignment, priority, and manual
review indexing:

- `source_path`
- `cache_path`
- `source_sha256`
- `sample_index`
- `split`
- review rank/category
- Loop57 probabilities
- model error type

They are not verdict evidence, model evidence, replacement sampling keys, or
threshold/fusion inputs.

## Commands

Unit tests:

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_import_loop87_review_evidence_verdicts.py -q
```

Result: `6 passed`.

Guard:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\import_loop87_review_evidence_verdicts.py `
  --output-json reports\random_20w_split\loop87_review_evidence_verdict_import_guard.json
```

Result: `decision=pass`, static findings `0`.

Real no-op import:

```powershell
.\vnev\Scripts\python.exe scripts\import_loop87_review_evidence_verdicts.py `
  --evidence-csv reports\random_20w_split\loop86_review_evidence_package.csv `
  --output-csv reports\random_20w_split\loop87_review_evidence_verdict_import.csv `
  --output-json reports\random_20w_split\loop87_review_evidence_verdict_import.json `
  --expected-rows 62
```

Result: exit code `0`.

Regression:

```powershell
.\vnev\Scripts\python.exe -m pytest `
  tests\test_import_loop87_review_evidence_verdicts.py `
  tests\test_build_loop86_review_evidence_package.py `
  tests\test_build_loop85_noise_strategy_gate.py `
  tests\test_identity_feature_guard.py -q
```

Result: `14 passed`.

## Result

Real Loop86 evidence import:

- rows: `62`
- expected rows: `62`
- import ready: `true`
- decision: `ready_noop_no_actionable_verdicts`
- blank verdict rows: `62`
- invalid rows: `0`
- actionable rows: `0`
- replacement required rows: `0`
- training policy rows: `0`
- duplicate `sample_index` rows: `0`
- duplicate review-rank rows: `0`
- missing required columns: `0`

No redraw, relabel, training, or evaluation is allowed from this no-op state.

## Valid Action Semantics

`label_correct` with `keep_label` or `model_blindspot` means the row remains a
model error or blind spot, not data noise.

`label_wrong`, `feature_broken`, or `out_of_scope` must pair with a replace or
quarantine action. If accepted, Loop87 records only a redraw request from the
locked-manifest original-label pool. It never self-fills from the bad row and
does not directly edit the split.

`uncertain` can only request more evidence or no action.

## Verdict

Loop87 is ready as the safe gate after Loop86, but the current real file is a
no-op because all manual/external verdict fields are blank.

The next useful work is to obtain independent content or external verdicts for
the `62` Loop86 rows. Until then, there is no authorized replacement plan, no
corrected split, no Train/Val rerun, and no Test-10k access.
