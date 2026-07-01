# Phase 3 Loop40: Loop39 Replacement Preflight

Date: 2026-07-02

## Objective

Loop40 adds a read-only gate between the Loop39 high-confidence conflict queue
and any corrected-split rebuild. It prevents accidental self-fill, wrong-label
replacement, short splits, or silent use of blank manual verdicts.

This loop does not train a model, does not tune on Test/Test-10k, does not edit
the split, and does not rebuild cache.

## Rule Clarification

Files that are confirmed `feature_broken`, `out_of_scope`, or `label_wrong`
must not be used to fill themselves. They also must not be replaced by “补齐”
from the bad rows. The only valid action is to re-sample fresh valid candidates
from the same intended label pool and preserve the exact 200000-row split.

Filename, path, extension, directory, source hash, sample id, split, and row
order remain identity/audit fields only. They are not model features and not
relabel evidence.

## New Tool

- Script: `scripts/build_loop39_replacement_preflight.py`
- Test: `tests/test_build_loop39_replacement_preflight.py`

The preflight checks:

- active split shape: `20000 train / 20000 val / 160000 test`
- label balance: `10000/10000`, `10000/10000`, `80000/80000`
- every review row still maps to the split
- manual verdict/action values are valid and mutually consistent
- replacement verdicts require `replace_with_fresh_same_label_candidate`
- if replacements are requested, a candidate CSV is required
- candidate rows must be fresh, not self-fill, not already in the split, and
  must cover requested counts by same label

## Real Loop39 Preflight

Command:

```powershell
.\vnev\Scripts\python.exe scripts\build_loop39_replacement_preflight.py `
  --review-csv reports\random_20w_split\loop39_loop28_conflict_adjudication\loop28_conflict_adjudication_queue.csv `
  --split-csv reports\random_20w_split\random_20w_split.csv `
  --output-json reports\random_20w_split\loop40_loop39_replacement_preflight\loop39_replacement_preflight.json `
  --detail-output-csv reports\random_20w_split\loop40_loop39_replacement_preflight\loop39_replacement_preflight_details.csv
```

Result:

- `review_rows`: `649`
- split rows: `200000`
- split counts: `train=20000`, `val=20000`, `test=160000`
- label balance: preserved exactly
- `blank_manual_rows`: `649`
- `replacement_required`: `0`
- `blocking_issues`: `blocked_no_verdicts`
- `preflight_ok`: `false`

Interpretation: the current Loop39 queue is correctly blocked. No corrected
split may be built from it yet, because every manual verdict/action field is
blank. This is intentional: the queue identifies high-value review targets, not
automatic replacements.

## Validation

Commands:

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\build_loop39_replacement_preflight.py
.\vnev\Scripts\python.exe -m pytest tests\test_build_loop39_replacement_preflight.py
```

Results:

- preflight compile: passed
- preflight tests: `4 passed`

## Next Gate

Only after manual verdicts are filled:

1. Run `scripts/build_loop39_replacement_preflight.py` again.
2. If `replacement_required > 0`, build a fresh same-label candidate pool with
   `scripts/build_replacement_candidate_pool.py`.
3. Rerun preflight with `--candidate-csv`.
4. Only if `preflight_ok=true`, build the corrected split.
5. Run `audit_corrected_split_replacements.py --enforce-label-balance --strict`.
6. Run corrected split cache readiness before any Val experiment.

If the candidate pool has shortfall, stop and collect more valid raw samples.
Do not reduce the split below `200000`.
