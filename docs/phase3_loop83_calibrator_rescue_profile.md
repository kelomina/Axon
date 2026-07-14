# Phase 3 Loop83 Calibrator Rescue Profile

## Purpose

Loop83 profiles the Loop82 Val overlap to answer a narrow question:

Can a simple score-only rule identify the rows where the probability calibrator
fixes Loop57, without damaging many rows Loop57 already gets right?

This is a Val-only diagnostic. It does not train a model, does not tune on
Test/Test-10k, and does not use identity fields as evidence.

## Inputs

- overlap CSV:
  `reports/random_20w_split/loop82_same_manifest_val/loop82_val_complementarity_overlap.csv`
- rows: `20000`
- prior alignment gate: passed in Loop82 with `20000/20000` unique
  `source_sha256` alignment

`source_sha256`, `sample_index`, split, path, filename, directory, and row order
are row audit metadata only. The diagnostic rule uses only `abs_score_delta`,
the absolute difference between Loop57 and calibrator scores.

## Commands

Guard:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\analyze_loop83_calibrator_rescue_profile.py `
  --output-json reports\random_20w_split\loop82_same_manifest_val\loop83_rescue_profile_guard.json
```

Result: `decision=pass`.

Tests:

```powershell
.\vnev\Scripts\python.exe -m pytest tests\test_analyze_loop83_calibrator_rescue_profile.py -q
```

Result: `2 passed`.

Profile:

```powershell
.\vnev\Scripts\python.exe scripts\analyze_loop83_calibrator_rescue_profile.py `
  --overlap-csv reports\random_20w_split\loop82_same_manifest_val\loop82_val_complementarity_overlap.csv `
  --output-json reports\random_20w_split\loop82_same_manifest_val\loop83_calibrator_rescue_profile.json
```

## Result

Baseline metrics on the same `20000` Val rows:

- Loop57: F1 `0.9926635723910766`, errors `147`, FP/FN `92 / 55`
- calibrator: F1 `0.9723470100828591`, errors `554`, FP/FN `294 / 260`

Overlap:

- both correct: `19390`
- both wrong: `91`
- Loop57-only-correct: `463`
- calibrator-only-correct: `56`

Score-delta rule scan:

- feature: `abs_score_delta`
- best threshold: `0.90`
- best rule F1: `0.9909631034999251`
- best rule errors: `181`
- calibrator-only-correct captured: `0`
- Loop57-only-correct harmed: `34`

The best score-only rule is worse than Loop57 by `+34` errors and captures none
of the `56` rescue rows. Lower thresholds capture a few rescue rows but damage
many more Loop57-only-correct rows:

- threshold `0.85`: captures `1`, harms `53`, errors `199`
- threshold `0.80`: captures `7`, harms `86`, errors `226`
- threshold `0.75`: captures `12`, harms `131`, errors `266`

## Interpretation

Simple score disagreement is not a useful selector. The calibrator rescue rows
and calibrator regression rows both often have large score differences, so
trusting the calibrator when it strongly disagrees with Loop57 mostly imports
calibrator mistakes.

The useful evidence from Loop82 remains real, but the selector must use
additional non-identity content signals if we continue. A learned probe should
be allowed only if its feature names pass identity guard and it is evaluated
strictly on Val before any Test-10k access.

## Verdict

Reject score-delta fusion. Do not run Test-10k. Do not replace Loop57 with the
calibrator and do not do simple score averaging.

The next acceptable step is a Val-only content-feature rescue probe focused on
separating:

- `56` calibrator-only-correct rows
- `463` Loop57-only-correct rows

It must not use filename, path, directory, extension, hash, `source_sha256`,
`sample_index`, split, cache path, or row order as model evidence.
