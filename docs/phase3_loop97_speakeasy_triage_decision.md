# Phase 3 Loop97 Speakeasy Triage Decision

## Purpose

Loop97 closes the remaining SpeakeasyX P2 question as an automatic-classifier
candidate. It does not rerun emulation. It reads the existing Speakeasy Val,
confirmation, and random-Val summaries and turns them into a strict decision
gate.

This is read-only evidence consolidation: no emulation, no model fitting, no
threshold selection, no training, no split/cache mutation, and no Test-10k
authorization.

## Inputs

- Val expanded filter probe:
  `reports/hard_family_finetune/clean_hyperparam_search/speakeasy_val_expanded_filter_probe/summary.json`
- Test confirmation subset:
  `reports/hard_family_finetune/clean_hyperparam_search/speakeasy_test_fixed_filter_confirmation/summary.json`
- Random Val sanity probe:
  `reports/hard_family_finetune/clean_hyperparam_search/speakeasy_random_val_probe_30_t35/summary.json`

Evaluated fixed rule:

```text
timeout_filter_score_lt_0.95
```

This rule was selected before Loop97. Loop97 does not tune it.

## Resource Guard

The original Speakeasy runner was not rerun. A preflight guard on that runner
blocked by default because it contains `np.load()` for small calibrator feature
reads. Rather than bypass that risk and rerun external emulation, Loop97 uses
existing JSON summaries only.

Guard for the Loop97 summarizer:

```powershell
.\vnev\Scripts\python.exe scripts\pre_run_resource_leak_guard.py `
  --target-script scripts\build_loop97_speakeasy_triage_decision.py `
  --output-json reports\random_20w_split\loop97_speakeasy_decision_guard.json
```

Result: `decision=pass`, static findings `0`.

## Command

```powershell
.\vnev\Scripts\python.exe scripts\build_loop97_speakeasy_triage_decision.py `
  --val-summary-json reports\hard_family_finetune\clean_hyperparam_search\speakeasy_val_expanded_filter_probe\summary.json `
  --test-confirmation-json reports\hard_family_finetune\clean_hyperparam_search\speakeasy_test_fixed_filter_confirmation\summary.json `
  --random-val-summary-json reports\hard_family_finetune\clean_hyperparam_search\speakeasy_random_val_probe_30_t35\summary.json `
  --output-json reports\random_20w_split\loop97_speakeasy_triage_decision.json `
  --rule-name timeout_filter_score_lt_0.95
```

## Results

Val expanded subset:

- sample count: `172`
- baseline errors: `36`, FP/FN `31/5`, F1 `0.7534246575`
- rule errors: `16`, FP/FN `10/6`, F1 `0.8709677419`
- delta: errors `-20`, FP `-21`, FN `+1`
- new FN from baseline TP: `1`

Confirmation subset:

- sample count: `700`
- baseline errors: `242`, FP/FN `122/120`, F1 `0.7055961071`
- rule errors: `168`, FP/FN `0/168`, F1 `0.7423312883`
- delta: errors `-74`, FP `-122`, FN `+48`
- new FN from baseline TP: `48`
- new FN rate on confirmation subset: `6.857%`

Timeout also hit true malicious rows:

- matched correct malicious for FN: `20`
- ordinary malicious: `6`
- rule-risk correct malicious: `27`

Random Val sanity:

- sample count: `30`
- existing probability calibrator errors: `0`
- existing probability calibrator F1: `1.0`

## Decision

Automatic merge is blocked:

- `confirmation_new_fn_exceeds_zero_tolerance`
- `confirmation_new_fn_rate_too_high`
- `confirmation_fn_delta_positive`
- `timeout_signal_also_hits_true_malicious_rows`

Allowed use:

- Speakeasy timeout/dynamic behavior may be used as manual or external-review
  context for likely FP triage.

Disallowed use:

- no automatic classifier merge
- no automatic threshold override
- no training authorization
- no Test-10k authorization
- no rule that downgrades malicious predictions automatically

This preserves the user requirement that noise must be handled, but not by
trading away true malicious coverage. Speakeasy remains useful evidence for
manual/external review, especially inside the Loop96 blinded package workflow,
but it is not a safe automatic production classifier component under the current
evidence.
