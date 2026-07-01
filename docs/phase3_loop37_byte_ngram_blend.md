# Loop37 Byte N-gram Blend

Date: 2026-07-02

## Objective

Loop37 tested whether a diverse byte n-gram SGD model could complement the
current Loop28 content PE Stage-2 model.

This was a content-only experiment. The byte n-gram model used bytes from the
feature cache plus cache-resident PE/stat/lightweight vectors. Filename, path,
extension, directory, source hash, sample id, split, prediction, and correctness
fields were not used as model features. `source_path`, `cache_path`,
`source_sha256`, and `sample_index` were used only for cache loading and
prediction-table alignment audits.

## Protocol

1. Train byte n-gram SGD only on the corrected Loop27 train rows.
2. Evaluate byte n-gram on the corrected Loop27/Loop28 Val rows.
3. Run Val-only complementarity analysis against Loop28.
4. Select blend weights and threshold only from Val.
5. If Val beats Loop28, run frozen Test-10k confirmation.
6. If Test-10k beats Loop28, run frozen full-test evaluation.

The selected Val blend was:

- `0.8 * Loop28 + 0.2 * byte_ngram`
- threshold `0.48`

Prediction joins were guarded by alignment audits. The audit required matching
labels and, when present, matching `source_sha256`; `source_path` was only a
fallback audit field.

## Code

New or updated tools:

- `scripts/train_byte_ngram_sgd.py`
- `scripts/evaluate_byte_ngram_sgd.py`
- `scripts/analyze_val_prediction_ensemble.py`
- `scripts/evaluate_prediction_blend.py`
- `tests/test_train_byte_ngram_sgd_predictions.py`
- `tests/test_analyze_val_prediction_ensemble_alignment.py`
- `tests/test_evaluate_prediction_blend.py`

## Results

| Candidate | Split | F1 | Errors | FP/FN | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Loop28 content PE | Val | `0.9919048571` | `162` | `87/75` | Baseline |
| Byte n-gram SGD | Val | `0.9383872240` | `1250` | `769/481` | Weak standalone |
| Loop37 blend | Val | `0.9920591320` | `159` | `91/68` | Enter Test-10k |
| Loop28 content PE | Test-10k | `0.9888677164` | `111` | `61/50` | Baseline |
| Loop37 blend | Test-10k | `0.9889713254` | `110` | `62/48` | Enter full test |
| Loop28 content PE | Full test | `0.9878358558` | `1949` | `1087/862` | Current best |
| Loop37 blend | Full test | `0.9877738410` | `1960` | `1136/824` | Reject |

The byte n-gram model had low error overlap with Loop28 on Val:

- Loop28 Val errors: `162`
- byte n-gram Val errors: `1250`
- shared Val errors: `91`
- Jaccard: `0.0688872067`

That complementarity was real enough to pass Val and Test-10k by a very narrow
margin, but it did not survive the 160k full-test evaluation.

## Interpretation

Loop37 is rejected. The full test lost 11 errors compared with Loop28
(`1960` vs `1949`) despite Val improving by 3 errors and Test-10k improving by
1 error.

The practical lesson is important: when the improvement is only a few samples,
Test-10k is not enough evidence. It is useful as a funnel, but not as proof of
final improvement. For this project's current error scale, a candidate should
ideally clear Val and Test-10k by a wider margin before we expect it to hold on
the 160k final test.

## Artifacts

- Val byte n-gram:
  `reports/random_20w_split/byte_ngram_sgd_loop37_loop27_valonly/byte_ngram_sgd_report.json`
- Val ensemble:
  `reports/random_20w_split/loop37_val_ensemble_loop28_byte_ngram/val_prediction_ensemble_analysis.json`
- Test-10k byte n-gram:
  `reports/random_20w_split/byte_ngram_sgd_loop37_loop27_test10k_frozen/byte_ngram_sgd_frozen_test10k_eval.json`
- Test-10k blend:
  `reports/random_20w_split/loop37_frozen_test10k_blend_loop28_byte_ngram/loop37_blend_test10k_eval.json`
- Full-test byte n-gram:
  `reports/random_20w_split/byte_ngram_sgd_loop37_loop27_full_test_frozen/byte_ngram_sgd_frozen_full_test_eval.json`
- Full-test blend:
  `reports/random_20w_split/loop37_frozen_full_test_blend_loop28_byte_ngram/loop37_blend_full_test_eval.json`

## Verification

Commands run:

```powershell
.\vnev\Scripts\python.exe -m py_compile scripts\train_byte_ngram_sgd.py scripts\evaluate_byte_ngram_sgd.py scripts\analyze_val_prediction_ensemble.py scripts\evaluate_prediction_blend.py
.\vnev\Scripts\python.exe -m pytest tests\test_train_byte_ngram_sgd_predictions.py tests\test_analyze_val_prediction_ensemble.py tests\test_analyze_val_prediction_ensemble_alignment.py tests\test_evaluate_prediction_blend.py
```
