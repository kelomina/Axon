import csv
import json

from scripts.evaluate_authenticode_trusted_signer_guard import evaluate_trusted_signer_guard, main


PRED_FIELDS = ["source_path", "source_sha256", "sample_index", "split", "label", "prediction", "stage2_prob_malicious"]
SIG_FIELDS = ["source_path", "source_sha256", "sample_index", "auth_status", "signer_subject"]


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pred(index, label, prediction, score=0.9):
    return {
        "source_path": f"data/sample-{index}.exe",
        "source_sha256": f"{index:064x}",
        "sample_index": str(index),
        "split": "val",
        "label": str(label),
        "prediction": str(prediction),
        "stage2_prob_malicious": str(score),
    }


def _sig(index, status, subject):
    return {
        "source_path": f"data/sample-{index}.exe",
        "source_sha256": f"{index:064x}",
        "sample_index": str(index),
        "auth_status": status,
        "signer_subject": subject,
    }


def test_trusted_signer_guard_only_flips_valid_matching_predicted_positive(tmp_path):
    predictions = tmp_path / "predictions.csv"
    signatures = tmp_path / "signatures.csv"
    output_json = tmp_path / "report.json"
    output_csv = tmp_path / "guard_predictions.csv"
    _write_csv(
        predictions,
        PRED_FIELDS,
        [
            _pred(1, 0, 1, 0.99),
            _pred(2, 1, 1, 0.99),
            _pred(3, 0, 1, 0.99),
            _pred(4, 0, 0, 0.01),
        ],
    )
    _write_csv(
        signatures,
        SIG_FIELDS,
        [
            _sig(1, "Valid", "CN=Microsoft Corporation, O=Microsoft Corporation"),
            _sig(2, "Valid", "CN=Unknown Publisher"),
            _sig(3, "HashMismatch", "CN=Microsoft Corporation"),
        ],
    )

    report = evaluate_trusted_signer_guard(
        predictions_csv=predictions,
        signature_csv=signatures,
        output_json=output_json,
        output_predictions_csv=output_csv,
        trusted_terms=["Microsoft Corporation"],
        reference_errors=2,
        min_error_improvement=1,
    )

    assert report["baseline"]["errors"] == 2
    assert report["candidate"]["errors"] == 1
    assert report["fixed_fp"] == 1
    assert report["introduced_fn"] == 0
    assert report["decision"] == "allow_next_funnel_step"
    rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    assert rows[0]["trusted_signer_guard_downgrade"] == "True"
    assert rows[1]["trusted_signer_guard_downgrade"] == "False"


def test_main_writes_report(tmp_path):
    predictions = tmp_path / "predictions.csv"
    signatures = tmp_path / "signatures.csv"
    output_json = tmp_path / "report.json"
    _write_csv(predictions, PRED_FIELDS, [_pred(1, 0, 1)])
    _write_csv(signatures, SIG_FIELDS, [_sig(1, "Valid", "CN=Microsoft Windows")])

    assert main(
        [
            "--predictions-csv",
            str(predictions),
            "--signature-csv",
            str(signatures),
            "--trusted-term",
            "Microsoft Windows",
            "--reference-errors",
            "1",
            "--min-error-improvement",
            "1",
            "--output-json",
            str(output_json),
        ]
    ) == 0
    assert json.loads(output_json.read_text(encoding="utf-8"))["candidate"]["errors"] == 0
