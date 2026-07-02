from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.probe_loop73_authenticode_val import evaluate_valid_signature_downgrade


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_loop73_valid_signature_downgrade_counts_fp_fix_and_fn_harm(tmp_path: Path):
    predictions = tmp_path / "predictions.csv"
    signatures = tmp_path / "signatures.csv"
    out_json = tmp_path / "report.json"
    out_predictions = tmp_path / "out_predictions.csv"

    _write_csv(
        predictions,
        [
            {"sample_index": "1", "source_sha256": "a", "source_path": "a.exe", "label": "0", "prediction": "1"},
            {"sample_index": "2", "source_sha256": "b", "source_path": "b.exe", "label": "1", "prediction": "1"},
            {"sample_index": "3", "source_sha256": "c", "source_path": "c.exe", "label": "1", "prediction": "0"},
            {"sample_index": "4", "source_sha256": "d", "source_path": "d.exe", "label": "0", "prediction": "0"},
        ],
    )
    _write_csv(
        signatures,
        [
            {"sample_index": "1", "source_sha256": "a", "source_path": "a.exe", "auth_status": "Valid", "signer_thumbprint": "ta"},
            {"sample_index": "2", "source_sha256": "b", "source_path": "b.exe", "auth_status": "Valid", "signer_thumbprint": "tb"},
        ],
    )

    report = evaluate_valid_signature_downgrade(
        predictions_csv=predictions,
        signature_csv=signatures,
        output_json=out_json,
        output_predictions_csv=out_predictions,
        reference_val_errors=2,
        min_val_error_improvement=1,
    )
    rows = list(csv.DictReader(out_predictions.open("r", encoding="utf-8-sig", newline="")))

    assert report["baseline"]["errors"] == 2
    assert report["candidate"]["errors"] == 2
    assert report["fixed_fp"] == 1
    assert report["introduced_fn"] == 1
    assert report["valid_signed_flips"] == 2
    assert report["decision"] == "reject_val_margin_too_small"
    assert rows[0]["authenticode_valid_downgrade"] == "True"
    assert rows[1]["authenticode_valid_downgrade"] == "True"
    assert json.loads(out_json.read_text(encoding="utf-8"))["schema"] == "axon_loop73_authenticode_val_probe_v1"


def test_loop73_protocol_and_identity_policy_are_explicit(tmp_path: Path):
    predictions = tmp_path / "predictions.csv"
    signatures = tmp_path / "signatures.csv"
    out_json = tmp_path / "report.json"
    _write_csv(
        predictions,
        [{"sample_index": "1", "source_sha256": "a", "source_path": "a.exe", "label": "0", "prediction": "1"}],
    )
    _write_csv(
        signatures,
        [{"sample_index": "1", "source_sha256": "a", "source_path": "a.exe", "auth_status": "NotSigned"}],
    )

    report = evaluate_valid_signature_downgrade(
        predictions_csv=predictions,
        signature_csv=signatures,
        output_json=out_json,
        output_predictions_csv=None,
        reference_val_errors=1,
        min_val_error_improvement=1,
    )

    for text in ("Val-only", "no Test-10k", "no model fitting", "no automatic relabeling", "no split mutation"):
        assert text in report["protocol"]
    for text in ("filename", "path", "extension", "directory", "source hash", "sample_index", "split", "row order"):
        assert text in report["identity_feature_policy"]
    assert "not model evidence" in report["identity_feature_policy"]
