from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

from scripts.audit_loop68_residual_oof_readiness import run_audit


def _write_report(path: Path, artifacts: dict | None = None) -> None:
    report = {
        "schema": "unit_report",
        "protocol": "base/candidate train scores are strict OOF; Val selects gate model",
        "records": {
            "train": {"kept": 3},
            "val": {"kept": 2},
        },
        "gate_feature_names": ["gate_base_score", "gate_content_feature_0"],
        "selected_by_val": {"val_best": {"f1": 0.9, "errors": 1}},
        "artifacts": artifacts or {},
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def _write_train_only_report(path: Path, artifacts: dict | None = None) -> None:
    report = {
        "schema": "unit_nested_oof_report",
        "protocol": "train-only nested OOF materialization; no Val selection and no Test",
        "records": {"total": 3, "kept": 3, "skipped_missing_cache": 0},
        "gate_feature_names": ["gate_base_score", "gate_content_feature_0"],
        "metrics": {"f1": 0.9, "errors": 1},
        "artifacts": artifacts or {},
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def _write_payload(path: Path, feature_names: list[str] | None = None) -> None:
    with path.open("wb") as handle:
        pickle.dump(
            {
                "schema": "unit_payload",
                "gate_feature_names": feature_names or ["gate_base_score"],
                "gate_threshold": 0.5,
            },
            handle,
        )


def _write_oof_csv(path: Path) -> None:
    fieldnames = [
        "sample_index",
        "label",
        "base_oof_prob_malicious",
        "candidate_oof_prob_malicious",
        "gate_oof_prob_override",
        "final_oof_prob_malicious",
        "final_oof_prediction",
        "oof_override_flag",
        "oof_fold",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            [
                {
                    "sample_index": "1",
                    "label": "1",
                    "base_oof_prob_malicious": "0.4",
                    "candidate_oof_prob_malicious": "0.9",
                    "gate_oof_prob_override": "0.8",
                    "final_oof_prob_malicious": "0.9",
                    "final_oof_prediction": "1",
                    "oof_override_flag": "1",
                    "oof_fold": "0",
                },
                {
                    "sample_index": "2",
                    "label": "0",
                    "base_oof_prob_malicious": "0.1",
                    "candidate_oof_prob_malicious": "0.2",
                    "gate_oof_prob_override": "0.1",
                    "final_oof_prob_malicious": "0.1",
                    "final_oof_prediction": "0",
                    "oof_override_flag": "0",
                    "oof_fold": "1",
                },
                {
                    "sample_index": "3",
                    "label": "1",
                    "base_oof_prob_malicious": "0.8",
                    "candidate_oof_prob_malicious": "0.7",
                    "gate_oof_prob_override": "0.2",
                    "final_oof_prob_malicious": "0.8",
                    "final_oof_prediction": "1",
                    "oof_override_flag": "0",
                    "oof_fold": "2",
                },
            ]
        )


def test_loop68_blocks_when_final_whole_pipeline_train_oof_is_missing(tmp_path: Path):
    report_path = tmp_path / "report.json"
    model_path = tmp_path / "model.pkl"
    output_json = tmp_path / "audit.json"
    _write_report(report_path)
    _write_payload(model_path)

    report = run_audit(
        candidates=[f"{report_path}::{model_path}"],
        output_json=output_json,
        expected_train_rows=3,
        expected_val_rows=2,
    )

    audit = report["audits"][0]
    assert report["overall_decision"] == "third_layer_residual_training_blocked"
    assert audit["readiness"] == "not_ready"
    assert "missing_row_level_train_final_whole_pipeline_oof_predictions" in audit["missing_requirements"]
    assert json.loads(output_json.read_text(encoding="utf-8"))["ready_candidate_count"] == 0


def test_loop68_allows_when_train_oof_csv_has_required_columns(tmp_path: Path):
    oof_csv = tmp_path / "train_final_oof_predictions.csv"
    report_path = tmp_path / "report.json"
    model_path = tmp_path / "model.pkl"
    output_json = tmp_path / "audit.json"
    _write_oof_csv(oof_csv)
    _write_report(report_path, {"train_final_oof_predictions": str(oof_csv)})
    _write_payload(model_path)

    report = run_audit(
        candidates=[f"{report_path}::{model_path}"],
        output_json=output_json,
        expected_train_rows=3,
        expected_val_rows=2,
    )

    assert report["overall_decision"] == "third_layer_residual_training_allowed"
    assert report["audits"][0]["usable_train_oof_artifact_count"] == 1


def test_loop68_rejects_identity_like_model_feature_names(tmp_path: Path):
    oof_csv = tmp_path / "train_final_oof_predictions.csv"
    report_path = tmp_path / "report.json"
    model_path = tmp_path / "model.pkl"
    output_json = tmp_path / "audit.json"
    _write_oof_csv(oof_csv)
    _write_report(report_path, {"train_final_oof_predictions": str(oof_csv)})
    _write_payload(model_path, ["source_path_bucket"])

    report = run_audit(
        candidates=[f"{report_path}::{model_path}"],
        output_json=output_json,
        expected_train_rows=3,
        expected_val_rows=2,
    )

    audit = report["audits"][0]
    assert report["overall_decision"] == "third_layer_residual_training_blocked"
    assert "identity_like_model_feature_name_detected" in audit["missing_requirements"]
    assert audit["payload_summary"]["feature_name_violations"] == ["source_path_bucket"]


def test_loop68_accepts_train_only_nested_oof_report_shape(tmp_path: Path):
    oof_csv = tmp_path / "train_final_oof_predictions.csv"
    report_path = tmp_path / "report.json"
    output_json = tmp_path / "audit.json"
    _write_oof_csv(oof_csv)
    _write_train_only_report(report_path, {"train_oof_predictions": str(oof_csv)})

    report = run_audit(
        candidates=[str(report_path)],
        output_json=output_json,
        expected_train_rows=3,
        expected_val_rows=0,
    )

    assert report["overall_decision"] == "third_layer_residual_training_allowed"
    assert report["audits"][0]["records"]["train_kept"] == 3
    assert report["audits"][0]["records"]["val_kept"] == 0
