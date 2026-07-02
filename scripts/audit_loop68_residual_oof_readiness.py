#!/usr/bin/env python3
"""Audit whether prior residual artifacts can safely train another residual layer.

Loop68 is deliberately read-only. A third-layer residual learner is only safe
when the previous full pipeline has row-level train out-of-fold final scores.
Base/candidate OOF scores alone are not enough once a gate or override model was
trained on those same train rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for item in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from identity_feature_guard import identity_feature_violations  # noqa: E402


IDENTITY_ALIGNMENT_FIELDS = [
    "source_path",
    "cache_path",
    "source_sha256",
    "sample_index",
    "split",
]

BASE_SCORE_COLUMNS = [
    "base_oof_prob_malicious",
    "base_prob_malicious",
]

CANDIDATE_SCORE_COLUMNS = [
    "candidate_oof_prob_malicious",
    "candidate_prob_malicious",
]

GATE_SCORE_COLUMNS = [
    "gate_oof_prob_override",
    "allow_oof_prob",
    "gate_prob_override",
]

FINAL_SCORE_COLUMNS = [
    "final_oof_prob_malicious",
    "final_prob_malicious",
    "stage2_prob_malicious",
    "prob_malicious",
    "score_malicious",
]

PREDICTION_COLUMNS = ["final_oof_prediction", "prediction", "predicted_label", "pred"]
OVERRIDE_COLUMNS = ["oof_override_flag", "fn_override", "override_flag"]
OOF_PROVENANCE_COLUMNS = ["oof_fold", "fold", "fold_id", "oof_model_id", "out_of_fold"]


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def read_json(path: Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _first_present(mapping: dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _selected_metrics(report: dict[str, Any]) -> dict[str, Any]:
    selected = report.get("selected_by_val") or {}
    metrics = (
        selected.get("val_gate_best")
        or selected.get("val_best")
        or selected.get("metrics")
        or report.get("metrics")
        or {}
    )
    return {
        "f1": metrics.get("f1"),
        "errors": metrics.get("errors"),
        "false_positive": metrics.get("false_positive"),
        "false_negative": metrics.get("false_negative"),
    }


def _flatten_feature_names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        names: list[str] = []
        for item in value.values():
            names.extend(_flatten_feature_names(item))
        return names
    if isinstance(value, (list, tuple)):
        names = []
        for item in value:
            names.extend(_flatten_feature_names(item))
        return names
    return []


def _feature_names_from_report(report: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key in (
        "gate_feature_names",
        "override_feature_names",
        "overlay_boundary_feature_names",
        "feature_names",
        "feature_name_groups",
    ):
        if key in report:
            names.extend(_flatten_feature_names(report[key]))
    return names


def _payload_summary(model_path: Optional[Path]) -> dict[str, Any]:
    if model_path is None:
        return {
            "model_path": None,
            "loaded": False,
            "schema": None,
            "keys": [],
            "array_like_train_rows": [],
            "feature_name_violations": [],
        }

    resolved = resolve_path(model_path)
    if not resolved.exists():
        return {
            "model_path": str(resolved),
            "loaded": False,
            "missing": True,
            "schema": None,
            "keys": [],
            "array_like_train_rows": [],
            "feature_name_violations": [],
        }

    with resolved.open("rb") as handle:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            payload = pickle.load(handle)

    if not isinstance(payload, dict):
        return {
            "model_path": str(resolved),
            "loaded": True,
            "schema": type(payload).__name__,
            "keys": [],
            "array_like_train_rows": [],
            "feature_name_violations": [],
        }

    feature_names: list[str] = []
    for key in ("gate_feature_names", "override_feature_names", "feature_names", "feature_name_groups"):
        if key in payload:
            feature_names.extend(_flatten_feature_names(payload[key]))

    array_like = []
    for key, value in payload.items():
        shape = getattr(value, "shape", None)
        if shape is not None:
            array_like.append({"key": key, "shape": [int(dim) for dim in shape]})
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], (int, float, np.integer, np.floating)):
            array_like.append({"key": key, "shape": [len(value)]})

    return {
        "model_path": str(resolved),
        "loaded": True,
        "schema": payload.get("schema"),
        "keys": sorted(str(key) for key in payload.keys()),
        "array_like_train_rows": array_like,
        "feature_name_violations": identity_feature_violations(feature_names),
    }


def _collect_train_oof_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{prefix}.{key}" if prefix else str(key))
            return
        if isinstance(value, str):
            normalized = prefix.lower()
            text = value.lower()
            combined = f"{normalized} {text}"
            if "train" in combined and ("oof" in combined or "out_of_fold" in combined):
                candidates.append({"field": prefix, "path": value})

    visit(report.get("artifacts") or {}, "artifacts")
    for key in ("train_oof_predictions", "train_final_oof_predictions", "nested_train_oof_predictions"):
        if report.get(key):
            candidates.append({"field": key, "path": report[key]})
    return candidates


def _inspect_prediction_csv(path: Path, expected_rows: int) -> dict[str, Any]:
    resolved = resolve_path(path)
    if not resolved.exists():
        return {"path": str(resolved), "exists": False, "usable": False, "reason": "missing_file"}

    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        row_count = sum(1 for _ in reader)

    has_base_score = any(column in fieldnames for column in BASE_SCORE_COLUMNS)
    has_candidate_score = any(column in fieldnames for column in CANDIDATE_SCORE_COLUMNS)
    has_gate_score = any(column in fieldnames for column in GATE_SCORE_COLUMNS)
    has_final_score = any(column in fieldnames for column in FINAL_SCORE_COLUMNS)
    has_prediction = any(column in fieldnames for column in PREDICTION_COLUMNS)
    has_override_flag = any(column in fieldnames for column in OVERRIDE_COLUMNS)
    has_label = "label" in fieldnames
    has_oof_provenance = any(column in fieldnames for column in OOF_PROVENANCE_COLUMNS)
    identity_columns_present = [column for column in IDENTITY_ALIGNMENT_FIELDS if column in fieldnames]
    usable = (
        row_count == expected_rows
        and has_label
        and has_base_score
        and has_candidate_score
        and has_gate_score
        and has_final_score
        and has_prediction
        and has_override_flag
        and has_oof_provenance
    )
    missing = []
    if row_count != expected_rows:
        missing.append(f"expected_{expected_rows}_rows")
    if not has_label:
        missing.append("label_column")
    if not has_base_score:
        missing.append("base_oof_score_column")
    if not has_candidate_score:
        missing.append("candidate_oof_score_column")
    if not has_gate_score:
        missing.append("gate_or_allow_oof_score_column")
    if not has_final_score:
        missing.append("final_score_column")
    if not has_prediction:
        missing.append("prediction_column")
    if not has_override_flag:
        missing.append("override_flag_column")
    if not has_oof_provenance:
        missing.append("oof_provenance_column")

    return {
        "path": str(resolved),
        "exists": True,
        "row_count": row_count,
        "columns": fieldnames,
        "identity_columns_present_for_alignment_only": identity_columns_present,
        "has_label": has_label,
        "has_base_score": has_base_score,
        "has_candidate_score": has_candidate_score,
        "has_gate_score": has_gate_score,
        "has_final_score": has_final_score,
        "has_prediction": has_prediction,
        "has_override_flag": has_override_flag,
        "has_oof_provenance": has_oof_provenance,
        "usable": usable,
        "missing": missing,
    }


def audit_candidate(
    *,
    report_path: Path,
    model_path: Optional[Path],
    expected_train_rows: int,
    expected_val_rows: int,
) -> dict[str, Any]:
    report_resolved = resolve_path(report_path)
    report = read_json(report_resolved)
    records = report.get("records") or {}
    if isinstance(records.get("train"), dict) or isinstance(records.get("val"), dict):
        train_records = records.get("train") if isinstance(records.get("train"), dict) else {}
        val_records = records.get("val") if isinstance(records.get("val"), dict) else {}
    else:
        train_records = records if isinstance(records, dict) else {}
        val_records = {"kept": 0}
    train_kept = int(train_records.get("kept") or 0)
    val_kept = int(val_records.get("kept") or 0)

    report_feature_violations = identity_feature_violations(_feature_names_from_report(report))
    payload = _payload_summary(model_path)
    train_oof_candidates = _collect_train_oof_candidates(report)
    inspected_oof = [
        _inspect_prediction_csv(resolve_path(candidate["path"]), expected_train_rows)
        for candidate in train_oof_candidates
    ]
    usable_oof_artifacts = [item for item in inspected_oof if item.get("usable")]

    missing_requirements = []
    if train_kept != expected_train_rows:
        missing_requirements.append("report_does_not_confirm_expected_train_rows")
    if val_kept != expected_val_rows:
        missing_requirements.append("report_does_not_confirm_expected_val_rows")
    if report_feature_violations or payload.get("feature_name_violations"):
        missing_requirements.append("identity_like_model_feature_name_detected")
    if not usable_oof_artifacts:
        missing_requirements.append("missing_row_level_train_final_whole_pipeline_oof_predictions")
    protocol = str(report.get("protocol") or "")
    protocol_lower = protocol.lower()
    if "strict oof" not in protocol_lower and "nested oof" not in protocol_lower:
        missing_requirements.append("report_protocol_does_not_claim_train_oof")

    readiness = "ready_for_third_layer_residual_training" if not missing_requirements else "not_ready"
    if readiness == "not_ready":
        recommendation = (
            "Do not train a third-layer residual learner from this artifact. "
            "Export nested train OOF final predictions for the full selected pipeline first."
        )
    else:
        recommendation = "Train/Val residual learning is protocol-eligible; still keep Test-10k locked until Val margin passes."

    return {
        "report_path": str(report_resolved),
        "model_path": payload.get("model_path"),
        "schema": report.get("schema"),
        "protocol": protocol,
        "records": {
            "train_kept": train_kept,
            "val_kept": val_kept,
            "expected_train_rows": expected_train_rows,
            "expected_val_rows": expected_val_rows,
        },
        "selected_val_metrics": _selected_metrics(report),
        "test_gate_decision": report.get("test_gate_decision"),
        "payload_summary": payload,
        "report_feature_name_violations": report_feature_violations,
        "train_oof_candidate_artifacts": train_oof_candidates,
        "inspected_train_oof_artifacts": inspected_oof,
        "usable_train_oof_artifact_count": len(usable_oof_artifacts),
        "missing_requirements": missing_requirements,
        "readiness": readiness,
        "recommendation": recommendation,
        "identity_field_policy": (
            "source_path/cache_path/source_sha256/sample_index/split may be present in prediction CSVs for "
            "alignment and audit only; they are forbidden as model features or residual learner inputs."
        ),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit residual OOF readiness for Loop68.")
    parser.add_argument("--candidate", action="append", required=True, help="report.json[::model.pkl]")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-train-rows", type=int, default=20000)
    parser.add_argument("--expected-val-rows", type=int, default=20000)
    return parser.parse_args(argv)


def _parse_candidate(text: str) -> tuple[Path, Optional[Path]]:
    if "::" not in text:
        return Path(text), None
    report, model = text.split("::", 1)
    return Path(report), Path(model) if model else None


def run_audit(
    *,
    candidates: Sequence[str],
    output_json: Path,
    expected_train_rows: int = 20000,
    expected_val_rows: int = 20000,
) -> dict[str, Any]:
    audits = []
    for candidate in candidates:
        report_path, model_path = _parse_candidate(candidate)
        audits.append(
            audit_candidate(
                report_path=report_path,
                model_path=model_path,
                expected_train_rows=expected_train_rows,
                expected_val_rows=expected_val_rows,
            )
        )

    ready = [item for item in audits if item["readiness"] == "ready_for_third_layer_residual_training"]
    report = {
        "schema": "axon_loop68_residual_oof_readiness_v1",
        "protocol": (
            "read-only protocol audit; no fitting, no threshold sweep, no Val selection, no Test-10k/full-test"
        ),
        "identity_feature_policy": (
            "filename/path/extension/directory/source hash/sample id/split/row order are alignment/cache/audit "
            "fields only and are not model features"
        ),
        "expected_train_rows": int(expected_train_rows),
        "expected_val_rows": int(expected_val_rows),
        "candidate_count": len(audits),
        "ready_candidate_count": len(ready),
        "overall_decision": (
            "third_layer_residual_training_allowed" if ready else "third_layer_residual_training_blocked"
        ),
        "audits": audits,
        "next_safe_action": (
            "If another residual layer is still desired, implement nested OOF export for the entire selected "
            "Loop57/Loop61-style pipeline first; otherwise continue with noise review and new independent "
            "content evidence sources."
        ),
    }
    output_resolved = resolve_path(output_json)
    output_resolved.parent.mkdir(parents=True, exist_ok=True)
    output_resolved.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_audit(
        candidates=args.candidate,
        output_json=args.output_json,
        expected_train_rows=args.expected_train_rows,
        expected_val_rows=args.expected_val_rows,
    )
    print(json.dumps({"overall_decision": report["overall_decision"], "ready": report["ready_candidate_count"]}, indent=2))
    print(f"JSON: {resolve_path(args.output_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
