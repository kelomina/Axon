#!/usr/bin/env python3
"""Freeze the current Loop151 evidence chain into a machine-readable manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_F1 = 0.9997
CHAMPION_ID = "current_strict_best_loop151"

FORBIDDEN_SUPPLEMENTAL_ARTIFACT_PATHS = frozenset(
    {
        Path("manifests/roadmap_9997/p0_loop28_parity_diagnostic/run_authorization.json"),
        Path("manifests/roadmap_9997/p0_loop28_parity_diagnostic/run_attempt.final.json"),
        Path("manifests/roadmap_9997/p0_loop28_parity_diagnostic/diagnostic_receipt.json"),
        Path("manifests/roadmap_9997/p0_loop28_parity_diagnostic/post_diagnostic_manifest.json"),
        Path("manifests/roadmap_9997/p0_loop28_parity_diagnostic/post_manifest.json"),
        Path(
            "reports/roadmap_9997/p0_loop28_parity_diagnostic/diagnostic_receipt.provisional.json"
        ),
        Path("reports/roadmap_9997/p0_loop28_parity_diagnostic/diagnostic_receipt.final.json"),
        Path("reports/roadmap_9997/p0_loop28_parity_diagnostic/diagnostic_receipt.json"),
    }
)

DEFAULT_METRIC_REPORTS = {
    "val": Path("reports/phase3_loop151/loop151_trusted_signer_guard_val_eval.json"),
    "test10k": Path("reports/phase3_loop151/loop151_trusted_signer_guard_test10k_eval.json"),
    "legacy_full_test": Path("reports/phase3_loop151/loop151_trusted_signer_guard_full_eval.json"),
}

EXPECTED_SPLIT_ROWS = {"val": 20_000, "test10k": 10_000, "legacy_full_test": 160_000}


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    role: str
    path: Path
    required: bool = True


DEFAULT_ARTIFACTS = (
    ArtifactSpec("goal_contract", "execution_contract", Path("goal.md")),
    ArtifactSpec("manifest_tracking_policy", "repository_policy", Path(".gitignore")),
    ArtifactSpec(
        "truth_manifest_builder",
        "fact_freeze_implementation",
        Path("scripts/build_roadmap_9997_truth_manifest.py"),
    ),
    ArtifactSpec(
        "truth_manifest_builder_test",
        "fact_freeze_test",
        Path("tests/test_build_roadmap_9997_truth_manifest.py"),
    ),
    ArtifactSpec(
        "p0_proposal",
        "proposal",
        Path("manifests/roadmap_9997/p0_truth_freeze/proposal.json"),
    ),
    ArtifactSpec(
        "p0_authorization",
        "authorization",
        Path("manifests/roadmap_9997/p0_truth_freeze/authorization.json"),
    ),
    ArtifactSpec(
        "p0_preflight",
        "preflight",
        Path("manifests/roadmap_9997/p0_truth_freeze/preflight.json"),
    ),
    ArtifactSpec(
        "raw_replay_runner",
        "raw_replay_implementation",
        Path("scripts/replay_loop151_raw.py"),
    ),
    ArtifactSpec(
        "raw_replay_runner_test",
        "raw_replay_test",
        Path("tests/test_replay_loop151_raw.py"),
    ),
    ArtifactSpec("predict_api", "raw_runtime_source", Path("src/predict_api.py")),
    ArtifactSpec("predict_api_test", "raw_runtime_test", Path("tests/test_predict_api_loop28.py")),
    ArtifactSpec("runtime_config", "raw_runtime_source", Path("src/config.py")),
    ArtifactSpec("runtime_model", "raw_runtime_source", Path("src/model.py")),
    ArtifactSpec("runtime_security", "raw_runtime_source", Path("src/security.py")),
    ArtifactSpec("runtime_archive_scanner", "raw_runtime_source", Path("src/archive_scanner.py")),
    ArtifactSpec(
        "runtime_feature_exports",
        "raw_runtime_source",
        Path("src/kvd_features/__init__.py"),
    ),
    ArtifactSpec(
        "runtime_feature_extractor",
        "raw_runtime_source",
        Path("src/kvd_features/extractor.py"),
    ),
    ArtifactSpec(
        "runtime_content_pe_v1",
        "raw_runtime_source",
        Path("src/kvd_features/content_pe_v1.py"),
    ),
    ArtifactSpec(
        "runtime_mhdsra2",
        "raw_runtime_source",
        Path("src/dsra/mhdsra2/improved_dsra_mha.py"),
    ),
    ArtifactSpec(
        "runtime_paged_memory",
        "raw_runtime_source",
        Path("src/dsra/mhdsra2/paged_exact_memory.py"),
    ),
    ArtifactSpec(
        "stage2_metadata_generator",
        "raw_runtime_source",
        Path("scripts/train_stage2_cache_matrix.py"),
    ),
    ArtifactSpec("python_project_metadata", "environment_contract", Path("pyproject.toml")),
    ArtifactSpec("python_requirements", "environment_contract", Path("requirements.txt")),
    ArtifactSpec(
        "raw_replay_proposal",
        "proposal",
        Path("manifests/roadmap_9997/p0_raw_replay/proposal.json"),
    ),
    ArtifactSpec(
        "raw_replay_authorization",
        "authorization",
        Path("manifests/roadmap_9997/p0_raw_replay/authorization.json"),
    ),
    ArtifactSpec(
        "raw_replay_preflight",
        "preflight",
        Path("manifests/roadmap_9997/p0_raw_replay/preflight.json"),
    ),
    ArtifactSpec(
        "raw_replay_pickle_allowlist",
        "deserialization_policy",
        Path("manifests/roadmap_9997/p0_raw_replay/pickle_sha256_allowlist.json"),
    ),
    ArtifactSpec(
        "loop28_stage2_guarded_metadata",
        "research_model_metadata",
        Path("manifests/roadmap_9997/p0_raw_replay/loop28_stage2.metadata.json"),
    ),
    ArtifactSpec(
        "experiment_journal",
        "durable_journal",
        Path("reports/hard_family_finetune/experiment_journal.md"),
    ),
    ArtifactSpec(
        "status_ledger",
        "registry",
        Path("reports/model_review/final_model_selection/ml_recommendation_status.json"),
    ),
    ArtifactSpec("experiment_config", "config", Path("config/random_20w_8192.toml")),
    ArtifactSpec(
        "legacy_split",
        "development_split",
        Path("reports/random_20w_split/loop127_full_duplicate_corrected_split.csv"),
    ),
    ArtifactSpec(
        "loop136_selector",
        "research_model",
        Path(
            "reports/phase3_loop136/r5_oof_noise_pairwise_selector_recall_valonly/"
            "loop135_pairwise_selector.pkl"
        ),
    ),
    ArtifactSpec(
        "loop127_primary_model",
        "research_model",
        Path(
            "reports/phase3_loop127/oof_fixed_v2_all_valonly_with_logreg/"
            "stage2_oof_stacker_selected_model.pkl"
        ),
    ),
    ArtifactSpec(
        "loop127_primary_report",
        "research_model_metadata",
        Path(
            "reports/phase3_loop127/oof_fixed_v2_all_valonly_with_logreg/"
            "stage2_oof_stacker_report.json"
        ),
    ),
    ArtifactSpec(
        "loop127_conservative_model",
        "research_model",
        Path(
            "reports/phase3_loop127/oof_fixed_v2_all_valonly_no_logreg/"
            "stage2_oof_stacker_selected_model.pkl"
        ),
    ),
    ArtifactSpec(
        "loop127_conservative_report",
        "research_model_metadata",
        Path(
            "reports/phase3_loop127/oof_fixed_v2_all_valonly_no_logreg/"
            "stage2_oof_stacker_report.json"
        ),
    ),
    ArtifactSpec(
        "loop127_content_cross_model",
        "research_model",
        Path(
            "reports/phase3_loop127/phase1_content_cross_hgb_local_valonly/"
            "loop43_content_cross_selected_model.pkl"
        ),
    ),
    ArtifactSpec(
        "loop127_content_cross_report",
        "research_model_metadata",
        Path(
            "reports/phase3_loop127/phase1_content_cross_hgb_local_valonly/"
            "loop43_content_cross_report.json"
        ),
    ),
    ArtifactSpec(
        "loop130_r5_policy",
        "research_policy",
        Path("reports/phase3_loop128/loop130_content_string_guard_val_eval.json"),
    ),
    ArtifactSpec(
        "loop130_r5_evaluator",
        "policy_implementation",
        Path("scripts/evaluate_loop130_content_string_guard_rules.py"),
    ),
    ArtifactSpec(
        "loop134_oof_noise_model",
        "research_model",
        Path(
            "reports/phase3_loop134/oof_fixed_v2_string_noise_valonly/"
            "stage2_oof_stacker_selected_model.pkl"
        ),
    ),
    ArtifactSpec(
        "loop134_oof_noise_report",
        "research_model_metadata",
        Path(
            "reports/phase3_loop134/oof_fixed_v2_string_noise_valonly/"
            "stage2_oof_stacker_report.json"
        ),
    ),
    ArtifactSpec(
        "oof_stacker_evaluator",
        "research_model_implementation",
        Path("scripts/evaluate_stage2_oof_stacker.py"),
    ),
    ArtifactSpec(
        "content_cross_evaluator",
        "research_model_implementation",
        Path("scripts/evaluate_stage2_cache_model.py"),
    ),
    ArtifactSpec(
        "loop136_selector_evaluator",
        "research_model_implementation",
        Path("scripts/evaluate_loop135_pairwise_selector.py"),
    ),
    ArtifactSpec("pytorch_base", "research_model", Path("models/random_20w_8192/best_model.pt")),
    ArtifactSpec(
        "loop151_evaluator",
        "policy_implementation",
        Path("scripts/evaluate_authenticode_trusted_signer_guard.py"),
    ),
    ArtifactSpec(
        "loop151_evaluator_test",
        "policy_test",
        Path("tests/test_evaluate_authenticode_trusted_signer_guard.py"),
    ),
    ArtifactSpec(
        "loop151_report",
        "human_report",
        Path("docs/phase3_loop151_trusted_signer_guard_report.md"),
    ),
    ArtifactSpec(
        "native_loop28_onnx", "native_product", Path("models/random_20w_8192/axon_loop28_base.onnx")
    ),
    ArtifactSpec(
        "native_loop28_onnx_data",
        "native_product",
        Path("models/random_20w_8192/axon_loop28_base.onnx.data"),
    ),
    ArtifactSpec(
        "native_loop28_stage2",
        "native_product",
        Path("models/random_20w_8192/loop28_stage2_hgb.json"),
    ),
    ArtifactSpec(
        "native_loop28_runtime_source",
        "native_product_source",
        Path("tools/axon_onnx_dll/src/axon_onnx_predict.cpp"),
    ),
    ArtifactSpec(
        "native_loop28_cmake",
        "native_product_build_definition",
        Path("tools/axon_onnx_dll/CMakeLists.txt"),
    ),
    ArtifactSpec(
        "native_loop28_selftest_source",
        "native_product_test_source",
        Path("tools/axon_onnx_dll/examples/axon_onnx_selftest.cpp"),
    ),
    ArtifactSpec(
        "native_loop28_public_header",
        "native_product_abi",
        Path("tools/axon_onnx_dll/include/axon_onnx_predict.h"),
    ),
    ArtifactSpec(
        "native_loop28_readme",
        "native_product_documentation",
        Path("tools/axon_onnx_dll/README.md"),
    ),
    ArtifactSpec(
        "native_loop28_dll_binary",
        "native_runtime_build",
        Path("tools/axon_onnx_dll/build/bin/Release/axon_onnx_predict.dll"),
    ),
    ArtifactSpec(
        "native_loop28_selftest_binary",
        "native_runtime_build",
        Path("tools/axon_onnx_dll/build/bin/Release/axon_onnx_selftest.exe"),
    ),
    ArtifactSpec(
        "native_onnxruntime_binary",
        "native_runtime_dependency",
        Path("tools/axon_onnx_dll/build/bin/Release/onnxruntime.dll"),
    ),
)

REQUIRED_METRIC_KEYS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
    "errors",
)


def resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _relative_display(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _normalize_report_path(value: object, project_root: Path) -> Optional[Path]:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return None
    marker = f"/{project_root.name.casefold()}/"
    lowered = text.casefold()
    marker_index = lowered.rfind(marker)
    if marker_index >= 0:
        return Path(text[marker_index + len(marker) :])
    if len(text) >= 3 and text[1:3] == ":/":
        return None
    return Path(text.lstrip("/"))


def _metric_values(payload: Mapping[str, object], section: str) -> tuple[dict, list[str]]:
    raw = payload.get(section)
    if not isinstance(raw, dict):
        return {}, [f"missing_{section}_metrics"]
    missing = [key for key in REQUIRED_METRIC_KEYS if key not in raw]
    if missing:
        return dict(raw), [f"{section}_missing_keys:{','.join(missing)}"]

    metrics = {key: raw[key] for key in REQUIRED_METRIC_KEYS}
    errors: list[str] = []
    try:
        true_positive = int(metrics["true_positive"])
        true_negative = int(metrics["true_negative"])
        false_positive = int(metrics["false_positive"])
        false_negative = int(metrics["false_negative"])
        sample_count = true_positive + true_negative + false_positive + false_negative
        expected_errors = false_positive + false_negative
        expected_accuracy = (true_positive + true_negative) / sample_count
        expected_precision = true_positive / (true_positive + false_positive)
        expected_recall = true_positive / (true_positive + false_negative)
        expected_f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return metrics, [f"{section}_invalid_confusion_matrix:{exc}"]

    # 使用混淆矩阵重新计算指标，避免把手填 JSON 当成事实源。
    checks = {
        "errors": (int(metrics["errors"]), expected_errors),
        "accuracy": (float(metrics["accuracy"]), expected_accuracy),
        "precision": (float(metrics["precision"]), expected_precision),
        "recall": (float(metrics["recall"]), expected_recall),
        "f1": (float(metrics["f1"]), expected_f1),
    }
    for key, (actual, expected) in checks.items():
        tolerance = 0 if key == "errors" else 1e-12
        if abs(actual - expected) > tolerance:
            errors.append(f"{section}_{key}_mismatch:{actual}!={expected}")

    metrics["sample_count"] = sample_count
    metrics["positive_count"] = true_positive + false_negative
    metrics["negative_count"] = true_negative + false_positive
    metrics["recomputed_f1"] = expected_f1
    return metrics, errors


def _recompute_prediction_csv(path: Path) -> tuple[dict, list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return {"exists": False, "path": str(path)}, ["missing_output_predictions_csv"]

    required_columns = {
        "source_sha256",
        "sample_index",
        "label",
        "trusted_signer_guard_prediction",
    }
    true_positive = true_negative = false_positive = false_negative = 0
    row_count = 0
    identity_keys: set[tuple[str, str]] = set()
    duplicate_identity_count = 0
    invalid_sha_count = 0
    sha_labels: dict[str, int] = {}
    sha_label_conflict_count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = sorted(required_columns - set(reader.fieldnames or []))
        if missing_columns:
            return {
                "exists": True,
                "path": str(path),
                "columns": reader.fieldnames or [],
            }, [f"output_predictions_missing_columns:{','.join(missing_columns)}"]
        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            try:
                label = int(row["label"])
                prediction = int(row["trusted_signer_guard_prediction"])
            except (TypeError, ValueError):
                errors.append(f"invalid_label_or_prediction_row:{row_number}")
                continue
            if label not in {0, 1} or prediction not in {0, 1}:
                errors.append(f"non_binary_label_or_prediction_row:{row_number}")
                continue

            source_sha = str(row["source_sha256"] or "").strip().casefold()
            sample_index = str(row["sample_index"] or "").strip()
            if len(source_sha) != 64 or any(
                character not in "0123456789abcdef" for character in source_sha
            ):
                invalid_sha_count += 1
            identity = (source_sha, sample_index)
            if identity in identity_keys:
                duplicate_identity_count += 1
            identity_keys.add(identity)
            previous_label = sha_labels.setdefault(source_sha, label)
            if previous_label != label:
                sha_label_conflict_count += 1

            if label == 1 and prediction == 1:
                true_positive += 1
            elif label == 0 and prediction == 0:
                true_negative += 1
            elif label == 0 and prediction == 1:
                false_positive += 1
            else:
                false_negative += 1

    if duplicate_identity_count:
        errors.append(f"duplicate_prediction_identity_keys:{duplicate_identity_count}")
    if invalid_sha_count:
        errors.append(f"invalid_prediction_source_sha256:{invalid_sha_count}")
    if sha_label_conflict_count:
        errors.append(f"prediction_sha_label_conflicts:{sha_label_conflict_count}")

    raw_metrics = {
        "accuracy": (true_positive + true_negative) / row_count if row_count else 0.0,
        "precision": true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0,
        "recall": true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0,
        "f1": 2 * true_positive / (2 * true_positive + false_positive + false_negative)
        if true_positive or false_positive or false_negative
        else 0.0,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "errors": false_positive + false_negative,
    }
    metrics, metric_errors = _metric_values({"csv": raw_metrics}, "csv")
    errors.extend(metric_errors)
    return {
        "exists": True,
        "path": str(path),
        "sha256": file_sha256(path),
        "row_count": row_count,
        "unique_identity_count": len(identity_keys),
        "duplicate_identity_count": duplicate_identity_count,
        "invalid_source_sha256_count": invalid_sha_count,
        "sha_label_conflict_count": sha_label_conflict_count,
        "metrics": metrics,
    }, errors


def _compare_metrics(declared: Mapping[str, object], recomputed: Mapping[str, object]) -> list[str]:
    errors = []
    for key in REQUIRED_METRIC_KEYS:
        if key not in declared or key not in recomputed:
            errors.append(f"prediction_metric_missing:{key}")
            continue
        actual = declared[key]
        expected = recomputed[key]
        tolerance = (
            0
            if key
            in {
                "true_positive",
                "true_negative",
                "false_positive",
                "false_negative",
                "errors",
            }
            else 1e-12
        )
        try:
            mismatch = abs(float(actual) - float(expected)) > tolerance
        except (TypeError, ValueError):
            mismatch = True
        if mismatch:
            errors.append(f"prediction_metric_mismatch:{key}:{actual}!={expected}")
    return errors


def _read_metric_report(
    split_name: str,
    path: Path,
    expected_rows: Optional[int],
    project_root: Path,
) -> tuple[dict, list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return {"path": str(path), "exists": False, "valid": False}, [
            f"missing_metric_report:{split_name}"
        ]
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": str(path), "exists": True, "valid": False}, [
            f"invalid_metric_report:{split_name}:{exc}"
        ]
    if not isinstance(payload, dict):
        return {"path": str(path), "exists": True, "valid": False}, [
            f"metric_report_not_object:{split_name}"
        ]

    baseline, baseline_errors = _metric_values(payload, "baseline")
    candidate, candidate_errors = _metric_values(payload, "candidate")
    errors.extend(baseline_errors)
    errors.extend(candidate_errors)
    if expected_rows is not None and candidate.get("sample_count") != expected_rows:
        errors.append(f"{split_name}_row_count:{candidate.get('sample_count')}!={expected_rows}")

    lineage = {}
    for key, value in {
        "input_predictions": payload.get("predictions_csv"),
        "signature_cache": payload.get("signature_csv"),
        "output_predictions": (payload.get("artifacts") or {}).get("output_predictions_csv")
        if isinstance(payload.get("artifacts"), dict)
        else None,
    }.items():
        normalized = _normalize_report_path(value, project_root)
        lineage[key] = {
            "declared": value,
            "project_path": normalized.as_posix() if normalized else None,
        }

    output_path_value = lineage.get("output_predictions", {}).get("project_path")
    if output_path_value:
        recomputed, prediction_errors = _recompute_prediction_csv(
            resolve_path(project_root, Path(output_path_value))
        )
        errors.extend(prediction_errors)
        errors.extend(_compare_metrics(candidate, recomputed.get("metrics", {})))
        if expected_rows is not None and recomputed.get("row_count") != expected_rows:
            errors.append(
                f"{split_name}_prediction_row_count:{recomputed.get('row_count')}!={expected_rows}"
            )
    else:
        recomputed = {"exists": False, "path": None}
        errors.append(f"{split_name}_missing_output_predictions_reference")

    policy = {
        "schema": payload.get("schema"),
        "protocol": payload.get("protocol"),
        "identity_feature_policy": payload.get("identity_feature_policy"),
        "trusted_terms": payload.get("trusted_terms"),
        "score_column": payload.get("score_column"),
        "score_threshold": payload.get("score_threshold"),
    }
    report = {
        "path": str(path),
        "exists": True,
        "sha256": file_sha256(path),
        "valid": not errors,
        "decision": payload.get("decision"),
        "baseline": baseline,
        "candidate": candidate,
        "policy": policy,
        "policy_sha256": _canonical_sha256(policy),
        "lineage": lineage,
        "recomputed_from_predictions": recomputed,
    }
    return report, errors


def _git_executable() -> str:
    windows_git = Path(r"C:\Program Files\Git\bin\git.exe")
    if os.name == "nt" and windows_git.is_file():
        return str(windows_git)
    return "git"


def _git_bytes(project_root: Path, args: Sequence[str]) -> tuple[int, bytes, str]:
    try:
        completed = subprocess.run(
            [_git_executable(), *args],
            cwd=project_root,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        return 127, b"", str(exc)
    return (
        completed.returncode,
        completed.stdout,
        completed.stderr.decode("utf-8", errors="replace").strip(),
    )


def _git_snapshot(project_root: Path, output_json: Path) -> dict:
    # 输出文件从 dirty snapshot 排除，避免 manifest 因记录自身而在二次运行时漂移。
    try:
        output_relative = output_json.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        output_relative = None
    pathspec = ["--", "."]
    if output_relative:
        pathspec.append(f":(exclude){output_relative}")

    head_rc, head, head_error = _git_bytes(project_root, ["rev-parse", "HEAD"])
    branch_rc, branch, branch_error = _git_bytes(project_root, ["branch", "--show-current"])
    status_rc, status, status_error = _git_bytes(
        project_root,
        ["status", "--no-renames", "--porcelain=v1", "-z", "--untracked-files=all", *pathspec],
    )
    diff_rc, patch, diff_error = _git_bytes(
        project_root,
        ["diff", "--no-renames", "--binary", "HEAD", *pathspec],
    )
    available = all(code == 0 for code in (head_rc, branch_rc, status_rc, diff_rc))
    entries = [
        entry.decode("utf-8", errors="surrogateescape") for entry in status.split(b"\0") if entry
    ]
    return {
        "available": available,
        "head": head.decode("ascii", errors="replace").strip() if head_rc == 0 else None,
        "branch": branch.decode("utf-8", errors="replace").strip() if branch_rc == 0 else None,
        "is_dirty": bool(entries) if status_rc == 0 else None,
        "status_entry_count": len(entries),
        "status_entries": entries,
        "status_sha256": hashlib.sha256(status).hexdigest() if status_rc == 0 else None,
        "tracked_patch_sha256": hashlib.sha256(patch).hexdigest() if diff_rc == 0 else None,
        "untracked_content_hashed": False,
        "excluded_output_path": output_relative,
        "errors": [
            error for error in (head_error, branch_error, status_error, diff_error) if error
        ],
    }


def _artifact_record(
    spec: ArtifactSpec,
    project_root: Path,
    hash_cache: dict[Path, tuple[int, str]],
) -> dict:
    resolved = resolve_path(project_root, spec.path)
    record = {
        "name": spec.name,
        "role": spec.role,
        "path": _relative_display(project_root, resolved),
        "required": spec.required,
        "exists": resolved.is_file(),
    }
    if resolved.is_file():
        canonical = resolved.resolve()
        if canonical not in hash_cache:
            hash_cache[canonical] = (resolved.stat().st_size, file_sha256(resolved))
        size_bytes, digest = hash_cache[canonical]
        record.update({"size_bytes": size_bytes, "sha256": digest})
    return record


def _load_champion_evidence(
    project_root: Path, status_path: Path, champion_id: str
) -> tuple[dict, list[str]]:
    errors: list[str] = []
    if not status_path.is_file():
        return {"champion_id": champion_id, "found": False, "evidence": []}, [
            "missing_status_ledger"
        ]
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"champion_id": champion_id, "found": False, "evidence": []}, [
            f"invalid_status_ledger:{exc}"
        ]
    recommendations = payload.get("recommendations", []) if isinstance(payload, dict) else []
    champion = next(
        (row for row in recommendations if isinstance(row, dict) and row.get("id") == champion_id),
        None,
    )
    if champion is None:
        return {"champion_id": champion_id, "found": False, "evidence": []}, [
            "champion_missing_from_status_ledger"
        ]

    evidence_rows = []
    for value in champion.get("evidence", []):
        evidence_path = resolve_path(project_root, Path(str(value)))
        exists = evidence_path.is_file()
        evidence_rows.append({"path": str(value).replace("\\", "/"), "exists": exists})
        if not exists:
            errors.append(f"missing_champion_evidence:{value}")
    return {
        "champion_id": champion_id,
        "found": True,
        "status": champion.get("status"),
        "evidence": evidence_rows,
    }, errors


def _lineage_artifacts(metric_reports: Mapping[str, dict]) -> list[ArtifactSpec]:
    artifacts = []
    for split_name, report in metric_reports.items():
        for lineage_name, lineage in report.get("lineage", {}).items():
            path = lineage.get("project_path") if isinstance(lineage, dict) else None
            if path:
                artifacts.append(
                    ArtifactSpec(
                        name=f"{split_name}_{lineage_name}",
                        role=f"metric_{lineage_name}",
                        path=Path(path),
                    )
                )
    return artifacts


def _target_gap(full_metrics: Mapping[str, object], target_f1: float) -> dict:
    positive_count = int(full_metrics["positive_count"])
    current_errors = int(full_metrics["errors"])
    weighted_budget = 2 * positive_count * (1 - target_f1)
    max_errors = int(weighted_budget / min(target_f1, 2 - target_f1))
    conservative_errors = int(weighted_budget / max(target_f1, 2 - target_f1))
    required_reduction = max(0, current_errors - max_errors)
    return {
        "target_f1": target_f1,
        "positive_count": positive_count,
        "weighted_error_equation": f"{2 - target_f1:.4f}*FN + {target_f1:.4f}*FP <= {weighted_budget:.12g}",
        "max_total_errors_any_fp_fn_mix": max_errors,
        "conservative_all_fn_error_budget": conservative_errors,
        "current_errors": current_errors,
        "minimum_errors_to_remove": required_reduction,
        "minimum_residual_elimination_fraction": required_reduction / current_errors
        if current_errors
        else 0.0,
        "point_target_met": float(full_metrics["f1"]) >= target_f1,
    }


def build_truth_manifest(
    *,
    project_root: Path,
    output_json: Path,
    metric_report_paths: Mapping[str, Path],
    artifact_specs: Iterable[ArtifactSpec],
    target_f1: float = TARGET_F1,
    champion_id: str = CHAMPION_ID,
    source_commands: Optional[Sequence[str]] = None,
    invocation: Optional[Sequence[str]] = None,
    expected_split_rows: Optional[Mapping[str, int]] = None,
) -> dict:
    project_root = project_root.resolve()
    output_json = resolve_path(project_root, output_json)
    blockers: list[str] = []
    base_artifact_specs = list(artifact_specs)

    expected_rows_by_split = dict(
        EXPECTED_SPLIT_ROWS if expected_split_rows is None else expected_split_rows
    )
    metric_reports = {}
    for split_name, declared_path in metric_report_paths.items():
        resolved = resolve_path(project_root, declared_path)
        report, report_errors = _read_metric_report(
            split_name,
            resolved,
            expected_rows_by_split.get(split_name),
            project_root,
        )
        report["path"] = _relative_display(project_root, resolved)
        metric_reports[split_name] = report
        blockers.extend(report_errors)

    policy_hashes = {
        report.get("policy_sha256")
        for report in metric_reports.values()
        if report.get("policy_sha256")
    }
    if len(policy_hashes) != 1:
        blockers.append("metric_reports_do_not_share_one_frozen_policy")

    status_spec = next((spec for spec in base_artifact_specs if spec.name == "status_ledger"), None)
    status_path = (
        resolve_path(project_root, status_spec.path)
        if status_spec
        else project_root / "missing-status-ledger"
    )
    champion_evidence, evidence_errors = _load_champion_evidence(
        project_root, status_path, champion_id
    )
    blockers.extend(evidence_errors)

    expanded_specs = list(base_artifact_specs)
    expanded_specs.extend(
        ArtifactSpec(f"metric_report_{name}", "metric_report", path)
        for name, path in metric_report_paths.items()
    )
    expanded_specs.extend(_lineage_artifacts(metric_reports))
    expanded_specs.extend(
        ArtifactSpec(f"champion_evidence_{index:02d}", "champion_evidence", Path(row["path"]))
        for index, row in enumerate(champion_evidence.get("evidence", []), start=1)
    )

    hash_cache: dict[Path, tuple[int, str]] = {}
    artifacts = [_artifact_record(spec, project_root, hash_cache) for spec in expanded_specs]
    for artifact in artifacts:
        if artifact["required"] and not artifact["exists"]:
            blockers.append(f"missing_required_artifact:{artifact['name']}:{artifact['path']}")

    legacy_full = metric_reports.get("legacy_full_test", {}).get("candidate")
    target_gap = (
        _target_gap(legacy_full, target_f1)
        if isinstance(legacy_full, dict) and legacy_full
        else None
    )
    if target_gap is None:
        blockers.append("missing_legacy_full_test_candidate_metrics")

    git_snapshot = _git_snapshot(project_root, output_json)
    if not git_snapshot["available"]:
        blockers.append("git_snapshot_unavailable")

    unique_blockers = sorted(set(blockers))
    artifact_freeze_complete = not unique_blockers
    generated_command = list(invocation or [sys.executable, *sys.argv])
    manifest = {
        "schema": "axon_roadmap_9997_truth_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "goal": "Full-test F1 >= 0.9997 without leakage, sample filtering, or abstain deletion.",
            "current_protocol_role": "legacy_development_leaderboard",
            "champion_scope": "research_champion",
        },
        "generation": {
            "argv": generated_command,
            "source_commands": list(source_commands or []),
            "historical_raw_to_report_command_available": False,
        },
        "git": git_snapshot,
        "champion_registry": champion_evidence,
        "metrics": metric_reports,
        "frozen_policy_sha256": next(iter(policy_hashes)) if len(policy_hashes) == 1 else None,
        "target_gap": target_gap,
        "artifacts": artifacts,
        "integrity": {
            "artifact_freeze_complete": artifact_freeze_complete,
            "blockers": unique_blockers,
            "required_artifact_count": sum(bool(row["required"]) for row in artifacts),
            "present_required_artifact_count": sum(
                bool(row["required"] and row["exists"]) for row in artifacts
            ),
        },
        "capability_boundary": {
            "classification": "prediction_level_policy_freeze",
            "raw_file_to_report_replay_ready": False,
            "native_loop151_ready": False,
            "connected_system_ready": False,
            "certification_ready": False,
            "blockers": [
                "The fail-closed raw replay runner exists, but Loop127 through Loop151 are not connected as one raw runtime.",
                "Loop151 still consumes frozen Loop136 prediction CSVs rather than raw files.",
                "Guarded Python/native Loop28 decisions match on the first train smoke row, but probability deltas exceed the <=1e-6 contract.",
                "The Authenticode signer guard is not serialized into the Loop28 native bundle.",
                "The legacy full-test is a repeatedly observed development leaderboard, not a sealed holdout.",
                "An immutable clean execution bundle and independently repeated empty-workdir replay are still required.",
            ],
        },
        "decision": (
            "artifact_freeze_complete_raw_replay_pending"
            if artifact_freeze_complete
            else "artifact_freeze_blocked"
        ),
        "next_gate": "close_loop127_to_loop151_raw_runtime_edges_and_pass_loop28_parity",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_named_paths(values: Optional[Sequence[str]], option_name: str) -> dict[str, Path]:
    parsed = {}
    for value in values or []:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path.strip():
            raise ValueError(f"{option_name} must use NAME=PATH: {value!r}")
        parsed[name.strip()] = Path(path.strip())
    return parsed


def _reject_cyclic_supplemental_artifacts(
    project_root: Path,
    output_json: Path,
    supplemental: Mapping[str, Path],
) -> None:
    root = project_root.resolve()
    forbidden = {
        resolve_path(root, path).resolve() for path in FORBIDDEN_SUPPLEMENTAL_ARTIFACT_PATHS
    }
    forbidden.add(resolve_path(root, output_json).resolve())
    for name, path in supplemental.items():
        resolved = resolve_path(root, path).resolve()
        if resolved in forbidden:
            raise ValueError(
                f"--artifact cannot include generated authorization/output path: {name}={path}"
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze Loop151 facts and artifacts for the 99.97 roadmap."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("manifests/roadmap_9997/p0_truth_freeze/loop151_truth_manifest.json"),
    )
    parser.add_argument("--metric-report", action="append", default=None, metavar="SPLIT=PATH")
    parser.add_argument("--artifact", action="append", default=None, metavar="NAME=PATH")
    parser.add_argument("--source-command", action="append", default=None)
    parser.add_argument("--target-f1", type=float, default=TARGET_F1)
    parser.add_argument("--champion-id", default=CHAMPION_ID)
    parser.add_argument("--fail-on-blocker", action="store_true")
    args = parser.parse_args(argv)

    try:
        metric_reports = _parse_named_paths(args.metric_report, "--metric-report") or dict(
            DEFAULT_METRIC_REPORTS
        )
        supplemental = _parse_named_paths(args.artifact, "--artifact")
        _reject_cyclic_supplemental_artifacts(
            args.project_root,
            args.output_json,
            supplemental,
        )
    except ValueError as exc:
        parser.error(str(exc))
    artifact_specs = [*DEFAULT_ARTIFACTS]
    artifact_specs.extend(
        ArtifactSpec(name=name, role="supplemental", path=path)
        for name, path in supplemental.items()
    )
    invocation = [
        sys.executable,
        str(Path(__file__).resolve()),
        *(argv if argv is not None else sys.argv[1:]),
    ]
    manifest = build_truth_manifest(
        project_root=args.project_root,
        output_json=args.output_json,
        metric_report_paths=metric_reports,
        artifact_specs=artifact_specs,
        target_f1=args.target_f1,
        champion_id=args.champion_id,
        source_commands=args.source_command,
        invocation=invocation,
    )
    summary = {
        "decision": manifest["decision"],
        "artifact_freeze_complete": manifest["integrity"]["artifact_freeze_complete"],
        "blocker_count": len(manifest["integrity"]["blockers"]),
        "legacy_full_test_f1": (
            manifest.get("metrics", {}).get("legacy_full_test", {}).get("candidate", {}).get("f1")
        ),
        "minimum_errors_to_remove": (manifest.get("target_gap") or {}).get(
            "minimum_errors_to_remove"
        ),
        "output_json": str(resolve_path(args.project_root, args.output_json)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.fail_on_blocker and manifest["integrity"]["blockers"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
