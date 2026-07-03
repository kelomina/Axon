#!/usr/bin/env python3
"""Build content-only evidence rows for strict prediction errors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kvd_features.extractor import FEATURE_NAMES as LEGACY_PE_FEATURE_NAMES  # noqa: E402
from kvd_features.schema_names import fixed_v2_feature_names  # noqa: E402


FIXED_V2_PE_ALIASES = {
    "pe_file_size": "fixed_v2_file_size",
    "pe_log_size": "fixed_v2_log_size",
    "pe_size_of_optional_header": "fixed_v2_size_of_optional_header",
    "pe_header_size_ratio": "fixed_v2_header_size_ratio",
    "pe_subsystem": "fixed_v2_subsystem",
    "pe_dll_characteristics": "fixed_v2_dll_characteristics",
    "pe_checksum_zero_flag": "fixed_v2_checksum_zero_flag",
    "pe_has_aslr": "fixed_v2_has_aslr",
    "pe_has_nx_compat": "fixed_v2_has_nx_compat",
    "pe_has_guard_cf": "fixed_v2_has_guard_cf",
    "pe_has_seh": "fixed_v2_has_seh",
    "pe_has_debug_info": "fixed_v2_has_debug_info",
    "pe_has_relocs": "fixed_v2_has_relocs",
    "pe_has_tls": "fixed_v2_has_tls",
    "pe_has_exceptions": "fixed_v2_has_exceptions",
    "has_signature": "fixed_v2_has_signature",
    "sections_count": "fixed_v2_sections_count",
    "section_entropy_max": "fixed_v2_section_entropy_max",
    "section_entropy_min": "fixed_v2_section_entropy_min",
    "section_entropy_avg": "fixed_v2_section_entropy_avg",
    "section_entropy_std": "fixed_v2_section_entropy_std",
    "section_high_entropy_ratio": "fixed_v2_section_high_entropy_ratio",
    "section_total_raw_size": "fixed_v2_section_total_raw_size",
    "section_total_virtual_size": "fixed_v2_section_total_virtual_size",
    "section_avg_raw_size": "fixed_v2_section_avg_raw_size",
    "section_avg_virtual_size": "fixed_v2_section_avg_virtual_size",
    "section_min_raw_size": "fixed_v2_section_min_raw_size",
    "section_max_raw_size": "fixed_v2_section_max_raw_size",
    "section_raw_size_std": "fixed_v2_section_raw_size_std",
    "section_raw_size_cv": "fixed_v2_section_raw_size_cv",
    "section_names_count": "fixed_v2_section_names_count",
    "section_name_avg_length": "fixed_v2_section_name_avg_length",
    "section_name_max_length": "fixed_v2_section_name_max_length",
    "section_name_min_length": "fixed_v2_section_name_min_length",
    "long_sections_count": "fixed_v2_long_sections_count",
    "long_sections_ratio": "fixed_v2_long_sections_ratio",
    "short_sections_count": "fixed_v2_short_sections_count",
    "short_sections_ratio": "fixed_v2_short_sections_ratio",
    "api_network_ratio": "fixed_v2_api_network_ratio",
    "api_process_ratio": "fixed_v2_api_process_ratio",
    "api_filesystem_ratio": "fixed_v2_api_filesystem_ratio",
    "api_registry_ratio": "fixed_v2_api_registry_ratio",
    "api_crypto_ratio": "fixed_v2_api_crypto_ratio",
    "api_injection_ratio": "fixed_v2_api_injection_ratio",
    "packer_keyword_hits_count": "fixed_v2_packer_keyword_hits_count",
    "packer_keyword_hits_ratio": "fixed_v2_packer_keyword_hits_ratio",
}

LEGACY_PE_ALIASES = {
    "pe_file_size": "size",
    "pe_log_size": "log_size",
    "pe_entropy": "entropy",
    "sections_count": "sections_count",
    "section_entropy_max": "section_entropy_max",
    "section_entropy_min": "section_entropy_min",
    "section_entropy_avg": "section_entropy_avg",
    "section_entropy_std": "section_entropy_std",
    "packed_sections_ratio": "packed_sections_ratio",
    "section_total_raw_size": "section_total_size",
    "section_total_virtual_size": "section_total_vsize",
    "section_avg_raw_size": "avg_section_size",
    "section_avg_virtual_size": "avg_section_vsize",
    "section_min_raw_size": "min_section_size",
    "section_max_raw_size": "max_section_size",
    "section_raw_size_std": "section_size_std",
    "section_raw_size_cv": "section_size_cv",
    "section_names_count": "section_names_count",
    "section_name_avg_length": "section_name_avg_length",
    "section_name_max_length": "section_name_max_length",
    "section_name_min_length": "section_name_min_length",
    "long_sections_count": "long_sections_count",
    "long_sections_ratio": "long_sections_ratio",
    "short_sections_count": "short_sections_count",
    "short_sections_ratio": "short_sections_ratio",
    "executable_sections_ratio": "executable_sections_ratio",
    "writable_sections_ratio": "writable_sections_ratio",
    "readable_sections_ratio": "readable_sections_ratio",
    "rwx_sections_ratio": "rwx_sections_ratio",
    "imports_count": "imports_count",
    "unique_imports": "unique_imports",
    "unique_dlls": "unique_dlls",
    "has_resources": "has_resources",
    "resources_count": "resources_count",
    "has_signature": "has_signature",
    "entry_point_ratio": "entry_point_ratio",
    "entry_in_nonstandard_section_flag": "entry_in_nonstandard_section_flag",
    "trailing_data_size": "trailing_data_size",
    "trailing_data_ratio": "trailing_data_ratio",
    "has_large_trailing_data": "has_large_trailing_data",
    "overlay_entropy": "overlay_entropy",
    "overlay_high_entropy_flag": "overlay_high_entropy_flag",
    "api_network_ratio": "api_network_ratio",
    "api_process_ratio": "api_process_ratio",
    "api_filesystem_ratio": "api_filesystem_ratio",
    "api_registry_ratio": "api_registry_ratio",
    "packer_keyword_hits_count": "packer_keyword_hits_count",
    "packer_keyword_hits_ratio": "packer_keyword_hits_ratio",
}

DERIVED_FIXED_V2_PE_NAMES = [
    "executable_sections_ratio",
    "writable_sections_ratio",
    "readable_sections_ratio",
    "rwx_sections_ratio",
]

PE_EVIDENCE_NAMES = list(
    dict.fromkeys(
        [
            *LEGACY_PE_ALIASES.keys(),
            *FIXED_V2_PE_ALIASES.keys(),
            *DERIVED_FIXED_V2_PE_NAMES,
        ]
    )
)

STAT_EVIDENCE_INDICES = {
    "stat_mean_byte": 0,
    "stat_std_byte": 1,
    "stat_min_byte": 2,
    "stat_max_byte": 3,
    "stat_median_byte": 4,
    "stat_q25_byte": 5,
    "stat_q75_byte": 6,
    "stat_count_0x00": 7,
    "stat_count_0xff": 8,
    "stat_count_0x90": 9,
    "stat_count_ascii_printable": 10,
    "stat_byte_entropy": 11,
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def is_valid_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _bool_text(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _prediction_from_row(row: dict[str, str], prefix: str, threshold_fallback: Optional[float]) -> tuple[float, int, bool]:
    if prefix == "calibrated":
        score_key = "calibrated_prob_malicious"
        pred_key = "calibrated_prediction"
        correct_key = "calibrated_correct"
    else:
        score_key = "prob_malicious" if "prob_malicious" in row else "baseline_prob_malicious"
        pred_key = "prediction" if "prediction" in row else "baseline_prediction"
        correct_key = "correct" if "correct" in row else "baseline_correct"

    score = float(row[score_key])
    if pred_key in row and str(row.get(pred_key, "")).strip() != "":
        prediction = int(float(row[pred_key]))
    elif threshold_fallback is not None:
        prediction = int(score >= threshold_fallback)
    else:
        raise ValueError(f"Prediction row missing {pred_key!r} and no threshold fallback was supplied")
    label = int(float(row["label"]))
    if correct_key in row and str(row.get(correct_key, "")).strip() != "":
        correct = _bool_text(row[correct_key])
    else:
        correct = prediction == label
    return score, prediction, correct


def _load_cache_arrays(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    cache_path = resolve_path(Path(row["cache_path"]))
    source_sha = str(row.get("source_sha256") or "").strip().casefold()
    label = int(float(row["label"]))
    with np.load(cache_path, allow_pickle=False) as data:
        required = {"pe_features", "label", "source_sha256"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Cache missing required fields {missing}: {cache_path}")
        cache_label = int(np.asarray(data["label"]).reshape(-1)[0])
        if cache_label != label:
            raise ValueError(f"Cache label mismatch for {cache_path}: expected {label}, got {cache_label}")
        cache_sha = str(np.asarray(data["source_sha256"]).reshape(-1)[0]).strip().casefold()
        if cache_sha != source_sha:
            raise ValueError(f"Cache source_sha256 mismatch for {cache_path}")
        pe_features = data["pe_features"].astype(np.float32)
        stat_features = (
            data["stat_features"].astype(np.float32)
            if "stat_features" in data.files
            else np.zeros(49, dtype=np.float32)
        )
    return pe_features, stat_features


def _infer_pe_schema_version(pe_features: np.ndarray, requested: str, pe_fixed_section_slots: int) -> str:
    if requested != "auto":
        return requested
    fixed_v2_used_dim = len(fixed_v2_feature_names(section_slots=pe_fixed_section_slots))
    if len(pe_features) <= 512 and len(pe_features) >= fixed_v2_used_dim:
        return "fixed_v2"
    return "legacy_dynamic"


def _named_pe_value(pe_features: np.ndarray, columns: dict[str, int], feature_name: str) -> Optional[float]:
    index = columns.get(feature_name)
    if index is None or index >= len(pe_features):
        return None
    return float(pe_features[index])


def _fixed_v2_section_flag_ratios(
    pe_features: np.ndarray,
    columns: dict[str, int],
    *,
    section_slots: int,
    section_count: float,
) -> dict[str, float]:
    active_slots = min(max(int(round(section_count)), 0), section_slots)
    if active_slots <= 0:
        return {
            "executable_sections_ratio": 0.0,
            "writable_sections_ratio": 0.0,
            "readable_sections_ratio": 0.0,
            "rwx_sections_ratio": 0.0,
        }

    executable = 0
    writable = 0
    readable = 0
    rwx = 0
    for slot in range(active_slots):
        exec_flag = _safe_float(
            _named_pe_value(pe_features, columns, f"fixed_v2_section_{slot:02d}_is_executable")
        ) >= 0.5
        write_flag = _safe_float(
            _named_pe_value(pe_features, columns, f"fixed_v2_section_{slot:02d}_is_writable")
        ) >= 0.5
        read_flag = _safe_float(
            _named_pe_value(pe_features, columns, f"fixed_v2_section_{slot:02d}_is_readable")
        ) >= 0.5
        executable += int(exec_flag)
        writable += int(write_flag)
        readable += int(read_flag)
        rwx += int(exec_flag and write_flag and read_flag)

    denominator = float(active_slots)
    return {
        "executable_sections_ratio": executable / denominator,
        "writable_sections_ratio": writable / denominator,
        "readable_sections_ratio": readable / denominator,
        "rwx_sections_ratio": rwx / denominator,
    }


def _evidence_from_features(
    pe_features: np.ndarray,
    stat_features: np.ndarray,
    *,
    pe_schema_version: str = "auto",
    pe_fixed_section_slots: int = 32,
) -> tuple[dict[str, Optional[float]], str]:
    resolved_schema = _infer_pe_schema_version(pe_features, pe_schema_version, pe_fixed_section_slots)
    evidence: dict[str, Optional[float]] = {name: None for name in PE_EVIDENCE_NAMES}

    if resolved_schema == "fixed_v2":
        fixed_names = fixed_v2_feature_names(
            section_slots=pe_fixed_section_slots,
            pe_feature_dim=len(pe_features),
        )
        columns = {name: index for index, name in enumerate(fixed_names)}
        for output_name, schema_name in FIXED_V2_PE_ALIASES.items():
            evidence[output_name] = _named_pe_value(pe_features, columns, schema_name)
        evidence.update(
            _fixed_v2_section_flag_ratios(
                pe_features,
                columns,
                section_slots=pe_fixed_section_slots,
                section_count=_safe_float(evidence.get("sections_count")),
            )
        )
    elif resolved_schema == "legacy_dynamic":
        columns = {name: index for index, name in enumerate(LEGACY_PE_FEATURE_NAMES)}
        for output_name, schema_name in LEGACY_PE_ALIASES.items():
            evidence[output_name] = _named_pe_value(pe_features, columns, schema_name)
    else:
        raise ValueError(f"Unsupported PE schema version: {resolved_schema}")

    for name, index in STAT_EVIDENCE_INDICES.items():
        evidence[name] = float(stat_features[index]) if index < len(stat_features) else None
    return evidence, resolved_schema


def _severity_score(*, label: int, score: float, prediction: int, evidence: dict[str, Optional[float]]) -> float:
    confidence = score if prediction == 1 else 1.0 - score
    anomaly = 0.0
    anomaly += 0.20 if _safe_float(evidence.get("overlay_high_entropy_flag")) >= 0.5 else 0.0
    anomaly += 0.15 if _safe_float(evidence.get("has_large_trailing_data")) >= 0.5 else 0.0
    anomaly += min(max(_safe_float(evidence.get("section_high_entropy_ratio")), 0.0), 1.0) * 0.15
    anomaly += min(max(_safe_float(evidence.get("packed_sections_ratio")), 0.0), 1.0) * 0.15
    anomaly += min(max(_safe_float(evidence.get("rwx_sections_ratio")), 0.0), 1.0) * 0.10
    anomaly += min(max(_safe_float(evidence.get("packer_keyword_hits_ratio")), 0.0), 1.0) * 0.10
    if label == 0 and prediction == 1:
        return confidence + anomaly
    if label == 1 and prediction == 0:
        return confidence + max(0.0, 0.5 - anomaly)
    return confidence


def build_evidence_package(
    *,
    predictions_csv: Path,
    output_csv: Path,
    output_json: Path,
    score_prefix: str = "calibrated",
    threshold_fallback: Optional[float] = None,
    max_rows: Optional[int] = None,
    pe_schema_version: str = "auto",
    pe_fixed_section_slots: int = 32,
) -> dict:
    with resolve_path(predictions_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    output_rows: list[dict[str, object]] = []
    issue_counts: Counter = Counter()
    transition_counts: Counter = Counter()
    error_counts: Counter = Counter()
    pe_schema_counts: Counter = Counter()

    for row in rows:
        source_sha = str(row.get("source_sha256") or "").strip().casefold()
        if not is_valid_sha256(source_sha):
            issue_counts["invalid_source_sha256"] += 1
            continue
        label = int(float(row["label"]))
        score, prediction, correct = _prediction_from_row(row, score_prefix, threshold_fallback)
        if correct:
            continue
        pe_features, stat_features = _load_cache_arrays(row)
        evidence, resolved_schema = _evidence_from_features(
            pe_features,
            stat_features,
            pe_schema_version=pe_schema_version,
            pe_fixed_section_slots=pe_fixed_section_slots,
        )
        pe_schema_counts[resolved_schema] += 1
        error_type = "FP" if label == 0 and prediction == 1 else "FN"
        error_counts[error_type] += 1
        transition = row.get("error_transition", "error")
        transition_counts[transition] += 1
        evidence_row = {
            "source_path": row.get("source_path", ""),
            "source_sha256": source_sha,
            "cache_path": row.get("cache_path", ""),
            "label": label,
            "split": row.get("split", ""),
            "sample_index": row.get("sample_index", ""),
            "score_prefix": score_prefix,
            "score": score,
            "prediction": prediction,
            "error_type": error_type,
            "error_transition": transition,
            "pe_schema_version": resolved_schema,
            "severity_score": _severity_score(label=label, score=score, prediction=prediction, evidence=evidence),
            **evidence,
        }
        output_rows.append(evidence_row)
        if max_rows is not None and len(output_rows) >= max_rows:
            break

    output_rows.sort(key=lambda item: (-float(item["severity_score"]), str(item["error_type"]), str(item["sample_index"])))
    fieldnames = [
        "source_path",
        "source_sha256",
        "cache_path",
        "label",
        "split",
        "sample_index",
        "score_prefix",
        "score",
        "prediction",
        "error_type",
        "error_transition",
        "pe_schema_version",
        "severity_score",
        *PE_EVIDENCE_NAMES,
        *STAT_EVIDENCE_INDICES.keys(),
    ]
    resolved_output_csv = resolve_path(output_csv)
    resolved_output_csv.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    payload = {
        "schema": "axon_strict_error_content_evidence_package_v2",
        "identity_feature_policy": (
            "source_sha256/cache_path/source_path/sample_index are retained only for alignment and human lookup; "
            "ranking uses prediction confidence plus content PE/stat evidence, not filename/path/directory/extension."
        ),
        "feature_schema_policy": (
            "PE evidence values are resolved by schema names. fixed_v2 uses kvd_features.schema_names; "
            "legacy_dynamic uses kvd_features.extractor.FEATURE_NAMES. Unavailable schema fields are left blank."
        ),
        "predictions_csv": str(resolve_path(predictions_csv)),
        "output_csv": str(resolved_output_csv),
        "score_prefix": score_prefix,
        "threshold_fallback": threshold_fallback,
        "requested_pe_schema_version": pe_schema_version,
        "pe_fixed_section_slots": pe_fixed_section_slots,
        "input_rows": len(rows),
        "evidence_rows": len(output_rows),
        "error_counts": dict(sorted(error_counts.items())),
        "transition_counts": dict(sorted(transition_counts.items())),
        "pe_schema_counts": dict(sorted(pe_schema_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "max_rows": max_rows,
    }
    resolved_output_json = resolve_path(output_json)
    resolved_output_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build content-only evidence package for prediction errors.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--score-prefix", choices=["baseline", "calibrated"], default="calibrated")
    parser.add_argument("--threshold-fallback", type=float, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--pe-schema-version", choices=["auto", "fixed_v2", "legacy_dynamic"], default="auto")
    parser.add_argument("--pe-fixed-section-slots", type=int, default=32)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_evidence_package(
        predictions_csv=args.predictions_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
        score_prefix=args.score_prefix,
        threshold_fallback=args.threshold_fallback,
        max_rows=args.max_rows,
        pe_schema_version=args.pe_schema_version,
        pe_fixed_section_slots=args.pe_fixed_section_slots,
    )
    print(json.dumps({key: payload[key] for key in ["evidence_rows", "error_counts", "transition_counts", "pe_schema_counts", "issue_counts"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
