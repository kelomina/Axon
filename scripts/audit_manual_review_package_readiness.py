#!/usr/bin/env python3
"""Audit whether manual-review package rows have complete source/cache evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for item in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from apply_manual_review_verdicts import (  # noqa: E402
    EXCLUDE_ACTIONS,
    EXCLUDE_VERDICTS,
    KEEP_ACTIONS,
    KEEP_VERDICTS,
    RELABEL_ACTIONS,
    RELABEL_VERDICTS,
    UNCERTAIN_VERDICTS,
    normalize_text,
)
from audit_pe_metadata_queue import parse_pe  # noqa: E402


REQUIRED_NPZ_FIELDS = [
    "byte_sequence",
    "pe_features",
    "stat_features",
    "lightweight_features",
    "label",
    "source_sha256",
]
EXPECTED_NEIGHBOR_COUNT = 5
VALID_MANUAL_VERDICTS = KEEP_VERDICTS | RELABEL_VERDICTS | EXCLUDE_VERDICTS | UNCERTAIN_VERDICTS
VALID_RECOMMENDED_ACTIONS = KEEP_ACTIONS | RELABEL_ACTIONS | EXCLUDE_ACTIONS


def manual_verdict_category(verdict: str) -> str:
    if verdict in KEEP_VERDICTS:
        return "keep"
    if verdict in RELABEL_VERDICTS:
        return "relabel"
    if verdict in EXCLUDE_VERDICTS:
        return "exclude"
    if verdict in UNCERTAIN_VERDICTS:
        return "uncertain"
    return "invalid"


def recommended_action_category(action: str) -> str:
    if action in RELABEL_ACTIONS:
        return "relabel"
    if action in EXCLUDE_ACTIONS:
        return "exclude"
    if action in {"needs_more_evidence", "model_blindspot"}:
        return "uncertain"
    if action in KEEP_ACTIONS:
        return "keep"
    return "invalid"


def manual_fields_consistent(verdict: str, action: str) -> bool:
    verdict_category = manual_verdict_category(verdict)
    action_category = recommended_action_category(action)
    if verdict_category == "invalid" or action_category == "invalid":
        return False
    if verdict_category == "exclude":
        return action_category == "exclude"
    if verdict_category == "relabel":
        return action_category == "relabel"
    if verdict_category == "keep":
        return action_category == "keep"
    if verdict_category == "uncertain":
        return action_category in {"keep", "uncertain"}
    return False


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_path(path_text: str) -> str:
    return str(Path(path_text)).replace("/", "\\").casefold()


def project_relative_path_key(path_text: str) -> Optional[str]:
    if not path_text:
        return None
    normalized = normalize_path(path_text)
    root = normalize_path(str(PROJECT_ROOT)).rstrip("\\")
    prefix = root + "\\"
    if normalized.startswith(prefix):
        return normalized[len(prefix):]
    return None


def source_path_keys(path_text: str) -> list[str]:
    if not path_text:
        return []
    keys = {normalize_path(path_text)}
    path = Path(path_text)
    if not path.is_absolute():
        keys.add(normalize_path(str(PROJECT_ROOT / path)))
    relative_key = project_relative_path_key(path_text)
    if relative_key:
        keys.add(relative_key)
    keys.add(path.name.casefold())
    return list(keys)


def _looks_sha256(value: str) -> bool:
    text = value.casefold()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def source_sha_from_path(path_text: str) -> Optional[str]:
    stem = Path(path_text).stem.casefold()
    if _looks_sha256(stem):
        return stem
    name = Path(path_text).name.casefold()
    if _looks_sha256(name):
        return name
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_text(value: object) -> str:
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(value)


def load_manifest(manifest_path: Path) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    resolved = resolve_path(manifest_path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    by_source: dict[str, dict] = {}
    by_sha: dict[str, dict] = {}
    for sample in payload.get("samples", []):
        source_path = sample.get("source_path", "")
        for key in source_path_keys(source_path):
            by_source.setdefault(key, sample)
        source_sha256 = str(sample.get("source_sha256") or "").casefold()
        if source_sha256:
            by_sha.setdefault(source_sha256, sample)
        path_sha = source_sha_from_path(source_path)
        if path_sha:
            by_sha.setdefault(path_sha, sample)
    return payload, by_source, by_sha


def lookup_manifest_sample(row: dict, by_source: dict[str, dict], by_sha: dict[str, dict]) -> tuple[Optional[dict], str]:
    for key in source_path_keys(row.get("source_path", "")):
        sample = by_source.get(key)
        if sample is not None:
            return sample, "source_path"

    explicit_sha = str(row.get("source_sha256") or "").casefold()
    if explicit_sha:
        sample = by_sha.get(explicit_sha)
        if sample is not None:
            return sample, "source_sha256"

    path_sha = source_sha_from_path(row.get("source_path", ""))
    if path_sha:
        sample = by_sha.get(path_sha)
        if sample is not None:
            return sample, "source_sha256_from_path"
    return None, "manifest_missing"


def lookup_neighbor_sample(source_path: str, source_sha256: str, by_source: dict[str, dict], by_sha: dict[str, dict]) -> Optional[dict]:
    sha = str(source_sha256 or "").casefold()
    if sha:
        sample = by_sha.get(sha)
        if sample is not None:
            return sample
    for key in source_path_keys(source_path):
        sample = by_source.get(key)
        if sample is not None:
            return sample
    path_sha = source_sha_from_path(source_path)
    if path_sha:
        return by_sha.get(path_sha)
    return None


def resolve_cache_path(sample: dict, manifest_dir: Path) -> Path:
    cache_path = Path(sample.get("cache_path", ""))
    if cache_path.is_absolute():
        return cache_path
    return manifest_dir / cache_path.name


def shape_expected(manifest: dict, key: str) -> Optional[tuple[int, ...]]:
    value = int(manifest.get(key, 0) or 0)
    if value <= 0:
        return None
    return (value,)


def audit_npz(
    *,
    cache_path: Path,
    manifest: dict,
    row: dict,
    sample: dict,
    actual_source_sha256: str,
) -> dict:
    result = {
        "npz_loaded": False,
        "npz_error": "",
        "npz_missing_fields": "",
        "npz_label": "",
        "label_ok": False,
        "npz_source_sha256": "",
        "npz_source_sha256_ok": False,
        "npz_shape_ok": True,
        "npz_shape_errors": "",
    }
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            result["npz_loaded"] = True
            missing = [field for field in REQUIRED_NPZ_FIELDS if field not in data.files]
            if missing:
                result["npz_missing_fields"] = "|".join(missing)
                return result

            expected_label = int(row.get("label", sample.get("label")))
            manifest_label = int(sample.get("label", expected_label))
            npz_label = int(np.asarray(data["label"]).item())
            result["npz_label"] = str(npz_label)
            result["label_ok"] = npz_label == expected_label == manifest_label

            npz_sha = scalar_text(data["source_sha256"]).casefold()
            expected_sha = str(row.get("source_sha256") or sample.get("source_sha256") or actual_source_sha256).casefold()
            manifest_sha = str(sample.get("source_sha256") or "").casefold()
            result["npz_source_sha256"] = npz_sha
            result["npz_source_sha256_ok"] = bool(npz_sha) and npz_sha == expected_sha == manifest_sha == actual_source_sha256

            shape_checks = [
                ("byte_sequence", shape_expected(manifest, "max_byte_length")),
                ("pe_features", shape_expected(manifest, "pe_feature_dim")),
                ("stat_features", shape_expected(manifest, "stat_feature_dim")),
                ("lightweight_features", shape_expected(manifest, "lightweight_feature_dim")),
            ]
            shape_errors = []
            for field, expected_shape in shape_checks:
                if expected_shape is None:
                    continue
                actual_shape = tuple(data[field].shape)
                if actual_shape != expected_shape:
                    shape_errors.append(f"{field}:actual={actual_shape}:expected={expected_shape}")
            result["npz_shape_ok"] = not shape_errors
            result["npz_shape_errors"] = "|".join(shape_errors)
    except Exception as exc:  # pragma: no cover - operational detail is reported in CSV/JSON.
        result["npz_error"] = repr(exc)
    return result


def split_pipe_values(value: object) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def audit_neighbor_evidence(row: dict, by_source: dict[str, dict], by_sha: dict[str, dict], manifest_dir: Path) -> dict:
    labels = split_pipe_values(row.get("top5_neighbor_labels"))
    similarities = split_pipe_values(row.get("top5_neighbor_similarities"))
    shas = split_pipe_values(row.get("top5_neighbor_sha256"))
    paths = split_pipe_values(row.get("top5_neighbor_paths"))
    top5_lengths = {
        "labels": len(labels),
        "similarities": len(similarities),
        "sha256": len(shas),
        "paths": len(paths),
    }
    lengths_ok = all(count == EXPECTED_NEIGHBOR_COUNT for count in top5_lengths.values())
    labels_ok = all(label in {"0", "1"} for label in labels) and len(labels) == EXPECTED_NEIGHBOR_COUNT
    similarities_ok = len(similarities) == EXPECTED_NEIGHBOR_COUNT
    if similarities_ok:
        for value in similarities:
            try:
                float(value)
            except ValueError:
                similarities_ok = False
                break

    manifest_found = 0
    path_exists = 0
    cache_exists = 0
    for index in range(max(len(paths), len(shas))):
        path_text = paths[index] if index < len(paths) else ""
        sha_text = shas[index] if index < len(shas) else ""
        sample = lookup_neighbor_sample(path_text, sha_text, by_source, by_sha)
        if sample is not None:
            manifest_found += 1
            cache_path = resolve_cache_path(sample, manifest_dir)
            if cache_path.exists():
                cache_exists += 1
        resolved_path = resolve_path(Path(path_text)) if path_text else Path("")
        if path_text and resolved_path.exists():
            path_exists += 1

    evidence_ok = (
        lengths_ok
        and labels_ok
        and similarities_ok
        and manifest_found == EXPECTED_NEIGHBOR_COUNT
        and path_exists == EXPECTED_NEIGHBOR_COUNT
        and cache_exists == EXPECTED_NEIGHBOR_COUNT
    )
    return {
        "top5_lengths_ok": lengths_ok,
        "top5_neighbor_labels_ok": labels_ok,
        "top5_neighbor_similarities_ok": similarities_ok,
        "top5_neighbor_manifest_found_count": manifest_found,
        "top5_neighbor_path_exists_count": path_exists,
        "top5_neighbor_cache_exists_count": cache_exists,
        "top5_neighbor_evidence_ok": evidence_ok,
        "top5_neighbor_lengths": json.dumps(top5_lengths, sort_keys=True),
    }


def readiness_reasons(row: dict, require_pe: bool) -> list[str]:
    reasons = []
    checks = [
        ("source_missing", row["source_exists"]),
        ("source_sha256_mismatch", row["source_sha256_ok"]),
        ("manifest_missing", row["manifest_found"]),
        ("cache_missing", row["cache_exists"]),
        ("npz_load_failed", row["npz_loaded"]),
        ("npz_missing_fields", not bool(row["npz_missing_fields"])),
        ("label_mismatch", row["label_ok"]),
        ("cache_source_sha256_mismatch", row["npz_source_sha256_ok"]),
        ("npz_shape_mismatch", row["npz_shape_ok"]),
        ("top5_neighbor_evidence_incomplete", row["top5_neighbor_evidence_ok"]),
    ]
    for reason, ok in checks:
        if not ok:
            reasons.append(reason)
    if require_pe and not row["is_pe"]:
        reasons.append(f"not_valid_pe:{row['parse_error'] or 'unknown'}")
    return reasons


def _is_blank(value: object) -> bool:
    return str(value or "").strip() == ""


def audit_manual_fields(row: dict) -> dict:
    verdict = normalize_text(row.get("manual_label_verdict"))
    action = normalize_text(row.get("recommended_action"))
    verdict_valid = verdict in VALID_MANUAL_VERDICTS
    action_valid = action in VALID_RECOMMENDED_ACTIONS
    fields_consistent = not (verdict_valid and action_valid) or manual_fields_consistent(verdict, action)
    return {
        "manual_label_verdict_normalized": verdict,
        "recommended_action_normalized": action,
        "manual_label_verdict_blank": verdict == "",
        "recommended_action_blank": action == "",
        "manual_label_verdict_valid": verdict_valid,
        "recommended_action_valid": action_valid,
        "manual_verdict_category": manual_verdict_category(verdict),
        "recommended_action_category": recommended_action_category(action),
        "manual_fields_consistent": fields_consistent,
    }


def audit_manual_review_package(
    *,
    review_csv: Path,
    manifest_json: Path,
    output_csv: Path,
    output_json: Path,
    require_pe: bool = True,
) -> dict:
    rows = read_csv_rows(review_csv)
    manifest, by_source, by_sha = load_manifest(manifest_json)
    manifest_dir = resolve_path(manifest_json).parent
    output_rows: list[dict] = []
    match_counts: Counter[str] = Counter()

    for row in rows:
        source_path = resolve_path(Path(row.get("source_path", "")))
        source_exists = source_path.exists()
        actual_sha = sha256_file(source_path).casefold() if source_exists else ""
        expected_sha = str(row.get("source_sha256") or "").casefold()
        source_sha_ok = source_exists and (not expected_sha or actual_sha == expected_sha)

        sample, match_reason = lookup_manifest_sample(row, by_source, by_sha)
        manifest_found = sample is not None
        match_counts[match_reason] += 1
        cache_path = resolve_cache_path(sample, manifest_dir) if sample is not None else Path("")
        cache_exists = bool(sample) and cache_path.exists()

        if cache_exists:
            npz_result = audit_npz(
                cache_path=cache_path,
                manifest=manifest,
                row=row,
                sample=sample,
                actual_source_sha256=actual_sha,
            )
        else:
            npz_result = {
                "npz_loaded": False,
                "npz_error": "",
                "npz_missing_fields": "",
                "npz_label": "",
                "label_ok": False,
                "npz_source_sha256": "",
                "npz_source_sha256_ok": False,
                "npz_shape_ok": False,
                "npz_shape_errors": "",
            }

        pe_metadata = parse_pe(source_path)
        neighbor_evidence = audit_neighbor_evidence(row, by_source, by_sha, manifest_dir)
        manual_fields = audit_manual_fields(row)
        out = {
            **row,
            "source_exists": source_exists,
            "actual_source_sha256": actual_sha,
            "source_sha256_ok": source_sha_ok,
            "manifest_found": manifest_found,
            "manifest_match_reason": match_reason,
            "manifest_source_sha256": sample.get("source_sha256", "") if sample else "",
            "manifest_label": sample.get("label", "") if sample else "",
            "manifest_cache_path": str(cache_path) if sample else "",
            "cache_exists": cache_exists,
            **npz_result,
            "is_pe": bool(pe_metadata["is_pe"]),
            "parse_error": pe_metadata["parse_error"],
            "file_size": pe_metadata["file_size"],
            "machine": pe_metadata["machine"],
            "number_of_sections": pe_metadata["number_of_sections"],
            "section_names": pe_metadata["section_names"],
            "max_section_entropy": pe_metadata["max_section_entropy"],
            "overlay_size": pe_metadata["overlay_size"],
            **neighbor_evidence,
            **manual_fields,
        }
        reasons = readiness_reasons(out, require_pe=require_pe)
        out["manual_review_ready"] = not reasons
        out["readiness_reasons"] = "|".join(reasons)
        output_rows.append(out)

    fieldnames = [
        "review_rank",
        "support_bucket",
        "priority",
        "reason",
        "error_type",
        "source_path",
        "source_sha256",
        "label",
        "prediction",
        "prob_malicious",
        "score_column",
        "base_prob_malicious",
        "neighbor_label_counts",
        "opposite_label_ratio",
        "nearest_similarity",
        "source_exists",
        "actual_source_sha256",
        "source_sha256_ok",
        "manifest_found",
        "manifest_match_reason",
        "manifest_source_sha256",
        "manifest_label",
        "manifest_cache_path",
        "cache_exists",
        "npz_loaded",
        "npz_error",
        "npz_missing_fields",
        "npz_label",
        "label_ok",
        "npz_source_sha256",
        "npz_source_sha256_ok",
        "npz_shape_ok",
        "npz_shape_errors",
        "is_pe",
        "parse_error",
        "file_size",
        "machine",
        "number_of_sections",
        "section_names",
        "max_section_entropy",
        "overlay_size",
        "manual_review_ready",
        "readiness_reasons",
        "top5_neighbor_labels",
        "top5_neighbor_similarities",
        "top5_neighbor_sha256",
        "top5_neighbor_paths",
        "top5_lengths_ok",
        "top5_neighbor_labels_ok",
        "top5_neighbor_similarities_ok",
        "top5_neighbor_manifest_found_count",
        "top5_neighbor_path_exists_count",
        "top5_neighbor_cache_exists_count",
        "top5_neighbor_evidence_ok",
        "top5_neighbor_lengths",
        "manual_label_verdict_normalized",
        "recommended_action_normalized",
        "manual_label_verdict_blank",
        "recommended_action_blank",
        "manual_label_verdict_valid",
        "recommended_action_valid",
        "manual_verdict_category",
        "recommended_action_category",
        "manual_fields_consistent",
        "manual_label_verdict",
        "manual_verdict_note",
        "recommended_action",
    ]
    output_csv = resolve_path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    readiness_counter = Counter("ready" if row["manual_review_ready"] else "not_ready" for row in output_rows)
    verdict_blank_count = sum(1 for row in output_rows if row["manual_label_verdict_blank"])
    action_blank_count = sum(1 for row in output_rows if row["recommended_action_blank"])
    verdict_invalid_count = sum(1 for row in output_rows if not row["manual_label_verdict_valid"])
    action_invalid_count = sum(1 for row in output_rows if not row["recommended_action_valid"])
    manual_inconsistent_count = sum(1 for row in output_rows if not row["manual_fields_consistent"])
    verdict_package_ready = (
        readiness_counter["not_ready"] == 0
        and verdict_blank_count == 0
        and action_blank_count == 0
        and verdict_invalid_count == 0
        and action_invalid_count == 0
        and manual_inconsistent_count == 0
    )
    reason_counter: Counter[str] = Counter()
    for row in output_rows:
        for reason in str(row["readiness_reasons"]).split("|"):
            if reason:
                reason_counter[reason] += 1

    summary = {
        "schema": "axon_manual_review_package_readiness_audit_v1",
        "review_csv": str(resolve_path(review_csv)),
        "manifest_json": str(resolve_path(manifest_json)),
        "require_pe": bool(require_pe),
        "total_rows": len(output_rows),
        "ready_rows": readiness_counter["ready"],
        "not_ready_rows": readiness_counter["not_ready"],
        "review_queue_ready": readiness_counter["not_ready"] == 0,
        "verdict_package_ready": verdict_package_ready,
        "manual_review_ready": readiness_counter["not_ready"] == 0,
        "manual_label_verdict_blank_count": verdict_blank_count,
        "recommended_action_blank_count": action_blank_count,
        "manual_label_verdict_invalid_count": verdict_invalid_count,
        "recommended_action_invalid_count": action_invalid_count,
        "manual_fields_inconsistent_count": manual_inconsistent_count,
        "blocking_issues": [
            issue
            for issue, present in [
                ("review_queue_not_ready", readiness_counter["not_ready"] > 0),
                ("manual_verdict_empty", verdict_blank_count > 0),
                ("recommended_action_empty", action_blank_count > 0),
                ("manual_verdict_invalid", verdict_invalid_count > 0),
                ("recommended_action_invalid", action_invalid_count > 0),
                ("manual_fields_inconsistent", manual_inconsistent_count > 0),
            ]
            if present
        ],
        "label_counts": dict(sorted(Counter(str(row.get("label", "")) for row in output_rows).items())),
        "error_type_counts": dict(sorted(Counter(str(row.get("error_type", "")) for row in output_rows).items())),
        "priority_counts": dict(sorted(Counter(str(row.get("priority", "")) for row in output_rows).items())),
        "manifest_match_counts": dict(sorted(match_counts.items())),
        "source_exists_count": sum(1 for row in output_rows if row["source_exists"]),
        "source_sha256_ok_count": sum(1 for row in output_rows if row["source_sha256_ok"]),
        "cache_exists_count": sum(1 for row in output_rows if row["cache_exists"]),
        "npz_loaded_count": sum(1 for row in output_rows if row["npz_loaded"]),
        "npz_label_ok_count": sum(1 for row in output_rows if row["label_ok"]),
        "npz_source_sha256_ok_count": sum(1 for row in output_rows if row["npz_source_sha256_ok"]),
        "npz_shape_ok_count": sum(1 for row in output_rows if row["npz_shape_ok"]),
        "is_pe_count": sum(1 for row in output_rows if row["is_pe"]),
        "top5_lengths_ok_count": sum(1 for row in output_rows if row["top5_lengths_ok"]),
        "top5_neighbor_evidence_ok_count": sum(1 for row in output_rows if row["top5_neighbor_evidence_ok"]),
        "top5_neighbor_manifest_found_total": sum(int(row["top5_neighbor_manifest_found_count"]) for row in output_rows),
        "top5_neighbor_path_exists_total": sum(int(row["top5_neighbor_path_exists_count"]) for row in output_rows),
        "top5_neighbor_cache_exists_total": sum(int(row["top5_neighbor_cache_exists_count"]) for row in output_rows),
        "parse_error_counts": dict(sorted(Counter(str(row["parse_error"]) for row in output_rows).items())),
        "readiness_reason_counts": dict(sorted(reason_counter.items())),
        "manual_label_verdict_counts": dict(
            sorted(Counter(str(row["manual_label_verdict_normalized"]) for row in output_rows).items())
        ),
        "recommended_action_counts": dict(
            sorted(Counter(str(row["recommended_action_normalized"]) for row in output_rows).items())
        ),
        "manual_verdict_category_counts": dict(
            sorted(Counter(str(row["manual_verdict_category"]) for row in output_rows).items())
        ),
        "recommended_action_category_counts": dict(
            sorted(Counter(str(row["recommended_action_category"]) for row in output_rows).items())
        ),
        "duplicate_source_sha256_count": sum(
            1
            for _sha, count in Counter(str(row.get("source_sha256", "")).casefold() for row in output_rows).items()
            if _sha and count > 1
        ),
        "examples_not_ready": [row for row in output_rows if not row["manual_review_ready"]][:20],
        "outputs": {
            "readiness_csv": str(output_csv),
            "summary_json": str(resolve_path(output_json)),
        },
    }
    output_json = resolve_path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit manual-review package source/cache readiness.")
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--no-require-pe", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless evidence is ready and manual verdict/action fields are complete and consistent.",
    )
    args = parser.parse_args(argv)
    summary = audit_manual_review_package(
        review_csv=args.review_csv,
        manifest_json=args.manifest_json,
        output_csv=args.output_csv,
        output_json=args.output_json,
        require_pe=not bool(args.no_require_pe),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.strict and not summary["verdict_package_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
