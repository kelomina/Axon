#!/usr/bin/env python3
"""Build a train-only, SHA-bound local runtime probe bundle for Loop164.

The builder reads only split/cache metadata plus source-file stat metadata.  It
never opens source bytes, cache NPZ arrays, checkpoints, prediction rows, or
heldout split rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP_ID = "loop164_whole_file_residual_expert"
BUNDLE_SCHEMA = "axon_loop164_local_probe_bundle_v1"
BUNDLE_RECORD_SCHEMA = "axon_loop164_local_probe_record_v1"
SUMMARY_SCHEMA = "axon_loop164_local_probe_bundle_summary_v1"
CANONICAL_SPLIT_ROLE = "train"
RECORDS_PER_CLASS = 128
MIN_SOURCE_SIZE_BYTES = 64 * 1024
MAX_SOURCE_SIZE_BYTES = 8 * 1024 * 1024
SELECTION_SEED = "loop164_local_probe_bundle_v1"
REQUIRED_SPLIT_COLUMNS = {"source_path", "source_sha256", "label", "split"}

DEFAULT_SPLIT_CSV = (
    PROJECT_ROOT / "reports" / "random_20w_split" / "loop127_full_duplicate_corrected_split.csv"
)
DEFAULT_CACHE_MANIFEST = PROJECT_ROOT / "data" / ".cache" / "manifest_38672ba0.json"
DEFAULT_CANONICAL_SOURCE_ROOT = PROJECT_ROOT / "data"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "random_20w_worktree"
DEFAULT_BUNDLE_OUTPUT = PROJECT_ROOT / "reports" / "roadmap_9997" / "loop164" / "local_probe_bundle.jsonl"
DEFAULT_SUMMARY_OUTPUT = (
    PROJECT_ROOT / "reports" / "roadmap_9997" / "loop164" / "local_probe_bundle_summary.json"
)


@dataclass(frozen=True)
class ProbeCandidate:
    source_path: str
    source_sha256: str
    label: int
    source_size_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Non-finite JSON value: {value}")


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _is_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _normalize_label(value: object, *, context: str) -> int:
    text = "" if value is None else str(value).strip()
    if text not in {"0", "1"}:
        raise ValueError(f"Expected binary label for {context}: {value!r}")
    return int(text)


def _resolve_path(value: Path | str, *, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _ensure_below(path: Path, root: Path, *, label: str) -> Path:
    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes configured data root: {path}") from exc
    return resolved_path


def _lexical_relative_to(path: Path, root: Path) -> Path:
    absolute_path = path.absolute()
    absolute_root = root.absolute()
    try:
        return absolute_path.relative_to(absolute_root)
    except ValueError:
        path_parts = absolute_path.parts
        root_parts = absolute_root.parts
        if len(path_parts) < len(root_parts) or tuple(
            part.casefold() for part in path_parts[: len(root_parts)]
        ) != tuple(part.casefold() for part in root_parts):
            raise
        return Path(*path_parts[len(root_parts) :])


def _resolve_train_source(
    source_path_text: str,
    *,
    canonical_source_root: Path,
    data_root: Path,
) -> Path:
    source_path = Path(source_path_text)
    try:
        _lexical_relative_to(source_path, data_root)
    except ValueError:
        source_is_materialized = False
    else:
        source_is_materialized = True
    if source_is_materialized:
        return _ensure_below(source_path, data_root, label="Train source path")

    try:
        relative_path = _lexical_relative_to(source_path, canonical_source_root)
    except ValueError as exc:
        raise ValueError("Train source path is outside the canonical source root") from exc
    if not relative_path.parts or any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ValueError("Train source path has an invalid canonical relative path")
    materialized_path = data_root / relative_path
    return _ensure_below(materialized_path, data_root, label="Materialized train source path")


def _selection_key(source_sha256: str) -> str:
    material = f"{SELECTION_SEED}:{source_sha256}".encode("ascii")
    return hashlib.sha256(material).hexdigest()


def _iter_split_rows(split_csv: Path) -> Iterable[dict[str, str]]:
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_SPLIT_COLUMNS - fieldnames)
        if missing:
            raise ValueError(f"Split CSV missing required columns: {missing}")
        yield from reader


def _load_manifest_labels(
    manifest_path: Path,
    *,
    required_source_sha256: set[str],
) -> tuple[dict[str, int], int, int, int, int]:
    payload = _read_json_object(manifest_path)
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Cache manifest samples must be a list")

    all_labels_by_sha: dict[str, set[int]] = {}
    required_labels_by_sha: dict[str, int] = {}
    duplicate_same_label_records = 0
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Cache manifest sample {index} is not an object")
        source_sha256 = str(sample.get("source_sha256") or "").strip().casefold()
        if not _is_sha256(source_sha256):
            raise ValueError(f"Cache manifest sample {index} has invalid source_sha256")
        if not str(sample.get("cache_path") or "").strip():
            raise ValueError(f"Cache manifest sample {index} has empty cache_path")
        label = _normalize_label(sample.get("label"), context=f"cache manifest sample {index}")
        labels = all_labels_by_sha.setdefault(source_sha256, set())
        if label in labels:
            duplicate_same_label_records += 1
        labels.add(label)
        if source_sha256 in required_source_sha256:
            existing = required_labels_by_sha.get(source_sha256)
            if existing is not None and existing != label:
                raise ValueError(
                    "Cache manifest has cross-label canonical-train source_sha256: "
                    f"{source_sha256}"
                )
            required_labels_by_sha[source_sha256] = label
    cross_label_source_sha256 = sum(len(labels) > 1 for labels in all_labels_by_sha.values())
    return (
        required_labels_by_sha,
        len(samples),
        len(all_labels_by_sha),
        duplicate_same_label_records,
        cross_label_source_sha256,
    )


def _candidate_from_train_row(
    row: dict[str, str],
    *,
    manifest_labels_by_sha: dict[str, int],
    canonical_source_root: Path,
    data_root: Path,
) -> tuple[Optional[ProbeCandidate], Optional[str]]:
    source_path_text = str(row.get("source_path") or "").strip()
    source_sha256 = str(row.get("source_sha256") or "").strip().casefold()
    if not source_path_text:
        raise ValueError("Train split row has empty source_path")
    if not _is_sha256(source_sha256):
        raise ValueError(f"Train split row has invalid source_sha256: {source_path_text}")
    label = _normalize_label(row.get("label"), context=f"train split row {source_path_text}")

    manifest_label = manifest_labels_by_sha.get(source_sha256)
    if manifest_label is None:
        return None, "cache_manifest_missing"
    if manifest_label != label:
        return None, "cache_manifest_label_mismatch"

    try:
        source_path = _resolve_train_source(
            source_path_text,
            canonical_source_root=canonical_source_root,
            data_root=data_root,
        )
        source_size_bytes = int(source_path.stat().st_size)
    except (OSError, ValueError):
        return None, "source_stat_failed"
    if source_size_bytes < MIN_SOURCE_SIZE_BYTES:
        return None, "source_size_below_min"
    if source_size_bytes > MAX_SOURCE_SIZE_BYTES:
        return None, "source_size_above_max"
    return (
        ProbeCandidate(
            source_path=str(source_path),
            source_sha256=source_sha256,
            label=label,
            source_size_bytes=source_size_bytes,
        ),
        None,
    )


def _canonical_bundle_line(candidate: ProbeCandidate) -> str:
    payload = {
        "schema": BUNDLE_RECORD_SCHEMA,
        "loop_id": LOOP_ID,
        "bundle_role": "local_train_only_runtime_probe",
        "split_role": CANONICAL_SPLIT_ROLE,
        "label": candidate.label,
        "source_path": candidate.source_path,
        "source_sha256": candidate.source_sha256,
        "source_size_bytes": candidate.source_size_bytes,
        "metadata_not_model_features": [
            "source_path",
            "source_sha256",
            "source_size_bytes",
        ],
        "source_path_usage": "loader_identity_only_not_model_feature",
        "source_sha256_usage": "integrity_binding_only_not_model_feature",
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _write_new_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()


def build_local_probe_bundle(
    *,
    split_csv: Path,
    cache_manifest: Path,
    canonical_source_root: Path = DEFAULT_CANONICAL_SOURCE_ROOT,
    data_root: Path,
    bundle_output: Path,
    summary_output: Path,
    records_per_class: int = RECORDS_PER_CLASS,
) -> dict[str, Any]:
    """Create a deterministic, train-only probe bundle without opening source bytes."""
    if records_per_class <= 0:
        raise ValueError("records_per_class must be positive")

    split_csv = _resolve_path(split_csv, root=PROJECT_ROOT)
    cache_manifest = _resolve_path(cache_manifest, root=PROJECT_ROOT)
    canonical_source_root = _resolve_path(canonical_source_root, root=PROJECT_ROOT)
    data_root = _resolve_path(data_root, root=PROJECT_ROOT)
    bundle_output = _resolve_path(bundle_output, root=PROJECT_ROOT)
    summary_output = _resolve_path(summary_output, root=PROJECT_ROOT)
    if bundle_output == summary_output:
        raise ValueError("bundle_output and summary_output must differ")
    if bundle_output.exists() or summary_output.exists():
        raise FileExistsError("Probe bundle outputs already exist; refusing to overwrite")
    if not split_csv.is_file() or not cache_manifest.is_file():
        raise FileNotFoundError("Canonical split CSV or cache manifest is missing")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Configured data root is not a directory: {data_root}")
    if not canonical_source_root.is_dir():
        raise NotADirectoryError(
            f"Configured canonical source root is not a directory: {canonical_source_root}"
        )

    split_counts: Counter[str] = Counter()
    candidate_counts: Counter[int] = Counter()
    rejection_counts: Counter[str] = Counter()
    candidates_by_label: dict[int, list[ProbeCandidate]] = {0: [], 1: []}
    seen_train_sha256: set[str] = set()
    train_rows: list[dict[str, str]] = []

    # 先冻结 canonical train SHA 集；Val/Test 只计数，不进入 manifest/path 查询。
    for row in _iter_split_rows(split_csv):
        split_role = str(row.get("split") or "").strip()
        split_counts[split_role] += 1
        if split_role != CANONICAL_SPLIT_ROLE:
            continue
        source_sha256 = str(row.get("source_sha256") or "").strip().casefold()
        if not _is_sha256(source_sha256):
            raise ValueError("Canonical train split contains invalid source_sha256")
        if source_sha256 in seen_train_sha256:
            raise ValueError(f"Canonical train split repeats source_sha256: {source_sha256}")
        seen_train_sha256.add(source_sha256)
        train_rows.append(row)

    (
        manifest_labels_by_sha,
        manifest_record_count,
        manifest_unique_source_sha256,
        manifest_duplicate_same_label_records,
        manifest_cross_label_source_sha256,
    ) = _load_manifest_labels(
        cache_manifest,
        required_source_sha256=seen_train_sha256,
    )

    for row in train_rows:
        candidate, rejection = _candidate_from_train_row(
            row,
            manifest_labels_by_sha=manifest_labels_by_sha,
            canonical_source_root=canonical_source_root,
            data_root=data_root,
        )
        if rejection is not None:
            rejection_counts[rejection] += 1
            continue
        if candidate is None:
            raise AssertionError("Candidate must be present when no rejection is returned")
        candidates_by_label[candidate.label].append(candidate)
        candidate_counts[candidate.label] += 1

    selected_by_label: dict[int, list[ProbeCandidate]] = {}
    for label in (0, 1):
        ordered = sorted(
            candidates_by_label[label],
            key=lambda candidate: (_selection_key(candidate.source_sha256), candidate.source_sha256),
        )
        selected = ordered[:records_per_class]
        if len(selected) != records_per_class:
            raise ValueError(
                "Insufficient eligible canonical-train records for local probe: "
                f"label={label}, required={records_per_class}, eligible={len(ordered)}"
            )
        selected_by_label[label] = selected

    selected_records = [
        candidate
        for label in (0, 1)
        for candidate in selected_by_label[label]
    ]
    selected_records.sort(
        key=lambda candidate: (_selection_key(candidate.source_sha256), candidate.source_sha256)
    )
    bundle_bytes = (
        "\n".join(_canonical_bundle_line(candidate) for candidate in selected_records) + "\n"
    ).encode("utf-8")
    bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
    selected_sizes = [candidate.source_size_bytes for candidate in selected_records]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "loop_id": LOOP_ID,
        "bundle_schema": BUNDLE_SCHEMA,
        "bundle_role": "local_train_only_runtime_probe",
        "bundle": {
            "path": str(bundle_output),
            "sha256": bundle_sha256,
            "record_count": len(selected_records),
            "record_schema": BUNDLE_RECORD_SCHEMA,
        },
        "input_bindings": {
            "canonical_split_csv": {
                "path": str(split_csv),
                "sha256": sha256_file(split_csv),
            },
            "primary_cache_manifest": {
                "path": str(cache_manifest),
                "sha256": sha256_file(cache_manifest),
            },
        },
        "selection": {
            "canonical_split_role": CANONICAL_SPLIT_ROLE,
            "records_per_class": records_per_class,
            "labels": [0, 1],
            "selection_seed": SELECTION_SEED,
            "selection_order": "sha256(seed + ':' + source_sha256) ascending",
            "source_path_mapping": "canonical_relative_path_into_materialized_data_root",
            "canonical_source_root": str(canonical_source_root),
            "materialized_data_root": str(data_root),
            "source_size_min_bytes": MIN_SOURCE_SIZE_BYTES,
            "source_size_max_bytes": MAX_SOURCE_SIZE_BYTES,
            "source_content_opened": False,
            "source_metadata_access": ["stat_size_only"],
            "identity_fields_not_model_features": [
                "source_path",
                "source_sha256",
                "source_size_bytes",
            ],
        },
        "aggregate_counts": {
            "split_rows_by_role": dict(sorted(split_counts.items())),
            "manifest_records": manifest_record_count,
            "manifest_unique_source_sha256": manifest_unique_source_sha256,
            "manifest_duplicate_same_label_records": manifest_duplicate_same_label_records,
            "manifest_cross_label_source_sha256": manifest_cross_label_source_sha256,
            "eligible_train_rows_by_label": {str(label): candidate_counts[label] for label in (0, 1)},
            "selected_rows_by_label": {str(label): len(selected_by_label[label]) for label in (0, 1)},
            "rejected_train_rows_by_reason": dict(sorted(rejection_counts.items())),
        },
        "aggregate_source_size_bytes": {
            "minimum": min(selected_sizes),
            "maximum": max(selected_sizes),
            "total": sum(selected_sizes),
        },
        "ready_for": {
            "local_runtime_probe_bundle": True,
            "loop164_whole_file_training": False,
            "val_or_test_access": False,
            "f1_claim": False,
        },
        "decision": "local_train_only_probe_bundle_ready",
    }
    summary_bytes = (json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_new_file(bundle_output, bundle_bytes)
    try:
        _write_new_file(summary_output, summary_bytes)
    except Exception:
        bundle_output.unlink(missing_ok=True)
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a SHA-bound, canonical-train-only Loop164 local runtime probe bundle."
    )
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument(
        "--canonical-source-root", type=Path, default=DEFAULT_CANONICAL_SOURCE_ROOT
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--bundle-output", type=Path, default=DEFAULT_BUNDLE_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_local_probe_bundle(
        split_csv=args.split_csv,
        cache_manifest=args.cache_manifest,
        canonical_source_root=args.canonical_source_root,
        data_root=args.data_root,
        bundle_output=args.bundle_output,
        summary_output=args.summary_output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
