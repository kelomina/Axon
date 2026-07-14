#!/usr/bin/env python3
"""Build train-only content-similarity folds for local Loop164 diagnostics.

This is not the production purged-forward partition. It groups canonical train
rows by a bounded whole-file chunk sketch, then creates deterministic balanced
folds without reading any Val/Test source content.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP_ID = "loop164_whole_file_residual_expert"
SCHEMA = "axon_loop164_local_train_diagnostic_folds_v1"
RECORD_SCHEMA = "axon_loop164_local_train_diagnostic_fold_record_v1"
CLAIM_SCOPE = "local_train_content_similarity_diagnostic_not_family_or_time_isolation"
DEFAULT_SPLIT = (
    PROJECT_ROOT / "reports" / "random_20w_split" / "loop127_full_duplicate_corrected_split.csv"
)
DEFAULT_CANONICAL_SOURCE_ROOT = PROJECT_ROOT / "data"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "random_20w_worktree"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "reports" / "roadmap_9997" / "loop164" / "local_train_diagnostic_folds.jsonl"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "reports"
    / "roadmap_9997"
    / "loop164"
    / "local_train_diagnostic_folds_summary.json"
)
REQUIRED_COLUMNS = {"source_path", "source_sha256", "label", "sample_index", "split"}
DATE_BUCKET_PATTERN = re.compile(r"(?:^|[\\/])(20\d{2}-\d{2})(?:[\\/]|$)")
DEFAULT_EXPECTED_TRAIN_ROWS = 20000
MAX_TRAIN_PREFIX_BYTES = 32 * 1024 * 1024
MAX_CSV_LINE_BYTES = 64 * 1024


@dataclass(frozen=True)
class TrainContentRecord:
    train_row_index: int
    source_path: Path
    source_sha256: str
    label: int
    sample_index: int
    source_size_bytes: Optional[int]
    signature: tuple[int, ...]
    chunk_hashes: tuple[int, ...]
    availability: str
    missing_reason: Optional[str]
    path_date_bucket: Optional[str]


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _resolve_path(value: Path, *, root: Path = PROJECT_ROOT) -> Path:
    return value if value.is_absolute() else root / value


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


def _materialized_path(source_path: str, *, canonical_root: Path, data_root: Path) -> Path:
    path = Path(source_path)
    try:
        relative = _lexical_relative_to(path, data_root)
    except ValueError:
        relative = _lexical_relative_to(path, canonical_root)
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("Canonical train source path has invalid relative components")
    return data_root / relative


def _resolve_regular_source(path: Path, *, data_root: Path) -> Path:
    absolute_root = data_root.absolute()
    absolute_path = path.absolute()
    relative = _lexical_relative_to(absolute_path, absolute_root)
    cursor = absolute_root
    if cursor.is_symlink():
        raise ValueError("Materialized data root cannot be a symbolic link")
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError("Train source path cannot contain symbolic links")
    resolved_root = absolute_root.resolve(strict=True)
    resolved_path = absolute_path.resolve(strict=True)
    resolved_path.relative_to(resolved_root)
    if not resolved_path.is_file():
        raise OSError("Train source path is not a regular file")
    return resolved_path


def _chunk_hash(chunk: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(chunk, digest_size=8).digest(), "big")


def _inspect_source(
    path: Path,
    *,
    expected_sha256: str,
    chunk_size: int,
    signature_size: int,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    before = path.stat()
    digest = hashlib.sha256()
    unique_chunk_hashes: set[int] = set()
    bytes_read = 0
    with path.open("rb", buffering=0) as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
            bytes_read += len(chunk)
            unique_chunk_hashes.add(_chunk_hash(chunk))
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or bytes_read != int(before.st_size)
    ):
        raise RuntimeError("Train source changed during content-sketch scan")
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError("Train source SHA does not match the canonical split")
    chunk_hashes = tuple(sorted(unique_chunk_hashes))
    signature = chunk_hashes[:signature_size]
    return bytes_read, signature, chunk_hashes


def _iter_train_rows(
    split_csv: Path,
    *,
    expected_train_rows: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if expected_train_rows < 1:
        raise ValueError("expected_train_rows must be positive")
    prefix_lines: list[bytes] = []
    prefix_bytes = 0
    # 使用 unbuffered readline，只消费 header + 固定 Train 行数，不触碰下一条 heldout 行。
    with split_csv.open("rb", buffering=0) as handle:
        for line_index in range(expected_train_rows + 1):
            line = handle.readline(MAX_CSV_LINE_BYTES + 1)
            if not line:
                raise ValueError("Canonical split ended before the frozen train prefix")
            if len(line) > MAX_CSV_LINE_BYTES:
                raise ValueError("Canonical split line exceeds the bounded line size")
            prefix_bytes += len(line)
            if prefix_bytes > MAX_TRAIN_PREFIX_BYTES:
                raise ValueError("Canonical train prefix exceeds its bounded size")
            prefix_lines.append(line)
    prefix_raw = b"".join(prefix_lines)
    try:
        prefix_text = prefix_raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Canonical train prefix is not valid UTF-8") from exc
    reader = csv.DictReader(io.StringIO(prefix_text, newline=""))
    missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Canonical split is missing columns: {sorted(missing)}")
    train_rows: list[dict[str, str]] = []
    for row in reader:
        train_rows.append(row)
    if len(train_rows) != expected_train_rows:
        raise ValueError("Canonical train prefix row count drifted")
    for train_row_index, row in enumerate(train_rows):
        if str(row.get("split") or "").strip() != "train":
            raise ValueError("Canonical train prefix contains a non-train role")
        if int(str(row.get("sample_index") or "")) != train_row_index:
            raise ValueError("Canonical train prefix sample_index is not contiguous")
    return train_rows, {
        "sha256": hashlib.sha256(prefix_raw).hexdigest(),
        "bytes": len(prefix_raw),
        "line_count": len(prefix_lines),
        "train_rows": len(train_rows),
        "heldout_rows_read": 0,
        "stopped_before_next_line": True,
    }


def scan_train_content(
    *,
    split_csv: Path,
    canonical_source_root: Path,
    data_root: Path,
    chunk_size: int,
    signature_size: int,
    max_supported_file_bytes: int,
    expected_train_rows: int,
) -> tuple[list[TrainContentRecord], dict[str, int], dict[str, Any]]:
    train_rows, prefix_binding = _iter_train_rows(
        split_csv,
        expected_train_rows=expected_train_rows,
    )
    records: list[TrainContentRecord] = []
    seen_sha256: set[str] = set()
    total_scanned_bytes = 0
    verified_sources = 0
    for train_row_index, row in enumerate(train_rows):
        source_sha256 = str(row.get("source_sha256") or "").strip().casefold()
        if not _is_sha256(source_sha256):
            raise ValueError("Canonical train row has invalid source_sha256")
        if source_sha256 in seen_sha256:
            raise ValueError("Canonical train source_sha256 is not unique")
        seen_sha256.add(source_sha256)
        label_text = str(row.get("label") or "").strip()
        if label_text not in {"0", "1"}:
            raise ValueError("Canonical train label is invalid")
        sample_index = int(str(row.get("sample_index") or ""))
        canonical_path_text = str(row.get("source_path") or "")
        source_path = _materialized_path(
            canonical_path_text,
            canonical_root=canonical_source_root,
            data_root=data_root,
        )
        date_match = DATE_BUCKET_PATTERN.search(canonical_path_text)
        path_date_bucket = date_match.group(1) if date_match else None
        try:
            resolved_source = _resolve_regular_source(source_path, data_root=data_root)
            source_size_bytes, signature, chunk_hashes = _inspect_source(
                resolved_source,
                expected_sha256=source_sha256,
                chunk_size=chunk_size,
                signature_size=signature_size,
            )
            verified_sources += 1
            total_scanned_bytes += source_size_bytes
            if source_size_bytes < 1:
                availability = "parse_failure"
                missing_reason = "parse_failure"
            elif source_size_bytes > max_supported_file_bytes:
                availability = "oversize"
                missing_reason = "oversize"
            else:
                availability = "supported"
                missing_reason = None
        except (FileNotFoundError, OSError):
            source_size_bytes = None
            signature = ()
            chunk_hashes = ()
            availability = "read_failure"
            missing_reason = "read_failure"
        records.append(
            TrainContentRecord(
                train_row_index=train_row_index,
                source_path=source_path,
                source_sha256=source_sha256,
                label=int(label_text),
                sample_index=sample_index,
                source_size_bytes=source_size_bytes,
                signature=signature,
                chunk_hashes=chunk_hashes,
                availability=availability,
                missing_reason=missing_reason,
                path_date_bucket=path_date_bucket,
            )
        )
    return (
        records,
        {"verified_sources": verified_sources, "total_scanned_bytes": total_scanned_bytes},
        prefix_binding,
    )


def _band_keys(signature: tuple[int, ...], band_size: int) -> Iterable[tuple[int, tuple[int, ...]]]:
    if not signature:
        return
    if len(signature) <= band_size:
        yield 0, signature
        return
    usable = len(signature) - len(signature) % band_size
    for start in range(0, usable, band_size):
        yield start, signature[start : start + band_size]


def signature_coverage(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or not right:
        return 0.0
    return len(set(left) & set(right)) / min(len(left), len(right))


def exact_chunk_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    intersection = len(left_set & right_set)
    jaccard = intersection / len(left_set | right_set)
    containment = intersection / min(len(left_set), len(right_set))
    return max(jaccard, containment)


def build_similarity_components(
    records: Sequence[TrainContentRecord],
    *,
    band_size: int,
    max_bucket_size: int,
    similarity_threshold: float,
) -> tuple[list[list[int]], dict[str, int]]:
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for record_index, record in enumerate(records):
        for key in _band_keys(record.signature, band_size):
            buckets[key].append(record_index)
    candidate_pairs: set[tuple[int, int]] = set()
    skipped_buckets = 0
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        if len(indices) > max_bucket_size:
            skipped_buckets += 1
            continue
        candidate_pairs.update(combinations(indices, 2))

    union_find = UnionFind(len(records))
    accepted_pairs = 0
    for left_index, right_index in sorted(candidate_pairs):
        if exact_chunk_similarity(
            records[left_index].chunk_hashes, records[right_index].chunk_hashes
        ) >= similarity_threshold:
            union_find.union(left_index, right_index)
            accepted_pairs += 1
    by_root: dict[int, list[int]] = defaultdict(list)
    for record_index in range(len(records)):
        by_root[union_find.find(record_index)].append(record_index)
    components = list(by_root.values())
    return components, {
        "lsh_bucket_count": len(buckets),
        "lsh_candidate_pairs": len(candidate_pairs),
        "accepted_similarity_pairs": accepted_pairs,
        "skipped_oversized_lsh_buckets": skipped_buckets,
    }


def assign_components_to_folds(
    records: Sequence[TrainContentRecord],
    components: Sequence[Sequence[int]],
    *,
    fold_count: int,
    seed: int,
) -> tuple[list[int], list[str], dict[str, Any]]:
    label_totals = Counter(record.label for record in records)
    target_by_label = {
        label: label_totals[label] / fold_count
        for label in (0, 1)
    }
    fold_label_counts = [Counter() for _ in range(fold_count)]
    fold_total_counts = [0] * fold_count
    row_folds = [-1] * len(records)
    row_component_ids = [""] * len(records)

    def component_digest(indices: Sequence[int]) -> str:
        material = "|".join(sorted(records[index].source_sha256 for index in indices))
        return hashlib.sha256(f"{seed}:{material}".encode("ascii")).hexdigest()

    ordered_components = sorted(
        components,
        key=lambda indices: (-len(indices), component_digest(indices)),
    )
    for indices in ordered_components:
        group_labels = Counter(records[index].label for index in indices)
        component_id = component_digest(indices)[:24]

        def fold_score(fold_index: int) -> tuple[float, int, int]:
            imbalance = 0.0
            for candidate_fold in range(fold_count):
                for label in (0, 1):
                    prospective = fold_label_counts[candidate_fold][label]
                    if candidate_fold == fold_index:
                        prospective += group_labels[label]
                    imbalance += (
                        (prospective - target_by_label[label]) / target_by_label[label]
                    ) ** 2
            return imbalance, fold_total_counts[fold_index], fold_index

        selected_fold = min(range(fold_count), key=fold_score)
        for record_index in indices:
            row_folds[record_index] = selected_fold
            row_component_ids[record_index] = component_id
        fold_total_counts[selected_fold] += len(indices)
        fold_label_counts[selected_fold].update(group_labels)

    if any(fold < 0 for fold in row_folds) or any(not value for value in row_component_ids):
        raise AssertionError("Every canonical train row must receive one component and fold")
    component_folds: dict[str, set[int]] = defaultdict(set)
    for component_id, fold in zip(row_component_ids, row_folds):
        component_folds[component_id].add(fold)
    if any(len(folds) != 1 for folds in component_folds.values()):
        raise AssertionError("A content component crossed diagnostic folds")
    if any(
        fold_label_counts[index][label] < 1
        for index in range(fold_count)
        for label in (0, 1)
    ):
        raise AssertionError("Every diagnostic fold must contain both labels")
    summary = {
        "fold_total_counts": {str(index): fold_total_counts[index] for index in range(fold_count)},
        "fold_label_counts": {
            str(index): {str(label): fold_label_counts[index][label] for label in (0, 1)}
            for index in range(fold_count)
        },
        "component_cross_fold_count": 0,
    }
    return row_folds, row_component_ids, summary


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def build_local_train_diagnostic_folds(
    *,
    split_csv: Path,
    canonical_source_root: Path,
    data_root: Path,
    output_jsonl: Path,
    summary_json: Path,
    fold_count: int = 5,
    seed: int = 164,
    chunk_size: int = 4096,
    signature_size: int = 128,
    band_size: int = 8,
    max_bucket_size: int = 256,
    similarity_threshold: float = 0.85,
    max_supported_file_bytes: int = 8 * 1024 * 1024,
    expected_train_rows: int = DEFAULT_EXPECTED_TRAIN_ROWS,
) -> dict[str, Any]:
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    if chunk_size < 1 or signature_size < 1 or band_size < 1 or max_bucket_size < 2:
        raise ValueError("Content sketch bounds must be positive")
    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in (0, 1]")
    split_csv = _resolve_path(split_csv).resolve(strict=True)
    canonical_source_root = _resolve_path(canonical_source_root).resolve(strict=True)
    data_root = _resolve_path(data_root).resolve(strict=True)
    output_jsonl = _resolve_path(output_jsonl)
    summary_json = _resolve_path(summary_json)
    if output_jsonl == summary_json or output_jsonl.exists() or summary_json.exists():
        raise FileExistsError("Diagnostic fold outputs must be distinct and new")

    records, scan_counts, prefix_binding = scan_train_content(
        split_csv=split_csv,
        canonical_source_root=canonical_source_root,
        data_root=data_root,
        chunk_size=chunk_size,
        signature_size=signature_size,
        max_supported_file_bytes=max_supported_file_bytes,
        expected_train_rows=expected_train_rows,
    )
    components, similarity_counts = build_similarity_components(
        records,
        band_size=band_size,
        max_bucket_size=max_bucket_size,
        similarity_threshold=similarity_threshold,
    )
    row_folds, row_component_ids, fold_summary = assign_components_to_folds(
        records,
        components,
        fold_count=fold_count,
        seed=seed,
    )
    component_sizes = Counter(row_component_ids)
    availability_counts = Counter(record.availability for record in records)
    path_date_counts = Counter(
        record.label for record in records if record.path_date_bucket is not None
    )
    lines = []
    for record, fold, component_id in zip(records, row_folds, row_component_ids):
        payload = {
            "schema": RECORD_SCHEMA,
            "loop_id": LOOP_ID,
            "claim_scope": CLAIM_SCOPE,
            "split_role": "train",
            "train_row_index": record.train_row_index,
            "sample_index": record.sample_index,
            "source_path": str(record.source_path),
            "source_sha256": record.source_sha256,
            "source_size_bytes": record.source_size_bytes,
            "label": record.label,
            "availability": record.availability,
            "missing_reason": record.missing_reason,
            "content_component_id": component_id,
            "content_component_size": component_sizes[component_id],
            "diagnostic_fold": fold,
            "identity_metadata_not_model_features": [
                "train_row_index",
                "sample_index",
                "source_path",
                "source_sha256",
                "content_component_id",
                "diagnostic_fold",
            ],
        }
        lines.append(json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    output_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    component_label_conflicts = 0
    members_by_component: dict[str, list[TrainContentRecord]] = defaultdict(list)
    for record, component_id in zip(records, row_component_ids):
        members_by_component[component_id].append(record)
    for members in members_by_component.values():
        if len({member.label for member in members}) > 1:
            component_label_conflicts += 1
    summary = {
        "schema": SCHEMA,
        "loop_id": LOOP_ID,
        "claim_scope": CLAIM_SCOPE,
        "inputs": {
            "canonical_split_train_prefix": {
                "path": str(split_csv),
                **prefix_binding,
            },
            "canonical_source_root": str(canonical_source_root),
            "materialized_data_root": str(data_root),
        },
        "parameters": {
            "fold_count": fold_count,
            "seed": seed,
            "chunk_size": chunk_size,
            "signature_size": signature_size,
            "lsh_band_size": band_size,
            "max_lsh_bucket_size": max_bucket_size,
            "signature_coverage_threshold": similarity_threshold,
            "max_supported_file_bytes": max_supported_file_bytes,
        },
        "aggregate": {
            "split_rows_by_role_read": {"train": len(records)},
            "canonical_train_rows": len(records),
            "label_counts": {
                str(label): sum(record.label == label for record in records) for label in (0, 1)
            },
            "availability_counts": dict(sorted(availability_counts.items())),
            "verified_source_sha256": scan_counts["verified_sources"],
            "raw_bytes_scanned": scan_counts["total_scanned_bytes"],
            "content_components": len(members_by_component),
            "non_singleton_components": sum(
                len(members) > 1 for members in members_by_component.values()
            ),
            "largest_component_size": max(component_sizes.values(), default=0),
            "cross_label_components": component_label_conflicts,
            **similarity_counts,
        },
        "folds": fold_summary,
        "time_stress_metadata": {
            "path_date_is_not_first_seen_time": True,
            "rows_with_path_date_by_label": {
                str(label): path_date_counts[label] for label in (0, 1)
            },
            "used_for_fold_assignment": False,
        },
        "output": {
            "path": str(output_jsonl),
            "sha256": output_sha256,
            "record_count": len(records),
            "record_schema": RECORD_SCHEMA,
        },
        "limitations": [
            "no_authoritative_first_seen_time",
            "no_family_or_campaign_group",
            "no_custodian_source_group",
            "bounded_chunk_sketch_is_not_a_complete_near_duplicate_oracle",
            "not_purged_forward_oof",
        ],
        "ready_for": {
            "local_whole_file_randomized_oof_diagnostic": True,
            "loop164_production_oof": False,
            "a2_training_authority": False,
            "val_or_test_access": False,
            "candidate_promotion": False,
        },
        "decision": "local_content_group_diagnostic_folds_ready_not_production_scope",
    }
    summary_bytes = (json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _write_exclusive(output_jsonl, output_bytes)
    try:
        _write_exclusive(summary_json, summary_bytes)
    except BaseException:
        output_jsonl.unlink(missing_ok=True)
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build canonical-train-only content-similarity folds for local Loop164 diagnostics."
    )
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--canonical-source-root", type=Path, default=DEFAULT_CANONICAL_SOURCE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=164)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--signature-size", type=int, default=128)
    parser.add_argument("--band-size", type=int, default=8)
    parser.add_argument("--max-bucket-size", type=int, default=256)
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument(
        "--expected-train-rows", type=int, default=DEFAULT_EXPECTED_TRAIN_ROWS
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_local_train_diagnostic_folds(
        split_csv=args.split_csv,
        canonical_source_root=args.canonical_source_root,
        data_root=args.data_root,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        fold_count=args.fold_count,
        seed=args.seed,
        chunk_size=args.chunk_size,
        signature_size=args.signature_size,
        band_size=args.band_size,
        max_bucket_size=args.max_bucket_size,
        similarity_threshold=args.similarity_threshold,
        expected_train_rows=args.expected_train_rows,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
