#!/usr/bin/env python3
"""Analyze original PE files for duplicate and near-duplicate risks.

这个脚本直接扫描 data 下的原始文件，不读取 data/.cache、manifest 或 NPZ 缓存。
它只输出报告，不删除样本、不移动文件，也不触发特征提取或训练。
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
for path in [SCRIPT_DIR, SRC_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_similarity import (  # noqa: E402
    UnionFind,
    assign_splits,
    build_experiment_config,
    read_toml_config,
    resolve_path,
)


@dataclass
class RawSampleRecord:
    index: int
    source_path: str
    label: int
    file_size: int
    sha256: str
    chunk_signature: Tuple[int, ...]
    skipped_reason: str = ""
    split: str = "unknown"


@dataclass
class RawAnalysisOptions:
    config_path: Optional[Path] = None
    data_dir: Optional[Path] = None
    output_dir: Path = Path("reports/raw_similarity")
    similarity_threshold: float = 0.85
    max_samples_per_class: Optional[int] = None
    chunk_size: int = 4096
    minhash_size: int = 128
    lsh_band_size: int = 8
    max_bucket_size: int = 2048
    include_non_pe: bool = False
    max_file_size: Optional[int] = None
    seed: Optional[int] = None


def iter_sorted_files(root: Path) -> Iterable[Path]:
    skip_dirs = {".cache", "__pycache__", ".git", ".pytest_cache", "reports", "models", "swanlog"}
    for dirpath, dirnames, filenames in __import__("os").walk(root, followlinks=False):
        # 原始数据分析永远跳过缓存/输出目录和符号链接，避免越界扫描。
        filtered_dirs = []
        for name in sorted(dirnames):
            child = Path(dirpath) / name
            if name in skip_dirs or child.is_symlink():
                continue
            filtered_dirs.append(name)
        dirnames[:] = filtered_dirs
        for filename in sorted(filenames):
            file_path = Path(dirpath) / filename
            if file_path.is_symlink():
                continue
            yield file_path


def file_starts_with_mz(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(2) == b"MZ"
    except OSError:
        return False


def chunk_hash_to_int(chunk: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(chunk, digest_size=8).digest(), "big", signed=False)


def inspect_raw_file(path: Path, chunk_size: int, minhash_size: int) -> Tuple[int, str, Tuple[int, ...]]:
    """一次流式读取同时生成完整文件 SHA256 和 chunk bottom-k 签名。"""
    sha = hashlib.sha256()
    minhash_values = set()
    total_size = 0

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            total_size += len(chunk)
            sha.update(chunk)
            minhash_values.add(chunk_hash_to_int(chunk))

    signature = tuple(sorted(minhash_values)[:minhash_size])
    return total_size, sha.hexdigest(), signature


def _configured_roots(data_dir: Path, config) -> List[Tuple[Path, int]]:
    roots = []
    for dirname in getattr(config, "benign_dir_names_fs", ["benign", "待加入白名单"]):
        root = data_dir / dirname
        if root.exists():
            roots.append((root, 0))
    for dirname in getattr(config, "malicious_dir_names_fs", ["malicious", "待拉黑"]):
        root = data_dir / dirname
        if root.exists():
            roots.append((root, 1))
    return roots


def scan_raw_samples(options: RawAnalysisOptions, config) -> Tuple[List[RawSampleRecord], List[dict]]:
    data_dir = resolve_path(options.data_dir or Path(config.data_dir or "data"))
    roots = _configured_roots(data_dir, config)
    if not roots:
        raise FileNotFoundError(f"No configured raw dataset roots found under {data_dir}")

    max_file_size = options.max_file_size if options.max_file_size is not None else config.max_file_size
    counts = Counter()
    records: List[RawSampleRecord] = []
    skipped = []

    for root, label in roots:
        for path in iter_sorted_files(root):
            if options.max_samples_per_class is not None and counts[label] >= options.max_samples_per_class:
                break
            try:
                stat = path.stat()
            except OSError as exc:
                skipped.append({"path": str(path), "label": label, "reason": f"stat_failed:{exc}"})
                continue
            if stat.st_size <= 0:
                skipped.append({"path": str(path), "label": label, "reason": "empty_file"})
                continue
            if stat.st_size > max_file_size:
                skipped.append({"path": str(path), "label": label, "reason": "too_large"})
                continue
            if not options.include_non_pe and not file_starts_with_mz(path):
                skipped.append({"path": str(path), "label": label, "reason": "not_pe_mz"})
                continue

            try:
                file_size, sha256, signature = inspect_raw_file(
                    path,
                    chunk_size=options.chunk_size,
                    minhash_size=options.minhash_size,
                )
            except OSError as exc:
                skipped.append({"path": str(path), "label": label, "reason": f"read_failed:{exc}"})
                continue

            records.append(
                RawSampleRecord(
                    index=len(records),
                    source_path=str(path),
                    label=label,
                    file_size=file_size,
                    sha256=sha256,
                    chunk_signature=signature,
                )
            )
            counts[label] += 1

    return records, skipped


def exact_duplicate_pairs(records: Sequence[RawSampleRecord]) -> Iterable[Tuple[int, int]]:
    by_sha = defaultdict(list)
    for record in records:
        by_sha[(record.file_size, record.sha256)].append(record.index)
    for indices in by_sha.values():
        if len(indices) > 1:
            yield from combinations(indices, 2)


def _band_keys(signature: Tuple[int, ...], band_size: int) -> Iterable[Tuple[int, Tuple[int, ...]]]:
    if not signature:
        return
    if len(signature) <= band_size:
        yield 0, signature
        return
    usable = len(signature) - (len(signature) % band_size)
    for start in range(0, usable, band_size):
        yield start, signature[start:start + band_size]


def lsh_candidate_pairs(
    records: Sequence[RawSampleRecord],
    band_size: int,
    max_bucket_size: int,
) -> Tuple[set, int, int]:
    buckets = defaultdict(list)
    for record in records:
        for band_start, band_key in _band_keys(record.chunk_signature, band_size):
            buckets[(band_start, band_key)].append(record.index)

    candidates = set()
    skipped_buckets = 0
    oversized_samples = 0
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        if len(indices) > max_bucket_size:
            skipped_buckets += 1
            oversized_samples += len(indices)
            continue
        candidates.update(tuple(sorted(pair)) for pair in combinations(indices, 2))
    return candidates, skipped_buckets, oversized_samples


def signature_similarity(left: Tuple[int, ...], right: Tuple[int, ...]) -> float:
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    # 这里使用“覆盖率”而不是纯 Jaccard：如果一个小文件几乎被大文件包含，也会被视为高风险近亲。
    return len(left_set & right_set) / max(1, min(len(left_set), len(right_set)))


def _pair_flags(left: RawSampleRecord, right: RawSampleRecord) -> Dict[str, bool]:
    same_split = left.split == right.split
    leakage_pair = {left.split, right.split} in [
        {"train", "val"},
        {"train", "test"},
    ]
    return {
        "same_label": left.label == right.label,
        "same_split": same_split,
        "cross_split": not same_split,
        "leakage_pair": leakage_pair,
        "label_conflict": left.label != right.label,
    }


def build_pair_rows(
    records: Sequence[RawSampleRecord],
    duplicate_pairs: Iterable[Tuple[int, int]],
    candidate_pairs: Iterable[Tuple[int, int]],
    similarity_threshold: float,
) -> List[dict]:
    pair_methods = defaultdict(set)
    for left, right in duplicate_pairs:
        pair_methods[tuple(sorted((left, right)))].add("raw_duplicate")

    for left, right in candidate_pairs:
        similarity = signature_similarity(records[left].chunk_signature, records[right].chunk_signature)
        if similarity >= similarity_threshold:
            pair_methods[tuple(sorted((left, right)))].add("chunk_similar")

    rows = []
    for (left_idx, right_idx), methods in sorted(pair_methods.items()):
        left = records[left_idx]
        right = records[right_idx]
        similarity = signature_similarity(left.chunk_signature, right.chunk_signature)
        flags = _pair_flags(left, right)
        rows.append({
            "sample_i": left.index,
            "sample_j": right.index,
            "label_i": left.label,
            "label_j": right.label,
            "split_i": left.split,
            "split_j": right.split,
            "file_size_i": left.file_size,
            "file_size_j": right.file_size,
            "chunk_similarity": similarity,
            "methods": "+".join(sorted(methods)),
            "same_label": flags["same_label"],
            "same_split": flags["same_split"],
            "cross_split": flags["cross_split"],
            "leakage_pair": flags["leakage_pair"],
            "label_conflict": flags["label_conflict"],
            "sha256_i": left.sha256,
            "sha256_j": right.sha256,
            "source_path_i": left.source_path,
            "source_path_j": right.source_path,
        })
    return rows


def build_group_rows(records: Sequence[RawSampleRecord], pair_rows: Sequence[dict]) -> List[dict]:
    uf = UnionFind(len(records))
    for row in pair_rows:
        uf.union(int(row["sample_i"]), int(row["sample_j"]))

    grouped = defaultdict(list)
    for idx in range(len(records)):
        grouped[uf.find(idx)].append(idx)

    rows = []
    group_id = 1
    for indices in sorted(grouped.values(), key=lambda values: (-len(values), values[0])):
        if len(indices) < 2:
            continue
        group_records = [records[idx] for idx in indices]
        labels = Counter(record.label for record in group_records)
        splits = Counter(record.split for record in group_records)
        rows.append({
            "group_id": group_id,
            "size": len(indices),
            "labels": "|".join(f"{label}:{count}" for label, count in sorted(labels.items())),
            "splits": "|".join(f"{split}:{count}" for split, count in sorted(splits.items())),
            "has_label_conflict": len(labels) > 1,
            "has_cross_split": len(splits) > 1,
            "has_leakage": bool({"train"} & set(splits) and ({"val", "test"} & set(splits))),
            "sample_indices": "|".join(str(idx) for idx in indices),
            "source_paths": "|".join(record.source_path for record in group_records),
        })
        group_id += 1
    return rows


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_pairs_csv(path: Path, rows: Sequence[dict]) -> None:
    write_csv(
        path,
        rows,
        [
            "methods",
            "sample_i",
            "sample_j",
            "label_i",
            "label_j",
            "split_i",
            "split_j",
            "file_size_i",
            "file_size_j",
            "chunk_similarity",
            "same_label",
            "same_split",
            "cross_split",
            "leakage_pair",
            "label_conflict",
            "sha256_i",
            "sha256_j",
            "source_path_i",
            "source_path_j",
        ],
    )


def write_groups_csv(path: Path, rows: Sequence[dict]) -> None:
    write_csv(
        path,
        rows,
        [
            "group_id",
            "size",
            "labels",
            "splits",
            "has_label_conflict",
            "has_cross_split",
            "has_leakage",
            "sample_indices",
            "source_paths",
        ],
    )


def build_summary(
    options: RawAnalysisOptions,
    config,
    data_dir: Path,
    records: Sequence[RawSampleRecord],
    skipped: Sequence[dict],
    pair_rows: Sequence[dict],
    group_rows: Sequence[dict],
    candidate_count: int,
    skipped_lsh_buckets: int,
    oversized_lsh_samples: int,
    output_paths: Dict[str, Path],
) -> dict:
    label_counts = Counter(record.label for record in records)
    split_counts = Counter(record.split for record in records)
    method_counts = Counter()
    for row in pair_rows:
        for method in str(row["methods"]).split("+"):
            method_counts[method] += 1

    return {
        "mode": "raw_files",
        "data_dir": str(data_dir),
        "analyzed_samples": len(records),
        "skipped_samples": len(skipped),
        "max_samples_per_class": options.max_samples_per_class,
        "include_non_pe": options.include_non_pe,
        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "split_counts": {str(k): int(v) for k, v in sorted(split_counts.items())},
        "similarity_threshold": options.similarity_threshold,
        "chunk_size": options.chunk_size,
        "minhash_size": options.minhash_size,
        "lsh_band_size": options.lsh_band_size,
        "max_bucket_size": options.max_bucket_size,
        "lsh_candidate_pairs": int(candidate_count),
        "skipped_lsh_buckets": int(skipped_lsh_buckets),
        "oversized_lsh_samples": int(oversized_lsh_samples),
        "pair_counts": {
            "total": len(pair_rows),
            "raw_duplicate": int(method_counts.get("raw_duplicate", 0)),
            "chunk_similar": int(method_counts.get("chunk_similar", 0)),
            "leakage_pairs": sum(1 for row in pair_rows if row["leakage_pair"]),
            "label_conflicts": sum(1 for row in pair_rows if row["label_conflict"]),
        },
        "group_count": len(group_rows),
        "largest_group_size": max((int(row["size"]) for row in group_rows), default=0),
        "config": {
            "experiment_name": config.experiment_name,
            "seed": config.seed,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
        },
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }


def analyze_raw_similarity(options: RawAnalysisOptions) -> dict:
    raw_config = read_toml_config(resolve_path(options.config_path) if options.config_path else None)
    config = build_experiment_config(raw_config)
    if options.seed is not None:
        config.seed = options.seed

    data_dir = resolve_path(options.data_dir or Path(config.data_dir or "data"))
    records, skipped = scan_raw_samples(options, config)
    assign_splits(records, config)

    duplicate_pairs = list(exact_duplicate_pairs(records))
    candidate_pairs, skipped_lsh_buckets, oversized_lsh_samples = lsh_candidate_pairs(
        records,
        band_size=options.lsh_band_size,
        max_bucket_size=options.max_bucket_size,
    )
    candidate_pairs.update(tuple(sorted(pair)) for pair in duplicate_pairs)
    pair_rows = build_pair_rows(records, duplicate_pairs, candidate_pairs, options.similarity_threshold)
    group_rows = build_group_rows(records, pair_rows)

    output_dir = resolve_path(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "summary": output_dir / "raw_similarity_summary.json",
        "pairs": output_dir / "raw_similarity_pairs.csv",
        "groups": output_dir / "raw_similarity_groups.csv",
        "skipped": output_dir / "raw_similarity_skipped.csv",
    }

    summary = build_summary(
        options,
        config,
        data_dir,
        records,
        skipped,
        pair_rows,
        group_rows,
        len(candidate_pairs),
        skipped_lsh_buckets,
        oversized_lsh_samples,
        output_paths,
    )
    with output_paths["summary"].open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_pairs_csv(output_paths["pairs"], pair_rows)
    write_groups_csv(output_paths["groups"], group_rows)
    write_csv(output_paths["skipped"], skipped, ["path", "label", "reason"])
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> RawAnalysisOptions:
    parser = argparse.ArgumentParser(
        description="Analyze original Axon PE files for duplicate and near-duplicate risks."
    )
    parser.add_argument("--config", type=Path, default=Path("config/default_config.toml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/raw_similarity"))
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--minhash-size", type=int, default=128)
    parser.add_argument("--lsh-band-size", type=int, default=8)
    parser.add_argument("--max-bucket-size", type=int, default=2048)
    parser.add_argument("--include-non-pe", action="store_true", default=False)
    parser.add_argument("--max-file-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    return RawAnalysisOptions(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        similarity_threshold=args.similarity_threshold,
        max_samples_per_class=args.max_samples_per_class,
        chunk_size=args.chunk_size,
        minhash_size=args.minhash_size,
        lsh_band_size=args.lsh_band_size,
        max_bucket_size=args.max_bucket_size,
        include_non_pe=args.include_non_pe,
        max_file_size=args.max_file_size,
        seed=args.seed,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = parse_args(argv)
    summary = analyze_raw_similarity(options)
    print("=" * 60)
    print("Axon Raw File Similarity Analysis")
    print("=" * 60)
    print(f"Data dir: {summary['data_dir']}")
    print(f"Analyzed samples: {summary['analyzed_samples']}")
    print(f"Skipped samples: {summary['skipped_samples']}")
    print(f"Similarity pairs: {summary['pair_counts']['total']}")
    print(f"Raw duplicate pairs: {summary['pair_counts']['raw_duplicate']}")
    print(f"Chunk similar pairs: {summary['pair_counts']['chunk_similar']}")
    print(f"Leakage pairs: {summary['pair_counts']['leakage_pairs']}")
    print(f"Label conflicts: {summary['pair_counts']['label_conflicts']}")
    print(f"Groups: {summary['group_count']}")
    print(f"Summary: {summary['outputs']['summary']}")
    print(f"Pairs CSV: {summary['outputs']['pairs']}")
    print(f"Groups CSV: {summary['outputs']['groups']}")
    print(f"Skipped CSV: {summary['outputs']['skipped']}")
    if summary["skipped_lsh_buckets"]:
        print(
            "[Warning] Some LSH buckets were too large and skipped. "
            "Increase --max-bucket-size for a slower, broader scan."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
