#!/usr/bin/env python3
"""Analyze cached Axon samples for duplicate and near-duplicate risks.

这个脚本只读取 data/.cache 里的 manifest 和 NPZ 缓存，输出报告，不删除样本、
不修改数据划分，也不触发模型训练。
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import sys
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AxonExperimentConfig  # noqa: E402
from dataset import _load_cached_feature_npz, _resolve_manifest_cache_path  # noqa: E402


@dataclass
class SampleRecord:
    index: int
    source_path: str
    cache_path: Path
    label: int
    split: str = "unknown"


@dataclass
class AnalysisOptions:
    config_path: Optional[Path] = None
    data_dir: Optional[Path] = None
    manifest_path: Optional[Path] = None
    output_dir: Path = Path("reports/similarity")
    similarity_threshold: float = 0.985
    max_samples_per_class: Optional[int] = None
    simhash_bits: int = 64
    lsh_band_size: int = 8
    max_bucket_size: int = 512
    seed: Optional[int] = None


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

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


def read_toml_config(config_path: Optional[Path]) -> dict:
    if config_path is None:
        return {}
    with config_path.open("rb") as f:
        return tomllib.load(f)


def build_experiment_config(raw_config: dict) -> AxonExperimentConfig:
    """按 scripts/main.py 的配置合并方式生成实验配置。"""
    merged = {}
    for section_name in ["experiment", "model", "data", "device"]:
        section = raw_config.get(section_name, {})
        if section:
            merged.update(section)
    field_names = {field.name for field in dataclasses.fields(AxonExperimentConfig)}
    config = AxonExperimentConfig(**{k: v for k, v in merged.items() if k in field_names})
    if "name" in raw_config.get("experiment", {}):
        config.experiment_name = raw_config["experiment"]["name"]
    if "device" in raw_config.get("device", {}):
        config.device = raw_config["device"]["device"]
    return config


def resolve_path(path: Path, base_dir: Path = PROJECT_ROOT) -> Path:
    return path if path.is_absolute() else (base_dir / path)


def load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _manifest_score(manifest_path: Path, manifest: dict, config: AxonExperimentConfig) -> Optional[Tuple[int, float]]:
    """返回可排序分数；不匹配当前主配置的 manifest 直接跳过。"""
    if int(manifest.get("pe_feature_dim", -1)) != int(config.pe_feature_dim):
        return None
    if manifest.get("pe_schema_version", "legacy_dynamic") != config.pe_schema_version:
        return None
    if bool(manifest.get("strict_pe_parsing", True)) != bool(config.strict_pe_parsing):
        return None
    if bool(manifest.get("allow_pe_fallback", False)) != bool(config.allow_pe_fallback):
        return None
    if int(manifest.get("pe_fixed_section_slots", config.pe_fixed_section_slots)) != int(config.pe_fixed_section_slots):
        return None
    return (len(manifest.get("samples", [])), manifest_path.stat().st_mtime)


def select_manifest(
    raw_config: dict,
    config: AxonExperimentConfig,
    data_dir: Path,
    explicit_manifest: Optional[Path],
) -> Tuple[Path, dict, str]:
    if explicit_manifest is not None:
        manifest_path = resolve_path(explicit_manifest)
        manifest = load_manifest(manifest_path)
        return manifest_path, manifest, "explicit"

    data_section = raw_config.get("data", {})
    configured_cache = data_section.get("cache_dir")
    cache_dir = resolve_path(Path(configured_cache)) if configured_cache else data_dir / ".cache"
    if not cache_dir.exists():
        raise FileNotFoundError(f"Feature cache directory not found: {cache_dir}")

    candidates = []
    for manifest_path in cache_dir.glob("manifest_*.json"):
        try:
            manifest = load_manifest(manifest_path)
        except Exception:
            continue
        score = _manifest_score(manifest_path, manifest, config)
        if score is not None:
            candidates.append((score, manifest_path, manifest))

    if not candidates:
        raise FileNotFoundError(
            "No manifest matched the config. Use --manifest to analyze a specific cache manifest."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    _score, manifest_path, manifest = candidates[0]
    return manifest_path, manifest, "auto-matched-config"


def limit_samples_per_class(samples: Sequence[dict], max_samples_per_class: Optional[int]) -> List[dict]:
    if max_samples_per_class is None or max_samples_per_class <= 0:
        return list(samples)
    counts = Counter()
    limited = []
    for sample in samples:
        label = int(sample["label"])
        if counts[label] >= max_samples_per_class:
            continue
        limited.append(sample)
        counts[label] += 1
    return limited


def build_sample_records(samples: Sequence[dict], manifest_path: Path) -> List[SampleRecord]:
    records = []
    cache_dir = manifest_path.parent
    for index, sample in enumerate(samples):
        cache_path = _resolve_manifest_cache_path(sample.get("cache_path", ""), cache_dir)
        label = int(sample["label"])
        if label not in {0, 1}:
            raise ValueError(f"Manifest label must be 0 or 1: {label}")
        records.append(
            SampleRecord(
                index=index,
                source_path=str(sample.get("source_path", cache_path)),
                cache_path=cache_path,
                label=label,
            )
        )
    return records


def assign_splits(records: Sequence[SampleRecord], config: AxonExperimentConfig) -> None:
    """复现 dataset.create_stratified_split 的分层切分规则，只给报告打标签。"""
    rng = np.random.RandomState(config.seed)
    labels = [record.label for record in records]
    for label in list(set(labels)):
        label_indices = [idx for idx, item_label in enumerate(labels) if item_label == label]
        rng.shuffle(label_indices)

        n = len(label_indices)
        n_val = int(n * config.val_ratio)
        n_test = int(n * config.test_ratio)
        if config.val_ratio > 0 and n_val == 0 and n >= 3:
            n_val = 1
        if config.test_ratio > 0 and n_test == 0 and n - n_val >= 2:
            n_test = 1
        if n_val + n_test >= n:
            n_test = max(0, n - n_val - 1)

        for idx in label_indices[:n_val]:
            records[idx].split = "val"
        for idx in label_indices[n_val:n_val + n_test]:
            records[idx].split = "test"
        for idx in label_indices[n_val + n_test:]:
            records[idx].split = "train"


def _fit_length(array: np.ndarray, target_length: int, dtype) -> np.ndarray:
    array = np.asarray(array)
    if array.shape[0] > target_length:
        array = array[:target_length]
    elif array.shape[0] < target_length:
        array = np.pad(array, (0, target_length - array.shape[0]))
    return array.astype(dtype, copy=False)


def load_feature_matrix(
    records: Sequence[SampleRecord],
    manifest: dict,
) -> Tuple[np.ndarray, List[str]]:
    pe_dim = int(manifest["pe_feature_dim"])
    stat_dim = int(manifest.get("stat_feature_dim", 49))
    max_byte_length = int(manifest["max_byte_length"])
    feature_dim = pe_dim + stat_dim
    feature_matrix = np.zeros((len(records), feature_dim), dtype=np.float32)
    byte_hashes = []

    for row_idx, record in enumerate(records):
        byte_seq, pe_features, stat_features, _lightweight_features, label = _load_cached_feature_npz(
            record.cache_path,
            max_byte_length,
            pe_dim,
            stat_dim,
            int(manifest.get("lightweight_feature_dim", 256)),
            expected_label=record.label,
        )
        if label != record.label:
            raise ValueError(f"Cache label mismatch for {record.cache_path}")
        # 字节哈希代表“模型实际看到的截断/补齐后字节序列”是否重复。
        byte_hashes.append(hashlib.sha256(byte_seq.tobytes()).hexdigest())
        feature_matrix[row_idx, :pe_dim] = pe_features
        feature_matrix[row_idx, pe_dim:] = stat_features

    return feature_matrix, byte_hashes


def normalize_features(feature_matrix: np.ndarray) -> np.ndarray:
    """把不同量纲的特征拉到同一尺度，再转成可算余弦相似度的单位向量。"""
    mean = feature_matrix.mean(axis=0, keepdims=True)
    std = feature_matrix.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    normalized = (feature_matrix - mean) / std
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return (normalized / norms).astype(np.float32, copy=False)


def byte_duplicate_pairs(byte_hashes: Sequence[str]) -> Iterable[Tuple[int, int]]:
    by_hash = defaultdict(list)
    for idx, digest in enumerate(byte_hashes):
        by_hash[digest].append(idx)
    for indices in by_hash.values():
        if len(indices) > 1:
            yield from combinations(indices, 2)


def _hash_bits_to_band_key(bits: np.ndarray, start: int, end: int) -> int:
    value = 0
    for bit in bits[start:end]:
        value = (value << 1) | int(bit)
    return value


def lsh_candidate_pairs(
    normalized_features: np.ndarray,
    seed: int,
    simhash_bits: int,
    band_size: int,
    max_bucket_size: int,
) -> Tuple[set, int, int]:
    if simhash_bits <= 0:
        raise ValueError("simhash_bits must be positive")
    if band_size <= 0 or simhash_bits % band_size != 0:
        raise ValueError("lsh_band_size must be positive and divide simhash_bits")

    rng = np.random.RandomState(seed)
    planes = rng.normal(size=(normalized_features.shape[1], simhash_bits)).astype(np.float32)
    bit_matrix = normalized_features @ planes >= 0
    candidate_pairs = set()
    skipped_buckets = 0
    oversized_samples = 0

    for band_start in range(0, simhash_bits, band_size):
        buckets = defaultdict(list)
        band_end = band_start + band_size
        for idx, bits in enumerate(bit_matrix):
            key = _hash_bits_to_band_key(bits, band_start, band_end)
            buckets[(band_start, key)].append(idx)

        for indices in buckets.values():
            if len(indices) < 2:
                continue
            if len(indices) > max_bucket_size:
                skipped_buckets += 1
                oversized_samples += len(indices)
                continue
            candidate_pairs.update(combinations(indices, 2))

    return candidate_pairs, skipped_buckets, oversized_samples


def _pair_flags(left: SampleRecord, right: SampleRecord) -> Dict[str, bool]:
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
    records: Sequence[SampleRecord],
    normalized_features: np.ndarray,
    byte_pairs: Iterable[Tuple[int, int]],
    candidate_pairs: Iterable[Tuple[int, int]],
    similarity_threshold: float,
) -> List[dict]:
    pair_methods = defaultdict(set)
    for left, right in byte_pairs:
        pair_methods[tuple(sorted((left, right)))].add("byte_duplicate")

    for left, right in candidate_pairs:
        left, right = sorted((left, right))
        similarity = float(np.dot(normalized_features[left], normalized_features[right]))
        if similarity >= similarity_threshold:
            pair_methods[(left, right)].add("feature_similar")

    rows = []
    for (left_idx, right_idx), methods in sorted(pair_methods.items()):
        left = records[left_idx]
        right = records[right_idx]
        similarity = float(np.dot(normalized_features[left_idx], normalized_features[right_idx]))
        flags = _pair_flags(left, right)
        rows.append({
            "sample_i": left.index,
            "sample_j": right.index,
            "label_i": left.label,
            "label_j": right.label,
            "split_i": left.split,
            "split_j": right.split,
            "similarity": similarity,
            "methods": "+".join(sorted(methods)),
            "same_label": flags["same_label"],
            "same_split": flags["same_split"],
            "cross_split": flags["cross_split"],
            "leakage_pair": flags["leakage_pair"],
            "label_conflict": flags["label_conflict"],
            "source_path_i": left.source_path,
            "source_path_j": right.source_path,
            "cache_path_i": str(left.cache_path),
            "cache_path_j": str(right.cache_path),
        })
    return rows


def build_group_rows(records: Sequence[SampleRecord], pair_rows: Sequence[dict]) -> List[dict]:
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


def write_pairs_csv(path: Path, rows: Sequence[dict]) -> None:
    columns = [
        "methods",
        "sample_i",
        "sample_j",
        "label_i",
        "label_j",
        "split_i",
        "split_j",
        "similarity",
        "same_label",
        "same_split",
        "cross_split",
        "leakage_pair",
        "label_conflict",
        "source_path_i",
        "source_path_j",
        "cache_path_i",
        "cache_path_j",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_groups_csv(path: Path, rows: Sequence[dict]) -> None:
    columns = [
        "group_id",
        "size",
        "labels",
        "splits",
        "has_label_conflict",
        "has_cross_split",
        "has_leakage",
        "sample_indices",
        "source_paths",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def build_summary(
    options: AnalysisOptions,
    config: AxonExperimentConfig,
    manifest_path: Path,
    manifest: dict,
    selection_reason: str,
    records: Sequence[SampleRecord],
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
        "manifest_path": str(manifest_path),
        "selection_reason": selection_reason,
        "cache_config_hash": manifest.get("cache_config_hash"),
        "manifest_max_byte_length": manifest.get("max_byte_length"),
        "manifest_pe_feature_dim": manifest.get("pe_feature_dim"),
        "manifest_stat_feature_dim": manifest.get("stat_feature_dim"),
        "manifest_pe_schema_version": manifest.get("pe_schema_version"),
        "analyzed_samples": len(records),
        "manifest_samples": len(manifest.get("samples", [])),
        "max_samples_per_class": options.max_samples_per_class,
        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "split_counts": {str(k): int(v) for k, v in sorted(split_counts.items())},
        "similarity_threshold": options.similarity_threshold,
        "simhash_bits": options.simhash_bits,
        "lsh_band_size": options.lsh_band_size,
        "max_bucket_size": options.max_bucket_size,
        "lsh_candidate_pairs": int(candidate_count),
        "skipped_lsh_buckets": int(skipped_lsh_buckets),
        "oversized_lsh_samples": int(oversized_lsh_samples),
        "pair_counts": {
            "total": len(pair_rows),
            "byte_duplicate": int(method_counts.get("byte_duplicate", 0)),
            "feature_similar": int(method_counts.get("feature_similar", 0)),
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
            "pe_feature_dim": config.pe_feature_dim,
            "pe_schema_version": config.pe_schema_version,
        },
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }


def analyze_similarity(options: AnalysisOptions) -> dict:
    raw_config = read_toml_config(resolve_path(options.config_path) if options.config_path else None)
    config = build_experiment_config(raw_config)
    if options.seed is not None:
        config.seed = options.seed

    data_dir = resolve_path(options.data_dir or Path(config.data_dir or "data"))
    manifest_path, manifest, selection_reason = select_manifest(
        raw_config,
        config,
        data_dir,
        options.manifest_path,
    )

    samples = limit_samples_per_class(manifest.get("samples", []), options.max_samples_per_class)
    records = build_sample_records(samples, manifest_path)
    assign_splits(records, config)

    feature_matrix, byte_hashes = load_feature_matrix(records, manifest)
    normalized_features = normalize_features(feature_matrix)

    byte_pairs = list(byte_duplicate_pairs(byte_hashes))
    candidate_pairs, skipped_lsh_buckets, oversized_lsh_samples = lsh_candidate_pairs(
        normalized_features,
        seed=config.seed,
        simhash_bits=options.simhash_bits,
        band_size=options.lsh_band_size,
        max_bucket_size=options.max_bucket_size,
    )
    candidate_pairs.update(tuple(sorted(pair)) for pair in byte_pairs)

    pair_rows = build_pair_rows(
        records,
        normalized_features,
        byte_pairs,
        candidate_pairs,
        options.similarity_threshold,
    )
    group_rows = build_group_rows(records, pair_rows)

    output_dir = resolve_path(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "summary": output_dir / "sample_similarity_summary.json",
        "pairs": output_dir / "sample_similarity_pairs.csv",
        "groups": output_dir / "sample_similarity_groups.csv",
    }

    summary = build_summary(
        options,
        config,
        manifest_path,
        manifest,
        selection_reason,
        records,
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
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> AnalysisOptions:
    parser = argparse.ArgumentParser(
        description="Analyze Axon cached samples for duplicate and high-similarity risks."
    )
    parser.add_argument("--config", type=Path, default=Path("config/default_config.toml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/similarity"))
    parser.add_argument("--similarity-threshold", type=float, default=0.985)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--simhash-bits", type=int, default=64)
    parser.add_argument("--lsh-band-size", type=int, default=8)
    parser.add_argument("--max-bucket-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    return AnalysisOptions(
        config_path=args.config,
        data_dir=args.data_dir,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        similarity_threshold=args.similarity_threshold,
        max_samples_per_class=args.max_samples_per_class,
        simhash_bits=args.simhash_bits,
        lsh_band_size=args.lsh_band_size,
        max_bucket_size=args.max_bucket_size,
        seed=args.seed,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = parse_args(argv)
    summary = analyze_similarity(options)
    print("=" * 60)
    print("Axon Sample Similarity Analysis")
    print("=" * 60)
    print(f"Manifest: {summary['manifest_path']}")
    print(f"Analyzed samples: {summary['analyzed_samples']}")
    print(f"Similarity pairs: {summary['pair_counts']['total']}")
    print(f"Byte duplicate pairs: {summary['pair_counts']['byte_duplicate']}")
    print(f"Feature similar pairs: {summary['pair_counts']['feature_similar']}")
    print(f"Leakage pairs: {summary['pair_counts']['leakage_pairs']}")
    print(f"Label conflicts: {summary['pair_counts']['label_conflicts']}")
    print(f"Groups: {summary['group_count']}")
    print(f"Summary: {summary['outputs']['summary']}")
    print(f"Pairs CSV: {summary['outputs']['pairs']}")
    print(f"Groups CSV: {summary['outputs']['groups']}")
    if summary["skipped_lsh_buckets"]:
        print(
            "[Warning] Some LSH buckets were too large and skipped. "
            "Increase --max-bucket-size for a slower, broader scan."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
