#!/usr/bin/env python3
"""Export an Axon family classifier JSON from similarity group diagnostics.

这个脚本把离线调试阶段生成的相似样本组，转换成 DLL 可以加载的
family_classifier.json。它不训练模型，也不修改数据，只读取缓存特征和
group_members.csv，输出一份“家族中心点 + 判定半径”的 JSON 文件。
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for path in [SRC_DIR, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import AxonExperimentConfig  # noqa: E402
from dataset import _load_cached_feature_npz, _resolve_manifest_cache_path  # noqa: E402
from raw_group_tools import normalize_path_text, read_csv_rows, resolve_path  # noqa: E402


def read_toml_config(config_path: Optional[Path]) -> dict:
    if config_path is None:
        return {}
    with resolve_path(config_path).open("rb") as f:
        return tomllib.load(f)


def build_experiment_config(raw_config: dict) -> AxonExperimentConfig:
    merged = {}
    for section_name in ["experiment", "model", "data", "device"]:
        section = raw_config.get(section_name, {})
        if section:
            merged.update(section)
    field_names = {field.name for field in dataclasses.fields(AxonExperimentConfig)}
    config = AxonExperimentConfig(**{k: v for k, v in merged.items() if k in field_names})
    if "name" in raw_config.get("experiment", {}):
        config.experiment_name = raw_config["experiment"]["name"]
    return config


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def manifest_matches_config(manifest: dict, config: AxonExperimentConfig) -> bool:
    return (
        int(manifest.get("pe_feature_dim", -1)) == int(config.pe_feature_dim)
        and int(manifest.get("stat_feature_dim", config.stat_feature_dim)) == int(config.stat_feature_dim)
        and int(manifest.get("max_byte_length", -1)) == int(config.max_byte_length)
        and manifest.get("pe_schema_version", "legacy_dynamic") == config.pe_schema_version
        and bool(manifest.get("strict_pe_parsing", True)) == bool(config.strict_pe_parsing)
        and bool(manifest.get("allow_pe_fallback", False)) == bool(config.allow_pe_fallback)
        and int(manifest.get("pe_fixed_section_slots", config.pe_fixed_section_slots))
        == int(config.pe_fixed_section_slots)
    )


def select_manifest(data_dir: Path, config: AxonExperimentConfig, explicit_manifest: Optional[Path]) -> tuple[Path, dict]:
    if explicit_manifest is not None:
        manifest_path = resolve_path(explicit_manifest)
        return manifest_path, load_json(manifest_path)

    cache_dir = data_dir / ".cache"
    if not cache_dir.exists():
        raise FileNotFoundError(f"Feature cache directory not found: {cache_dir}")

    candidates = []
    for manifest_path in cache_dir.glob("manifest_*.json"):
        try:
            manifest = load_json(manifest_path)
        except Exception:
            continue
        if manifest_matches_config(manifest, config):
            candidates.append((len(manifest.get("samples", [])), manifest_path.stat().st_mtime, manifest_path, manifest))

    if not candidates:
        raise FileNotFoundError(
            "No cache manifest matched the current config. "
            "Use --manifest to point at the exact manifest used for the group report."
        )

    candidates.sort(reverse=True)
    _count, _mtime, manifest_path, manifest = candidates[0]
    return manifest_path, manifest


def source_path_keys(path_text: str) -> set[str]:
    path = Path(path_text)
    keys = {normalize_path_text(path_text), path.name.casefold()}
    if not path.is_absolute():
        keys.add(normalize_path_text(str((PROJECT_ROOT / path).resolve())))
    else:
        try:
            keys.add(normalize_path_text(str(path.resolve().relative_to(PROJECT_ROOT))))
        except ValueError:
            pass
    return keys


def build_cache_feature_map(manifest_path: Path, manifest: dict, config: AxonExperimentConfig) -> dict[str, dict]:
    cache_dir = manifest_path.parent
    mapped: dict[str, dict] = {}
    for sample in manifest.get("samples", []):
        source_path = str(sample.get("source_path", ""))
        if not source_path:
            continue
        cache_path = _resolve_manifest_cache_path(sample.get("cache_path", ""), cache_dir)
        label = int(sample["label"])
        byte_seq, pe_features, stat_features, _lightweight, loaded_label = _load_cached_feature_npz(
            cache_path,
            config.max_byte_length,
            config.pe_feature_dim,
            config.stat_feature_dim,
            config.lightweight_feature_dim,
            expected_label=label,
            expected_source_sha256=sample.get("source_sha256"),
        )
        if int(loaded_label) != label:
            raise ValueError(f"Cache label mismatch for {cache_path}")
        feature_vector = np.concatenate(
            [
                np.asarray(pe_features, dtype=np.float32),
                np.asarray(stat_features, dtype=np.float32),
            ]
        )
        entry = {
            "source_path": source_path,
            "cache_path": str(cache_path),
            "label": label,
            "feature": feature_vector,
        }
        for key in source_path_keys(source_path):
            mapped[key] = entry
    return mapped


def lookup_feature(cache_by_source: dict[str, dict], source_path: str) -> Optional[dict]:
    for key in source_path_keys(source_path):
        entry = cache_by_source.get(key)
        if entry is not None:
            return entry
    return None


def parse_bool(value) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def build_family_classifier(
    group_members_path: Path,
    cache_by_source: dict[str, dict],
    *,
    pe_feature_dim: int,
    stat_feature_dim: int,
    min_family_size: int,
    threshold_scale: float,
    threshold_margin: float,
    min_threshold: float,
    max_threshold: Optional[float],
    family_name_prefix: str,
    include_singletons: bool,
) -> tuple[dict, dict]:
    rows = read_csv_rows(resolve_path(group_members_path))
    grouped: dict[int, list[dict]] = defaultdict(list)
    skipped_missing = []
    skipped_non_malicious = 0

    for row in rows:
        label = int(row.get("label", 0))
        if label != 1:
            skipped_non_malicious += 1
            continue
        source_path = row["source_path"]
        entry = lookup_feature(cache_by_source, source_path)
        if entry is None:
            skipped_missing.append(source_path)
            continue
        group_id = int(row["group_id"])
        grouped[group_id].append({**row, "feature": entry["feature"]})

    selected_groups = []
    skipped_small = {}
    for group_id, members in sorted(grouped.items()):
        if len(members) < min_family_size and not include_singletons:
            skipped_small[str(group_id)] = len(members)
            continue
        selected_groups.append((group_id, members))

    if not selected_groups:
        raise ValueError(
            "No eligible malicious family groups found. "
            "Lower --min-family-size or pass --include-singletons if this is expected."
        )

    all_features = np.stack([member["feature"] for _group_id, members in selected_groups for member in members])
    scaler_mean = all_features.mean(axis=0)
    scaler_scale = all_features.std(axis=0)
    scaler_scale[scaler_scale < 1e-6] = 1.0

    cluster_ids = []
    centroids = []
    thresholds = []
    family_names = []
    family_stats = []

    for group_id, members in selected_groups:
        raw = np.stack([member["feature"] for member in members])
        scaled = (raw - scaler_mean) / scaler_scale
        centroid = scaled.mean(axis=0)
        distances = np.linalg.norm(scaled - centroid, axis=1)
        max_distance = float(distances.max()) if len(distances) else 0.0
        threshold = max(min_threshold, max_distance * threshold_scale + threshold_margin)
        if max_threshold is not None:
            threshold = min(threshold, max_threshold)

        cluster_ids.append(int(group_id))
        centroids.append(centroid.astype(float).tolist())
        thresholds.append(float(threshold))
        family_names.append(f"{family_name_prefix}{group_id}")
        family_stats.append(
            {
                "cluster_id": int(group_id),
                "family_name": f"{family_name_prefix}{group_id}",
                "sample_count": len(members),
                "max_member_distance": max_distance,
                "threshold": threshold,
                "source_paths": [member["source_path"] for member in members],
            }
        )

    payload = {
        "schema": "axon_family_classifier_v1",
        "feature_type": "fixed_v2_pe_stat",
        "feature_dim": int(all_features.shape[1]),
        "pe_feature_dim": int(pe_feature_dim),
        "stat_feature_dim": int(stat_feature_dim),
        "distance": "scaled_l2",
        "cluster_ids": cluster_ids,
        "centroids": centroids,
        "thresholds": thresholds,
        "family_names": family_names,
        "scaler_mean": scaler_mean.astype(float).tolist(),
        "scaler_scale": scaler_scale.astype(float).tolist(),
        "families": family_stats,
    }
    summary = {
        "input_group_members": str(resolve_path(group_members_path)),
        "eligible_family_count": len(selected_groups),
        "cluster_ids": cluster_ids,
        "family_sample_counts": {str(item["cluster_id"]): item["sample_count"] for item in family_stats},
        "skipped_non_malicious_rows": skipped_non_malicious,
        "skipped_missing_cache_rows": len(skipped_missing),
        "skipped_missing_cache_examples": skipped_missing[:20],
        "skipped_small_groups": skipped_small,
        "settings": {
            "min_family_size": min_family_size,
            "include_singletons": include_singletons,
            "threshold_scale": threshold_scale,
            "threshold_margin": threshold_margin,
            "min_threshold": min_threshold,
            "max_threshold": max_threshold,
        },
    }
    return payload, summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Export Axon family_classifier.json from group_members.csv and cached fixed_v2 features."
    )
    parser.add_argument("--config", type=Path, default=Path("config/default_config.toml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--group-members", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("resources/axon_family/family_classifier.json"))
    parser.add_argument("--min-family-size", type=int, default=2)
    parser.add_argument("--include-singletons", action="store_true")
    parser.add_argument("--threshold-scale", type=float, default=1.25)
    parser.add_argument("--threshold-margin", type=float, default=0.05)
    parser.add_argument("--min-threshold", type=float, default=0.25)
    parser.add_argument("--max-threshold", type=float, default=None)
    parser.add_argument("--family-name-prefix", type=str, default="axon_group_")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    raw_config = read_toml_config(args.config)
    config = build_experiment_config(raw_config)
    data_dir = resolve_path(args.data_dir or Path(config.data_dir or "data"))
    manifest_path, manifest = select_manifest(data_dir, config, args.manifest)
    cache_by_source = build_cache_feature_map(manifest_path, manifest, config)

    payload, summary = build_family_classifier(
        args.group_members,
        cache_by_source,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        min_family_size=args.min_family_size,
        threshold_scale=args.threshold_scale,
        threshold_margin=args.threshold_margin,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        family_name_prefix=args.family_name_prefix,
        include_singletons=args.include_singletons,
    )
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    summary.update(
        {
            "config": str(resolve_path(args.config)),
            "data_dir": str(data_dir),
            "manifest": str(manifest_path),
            "output": str(output_path),
            "feature_dim": payload["feature_dim"],
        }
    )
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("Axon Family Classifier Export")
    print("=" * 60)
    print(f"Families: {summary['eligible_family_count']}")
    print(f"Feature dim: {payload['feature_dim']}")
    print(f"Manifest: {manifest_path}")
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")
    if summary["skipped_missing_cache_rows"]:
        print(f"Missing cache rows: {summary['skipped_missing_cache_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
