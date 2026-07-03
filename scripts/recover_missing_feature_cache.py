#!/usr/bin/env python3
"""Recover only the missing feature-cache rows listed in bounded CSV inputs."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import sys
import tomllib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AxonExperimentConfig  # noqa: E402
from kvd_features.extractor import ExtractionConfig, extract_all_features  # noqa: E402


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def absolute_without_resolving_links(path: Path) -> Path:
    return path.absolute() if path.is_absolute() else (PROJECT_ROOT / path).absolute()


def load_checkpoint_config(checkpoint_path: Path) -> AxonExperimentConfig:
    from security import load_safe_checkpoint

    checkpoint = load_safe_checkpoint(checkpoint_path, map_location="cpu")
    return AxonExperimentConfig.from_dict(dict(checkpoint["config"]))


def load_toml_config(config_path: Path) -> AxonExperimentConfig:
    with resolve_path(config_path).open("rb") as handle:
        raw_config = tomllib.load(handle)

    merged = {}
    for section_name in ["experiment", "model", "data", "device"]:
        section = raw_config.get(section_name, {})
        if isinstance(section, dict):
            merged.update(section)
    field_names = {field.name for field in dataclasses.fields(AxonExperimentConfig)}
    config = AxonExperimentConfig(**{key: value for key, value in merged.items() if key in field_names})
    if "name" in raw_config.get("experiment", {}):
        config.experiment_name = raw_config["experiment"]["name"]
    if "output_dir" in raw_config.get("data", {}):
        config.model_save_dir = Path(raw_config["data"]["output_dir"])
    if "log_dir" in raw_config.get("data", {}):
        config.log_dir = Path(raw_config["data"]["log_dir"])
    return config


def load_recovery_config(*, checkpoint: Optional[Path], config_path: Optional[Path]) -> AxonExperimentConfig:
    if checkpoint is not None and config_path is not None:
        raise ValueError("Use either --checkpoint or --config, not both")
    if checkpoint is not None:
        return load_checkpoint_config(resolve_path(checkpoint))
    if config_path is not None:
        return load_toml_config(resolve_path(config_path))
    raise ValueError("Either --checkpoint or --config is required")


def _feature_cache_signature(
    max_byte_length: int,
    stat_feature_dim: int,
    pe_feature_dim: int,
    lightweight_feature_dim: int,
    strict_pe_parsing: bool,
    allow_pe_fallback: bool,
    pe_schema_version: str = "legacy_dynamic",
    pe_fixed_section_slots: int = 32,
) -> str:
    if pe_schema_version == "legacy_dynamic":
        return (
            f"{max_byte_length}_{stat_feature_dim}_{pe_feature_dim}_"
            f"{lightweight_feature_dim}_{strict_pe_parsing}_{allow_pe_fallback}"
        )
    return (
        f"{max_byte_length}_{stat_feature_dim}_{pe_feature_dim}_"
        f"{lightweight_feature_dim}_{strict_pe_parsing}_{allow_pe_fallback}_"
        f"{pe_schema_version}_{pe_fixed_section_slots}"
    )


def _feature_cache_hash(
    max_byte_length: int,
    stat_feature_dim: int,
    pe_feature_dim: int,
    lightweight_feature_dim: int,
    strict_pe_parsing: bool,
    allow_pe_fallback: bool,
    pe_schema_version: str = "legacy_dynamic",
    pe_fixed_section_slots: int = 32,
) -> str:
    signature = _feature_cache_signature(
        max_byte_length,
        stat_feature_dim,
        pe_feature_dim,
        lightweight_feature_dim,
        strict_pe_parsing,
        allow_pe_fallback,
        pe_schema_version,
        pe_fixed_section_slots,
    )
    return hashlib.md5(signature.encode()).hexdigest()[:8]


def _feature_cache_path_for_file(file_path: Path, cache_dir: Path, cache_config_hash: str) -> Path:
    try:
        stat = file_path.stat()
        file_sig = f"{stat.st_size}_{int(stat.st_mtime_ns)}"
    except OSError:
        file_sig = "missing"
    file_hash = hashlib.md5(f"{file_path.resolve()}_{file_sig}".encode()).hexdigest()
    return cache_dir / f"{file_hash}_{cache_config_hash}.npz"


def _is_valid_pe_sample_path(file_path: Path, max_file_size: int) -> bool:
    try:
        file_size = file_path.stat().st_size
    except OSError:
        return False
    if file_size == 0 or file_size > max_file_size:
        return False
    try:
        with file_path.open("rb") as f:
            return f.read(2) == b"MZ"
    except OSError:
        return False


def _file_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npz_scalar_to_text(value) -> str:
    import numpy as np

    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(value)


def _normalize_cached_arrays(
    byte_seq,
    pe_feat,
    stat_feat,
    lightweight_feat,
    max_byte_length: int,
    pe_feature_dim: int,
    stat_feature_dim: int,
    lightweight_feature_dim: int,
):
    import numpy as np

    if len(byte_seq) > max_byte_length:
        byte_seq = byte_seq[:max_byte_length]
    elif len(byte_seq) < max_byte_length:
        byte_seq = np.pad(byte_seq, (0, max_byte_length - len(byte_seq)))

    if len(pe_feat) < pe_feature_dim:
        pe_feat = np.pad(pe_feat, (0, pe_feature_dim - len(pe_feat)))
    elif len(pe_feat) > pe_feature_dim:
        pe_feat = pe_feat[:pe_feature_dim]

    if len(stat_feat) < stat_feature_dim:
        stat_feat = np.pad(stat_feat, (0, stat_feature_dim - len(stat_feat)))
    elif len(stat_feat) > stat_feature_dim:
        stat_feat = stat_feat[:stat_feature_dim]

    if len(lightweight_feat) < lightweight_feature_dim:
        lightweight_feat = np.pad(lightweight_feat, (0, lightweight_feature_dim - len(lightweight_feat)))
    elif len(lightweight_feat) > lightweight_feature_dim:
        lightweight_feat = lightweight_feat[:lightweight_feature_dim]

    return (
        byte_seq.astype(np.uint8, copy=False),
        pe_feat.astype(np.float32, copy=False),
        stat_feat.astype(np.float32, copy=False),
        lightweight_feat.astype(np.float32, copy=False),
    )


def _load_cache_metadata(cache_path: Path) -> Optional[dict]:
    import numpy as np

    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if "label" not in data.files:
                return None
            label = int(data["label"])
            source_sha256 = (
                _npz_scalar_to_text(data["source_sha256"])
                if "source_sha256" in data.files
                else None
            )
            return {"label": label, "source_sha256": source_sha256}
    except Exception:
        return None


def _load_cached_feature_npz(
    cache_path: Path,
    max_byte_length: int,
    pe_feature_dim: int,
    stat_feature_dim: int,
    lightweight_feature_dim: int,
    expected_label: Optional[int] = None,
    expected_source_sha256: Optional[str] = None,
):
    import numpy as np

    with np.load(cache_path, allow_pickle=False) as data:
        required_fields = {"byte_sequence", "pe_features", "label"}
        missing_fields = sorted(required_fields - set(data.files))
        if missing_fields:
            raise ValueError(f"Cache missing required fields {missing_fields}: {cache_path}")
        byte_seq = data["byte_sequence"]
        pe_features = data["pe_features"]
        stat_features = data.get("stat_features", np.zeros(stat_feature_dim, dtype=np.float32))
        lightweight_features = data.get(
            "lightweight_features",
            np.zeros(lightweight_feature_dim, dtype=np.float32),
        )
        label = int(data["label"])
        if expected_label is not None and label != int(expected_label):
            raise ValueError(f"Cache label mismatch for {cache_path}: expected {expected_label}, got {label}")
        if expected_source_sha256:
            if "source_sha256" not in data.files:
                raise ValueError(f"Cache missing source SHA for {cache_path}")
            cached_sha = _npz_scalar_to_text(data["source_sha256"])
            if cached_sha != expected_source_sha256:
                raise ValueError(f"Cache source SHA mismatch for {cache_path}")
    return _normalize_cached_arrays(
        byte_seq,
        pe_features,
        stat_features,
        lightweight_features,
        max_byte_length,
        pe_feature_dim,
        stat_feature_dim,
        lightweight_feature_dim,
    ) + (label,)


def cache_config_hash(config: AxonExperimentConfig) -> str:
    return _feature_cache_hash(
        config.max_byte_length,
        config.stat_feature_dim,
        config.pe_feature_dim,
        config.lightweight_feature_dim,
        config.strict_pe_parsing,
        config.allow_pe_fallback,
        config.pe_schema_version,
        config.pe_fixed_section_slots,
    )


def read_missing_rows(paths: Sequence[Path]) -> list[dict]:
    rows_by_key: dict[tuple[str, int], dict] = {}
    for csv_path in paths:
        with resolve_path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                raw_source_path = row.get("source_path", "").strip()
                source_path = (row.get("original_source_path") or raw_source_path).strip()
                if not source_path:
                    continue
                label = int(row["label"])
                source_sha256 = str(row.get("source_sha256") or "").strip().casefold()
                key = (str(absolute_without_resolving_links(Path(source_path))), label)
                merged = dict(row)
                merged["source_path"] = key[0]
                if raw_source_path and raw_source_path != source_path:
                    merged["materialized_source_path"] = str(
                        absolute_without_resolving_links(Path(raw_source_path))
                    )
                merged["label"] = label
                merged["source_sha256"] = source_sha256
                rows_by_key.setdefault(key, merged)
    return list(rows_by_key.values())


def build_payload(
    row: dict,
    config_dict: dict,
    cache_dir: Path,
    config_hash: str,
    storage_format: str,
) -> dict:
    source_path = Path(row["source_path"])
    return {
        "source_path": str(source_path),
        "label": int(row["label"]),
        "expected_source_sha256": str(row.get("source_sha256") or "").strip().casefold(),
        "cache_dir": str(cache_dir),
        "config_hash": config_hash,
        "config_dict": config_dict,
        "storage_format": storage_format,
    }


def save_feature_cache_npz(cache_path: Path, payload: dict[str, object], storage_format: str) -> None:
    """Write a cache NPZ either compressed or uncompressed."""
    import numpy as np

    temp_path = cache_path.with_name(cache_path.name + ".tmp.npz")
    if storage_format == "compressed":
        np.savez_compressed(temp_path, **payload)
    elif storage_format == "uncompressed":
        np.savez(temp_path, **payload)
    else:
        raise ValueError(f"Unsupported cache storage format: {storage_format}")
    temp_path.replace(cache_path)


def recover_one(payload: dict) -> dict:
    source_path = Path(payload["source_path"])
    label = int(payload["label"])
    cache_dir = Path(payload["cache_dir"])
    config = AxonExperimentConfig.from_dict(dict(payload["config_dict"]))
    cache_path = _feature_cache_path_for_file(source_path, cache_dir, payload["config_hash"])

    if not source_path.exists():
        return {"status": "missing_source", "source_path": str(source_path), "cache_path": str(cache_path)}
    if not _is_valid_pe_sample_path(source_path, int(config.max_file_size)):
        return {"status": "invalid_pe", "source_path": str(source_path), "cache_path": str(cache_path)}

    source_sha256 = _file_sha256(source_path)
    expected_source_sha256 = str(payload.get("expected_source_sha256") or "").strip().casefold()
    if not expected_source_sha256:
        return {
            "status": "missing_expected_source_sha256",
            "source_path": str(source_path),
            "cache_path": str(cache_path),
            "label": label,
            "source_sha256": source_sha256,
        }
    if source_sha256 != expected_source_sha256:
        return {
            "status": "source_sha256_mismatch",
            "source_path": str(source_path),
            "cache_path": str(cache_path),
            "label": label,
            "source_sha256": source_sha256,
            "expected_source_sha256": expected_source_sha256,
        }
    cached_meta = _load_cache_metadata(cache_path)
    if cached_meta is not None:
        if cached_meta.get("label") == label and cached_meta.get("source_sha256") == source_sha256:
            return {
                "status": "cache_hit",
                "source_path": str(source_path),
                "cache_path": str(cache_path),
                "label": label,
                "source_sha256": source_sha256,
            }
        return {
            "status": "cache_conflict",
            "source_path": str(source_path),
            "cache_path": str(cache_path),
            "label": label,
            "source_sha256": source_sha256,
        }

    extraction_config = ExtractionConfig.from_axon_config(
        config,
        max_file_size=config.max_byte_length,
        pe_feature_dim=config.pe_feature_dim,
    )
    byte_seq, pe_feat, stat_feat, lightweight_feat, _orig_len = extract_all_features(
        str(source_path),
        extraction_config,
        axon_config=config,
        allow_pe_fallback=config.allow_pe_fallback,
    )
    if byte_seq is None or pe_feat is None:
        return {"status": "feature_extract_failed", "source_path": str(source_path), "cache_path": str(cache_path)}

    byte_seq, pe_feat, stat_feat, lightweight_feat = _normalize_cached_arrays(
        byte_seq,
        pe_feat,
        stat_feat,
        lightweight_feat,
        config.max_byte_length,
        config.pe_feature_dim,
        config.stat_feature_dim,
        config.lightweight_feature_dim,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_feature_cache_npz(
        cache_path,
        {
            "byte_sequence": byte_seq,
            "pe_features": pe_feat,
            "stat_features": stat_feat,
            "lightweight_features": lightweight_feat,
            "label": label,
            "source_sha256": source_sha256,
        },
        str(payload.get("storage_format") or "compressed"),
    )
    _load_cached_feature_npz(
        cache_path,
        config.max_byte_length,
        config.pe_feature_dim,
        config.stat_feature_dim,
        config.lightweight_feature_dim,
        expected_label=label,
        expected_source_sha256=source_sha256,
    )
    return {
        "status": "extracted",
        "source_path": str(source_path),
        "cache_path": str(cache_path),
        "label": label,
        "source_sha256": source_sha256,
    }


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "samples": []}
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest(
    manifest_path: Path,
    config: AxonExperimentConfig,
    config_hash: str,
    recovered: Sequence[dict],
    *,
    dry_run: bool,
    storage_format: str = "compressed",
) -> int:
    manifest = load_manifest(manifest_path)
    samples = list(manifest.get("samples", []))
    by_cache = {str(Path(sample.get("cache_path", "")).resolve(strict=False)): sample for sample in samples}
    added = 0
    for item in recovered:
        if item.get("status") not in {"extracted", "cache_hit"}:
            continue
        cache_path = str(Path(item["cache_path"]).resolve(strict=False))
        if cache_path in by_cache:
            continue
        sample = {
            "source_path": item["source_path"],
            "cache_path": item["cache_path"],
            "label": int(item["label"]),
            "source_sha256": item["source_sha256"],
        }
        samples.append(sample)
        by_cache[cache_path] = sample
        added += 1
    if dry_run:
        return added

    manifest.update(
        {
            "version": manifest.get("version", 1),
            "data_dir": str(PROJECT_ROOT / "data"),
            "cache_config_hash": config_hash,
            "max_byte_length": config.max_byte_length,
            "pe_feature_dim": config.pe_feature_dim,
            "stat_feature_dim": config.stat_feature_dim,
            "lightweight_feature_dim": config.lightweight_feature_dim,
            "strict_pe_parsing": config.strict_pe_parsing,
            "allow_pe_fallback": config.allow_pe_fallback,
            "pe_schema_version": config.pe_schema_version,
            "pe_fixed_section_slots": config.pe_fixed_section_slots,
            "cache_storage_format": storage_format,
            "samples": samples,
        }
    )
    tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(manifest_path)
    return added


def recover_rows(
    *,
    missing_csvs: Sequence[Path],
    checkpoint: Optional[Path] = None,
    config_path: Optional[Path] = None,
    cache_dir: Path,
    workers: int,
    backend: str,
    limit: Optional[int] = None,
    dry_run: bool = False,
    storage_format: str = "compressed",
    progress_interval: int = 1000,
) -> dict:
    config = load_recovery_config(checkpoint=checkpoint, config_path=config_path)
    config_hash = cache_config_hash(config)
    cache_dir = resolve_path(cache_dir)
    manifest_path = cache_dir / f"manifest_{config_hash}.json"
    rows = read_missing_rows(missing_csvs)
    if limit is not None:
        rows = rows[:limit]

    planned = []
    missing_expected_sha = 0
    invalid_expected_sha = 0
    for row in rows:
        source_path = Path(row["source_path"])
        cache_path = _feature_cache_path_for_file(source_path, cache_dir, config_hash)
        expected_sha = str(row.get("source_sha256") or "").strip().casefold()
        if not expected_sha:
            missing_expected_sha += 1
        elif len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
            invalid_expected_sha += 1
        planned.append({**row, "cache_path": str(cache_path), "cache_exists": cache_path.exists()})
    if dry_run:
        counts = Counter("cache_exists" if row["cache_exists"] else "would_extract" for row in planned)
        return {
            "schema": "axon_missing_feature_cache_recovery_v1",
            "dry_run": True,
            "checkpoint": str(resolve_path(checkpoint)) if checkpoint is not None else None,
            "config_path": str(resolve_path(config_path)) if config_path is not None else None,
            "cache_config_hash": config_hash,
            "manifest_path": str(manifest_path),
            "storage_format": storage_format,
            "planned_rows": len(planned),
            "missing_expected_source_sha256_rows": missing_expected_sha,
            "invalid_expected_source_sha256_rows": invalid_expected_sha,
            "input_ready": missing_expected_sha == 0 and invalid_expected_sha == 0,
            "status_counts": dict(counts),
        }

    results = []
    executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    processed = 0
    config_dict = config.to_dict()
    if workers <= 1:
        for row in rows:
            payload = build_payload(row, config_dict, cache_dir, config_hash, storage_format)
            results.append(recover_one(payload))
            processed += 1
            if progress_interval > 0 and processed % progress_interval == 0:
                print(f"[recover] processed {processed}/{len(rows)}", flush=True)
    else:
        chunk_size = max(workers * 8, 64)
        with executor_cls(max_workers=workers) as executor:
            for start in range(0, len(rows), chunk_size):
                chunk = rows[start:start + chunk_size]
                futures = [
                    executor.submit(
                        recover_one,
                        build_payload(row, config_dict, cache_dir, config_hash, storage_format),
                    )
                    for row in chunk
                ]
                for future in as_completed(futures):
                    results.append(future.result())
                    processed += 1
                    if progress_interval > 0 and processed % progress_interval == 0:
                        print(f"[recover] processed {processed}/{len(rows)}", flush=True)
    status_counts = Counter(result["status"] for result in results)
    manifest_added = update_manifest(
        manifest_path,
        config,
        config_hash,
        results,
        dry_run=False,
        storage_format=storage_format,
    )
    return {
        "schema": "axon_missing_feature_cache_recovery_v1",
        "dry_run": False,
        "checkpoint": str(resolve_path(checkpoint)) if checkpoint is not None else None,
        "config_path": str(resolve_path(config_path)) if config_path is not None else None,
        "cache_config_hash": config_hash,
        "manifest_path": str(manifest_path),
        "storage_format": storage_format,
        "input_rows": len(rows),
        "status_counts": dict(status_counts),
        "manifest_added": manifest_added,
        "failed_examples": [r for r in results if r["status"] not in {"extracted", "cache_hit"}][:20],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recover missing feature cache rows from bounded CSV files.")
    parser.add_argument("--missing-csv", type=Path, action="append", required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--checkpoint", type=Path)
    source_group.add_argument("--config", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/.cache"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=["thread", "process"], default="process")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--storage-format", choices=["compressed", "uncompressed"], default="compressed")
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = recover_rows(
        missing_csvs=args.missing_csv,
        checkpoint=args.checkpoint,
        config_path=args.config,
        cache_dir=args.cache_dir,
        workers=max(1, int(args.workers)),
        backend=args.backend,
        limit=args.limit,
        dry_run=args.dry_run,
        storage_format=args.storage_format,
        progress_interval=max(0, int(args.progress_interval)),
    )
    output_path = resolve_path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"JSON: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
