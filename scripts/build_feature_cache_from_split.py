#!/usr/bin/env python3
"""Build a resumable strict feature cache for every row in a split manifest."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import sys
import time
import tomllib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AxonExperimentConfig  # noqa: E402
from dataset import (  # noqa: E402
    _feature_cache_hash,
    _iter_manifest_sample_entries,
    _load_cache_metadata,
    _prepare_sample_cache_worker,
    _write_cache_manifest_stream,
)
from kvd_features.extractor import ExtractionConfig  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_extraction_config(config_path: Path) -> AxonExperimentConfig:
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    merged: dict = {}
    for section_name in ("experiment", "model", "data", "device"):
        section = raw.get(section_name, {})
        if isinstance(section, dict):
            merged.update(section)
    fields = {field.name for field in dataclasses.fields(AxonExperimentConfig)}
    return AxonExperimentConfig(**{key: value for key, value in merged.items() if key in fields})


def load_split_rows(split_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_sha: set[str] = set()
    seen_paths: set[str] = set()
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"split CSV has no header: {split_csv}")
        required = {"source_path", "label", "split"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"split CSV missing columns: {sorted(required - set(reader.fieldnames))}")
        if "source_sha256" not in reader.fieldnames and "sha256" not in reader.fieldnames:
            raise ValueError("split CSV requires source_sha256 or legacy sha256")
        for row in reader:
            source_sha = str(row.get("source_sha256") or row.get("sha256") or "").strip().casefold()
            source_path = str(row.get("source_path") or "").strip()
            label = str(row.get("label") or "").strip()
            split = str(row.get("split") or "").strip().casefold()
            if not is_sha256(source_sha):
                raise ValueError(f"invalid source SHA-256: {source_sha}")
            if source_sha in seen_sha:
                raise ValueError(f"duplicate source SHA-256: {source_sha}")
            path_key = source_path.casefold()
            if not source_path or path_key in seen_paths:
                raise ValueError(f"missing or duplicate source path: {source_path}")
            if label not in {"0", "1"}:
                raise ValueError(f"invalid label for {source_path}: {label}")
            if split not in {"train", "val", "test"}:
                raise ValueError(f"invalid split for {source_path}: {split}")
            seen_sha.add(source_sha)
            seen_paths.add(path_key)
            rows.append(
                {
                    "source_path": source_path,
                    "source_sha256": source_sha,
                    "label": label,
                    "split": split,
                }
            )
    if not rows:
        raise ValueError(f"split CSV has no rows: {split_csv}")
    return rows


def cache_hash_for_config(config: AxonExperimentConfig) -> str:
    return _feature_cache_hash(
        config.max_byte_length,
        config.stat_feature_dim,
        config.pe_feature_dim,
        config.lightweight_feature_dim,
        config.strict_pe_parsing,
        False if config.strict_pe_parsing else config.allow_pe_fallback,
        config.pe_schema_version,
        config.pe_fixed_section_slots,
    )


def worker_payload(config: AxonExperimentConfig, cache_dir: Path) -> dict:
    allow_pe_fallback = False if config.strict_pe_parsing else config.allow_pe_fallback
    return {
        "cache_dir": str(cache_dir),
        "cache_config_hash": cache_hash_for_config(config),
        "max_file_size": config.max_file_size,
        "max_byte_length": config.max_byte_length,
        "pe_feature_dim": config.pe_feature_dim,
        "stat_feature_dim": config.stat_feature_dim,
        "lightweight_feature_dim": config.lightweight_feature_dim,
        "use_cache": True,
        "allow_pe_fallback": allow_pe_fallback,
        "extraction_config": ExtractionConfig.from_axon_config(
            config,
            max_file_size=config.max_byte_length,
            pe_feature_dim=config.pe_feature_dim,
        ),
        "axon_config": config,
    }


def prepare_row(
    row: dict[str, str],
    base_payload: dict,
    trust_split_sha256: bool = False,
) -> tuple[Optional[dict], Optional[dict], str]:
    source_path = Path(row["source_path"])
    payload = dict(base_payload)
    payload["file_path"] = str(source_path)
    payload["label"] = int(row["label"])
    if trust_split_sha256:
        payload["trust_source_sha256"] = row["source_sha256"]
    result = _prepare_sample_cache_worker(payload)
    status = str(result.get("status") or "other_failed_skipped")
    cache_path_text = result.get("cache_path")
    if status not in {"cache_hits", "extracted"} or not cache_path_text:
        reason = str(result.get("warning") or status)
        return None, {**row, "reason": reason}, status

    cache_path = Path(cache_path_text)
    metadata = _load_cache_metadata(cache_path)
    if metadata is None:
        return None, {**row, "reason": "invalid_cache_metadata"}, "invalid_cache_metadata"
    if int(metadata["label"]) != int(row["label"]):
        return None, {**row, "reason": "cache_label_mismatch"}, "cache_label_mismatch"
    if str(metadata.get("source_sha256") or "").casefold() != row["source_sha256"]:
        return None, {**row, "reason": "source_sha256_mismatch"}, "source_sha256_mismatch"
    sample = {
        "source_path": row["source_path"],
        "cache_path": cache_path.name,
        "label": int(row["label"]),
        "source_sha256": row["source_sha256"],
    }
    return sample, None, status


def load_existing_samples(manifest_path: Path, cache_dir: Path) -> dict[str, dict]:
    samples: dict[str, dict] = {}
    if not manifest_path.exists():
        return samples
    for sample in _iter_manifest_sample_entries(manifest_path):
        source_sha = str(sample.get("source_sha256") or "").strip().casefold()
        cache_name = Path(str(sample.get("cache_path") or "")).name
        if not is_sha256(source_sha) or not cache_name or not (cache_dir / cache_name).is_file():
            continue
        samples[source_sha] = {
            "source_path": str(sample.get("source_path") or ""),
            "cache_path": cache_name,
            "label": int(sample["label"]),
            "source_sha256": source_sha,
        }
    return samples


def write_failures(path: Path, failures: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source_path", "source_sha256", "label", "split", "reason"),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(failures)


def build_cache(
    *,
    split_csv: Path,
    config_path: Path,
    cache_dir: Path,
    workers: int,
    receipt_path: Path,
    failures_path: Path,
    trust_split_sha256: bool = False,
) -> dict:
    started = time.perf_counter()
    rows = load_split_rows(split_csv)
    config = load_extraction_config(config_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_hash = cache_hash_for_config(config)
    manifest_path = cache_dir / f"manifest_{cache_hash}.json"
    base_payload = worker_payload(config, cache_dir)

    samples: list[dict] = []
    failures: list[dict] = []
    statuses: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = pool.map(
            lambda row: prepare_row(row, base_payload, trust_split_sha256),
            rows,
            chunksize=1,
        )
        for index, (sample, failure, status) in enumerate(results, start=1):
            statuses[status] += 1
            if sample is not None:
                samples.append(sample)
            if failure is not None:
                failures.append(failure)
            if index % 100 == 0 or index == len(rows):
                print(
                    f"[cache] {index:,}/{len(rows):,} success={len(samples):,} "
                    f"failures={len(failures):,}",
                    flush=True,
                )

    write_failures(failures_path, failures)
    manifest_written = False
    if not failures:
        cumulative = load_existing_samples(manifest_path, cache_dir)
        cumulative.update({sample["source_sha256"]: sample for sample in samples})
        header = {
            "version": 1,
            "data_dir": "manifest-bound-multi-root",
            "cache_config_hash": cache_hash,
            "max_byte_length": config.max_byte_length,
            "pe_feature_dim": config.pe_feature_dim,
            "stat_feature_dim": config.stat_feature_dim,
            "lightweight_feature_dim": config.lightweight_feature_dim,
            "strict_pe_parsing": config.strict_pe_parsing,
            "allow_pe_fallback": False if config.strict_pe_parsing else config.allow_pe_fallback,
            "pe_schema_version": config.pe_schema_version,
            "pe_fixed_section_slots": config.pe_fixed_section_slots,
            "source_split": str(split_csv),
            "source_split_sha256": sha256_file(split_csv),
        }
        ordered_samples = sorted(cumulative.values(), key=lambda sample: sample["source_sha256"])
        _write_cache_manifest_stream(manifest_path, header, ordered_samples)
        manifest_written = True

    payload = {
        "schema": "axon_manifest_bound_feature_cache_build_v1",
        "split_csv": str(split_csv),
        "split_sha256": sha256_file(split_csv),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "cache_dir": str(cache_dir),
        "cache_config_hash": cache_hash,
        "manifest": str(manifest_path),
        "manifest_written": manifest_written,
        "manifest_sha256": sha256_file(manifest_path) if manifest_written else None,
        "rows": len(rows),
        "success": len(samples),
        "failures": len(failures),
        "status_counts": dict(sorted(statuses.items())),
        "failures_csv": str(failures_path),
        "workers": max(1, workers),
        "trust_split_sha256": trust_split_sha256,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "decision": "cache_ready" if not failures else "blocked_fail_closed",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--failures-csv", type=Path, required=True)
    parser.add_argument(
        "--trust-split-sha256",
        action="store_true",
        help=(
            "Trust the source_sha256 column from the split CSV instead of "
            "re-hashing each source file. Skips full-file reads on a stable "
            "corpus; cache metadata still records the same source_sha256."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config_path = resolve_project_path(args.config)
    config = load_extraction_config(config_path)
    configured_cache_dir = Path(config.cache_dir or "data/.cache")
    cache_dir = resolve_project_path(args.cache_dir or configured_cache_dir)
    payload = build_cache(
        split_csv=resolve_project_path(args.split_csv),
        config_path=config_path,
        cache_dir=cache_dir,
        workers=args.workers,
        receipt_path=resolve_project_path(args.receipt),
        failures_path=resolve_project_path(args.failures_csv),
        trust_split_sha256=args.trust_split_sha256,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["failures"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
