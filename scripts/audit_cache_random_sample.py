#!/usr/bin/env python3
"""Randomly audit cached NPZ samples against the split CSV, manifest, and source files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_split_from_cache import (  # noqa: E402
    iter_split_rows,
    load_manifest_lookup,
    lookup_manifest_sample,
    resolve_path,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_text(value: object) -> str:
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(value)


def _record_failure(row: dict, sample: Optional[dict], reason: str, detail: str) -> dict:
    return {
        "reason": reason,
        "detail": detail,
        "source_path": row.get("source_path", ""),
        "source_sha256": row.get("source_sha256", ""),
        "label": row.get("label", ""),
        "split": row.get("split", ""),
        "sample_index": row.get("sample_index", ""),
        "manifest_cache_path": sample.get("cache_path", "") if sample else "",
        "manifest_source_sha256": sample.get("source_sha256", "") if sample else "",
        "manifest_label": sample.get("label", "") if sample else "",
    }


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit_random_sample(
    *,
    split_csv: Path,
    manifest_path: Path,
    split: Optional[str],
    sample_fraction: float,
    sample_size: Optional[int],
    seed: int,
    output_json: Path,
    output_failures_csv: Path,
    output_sample_csv: Path,
) -> dict:
    split_csv = resolve_path(split_csv)
    manifest_path = resolve_path(manifest_path)
    output_json = resolve_path(output_json)
    output_failures_csv = resolve_path(output_failures_csv)
    output_sample_csv = resolve_path(output_sample_csv)

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_payload = json.load(handle)
    manifest_by_source, manifest_by_sha = load_manifest_lookup(manifest_path)
    rows = iter_split_rows(split_csv, None if split in (None, "all") else split)
    if sample_size is None:
        sample_size = max(1, int(round(len(rows) * sample_fraction)))
    sample_size = min(int(sample_size), len(rows))
    rng = random.Random(seed)
    sampled_rows = rng.sample(rows, sample_size)

    expected_byte_len = int(manifest_payload.get("max_byte_length", 0))
    expected_pe_dim = int(manifest_payload.get("pe_feature_dim", 0))
    expected_stat_dim = int(manifest_payload.get("stat_feature_dim", 0))
    expected_light_dim = int(manifest_payload.get("lightweight_feature_dim", 0))
    cache_dir = manifest_path.parent

    failures: list[dict] = []
    audited_rows: list[dict] = []
    match_counts: Counter[str] = Counter()
    checked_source_sha = 0
    checked_npz = 0

    for row in sampled_rows:
        sample, match_reason = lookup_manifest_sample(row, manifest_by_source, manifest_by_sha)
        audited_rows.append(
            {
                "source_path": row.get("source_path", ""),
                "label": row.get("label", ""),
                "split": row.get("split", ""),
                "sample_index": row.get("sample_index", ""),
                "match_reason": match_reason,
                "manifest_cache_path": sample.get("cache_path", "") if sample else "",
            }
        )
        if sample is None:
            failures.append(_record_failure(row, sample, "manifest_missing", match_reason))
            continue
        match_counts[match_reason] += 1

        cache_path = Path(sample["cache_path"])
        if not cache_path.is_absolute():
            cache_path = cache_dir / cache_path.name
        if not cache_path.exists():
            failures.append(_record_failure(row, sample, "cache_file_missing", str(cache_path)))
            continue

        source_path = Path(row.get("source_path") or sample.get("source_path", ""))
        if not source_path.is_absolute():
            source_path = resolve_path(source_path)
        if not source_path.exists():
            failures.append(_record_failure(row, sample, "source_file_missing", str(source_path)))
            continue

        expected_sha = str(row.get("source_sha256") or sample.get("source_sha256") or "").casefold()
        actual_sha = _sha256_file(source_path).casefold()
        checked_source_sha += 1
        if expected_sha and actual_sha != expected_sha:
            failures.append(_record_failure(row, sample, "source_sha256_mismatch", f"actual={actual_sha}"))

        try:
            with np.load(cache_path, allow_pickle=False) as data:
                checked_npz += 1
                required = ["byte_sequence", "pe_features", "stat_features", "lightweight_features", "label", "source_sha256"]
                missing = [name for name in required if name not in data.files]
                if missing:
                    failures.append(_record_failure(row, sample, "npz_missing_fields", ",".join(missing)))
                    continue

                label = int(np.asarray(data["label"]).item())
                expected_label = int(row.get("label", sample.get("label")))
                manifest_label = int(sample.get("label"))
                if label != expected_label or label != manifest_label:
                    failures.append(
                        _record_failure(
                            row,
                            sample,
                            "label_mismatch",
                            f"npz={label}; split={expected_label}; manifest={manifest_label}",
                        )
                    )

                npz_sha = _scalar_text(data["source_sha256"]).casefold()
                manifest_sha = str(sample.get("source_sha256", "")).casefold()
                if npz_sha != manifest_sha or npz_sha != actual_sha:
                    failures.append(
                        _record_failure(
                            row,
                            sample,
                            "cache_source_sha256_mismatch",
                            f"npz={npz_sha}; manifest={manifest_sha}; actual={actual_sha}",
                        )
                    )

                shape_checks = [
                    ("byte_sequence", data["byte_sequence"].shape, (expected_byte_len,)),
                    ("pe_features", data["pe_features"].shape, (expected_pe_dim,)),
                    ("stat_features", data["stat_features"].shape, (expected_stat_dim,)),
                    ("lightweight_features", data["lightweight_features"].shape, (expected_light_dim,)),
                ]
                for field, actual_shape, expected_shape in shape_checks:
                    if expected_shape[0] > 0 and tuple(actual_shape) != expected_shape:
                        failures.append(
                            _record_failure(
                                row,
                                sample,
                                "npz_shape_mismatch",
                                f"{field}: actual={tuple(actual_shape)} expected={expected_shape}",
                            )
                        )
        except Exception as exc:  # pragma: no cover - reported in JSON for operational audit.
            failures.append(_record_failure(row, sample, "npz_load_error", repr(exc)))

    failure_counts = Counter(item["reason"] for item in failures)
    pass_count = sample_size - len({(item["source_path"], item["sample_index"]) for item in failures})
    payload = {
        "schema": "axon_cache_random_sample_audit_v1",
        "split_csv": str(split_csv),
        "manifest": str(manifest_path),
        "split": split or "all",
        "seed": seed,
        "sample_fraction": sample_fraction,
        "sample_size": sample_size,
        "total_rows": len(rows),
        "checked_npz": checked_npz,
        "checked_source_sha256": checked_source_sha,
        "pass_count": pass_count,
        "failure_count": len(failures),
        "failed_sample_count": len({(item["source_path"], item["sample_index"]) for item in failures}),
        "failure_reason_counts": dict(sorted(failure_counts.items())),
        "manifest_match_counts": dict(sorted(match_counts.items())),
        "expected_shapes": {
            "byte_sequence": [expected_byte_len],
            "pe_features": [expected_pe_dim],
            "stat_features": [expected_stat_dim],
            "lightweight_features": [expected_light_dim],
        },
        "outputs": {
            "failures_csv": str(output_failures_csv),
            "sample_csv": str(output_sample_csv),
            "summary_json": str(output_json),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(
        output_failures_csv,
        failures,
        [
            "reason",
            "detail",
            "source_path",
            "source_sha256",
            "label",
            "split",
            "sample_index",
            "manifest_cache_path",
            "manifest_source_sha256",
            "manifest_label",
        ],
    )
    _write_csv(
        output_sample_csv,
        audited_rows,
        ["source_path", "label", "split", "sample_index", "match_reason", "manifest_cache_path"],
    )
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Randomly audit split rows against cache NPZ files and source SHA256.")
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="all")
    parser.add_argument("--sample-fraction", type=float, default=0.01)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-failures-csv", type=Path, required=True)
    parser.add_argument("--output-sample-csv", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = audit_random_sample(
        split_csv=args.split_csv,
        manifest_path=args.manifest,
        split=args.split,
        sample_fraction=args.sample_fraction,
        sample_size=args.sample_size,
        seed=args.seed,
        output_json=args.output_json,
        output_failures_csv=args.output_failures_csv,
        output_sample_csv=args.output_sample_csv,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"JSON: {resolve_path(args.output_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
