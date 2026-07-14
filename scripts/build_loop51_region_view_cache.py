#!/usr/bin/env python3
"""Build a fixed-length semantic-region byte-view feature cache.

Paths and hashes are used only for loading, joining, and audit. The generated
`byte_sequence` is built from file bytes selected by PE/content-derived offsets.
No filename, extension, directory, path text, sample id, split, or row order is
encoded into the tensor.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for item in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dataset import _load_cached_feature_npz  # noqa: E402
from train_loop44_region_byte_ngram import REGION_NAMES, _read_region, region_slices_from_path  # noqa: E402


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def read_split_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"Manifest has no samples list: {path}")
    return samples


def split_filter_set(split_text: str) -> set[str]:
    return {item.strip() for item in split_text.split(",") if item.strip()}


def region_slot_sizes(byte_length: int, region_names: Sequence[str]) -> dict[str, int]:
    if byte_length <= 0:
        raise ValueError("byte_length must be positive")
    if not region_names:
        raise ValueError("region_names must not be empty")
    base = byte_length // len(region_names)
    remainder = byte_length % len(region_names)
    return {
        region_name: base + (1 if index < remainder else 0)
        for index, region_name in enumerate(region_names)
    }


def build_region_view_sequence(
    payloads: Iterable[tuple[str, bytes]],
    *,
    byte_length: int,
    region_names: Sequence[str] = REGION_NAMES,
) -> tuple[np.ndarray, dict[str, int]]:
    """Pack region payloads into fixed slots and return the byte sequence."""

    slots = region_slot_sizes(byte_length, region_names)
    payload_by_name = {name: data for name, data in payloads if data}
    output = np.zeros(byte_length, dtype=np.uint8)
    offset = 0
    copied_lengths: dict[str, int] = {}
    for region_name in region_names:
        slot_size = slots[region_name]
        payload = payload_by_name.get(region_name, b"")
        copied = min(len(payload), slot_size)
        if copied:
            output[offset : offset + copied] = np.frombuffer(payload[:copied], dtype=np.uint8)
        copied_lengths[region_name] = copied
        offset += slot_size
    return output, copied_lengths


def payloads_for_source(source_path: Path, *, region_window: int, tail_window: int) -> list[tuple[str, bytes]]:
    payloads: list[tuple[str, bytes]] = []
    for region in region_slices_from_path(source_path, region_window=region_window, tail_window=tail_window):
        data = _read_region(source_path, region.start, region.size)
        if data:
            payloads.append((region.name, data))
    return payloads


def _sample_for_output(
    *,
    source_path_text: str,
    output_cache_path: Path,
    label: int,
    source_sha256: str,
) -> dict[str, Any]:
    return {
        "source_path": source_path_text,
        "cache_path": str(output_cache_path),
        "label": int(label),
        "source_sha256": source_sha256.lower(),
    }


def _process_region_cache_row(task: dict[str, Any]) -> dict[str, Any]:
    source_path_text = task["source_path"]
    label_text = str(task["label"])
    source_sha256 = str(task["source_sha256"]).lower()
    source_cache_path = Path(task["source_cache_path"])
    output_cache_path = Path(task["output_cache_path"])
    source_path = resolve_path(source_path_text)
    byte_length = int(task["byte_length"])
    region_window = int(task["region_window"])
    tail_window = int(task["tail_window"])
    skip_existing = bool(task["skip_existing"])

    if skip_existing and output_cache_path.exists():
        try:
            _, _, _, _, label = _load_cached_feature_npz(
                output_cache_path,
                byte_length,
                256,
                49,
                256,
                expected_label=int(label_text),
                expected_source_sha256=source_sha256 or None,
                allow_missing_source_sha256=False,
            )
            return {
                "ok": True,
                "skipped_existing": True,
                "sample": _sample_for_output(
                    source_path_text=source_path_text,
                    output_cache_path=output_cache_path,
                    label=label,
                    source_sha256=source_sha256,
                ),
                "region_copied_lengths": {},
                "issue": None,
            }
        except Exception:
            pass

    try:
        _, pe_feat, stat_feat, lightweight_feat, label = _load_cached_feature_npz(
            source_cache_path,
            byte_length,
            256,
            49,
            256,
            expected_label=int(label_text),
            expected_source_sha256=source_sha256 or None,
            allow_missing_source_sha256=False,
        )
    except Exception as exc:
        return {"ok": False, "issue": f"source_cache_load_failed:{type(exc).__name__}"}

    if not source_path.exists():
        return {"ok": False, "issue": "source_missing"}

    payloads = payloads_for_source(source_path, region_window=region_window, tail_window=tail_window)
    issue = None if payloads else "no_regions"
    region_view, copied_lengths = build_region_view_sequence(
        payloads,
        byte_length=byte_length,
        region_names=REGION_NAMES,
    )

    output_cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_cache_path.with_suffix(output_cache_path.suffix + ".tmp.npz")
    np.savez(
        tmp_path,
        byte_sequence=region_view,
        pe_features=pe_feat.astype(np.float32, copy=False),
        stat_features=stat_feat.astype(np.float32, copy=False),
        lightweight_features=lightweight_feat.astype(np.float32, copy=False),
        label=np.array(label, dtype=np.int64),
        source_sha256=np.array(source_sha256),
    )
    tmp_path.replace(output_cache_path)
    return {
        "ok": True,
        "skipped_existing": False,
        "sample": _sample_for_output(
            source_path_text=source_path_text,
            output_cache_path=output_cache_path,
            label=label,
            source_sha256=source_sha256,
        ),
        "region_copied_lengths": copied_lengths,
        "issue": issue,
    }


def write_region_cache(
    *,
    split_csv: Path,
    source_manifest: Path,
    output_cache_dir: Path,
    splits: set[str],
    byte_length: int,
    region_window: int,
    tail_window: int,
    limit: int | None,
    workers: int,
    skip_existing: bool,
    output_json: Path,
) -> dict[str, Any]:
    split_rows = read_split_rows(split_csv)
    split_by_key = {(row["source_path"], str(row["label"])): row for row in split_rows if row.get("split") in splits}
    manifest_samples = load_manifest(source_manifest)
    manifest_by_key = {
        (str(sample.get("source_path", "")), str(sample.get("label", ""))): sample
        for sample in manifest_samples
    }

    selected_keys = list(split_by_key)
    if limit is not None:
        selected_keys = selected_keys[:limit]

    output_cache_dir.mkdir(parents=True, exist_ok=True)
    output_samples: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    skipped_existing_count = 0
    region_present_counts: Counter[str] = Counter()
    total_region_bytes: Counter[str] = Counter()
    started = time.time()

    tasks = []
    task_meta = []
    for key in selected_keys:
        split_row = split_by_key[key]
        source_path_text, label_text = key
        manifest_sample = manifest_by_key.get(key)
        if manifest_sample is None:
            issue_counts["missing_manifest_row"] += 1
            continue
        source_cache_path = resolve_path(str(manifest_sample.get("cache_path", "")))
        output_cache_path = output_cache_dir / source_cache_path.name
        tasks.append(
            {
                "source_path": source_path_text,
                "label": label_text,
                "source_sha256": str(manifest_sample.get("source_sha256", "")).lower(),
                "source_cache_path": str(source_cache_path),
                "output_cache_path": str(output_cache_path),
                "byte_length": byte_length,
                "region_window": region_window,
                "tail_window": tail_window,
                "skip_existing": skip_existing,
            }
        )
        task_meta.append(
            {
                "split": split_row.get("split", ""),
                "label": label_text,
            }
        )

    def consume_result(result: dict[str, Any], meta: dict[str, str]) -> None:
        nonlocal skipped_existing_count
        if not result.get("ok"):
            issue = str(result.get("issue") or "unknown_failed")
            issue_counts[issue.split(":", 1)[0]] += 1
            return
        if result.get("issue"):
            issue_counts[str(result["issue"]).split(":", 1)[0]] += 1
        if result.get("skipped_existing"):
            skipped_existing_count += 1
        output_samples.append(result["sample"])
        split_counts[meta["split"]] += 1
        label_counts[str(result["sample"]["label"])] += 1
        for region_name, copied in result.get("region_copied_lengths", {}).items():
            if int(copied) > 0:
                region_present_counts[region_name] += 1
                total_region_bytes[region_name] += int(copied)

    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            consume_result(_process_region_cache_row(task), task_meta[index - 1])
            if index % 1000 == 0:
                print(f"[loop51] processed {index}/{len(tasks)} rows", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_meta = {
                executor.submit(_process_region_cache_row, task): task_meta[index]
                for index, task in enumerate(tasks)
            }
            for index, future in enumerate(as_completed(future_to_meta), start=1):
                consume_result(future.result(), future_to_meta[future])
                if index % 1000 == 0:
                    elapsed = time.time() - started
                    print(f"[loop51] processed {index}/{len(tasks)} rows in {elapsed:.1f}s", flush=True)

    output_manifest = output_cache_dir / source_manifest.name
    output_manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "schema": "axon_loop51_region_view_cache_v1",
                "source_manifest": str(source_manifest),
                "split_csv": str(split_csv),
                "byte_length": byte_length,
                "region_names": list(REGION_NAMES),
                "region_slot_sizes": region_slot_sizes(byte_length, REGION_NAMES),
                "region_window": region_window,
                "tail_window": tail_window,
                "identity_policy": (
                    "source_path/cache_path/source_sha256/split/sample_index are loading and audit metadata only"
                ),
                "samples": output_samples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = {
        "schema": "axon_loop51_region_view_cache_build_v1",
        "split_csv": str(split_csv),
        "source_manifest": str(source_manifest),
        "output_cache_dir": str(output_cache_dir),
        "output_manifest": str(output_manifest),
        "requested_splits": sorted(splits),
        "requested_rows": len(selected_keys),
        "written_rows": len(output_samples),
        "skipped_existing_rows": skipped_existing_count,
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "byte_length": byte_length,
        "region_names": list(REGION_NAMES),
        "region_slot_sizes": region_slot_sizes(byte_length, REGION_NAMES),
        "region_present_counts": dict(sorted(region_present_counts.items())),
        "total_region_bytes": dict(sorted(total_region_bytes.items())),
        "identity_policy": (
            "Paths, hashes, sample ids, split names, and row order are not encoded into byte_sequence."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", default="reports/random_20w_split/loop27_corrected_split.csv")
    parser.add_argument("--source-manifest", default="data/.cache/manifest_38672ba0.json")
    parser.add_argument("--output-cache-dir", default="data/.cache_loop51_region_view_8192")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--byte-length", type=int, default=8192)
    parser.add_argument("--region-window", type=int, default=1024)
    parser.add_argument("--tail-window", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-skip-existing", action="store_true", default=False)
    parser.add_argument(
        "--output-json",
        default="reports/random_20w_split/loop51_region_view_cache_train_val_audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = write_region_cache(
        split_csv=resolve_path(args.split_csv),
        source_manifest=resolve_path(args.source_manifest),
        output_cache_dir=resolve_path(args.output_cache_dir),
        splits=split_filter_set(args.splits),
        byte_length=args.byte_length,
        region_window=args.region_window,
        tail_window=args.tail_window,
        limit=args.limit,
        workers=max(1, int(args.workers)),
        skip_existing=not bool(args.no_skip_existing),
        output_json=resolve_path(args.output_json),
    )
    print(
        "[loop51] written_rows=",
        report["written_rows"],
        "split_counts=",
        report["split_counts"],
        "issue_counts=",
        report["issue_counts"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
