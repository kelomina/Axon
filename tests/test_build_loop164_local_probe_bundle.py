from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_loop164_local_probe_bundle import (  # noqa: E402
    BUNDLE_RECORD_SCHEMA,
    MAX_SOURCE_SIZE_BYTES,
    MIN_SOURCE_SIZE_BYTES,
    build_local_probe_bundle,
    sha256_file,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write_sized_file(path: Path, size_bytes: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size_bytes)
    return path


def write_split(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_path", "source_sha256", "label", "sample_index", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "samples": [
            {
                "source_path": row["source_path"],
                "source_sha256": row["source_sha256"],
                "label": int(row["label"]),
                "cache_path": f"data/.cache/{row['source_sha256'][:16]}.npz",
            }
            for row in rows
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_case(tmp_path: Path) -> dict[str, Path | list[dict[str, str]]]:
    data_root = tmp_path / "data"
    split_rows: list[dict[str, str]] = []
    for label, directory in ((0, "benign"), (1, "malicious")):
        for index in range(4):
            source = write_sized_file(
                data_root / directory / f"train-{index}.bin",
                MIN_SOURCE_SIZE_BYTES + index + label,
            )
            split_rows.append(
                {
                    "source_path": str(source),
                    "source_sha256": digest(f"train-{label}-{index}"),
                    "label": str(label),
                    "sample_index": str(index),
                    "split": "train",
                }
            )
        heldout = tmp_path / "heldout" / directory / "heldout.bin"
        split_rows.append(
            {
                "source_path": str(heldout),
                "source_sha256": digest(f"heldout-{label}"),
                "label": str(label),
                "sample_index": "100",
                "split": "val" if label == 0 else "test",
            }
        )
    split_path = tmp_path / "split.csv"
    manifest_path = tmp_path / "manifest.json"
    write_split(split_path, list(reversed(split_rows)))
    write_manifest(manifest_path, list(reversed(split_rows)))
    return {
        "data_root": data_root,
        "split": split_path,
        "manifest": manifest_path,
        "rows": split_rows,
    }


def test_builds_deterministic_sha_bound_train_only_bundle_without_reading_source_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case = build_case(tmp_path)
    raw_paths = {
        Path(row["source_path"])
        for row in case["rows"]
        if row["split"] == "train"
    }
    original_open = Path.open

    def forbid_raw_content_open(path: Path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if path in raw_paths and "r" in mode:
            raise AssertionError("builder must not open raw source content")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", forbid_raw_content_open)
    bundle_a = tmp_path / "bundle-a.jsonl"
    summary_a = tmp_path / "summary-a.json"
    summary = build_local_probe_bundle(
        split_csv=case["split"],
        cache_manifest=case["manifest"],
        data_root=case["data_root"],
        bundle_output=bundle_a,
        summary_output=summary_a,
        records_per_class=2,
    )

    records = [json.loads(line) for line in bundle_a.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert {record["label"] for record in records} == {0, 1}
    assert {record["split_role"] for record in records} == {"train"}
    assert all(record["schema"] == BUNDLE_RECORD_SCHEMA for record in records)
    assert all(record["source_path_usage"] == "loader_identity_only_not_model_feature" for record in records)
    assert all(record["source_sha256_usage"] == "integrity_binding_only_not_model_feature" for record in records)
    assert all(record["source_size_bytes"] >= MIN_SOURCE_SIZE_BYTES for record in records)
    assert summary["bundle"]["sha256"] == sha256_file(bundle_a)
    assert summary["input_bindings"]["canonical_split_csv"]["sha256"] == sha256_file(case["split"])
    assert summary["input_bindings"]["primary_cache_manifest"]["sha256"] == sha256_file(case["manifest"])
    assert summary["selection"]["source_content_opened"] is False
    assert summary["ready_for"]["val_or_test_access"] is False
    assert not any("heldout" in record["source_path"] for record in records)

    bundle_b = tmp_path / "bundle-b.jsonl"
    summary_b = tmp_path / "summary-b.json"
    build_local_probe_bundle(
        split_csv=case["split"],
        cache_manifest=case["manifest"],
        data_root=case["data_root"],
        bundle_output=bundle_b,
        summary_output=summary_b,
        records_per_class=2,
    )
    assert bundle_a.read_bytes() == bundle_b.read_bytes()


def test_does_not_stat_or_open_heldout_rows(tmp_path: Path):
    case = build_case(tmp_path)
    summary = build_local_probe_bundle(
        split_csv=case["split"],
        cache_manifest=case["manifest"],
        data_root=case["data_root"],
        bundle_output=tmp_path / "bundle.jsonl",
        summary_output=tmp_path / "summary.json",
        records_per_class=1,
    )

    assert summary["aggregate_counts"]["split_rows_by_role"] == {"test": 1, "train": 8, "val": 1}
    assert summary["aggregate_counts"]["selected_rows_by_label"] == {"0": 1, "1": 1}


def test_rejects_insufficient_or_manifest_mismatched_train_candidates_without_outputs(tmp_path: Path):
    case = build_case(tmp_path)
    manifest_payload = json.loads(Path(case["manifest"]).read_text(encoding="utf-8"))
    manifest_payload["samples"] = [
        sample
        for sample in manifest_payload["samples"]
        if sample["source_sha256"] != digest("train-1-3")
    ]
    Path(case["manifest"]).write_text(json.dumps(manifest_payload), encoding="utf-8")
    bundle_output = tmp_path / "bundle.jsonl"
    summary_output = tmp_path / "summary.json"

    with pytest.raises(ValueError, match="Insufficient eligible canonical-train"):
        build_local_probe_bundle(
            split_csv=case["split"],
            cache_manifest=case["manifest"],
            data_root=case["data_root"],
            bundle_output=bundle_output,
            summary_output=summary_output,
            records_per_class=4,
        )

    assert not bundle_output.exists()
    assert not summary_output.exists()


def test_filters_size_outliers_using_stat_metadata_only(tmp_path: Path):
    case = build_case(tmp_path)
    train_rows = [row for row in case["rows"] if row["split"] == "train" and row["label"] == "0"]
    Path(train_rows[0]["source_path"]).write_bytes(b"x")
    with Path(train_rows[1]["source_path"]).open("r+b") as handle:
        handle.truncate(MAX_SOURCE_SIZE_BYTES + 1)

    summary = build_local_probe_bundle(
        split_csv=case["split"],
        cache_manifest=case["manifest"],
        data_root=case["data_root"],
        bundle_output=tmp_path / "bundle.jsonl",
        summary_output=tmp_path / "summary.json",
        records_per_class=2,
    )

    assert summary["aggregate_counts"]["rejected_train_rows_by_reason"] == {
        "source_size_above_max": 1,
        "source_size_below_min": 1,
    }


def test_maps_stale_canonical_paths_into_materialized_worktree(tmp_path: Path):
    canonical_root = tmp_path / "canonical"
    worktree_root = tmp_path / "worktree"
    canonical_root.mkdir()
    rows = []
    for label, directory in ((0, "benign"), (1, "malicious")):
        canonical_path = canonical_root / directory / f"sample-{label}.bin"
        materialized_path = write_sized_file(
            worktree_root / directory / f"sample-{label}.bin",
            MIN_SOURCE_SIZE_BYTES + label,
        )
        rows.append(
            {
                "source_path": str(canonical_path),
                "source_sha256": digest(f"mapped-{label}"),
                "label": str(label),
                "sample_index": str(label),
                "split": "train",
            }
        )
        assert not canonical_path.exists()
        assert materialized_path.exists()
    split_path = tmp_path / "split.csv"
    manifest_path = tmp_path / "manifest.json"
    write_split(split_path, rows)
    write_manifest(manifest_path, rows)

    summary = build_local_probe_bundle(
        split_csv=split_path,
        cache_manifest=manifest_path,
        canonical_source_root=canonical_root,
        data_root=worktree_root,
        bundle_output=tmp_path / "bundle.jsonl",
        summary_output=tmp_path / "summary.json",
        records_per_class=1,
    )

    assert summary["aggregate_counts"]["selected_rows_by_label"] == {"0": 1, "1": 1}
    records = [
        json.loads(line)
        for line in (tmp_path / "bundle.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(Path(record["source_path"]).is_relative_to(worktree_root) for record in records)


def test_ignores_cross_label_manifest_sha_outside_canonical_train(tmp_path: Path):
    case = build_case(tmp_path)
    payload = json.loads(Path(case["manifest"]).read_text(encoding="utf-8"))
    unrelated_sha = digest("unrelated-cross-label")
    payload["samples"].extend(
        [
            {"source_sha256": unrelated_sha, "label": 0, "cache_path": "a.npz"},
            {"source_sha256": unrelated_sha, "label": 1, "cache_path": "b.npz"},
        ]
    )
    Path(case["manifest"]).write_text(json.dumps(payload), encoding="utf-8")

    summary = build_local_probe_bundle(
        split_csv=case["split"],
        cache_manifest=case["manifest"],
        data_root=case["data_root"],
        bundle_output=tmp_path / "bundle.jsonl",
        summary_output=tmp_path / "summary.json",
        records_per_class=1,
    )

    assert summary["aggregate_counts"]["manifest_cross_label_source_sha256"] == 1


def test_rejects_cross_label_manifest_sha_used_by_canonical_train(tmp_path: Path):
    case = build_case(tmp_path)
    payload = json.loads(Path(case["manifest"]).read_text(encoding="utf-8"))
    train_row = next(row for row in case["rows"] if row["split"] == "train")
    payload["samples"].append(
        {
            "source_sha256": train_row["source_sha256"],
            "label": 1 - int(train_row["label"]),
            "cache_path": "conflict.npz",
        }
    )
    Path(case["manifest"]).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cross-label canonical-train"):
        build_local_probe_bundle(
            split_csv=case["split"],
            cache_manifest=case["manifest"],
            data_root=case["data_root"],
            bundle_output=tmp_path / "bundle.jsonl",
            summary_output=tmp_path / "summary.json",
            records_per_class=1,
        )
