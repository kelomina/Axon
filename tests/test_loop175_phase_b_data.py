from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest

import src.loop175.phase_b_data as phase_b_data
from src.loop167_phase_b.contracts import canonical_json_bytes
from src.loop175.phase_b_data import (
    AlignedPhaseBData,
    IdentityFreePhaseBFitPayload,
    Loop175PhaseBDataError,
    RaggedRegionCache,
    length_bucket,
    load_aligned_phase_b_data,
    load_canonical_fold_manifest,
    load_canonical_train_prefix,
    load_ragged_region_cache,
    offset_bucket,
    save_ragged_region_cache,
    validate_ragged_region_cache,
)


def _sha256(content: bytes | str) -> str:
    material = content.encode("ascii") if isinstance(content, str) else content
    return hashlib.sha256(material).hexdigest()


@pytest.fixture
def small_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase_b_data, "FULL_TRAIN_ROWS", 10)
    monkeypatch.setattr(phase_b_data, "ROWS_PER_CLASS", 5)
    monkeypatch.setattr(phase_b_data, "ROWS_PER_FOLD", 2)


def _fold_record(ordinal: int) -> dict[str, object]:
    source_sha = _sha256(f"source-{ordinal}")
    return {
        "availability": "supported",
        "claim_scope": phase_b_data.FOLD_CLAIM_SCOPE,
        "content_component_id": f"{ordinal + 1:024x}",
        "content_component_size": 1,
        "diagnostic_fold": ordinal % phase_b_data.FOLD_COUNT,
        "identity_metadata_not_model_features": list(phase_b_data.IDENTITY_DECLARATION),
        "label": ordinal % 2,
        "loop_id": phase_b_data.FOLD_LOOP_ID,
        "missing_reason": None,
        "sample_index": ordinal,
        "schema": phase_b_data.FOLD_RECORD_SCHEMA,
        "source_path": f"C:\\sealed-train\\sample-{ordinal}.bin",
        "source_sha256": source_sha,
        "source_size_bytes": 16,
        "split_role": "train",
        "train_row_index": ordinal,
    }


def _write_fold_manifest(
    path: Path,
    *,
    mutate: Callable[[list[dict[str, object]]], object] | None = None,
) -> tuple[list[dict[str, object]], str]:
    records = [_fold_record(index) for index in range(phase_b_data.FULL_TRAIN_ROWS)]
    if mutate is not None:
        mutate(records)
    raw = b"".join(canonical_json_bytes(record) for record in records)
    path.write_bytes(raw)
    return records, _sha256(raw)


def _write_train_prefix(
    path: Path,
    records: list[dict[str, object]],
    *,
    trailing_heldout_bytes: bytes = b"",
) -> tuple[str, int]:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=phase_b_data.SPLIT_FIELDS)
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                "source_path": record["source_path"],
                "source_sha256": record["source_sha256"],
                "label": record["label"],
                "sample_index": record["sample_index"],
                "split": "train",
            }
        )
    raw = output.getvalue().encode("utf-8-sig")
    path.write_bytes(raw + trailing_heldout_bytes)
    return _sha256(raw), len(raw)


def _ragged_cache(rows: int, *, regions_per_row: int = 2) -> RaggedRegionCache:
    row_region_offsets = np.arange(
        0,
        rows * regions_per_row + 1,
        regions_per_row,
        dtype="<i8",
    )
    file_sizes = np.full(rows, 10, dtype="<i8")
    row_lengths = np.zeros(regions_per_row, dtype="<i8")
    row_lengths[0] = 4
    region_lengths = np.tile(row_lengths, rows)
    region_token_offsets = np.concatenate(
        [np.array([0], dtype="<i8"), np.cumsum(region_lengths, dtype="<i8")]
    )
    row_types = np.zeros(regions_per_row, dtype="u1")
    row_types[0] = 1
    region_types = np.tile(row_types, rows)
    region_starts = np.zeros(rows * regions_per_row, dtype="<i8")
    offset_buckets = np.zeros(rows * regions_per_row, dtype="u1")
    row_length_buckets = np.zeros(regions_per_row, dtype="u1")
    row_length_buckets[0] = length_bucket(4)
    length_buckets = np.tile(row_length_buckets, rows)
    return RaggedRegionCache(
        row_region_offsets=row_region_offsets,
        file_sizes=file_sizes,
        region_token_offsets=region_token_offsets,
        token_values=np.arange(rows * 4, dtype="u1"),
        region_types=region_types,
        region_starts=region_starts,
        offset_buckets=offset_buckets,
        length_buckets=length_buckets,
    )


def test_fold_and_train_prefix_are_strictly_sha_aligned_without_heldout_parse(
    tmp_path: Path,
    small_contract: None,
) -> None:
    fold_path = tmp_path / "folds.jsonl"
    split_path = tmp_path / "split.csv"
    records, fold_sha = _write_fold_manifest(fold_path)
    prefix_sha, prefix_bytes = _write_train_prefix(
        split_path,
        records,
        trailing_heldout_bytes=b"\xffthis-is-not-utf8-heldout-data",
    )

    folds = load_canonical_fold_manifest(fold_path, expected_sha256=fold_sha)
    train = load_canonical_train_prefix(
        split_path,
        expected_prefix_sha256=prefix_sha,
        expected_prefix_bytes=prefix_bytes,
    )

    assert folds.source_sha256 == train.source_sha256
    np.testing.assert_array_equal(folds.labels, train.labels)
    assert len(folds.sha_to_ordinal) == phase_b_data.FULL_TRAIN_ROWS
    assert not folds.labels.flags.writeable
    assert not hasattr(folds, "source_path")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda rows: rows[1].__setitem__("source_sha256", rows[0]["source_sha256"]),
            "repeats a source_sha256",
        ),
        (
            lambda rows: rows[1].__setitem__("train_row_index", 99),
            "row index",
        ),
        (
            lambda rows: (
                rows[0].update(content_component_id="f" * 24, content_component_size=2),
                rows[1].update(content_component_id="f" * 24, content_component_size=2),
            ),
            "crosses folds",
        ),
    ],
)
def test_fold_manifest_rejects_identity_and_component_tamper(
    tmp_path: Path,
    small_contract: None,
    mutate,
    message: str,
) -> None:
    path = tmp_path / "folds.jsonl"
    _records, digest = _write_fold_manifest(path, mutate=mutate)
    with pytest.raises(Loop175PhaseBDataError, match=message):
        load_canonical_fold_manifest(path, expected_sha256=digest)


def test_official_cache_loader_is_reused_and_fit_payload_has_no_identity(
    tmp_path: Path,
    small_contract: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fold_path = tmp_path / "folds.jsonl"
    split_path = tmp_path / "split.csv"
    records, fold_sha = _write_fold_manifest(fold_path)
    prefix_sha, prefix_bytes = _write_train_prefix(split_path, records)
    monkeypatch.setattr(phase_b_data, "CANONICAL_FOLD_SHA256", fold_sha)
    monkeypatch.setattr(phase_b_data, "CANONICAL_TRAIN_PREFIX_SHA256", prefix_sha)
    monkeypatch.setattr(phase_b_data, "CANONICAL_TRAIN_PREFIX_BYTES", prefix_bytes)
    b0_values = np.arange(10 * 571, dtype="<f4").reshape(10, 571)
    b0_missing = np.zeros((10, 6), dtype="<f4")
    b0_missing[3, 5] = 1
    verified = SimpleNamespace(
        cache_sha256="a" * 64,
        loaded_cache=SimpleNamespace(
            cache=SimpleNamespace(
                b0_values=b0_values,
                b0_missing_indicators=b0_missing,
            )
        ),
    )
    calls = []

    def load_official(root: Path):
        calls.append(root)
        return verified

    monkeypatch.setattr(phase_b_data, "load_verified_v12_cache_for_v13", load_official)
    aligned = load_aligned_phase_b_data(
        tmp_path,
        fold_manifest_path=fold_path,
        fold_manifest_sha256=fold_sha,
        canonical_split_path=split_path,
        train_prefix_sha256=prefix_sha,
        train_prefix_bytes=prefix_bytes,
    )
    payload = aligned.make_fit_payload(_ragged_cache(10, regions_per_row=16))

    assert calls == [tmp_path.resolve()]
    assert aligned.b0_values.shape == (10, 571)
    assert aligned.b0_missing_counts == (0, 0, 0, 0, 0, 1)
    assert tuple(field.name for field in fields(IdentityFreePhaseBFitPayload)) == (
        "b0_values",
        "labels",
        "folds",
        "regions",
    )
    assert not any(
        forbidden in field.name
        for field in fields(payload)
        for forbidden in ("sha", "path", "component", "source", "identity")
    )
    assert not payload.b0_values.flags.writeable


def test_bucket_formulas_and_ranges_fail_closed() -> None:
    assert offset_bucket(0, 0) == 0
    assert offset_bucket(0, 1) == 0
    assert offset_bucket(9, 10) == 63
    assert offset_bucket(5, 11) == 31
    assert length_bucket(0) == 0
    assert length_bucket(1) == 1
    assert length_bucket(8192) == 63
    with pytest.raises(Loop175PhaseBDataError, match="outside"):
        offset_bucket(10, 10)
    with pytest.raises(Loop175PhaseBDataError, match="exceeds"):
        length_bucket(8193)


def test_ragged_region_cache_roundtrip_is_identity_free_and_sha_bound(tmp_path: Path) -> None:
    cache = _ragged_cache(3)
    output = tmp_path / "regions.npz"
    receipt = save_ragged_region_cache(output, cache)
    loaded = load_ragged_region_cache(output, expected_sha256=receipt.sha256, expected_rows=3)

    for field in fields(RaggedRegionCache):
        np.testing.assert_array_equal(getattr(loaded, field.name), getattr(cache, field.name))
        assert not getattr(loaded, field.name).flags.writeable
    with zipfile.ZipFile(output) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        assert metadata["schema"] == phase_b_data.RAGGED_REGION_CACHE_SCHEMA
        assert "source_sha256" not in json.dumps(metadata)
        assert "source_path" not in json.dumps(metadata)
    with pytest.raises(Loop175PhaseBDataError, match="SHA-256"):
        load_ragged_region_cache(output, expected_sha256="0" * 64)


def test_ragged_region_cache_rejects_bucket_and_range_tamper() -> None:
    cache = _ragged_cache(2)
    bad_bucket = cache.offset_buckets.copy()
    bad_bucket[0] = 1
    with pytest.raises(Loop175PhaseBDataError, match="offset bucket"):
        validate_ragged_region_cache(replace(cache, offset_buckets=bad_bucket))

    bad_start = cache.region_starts.copy()
    bad_start[0] = 9
    with pytest.raises(Loop175PhaseBDataError, match="outside"):
        validate_ragged_region_cache(replace(cache, region_starts=bad_start))


def test_production_fit_rejects_less_than_sixteen_regions_per_row(
    small_contract: None,
) -> None:
    aligned = AlignedPhaseBData(
        labels=np.resize(np.array([0, 1], dtype="u1"), 10),
        folds=np.repeat(np.arange(5, dtype="i1"), 2),
        component_ids=tuple(f"{index + 1:024x}" for index in range(10)),
        source_sha256=tuple(_sha256(f"source-{index}") for index in range(10)),
        b0_values=np.zeros((10, 571), dtype="<f4"),
        b0_missing_counts=(0,) * 6,
        fold_manifest_sha256="e" * 64,
        train_prefix_sha256="f" * 64,
        b0_cache_sha256="0" * 64,
    )
    with pytest.raises(Loop175PhaseBDataError, match="exactly 16"):
        aligned.make_fit_payload(_ragged_cache(10, regions_per_row=15))


def test_production_train_prefix_constants_bind_current_artifact() -> None:
    project_root = Path(__file__).resolve().parents[1]
    train = load_canonical_train_prefix(
        project_root / phase_b_data.CANONICAL_SPLIT_RELATIVE_PATH,
        expected_prefix_sha256=phase_b_data.CANONICAL_TRAIN_PREFIX_SHA256,
        expected_prefix_bytes=phase_b_data.CANONICAL_TRAIN_PREFIX_BYTES,
    )
    assert train.prefix_bytes == 4_120_895
    assert train.prefix_sha256 == "dfbad6994605aa0fd9b7fa049b19cd87f15e50e37490a60efc43696c540dd54a"
    assert len(train.source_sha256) == 20_000


def test_aligned_control_plane_can_hold_identity_but_payload_cannot() -> None:
    rows = 2
    aligned = AlignedPhaseBData(
        labels=np.array([0, 1], dtype="u1"),
        folds=np.array([0, 1], dtype="i1"),
        component_ids=("a" * 24, "b" * 24),
        source_sha256=("c" * 64, "d" * 64),
        b0_values=np.zeros((rows, 571), dtype="<f4"),
        b0_missing_counts=(0,) * 6,
        fold_manifest_sha256="e" * 64,
        train_prefix_sha256="f" * 64,
        b0_cache_sha256="0" * 64,
    )
    assert aligned.source_sha256[0] == "c" * 64
