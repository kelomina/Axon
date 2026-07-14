from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest

import src.loop167_phase_b.feature_cache_v4 as feature_cache_v4
import src.loop167_phase_b.raw_manifest_adapter_v4 as manifest_adapter_v4
import src.loop167_phase_b.raw_worker as raw_worker
from src.loop167_phase_b.contracts import canonical_json_bytes
from src.loop167_phase_b.feature_cache_v4 import (
    ARCHIVE_NAMES,
    CACHE_SCHEMA,
    B1SamplingAudit,
    FeatureCacheV4Error,
    FeatureCacheWriteReceipt,
    load_phase_b_feature_cache_v4,
    write_phase_b_feature_cache_v4,
)
from src.loop167_phase_b.progress_ledger import (
    FEATURE_ROW_GENESIS_SHA256,
    _next_feature_rows_commitment,
)
from src.loop167_phase_b.raw_manifest_adapter_v4 import (
    FOLD_RECORD_SCHEMA,
    FULL_TRAIN_ROWS,
    RawManifestAdapterV4Error,
    load_train_only_manifest_v4,
)
from src.loop167_phase_b.raw_worker import RawPlanEntry, RawScanOutcome


def _sha256(value: bytes | str) -> str:
    material = value.encode("ascii") if isinstance(value, str) else value
    return hashlib.sha256(material).hexdigest()


def _record(ordinal: int, source_path: Path) -> dict[str, object]:
    return {
        "schema": FOLD_RECORD_SCHEMA,
        "loop_id": "loop164_whole_file_residual_expert",
        "claim_scope": "local_train_content_similarity_diagnostic_not_family_or_time_isolation",
        "split_role": "train",
        "train_row_index": ordinal,
        "sample_index": ordinal,
        "source_path": str(source_path),
        "source_sha256": _sha256(f"synthetic-source-{ordinal}"),
        "source_size_bytes": None,
        "label": ordinal % 2,
        "availability": "read_failure",
        "missing_reason": "read_failure",
        "content_component_id": f"{ordinal:024x}",
        "content_component_size": 1,
        "diagnostic_fold": ordinal // 4_000,
        "identity_metadata_not_model_features": [
            "train_row_index",
            "sample_index",
            "source_path",
            "source_sha256",
            "content_component_id",
            "diagnostic_fold",
        ],
    }


def _write_synthetic_authority(
    tmp_path: Path,
    *,
    mutate_first_record: Callable[[dict[str, object]], None] | None = None,
    manifest_path_text: str = "manifests/train_fold_authority.jsonl",
) -> tuple[Path, Path, dict[str, str]]:
    project_root = tmp_path / "project"
    data_root = tmp_path / "train-raw"
    manifest_path = project_root / "manifests" / "train_fold_authority.jsonl"
    project_root.mkdir()
    data_root.mkdir()
    manifest_path.parent.mkdir()

    lines: list[bytes] = []
    for ordinal in range(FULL_TRAIN_ROWS):
        record = _record(ordinal, data_root / f"missing-{ordinal:05d}.bin")
        if ordinal == 0 and mutate_first_record is not None:
            mutate_first_record(record)
        lines.append(canonical_json_bytes(record)[:-1])
    manifest_raw = b"\n".join(lines) + b"\n"
    manifest_path.write_bytes(manifest_raw)

    protocol = {
        "schema": "axon_loop167_phase_b_protocol_v1",
        "loop_id": "loop167_ember_v3_novel_delta",
        "status": "synthetic_static_only",
        "claim_scope": "synthetic_train_only_adapter_test",
        "phase_a_bindings": {},
        "input_contract": {
            "folds": {
                "path": manifest_path_text,
                "sha256": _sha256(manifest_raw),
                "record_schema": FOLD_RECORD_SCHEMA,
                "split_role": "train",
                "rows": FULL_TRAIN_ROWS,
                "folds": 5,
                "rows_per_fold": 4_000,
                "val_test_or_full_access": False,
            },
            "scope_drift_is_fatal": True,
            "source_sha256_verified_in_same_stream": True,
        },
        "feature_contract": {},
        "fit_contract": {},
        "evaluation_contract": {},
        "resource_contract": {},
        "runtime_contract": {},
        "one_shot_lease": {},
        "forbidden": [],
        "ready_for": {},
    }
    protocol_path = project_root / "manifests" / "phase_b_protocol.json"
    protocol_raw = canonical_json_bytes(protocol)
    protocol_path.write_bytes(protocol_raw)
    return project_root, data_root, {
        "path": "manifests/phase_b_protocol.json",
        "sha256": _sha256(protocol_raw),
    }


def _forbid_plan_build(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_plan_build(cls: type[object], entries: object) -> object:
        del cls, entries
        raise AssertionError("Invalid manifest reached RawScanPlan construction")

    monkeypatch.setattr(
        manifest_adapter_v4.RawScanPlan,
        "from_entries",
        classmethod(unexpected_plan_build),
    )


def test_manifest_adapter_accepts_only_the_exact_synthetic_20k_train_authority(tmp_path: Path) -> None:
    project_root, data_root, binding = _write_synthetic_authority(tmp_path)

    result = load_train_only_manifest_v4(
        project_root,
        phase_b_protocol_binding=binding,
        data_root=data_root,
    )

    assert len(result.raw_scan_plan.entries) == FULL_TRAIN_ROWS
    assert result.fit_targets.labels.shape == (FULL_TRAIN_ROWS,)
    assert result.fit_targets.folds.shape == (FULL_TRAIN_ROWS,)
    assert np.bincount(result.fit_targets.labels, minlength=2).tolist() == [10_000, 10_000]
    assert np.bincount(result.fit_targets.folds, minlength=5).tolist() == [4_000] * 5
    assert result.fit_targets.labels.flags.writeable is False
    assert result.fit_targets.folds.flags.writeable is False
    assert tuple(field.name for field in fields(RawPlanEntry)) == (
        "ordinal",
        "source_file",
        "source_audit_sha256",
        "declared_size",
        "expected_sha256",
    )
    assert not {"label", "fold", "diagnostic_fold", "content_component_id"}.intersection(
        field.name for field in fields(RawPlanEntry)
    )
    assert result.raw_scan_plan.entries[0].source_file == data_root / "missing-00000.bin"


def test_manifest_adapter_rejects_heldout_record_before_raw_plan_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root, data_root, binding = _write_synthetic_authority(
        tmp_path,
        mutate_first_record=lambda record: record.__setitem__("split_role", "val"),
    )
    _forbid_plan_build(monkeypatch)

    with pytest.raises(RawManifestAdapterV4Error, match="Train-only"):
        load_train_only_manifest_v4(
            project_root,
            phase_b_protocol_binding=binding,
            data_root=data_root,
        )


def test_manifest_adapter_rejects_source_path_escape_before_raw_plan_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside_source = tmp_path / "outside" / "missing.bin"
    project_root, data_root, binding = _write_synthetic_authority(
        tmp_path,
        mutate_first_record=lambda record: record.__setitem__("source_path", str(outside_source)),
    )
    _forbid_plan_build(monkeypatch)

    with pytest.raises(RawManifestAdapterV4Error, match="outside"):
        load_train_only_manifest_v4(
            project_root,
            phase_b_protocol_binding=binding,
            data_root=data_root,
        )


def test_manifest_adapter_rejects_source_ancestor_symlink_before_raw_plan_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked_source = tmp_path / "train-raw" / "linked" / "missing.bin"
    project_root, data_root, binding = _write_synthetic_authority(
        tmp_path,
        mutate_first_record=lambda record: record.__setitem__("source_path", str(linked_source)),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (data_root / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        linked_directory = data_root / "linked"
        linked_directory.mkdir()
        original_lstat = manifest_adapter_v4.os.lstat

        def reparse_lstat(path: str | bytes | os.PathLike[str]):
            result = original_lstat(path)
            if Path(path) == linked_directory:
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x0400)
            return result

        monkeypatch.setattr(manifest_adapter_v4.os, "lstat", reparse_lstat)
    _forbid_plan_build(monkeypatch)

    with pytest.raises(RawManifestAdapterV4Error, match="symlink"):
        load_train_only_manifest_v4(
            project_root,
            phase_b_protocol_binding=binding,
            data_root=data_root,
        )


def test_manifest_adapter_rejects_noncanonical_manifest_path_before_raw_plan_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root, data_root, binding = _write_synthetic_authority(
        tmp_path,
        manifest_path_text="manifests/../manifests/train_fold_authority.jsonl",
    )
    _forbid_plan_build(monkeypatch)

    with pytest.raises(RawManifestAdapterV4Error, match="manifest path"):
        load_train_only_manifest_v4(
            project_root,
            phase_b_protocol_binding=binding,
            data_root=data_root,
        )


def _synthetic_outcome(tmp_path: Path, *, rows: int) -> RawScanOutcome:
    raw_scope = _sha256("synthetic-raw-scope")
    feature_rows_commitment = FEATURE_ROW_GENESIS_SHA256
    raw_rows = []
    for ordinal in range(rows):
        entry = RawPlanEntry(
            ordinal=ordinal,
            source_file=tmp_path / f"synthetic-{ordinal}.bin",
            source_audit_sha256=_sha256(f"synthetic-source-audit-{ordinal}"),
            declared_size=ordinal + 1,
            expected_sha256=_sha256(f"synthetic-source-bytes-{ordinal}"),
        )
        row = raw_worker._feature_row(
            entry,
            result="available",
            b0_values=np.full(571, ordinal + 1, dtype=np.float32),
            b0_missing_indicators=np.zeros(6, dtype=np.float32),
            b1_values=np.full(536, ordinal + 2, dtype=np.float32),
            b1_missing_indicators=np.zeros(4, dtype=np.float32),
            b1_sampling_indicators=np.array(
                [1.0, float(ordinal % 2), float(ordinal == rows - 1)], dtype=np.float32
            ),
            novel_values=np.full(292, ordinal + 3, dtype=np.float32),
            novel_missing_indicators=np.zeros(1, dtype=np.float32),
            novel_complete=ordinal % 2 == 0,
        )
        raw_rows.append(row)
        feature_rows_commitment = _next_feature_rows_commitment(
            feature_rows_commitment,
            ordinal=ordinal,
            source_audit_sha256=row.source_audit_sha256,
            feature_row_commitment_sha256=row.feature_row_commitment_sha256,
        )
    return RawScanOutcome(
        rows=tuple(raw_rows),
        raw_scope_commitment_sha256=raw_scope,
        feature_rows_commitment_sha256=feature_rows_commitment,
        raw_ledger_final_record_sha256=_sha256("synthetic-raw-ledger"),
    )


def _write_small_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[RawScanOutcome, FeatureCacheWriteReceipt]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(feature_cache_v4, "FULL_TRAIN_ROWS", 5)
    outcome = _synthetic_outcome(tmp_path, rows=5)
    receipt = write_phase_b_feature_cache_v4(
        tmp_path / "phase-b-cache.npz",
        outcome,
        expected_raw_scope_commitment_sha256=outcome.raw_scope_commitment_sha256,
    )
    return outcome, receipt


def _load_small_cache(
    outcome: RawScanOutcome,
    receipt: FeatureCacheWriteReceipt,
    *,
    expected_cache_sha256: str | None = None,
):
    return load_phase_b_feature_cache_v4(
        receipt.cache_path,
        expected_cache_sha256=(
            receipt.cache_sha256 if expected_cache_sha256 is None else expected_cache_sha256
        ),
        expected_raw_scope_commitment_sha256=outcome.raw_scope_commitment_sha256,
        expected_feature_rows_commitment_sha256=outcome.feature_rows_commitment_sha256,
        expected_raw_ledger_final_record_sha256=outcome.raw_ledger_final_record_sha256,
    )


def test_feature_cache_is_o_excl_numeric_only_and_roundtrips_b1_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_cache_v4, "FULL_TRAIN_ROWS", 5)
    outcome = _synthetic_outcome(tmp_path, rows=5)
    output_path = tmp_path / "phase-b-cache.npz"
    original_open = feature_cache_v4.os.open
    observed_flags: list[int] = []

    def observed_open(path: str | bytes | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        if Path(path) == output_path:
            observed_flags.append(flags)
        return original_open(path, flags, mode)

    monkeypatch.setattr(feature_cache_v4.os, "open", observed_open)
    receipt = write_phase_b_feature_cache_v4(
        output_path,
        outcome,
        expected_raw_scope_commitment_sha256=outcome.raw_scope_commitment_sha256,
    )

    assert any(flags & os.O_EXCL for flags in observed_flags)
    assert receipt.cache_sha256 == _sha256(receipt.cache_path.read_bytes())
    with zipfile.ZipFile(receipt.cache_path, mode="r") as archive:
        assert set(archive.namelist()) == ARCHIVE_NAMES
        assert "b1_sampling_indicators.npy" not in archive.namelist()
        metadata_raw = archive.read("metadata.json")
    metadata = json.loads(metadata_raw)
    assert metadata["schema"] == CACHE_SCHEMA
    assert set(metadata) == {
        "schema",
        "row_count",
        "raw_scope_commitment_sha256",
        "feature_rows_commitment_sha256",
        "raw_ledger_final_record_sha256",
        "numeric_payload_sha256",
        "b1_sampling_audit",
    }
    assert not {"label", "labels", "fold", "folds", "path", "source_path"}.intersection(metadata)
    assert metadata["b1_sampling_audit"]["indicator_counts"] == [5, 2, 1]
    assert B1SamplingAudit(**{
        "row_count": 5,
        "indicator_counts": tuple(metadata["b1_sampling_audit"]["indicator_counts"]),
        "sha256": metadata["b1_sampling_audit"]["sha256"],
    }) == receipt.sampling_audit

    loaded = _load_small_cache(outcome, receipt)
    np.testing.assert_array_equal(loaded.cache.b0_values[0], outcome.rows[0].b0_values)
    np.testing.assert_array_equal(loaded.cache.b1_values[-1], outcome.rows[-1].b1_values)
    assert loaded.cache.b0_values.flags.writeable is False
    assert loaded.cache.novel_complete.flags.writeable is False
    assert loaded.sampling_audit == receipt.sampling_audit
    with pytest.raises(FeatureCacheV4Error, match="already exists"):
        write_phase_b_feature_cache_v4(
            output_path,
            outcome,
            expected_raw_scope_commitment_sha256=outcome.raw_scope_commitment_sha256,
        )


def _rewrite_archive_member(
    path: Path,
    *,
    member_name: str,
    replacement: bytes,
) -> None:
    with zipfile.ZipFile(path, mode="r") as archive:
        members = {info.filename: archive.read(info.filename) for info in archive.infolist()}
    members[member_name] = replacement
    replacement_path = path.with_name("tampered-cache.npz")
    with zipfile.ZipFile(replacement_path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(members):
            archive.writestr(name, members[name], compress_type=zipfile.ZIP_STORED)
    os.replace(replacement_path, path)


def test_feature_cache_rejects_numeric_and_sampling_audit_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, receipt = _write_small_cache(tmp_path, monkeypatch)
    with zipfile.ZipFile(receipt.cache_path, mode="r") as archive:
        b0_values = np.lib.format.read_array(io.BytesIO(archive.read("b0_values.npy")), allow_pickle=False)
    b0_values = np.array(b0_values, copy=True)
    b0_values[0, 0] += 1.0
    b0_buffer = io.BytesIO()
    np.lib.format.write_array(b0_buffer, b0_values, allow_pickle=False)
    _rewrite_archive_member(receipt.cache_path, member_name="b0_values.npy", replacement=b0_buffer.getvalue())
    with pytest.raises(FeatureCacheV4Error, match="file SHA-256 binding"):
        _load_small_cache(outcome, receipt)
    with pytest.raises(FeatureCacheV4Error, match="numeric payload hash"):
        _load_small_cache(
            outcome,
            receipt,
            expected_cache_sha256=_sha256(receipt.cache_path.read_bytes()),
        )

    outcome, receipt = _write_small_cache(tmp_path / "second", monkeypatch)
    with zipfile.ZipFile(receipt.cache_path, mode="r") as archive:
        metadata = json.loads(archive.read("metadata.json"))
    metadata["b1_sampling_audit"]["indicator_counts"][0] = 0
    _rewrite_archive_member(
        receipt.cache_path,
        member_name="metadata.json",
        replacement=canonical_json_bytes(metadata),
    )
    with pytest.raises(FeatureCacheV4Error, match="file SHA-256 binding"):
        _load_small_cache(outcome, receipt)
    with pytest.raises(FeatureCacheV4Error, match="sampling audit hash"):
        _load_small_cache(
            outcome,
            receipt,
            expected_cache_sha256=_sha256(receipt.cache_path.read_bytes()),
        )


def test_feature_cache_rejects_tampered_raw_row_commitment_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_cache_v4, "FULL_TRAIN_ROWS", 5)
    outcome = _synthetic_outcome(tmp_path, rows=5)
    changed_values = np.array(outcome.rows[0].b0_values, copy=True)
    changed_values[0] += 1.0
    tampered_row = replace(outcome.rows[0], b0_values=changed_values)
    tampered_outcome = replace(outcome, rows=(tampered_row, *outcome.rows[1:]))

    with pytest.raises(FeatureCacheV4Error, match="feature-row commitment"):
        write_phase_b_feature_cache_v4(
            tmp_path / "tampered-row-cache.npz",
            tampered_outcome,
            expected_raw_scope_commitment_sha256=outcome.raw_scope_commitment_sha256,
        )
