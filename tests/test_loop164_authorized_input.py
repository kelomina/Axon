from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loop164.authorized_input import (  # noqa: E402
    LOCAL_BUNDLE_RECORD_SCHEMA,
    LOCAL_BUNDLE_ROLE,
    LOCAL_BUNDLE_SUMMARY_SCHEMA,
    LOOP_ID,
    InputContractError,
    LocalProbeRecord,
    SourceIntegrityError,
    StreamingWholeFileByteSource,
    load_local_probe_bundle,
)
from loop164.whole_file_gcg import InMemoryByteSource  # noqa: E402


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _record(path: Path, raw: bytes, *, label: int = 0) -> LocalProbeRecord:
    return LocalProbeRecord(
        source_path=path,
        source_sha256=_sha256(raw),
        source_size_bytes=len(raw),
        label=label,
    )


def _source(record: LocalProbeRecord, *, data_root: Path) -> StreamingWholeFileByteSource:
    return StreamingWholeFileByteSource(
        record,
        data_root=data_root,
        receptive_field_bytes=7,
        output_stride_bytes=3,
        max_outputs_per_chunk=4,
        bounded_read_bytes=5,
        max_supported_file_bytes=1024,
        timeout_seconds=10.0,
    )


def test_streaming_source_matches_in_memory_chunks_and_verifies_each_pass(tmp_path: Path):
    raw = bytes(range(29))
    path = tmp_path / "samples" / "sample.bin"
    path.parent.mkdir()
    path.write_bytes(raw)
    streaming = _source(_record(path, raw), data_root=tmp_path)
    memory = InMemoryByteSource.from_raw_bytes(list(raw))

    expected = list(
        memory.iter_output_chunks(
            receptive_field_bytes=7,
            output_stride_bytes=3,
            max_outputs_per_chunk=4,
        )
    )
    for _pass_index in range(2):
        actual = list(
            streaming.iter_output_chunks(
                receptive_field_bytes=7,
                output_stride_bytes=3,
                max_outputs_per_chunk=4,
            )
        )
        assert [(chunk.output_start, chunk.output_count) for chunk in actual] == [
            (chunk.output_start, chunk.output_count) for chunk in expected
        ]
        assert all(torch.equal(left.tokens, right.tokens) for left, right in zip(actual, expected))

    streaming.assert_complete()
    assert [receipt.pass_index for receipt in streaming.scan_receipts] == [1, 2]
    assert all(receipt.bytes_read == len(raw) for receipt in streaming.scan_receipts)
    assert all(receipt.sha256 == _sha256(raw) for receipt in streaming.scan_receipts)


def test_streaming_source_rejects_hash_mismatch_without_pass_receipt(tmp_path: Path):
    raw = b"0123456789abcdef"
    path = tmp_path / "sample.bin"
    path.write_bytes(raw)
    record = LocalProbeRecord(path, "0" * 64, len(raw), 0)
    source = _source(record, data_root=tmp_path)

    with pytest.raises(SourceIntegrityError, match="do not match"):
        list(
            source.iter_output_chunks(
                receptive_field_bytes=7,
                output_stride_bytes=3,
                max_outputs_per_chunk=4,
            )
        )

    assert source.scan_receipts == ()


def test_streaming_source_rejects_change_between_passes(tmp_path: Path):
    raw = b"0123456789abcdef"
    path = tmp_path / "sample.bin"
    path.write_bytes(raw)
    source = _source(_record(path, raw), data_root=tmp_path)
    list(
        source.iter_output_chunks(
            receptive_field_bytes=7,
            output_stride_bytes=3,
            max_outputs_per_chunk=4,
        )
    )
    path.write_bytes(raw[::-1])

    with pytest.raises(SourceIntegrityError, match="fingerprint changed"):
        list(
            source.iter_output_chunks(
                receptive_field_bytes=7,
                output_stride_bytes=3,
                max_outputs_per_chunk=4,
            )
        )


def _bundle_record(path: Path, raw: bytes, label: int) -> dict[str, object]:
    return {
        "schema": LOCAL_BUNDLE_RECORD_SCHEMA,
        "loop_id": LOOP_ID,
        "bundle_role": LOCAL_BUNDLE_ROLE,
        "split_role": "train",
        "label": label,
        "source_path": str(path),
        "source_sha256": _sha256(raw),
        "source_size_bytes": len(raw),
        "metadata_not_model_features": ["source_path", "source_sha256", "source_size_bytes"],
        "source_path_usage": "loader_identity_only_not_model_feature",
        "source_sha256_usage": "integrity_binding_only_not_model_feature",
    }


def _write_bundle_case(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "data"
    data_root.mkdir()
    rows = []
    for label in (0, 1):
        raw = bytes([label + 1]) * 32
        path = data_root / f"sample-{label}.bin"
        path.write_bytes(raw)
        rows.append(_bundle_record(path, raw, label))
    bundle_path = tmp_path / "bundle.jsonl"
    bundle_raw = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()
    bundle_path.write_bytes(bundle_raw)
    summary_path = tmp_path / "summary.json"
    summary = {
        "schema": LOCAL_BUNDLE_SUMMARY_SCHEMA,
        "loop_id": LOOP_ID,
        "bundle_role": LOCAL_BUNDLE_ROLE,
        "bundle": {
            "path": str(bundle_path),
            "sha256": _sha256(bundle_raw),
            "record_count": 2,
            "record_schema": LOCAL_BUNDLE_RECORD_SCHEMA,
        },
        "selection": {"canonical_split_role": "train", "records_per_class": 1, "labels": [0, 1]},
        "ready_for": {
            "local_runtime_probe_bundle": True,
            "loop164_whole_file_training": False,
            "val_or_test_access": False,
            "f1_claim": False,
        },
        "decision": "local_train_only_probe_bundle_ready",
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return bundle_path, summary_path, data_root


def test_bundle_loader_accepts_only_balanced_train_role(tmp_path: Path):
    bundle_path, summary_path, data_root = _write_bundle_case(tmp_path)

    records, _summary = load_local_probe_bundle(
        bundle_path=bundle_path,
        summary_path=summary_path,
        data_root=data_root,
        expected_records_per_class=1,
    )

    assert [record.label for record in records] == [0, 1]


def test_bundle_loader_rejects_heldout_role_even_with_matching_summary_hash(tmp_path: Path):
    bundle_path, summary_path, data_root = _write_bundle_case(tmp_path)
    rows = [json.loads(line) for line in bundle_path.read_text().splitlines()]
    rows[0]["split_role"] = "val"
    bundle_raw = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()
    bundle_path.write_bytes(bundle_raw)
    summary = json.loads(summary_path.read_text())
    summary["bundle"]["sha256"] = _sha256(bundle_raw)
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(InputContractError, match="role or identity"):
        load_local_probe_bundle(
            bundle_path=bundle_path,
            summary_path=summary_path,
            data_root=data_root,
            expected_records_per_class=1,
        )
