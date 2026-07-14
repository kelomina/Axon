from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from scripts import train_loop46_cert_structure as loop46
from scripts.train_loop46_cert_structure import (
    CERT_STRUCTURE_FEATURE_NAMES,
    CertStructureConfig,
    build_cert_structure_matrix,
    cert_structure_features_from_blob,
)


def _der_len(length: int) -> bytes:
    if length < 0x80:
        return bytes([length])
    encoded = length.to_bytes((length.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(encoded)]) + encoded


def _der(tag: int, value: bytes) -> bytes:
    return bytes([tag]) + _der_len(len(value)) + value


def _oid_value(text: str) -> bytes:
    arcs = [int(item) for item in text.split(".")]
    payload = bytearray([arcs[0] * 40 + arcs[1]])
    for arc in arcs[2:]:
        parts = [arc & 0x7F]
        arc >>= 7
        while arc:
            parts.append(0x80 | (arc & 0x7F))
            arc >>= 7
        payload.extend(reversed(parts))
    return bytes(payload)


def _win_cert(payload: bytes) -> bytes:
    total_len = len(payload) + 8
    return struct.pack("<IHH", total_len, 0x0200, 0x0002) + payload


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_cert_structure_features_parse_oid_and_time():
    payload = _der(
        0x30,
        b"".join(
            [
                _der(0x06, _oid_value("1.2.840.113549.1.7.2")),
                _der(0x06, _oid_value("2.16.840.1.101.3.4.2.1")),
                _der(0x13, b"Example Cert"),
                _der(0x17, b"240101000000Z"),
            ]
        ),
    )

    features = cert_structure_features_from_blob(_win_cert(payload), len(payload) + 8, 4096)
    by_name = {name: features[index] for index, name in enumerate(CERT_STRUCTURE_FEATURE_NAMES)}

    assert features.shape == (len(CERT_STRUCTURE_FEATURE_NAMES),)
    assert by_name["cert_struct_present"] == 1.0
    assert by_name["cert_struct_root_sequence"] == 1.0
    assert by_name["cert_struct_parse_ok"] == 1.0
    assert by_name["cert_struct_oid_pkcs7_signed_data_present"] == 1.0
    assert by_name["cert_struct_oid_sha256_present"] == 1.0
    assert by_name["cert_struct_utc_time_count_log"] > 0.0
    assert by_name["cert_struct_string_count_log"] > 0.0
    assert by_name["cert_struct_min_year_norm"] > 0.0


def test_cert_structure_features_return_zero_for_unsigned_blob():
    features = cert_structure_features_from_blob(b"", 0, 100)

    assert np.count_nonzero(features) == 0


def test_cert_structure_feature_names_are_identity_safe():
    assert CERT_STRUCTURE_FEATURE_NAMES
    forbidden_fragments = ["filename", "file_path", "extension", "source_path", "sample_index", "split"]
    assert not any(
        any(fragment in name for fragment in forbidden_fragments)
        for name in CERT_STRUCTURE_FEATURE_NAMES
    )


def test_build_cert_structure_matrix_preallocates_stable_width(monkeypatch):
    def fake_features(row, _cache_dir):
        return np.full(len(CERT_STRUCTURE_FEATURE_NAMES), float(row["value"]), dtype=np.float32)

    monkeypatch.setattr(loop46, "cert_structure_features_for_row", fake_features)

    matrix = build_cert_structure_matrix(
        [{"source_sha256": "a" * 64, "value": 1}, {"source_sha256": "b" * 64, "value": 2}],
        CertStructureConfig(cache_dir=None),
    )

    assert matrix.shape == (2, len(CERT_STRUCTURE_FEATURE_NAMES))
    assert matrix.dtype == np.float32
    assert matrix[0, 0] == 1.0
    assert matrix[1, 0] == 2.0


def test_cert_structure_cache_path_rejects_invalid_source_sha256(tmp_path):
    with pytest.raises(ValueError, match="invalid source_sha256"):
        loop46._cert_structure_cache_path(
            {"source_path": str(tmp_path / "sample.exe"), "source_sha256": "../escape"},
            str(tmp_path / "cache"),
        )

    assert not (tmp_path / "escape.npz").exists()


def test_cert_structure_cache_path_is_namespaced(tmp_path):
    source_sha = "a" * 64
    cache_path = loop46._cert_structure_cache_path(
        {"source_path": str(tmp_path / "sample.exe"), "source_sha256": source_sha},
        str(tmp_path / "cache"),
    )

    assert cache_path is not None
    assert cache_path.name == f"cert_structure_v1_{source_sha}.npz"


def test_cert_structure_features_reject_source_sha256_mismatch_before_writing(tmp_path, monkeypatch):
    source_path = tmp_path / "sample.exe"
    source_path.write_bytes(b"actual-content")
    wrong_sha = _sha256_bytes(b"different-content")
    cache_dir = tmp_path / "cache"

    def fail_if_extractor_is_called(_path):
        raise AssertionError("extractor should not run when source_sha256 mismatches source_path bytes")

    monkeypatch.setattr(loop46, "cert_structure_features_from_path", fail_if_extractor_is_called)

    with pytest.raises(ValueError, match="source_sha256_mismatch"):
        loop46.cert_structure_features_for_row(
            {"source_path": str(source_path), "source_sha256": wrong_sha},
            str(cache_dir),
        )

    assert not (cache_dir / f"{wrong_sha}.npz").exists()


def test_cert_structure_features_reject_existing_cache_when_source_sha256_mismatches(tmp_path, monkeypatch):
    source_path = tmp_path / "sample.exe"
    source_path.write_bytes(b"actual-content")
    wrong_sha = _sha256_bytes(b"different-content")
    cache_dir = tmp_path / "cache"
    row = {"source_path": str(source_path), "source_sha256": wrong_sha}
    cache_path = loop46._cert_structure_cache_path(row, str(cache_dir))
    assert cache_path is not None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, features=np.ones(len(CERT_STRUCTURE_FEATURE_NAMES), dtype=np.float32))

    def fail_if_extractor_is_called(_path):
        raise AssertionError("extractor should not run when source_sha256 mismatches source_path bytes")

    monkeypatch.setattr(loop46, "cert_structure_features_from_path", fail_if_extractor_is_called)

    with pytest.raises(ValueError, match="source_sha256_mismatch"):
        loop46.cert_structure_features_for_row(row, str(cache_dir))
