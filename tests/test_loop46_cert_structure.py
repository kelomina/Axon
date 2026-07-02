from __future__ import annotations

import struct

import numpy as np

from scripts.train_loop46_cert_structure import (
    CERT_STRUCTURE_FEATURE_NAMES,
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
