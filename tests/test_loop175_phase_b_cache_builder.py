from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop175.phase_b_cache_builder import (  # noqa: E402
    RegionCacheBuildError,
    RegionSource,
    build_region_cache,
    load_region_cache_progress,
)
from src.loop175.phase_b_data import load_ragged_region_cache  # noqa: E402


def _pe(marker: int) -> bytes:
    payload = bytearray(0x1400)
    payload[:2] = b"MZ"
    struct.pack_into("<I", payload, 0x3C, 0x80)
    payload[0x80:0x84] = b"PE\0\0"
    file_header = 0x84
    struct.pack_into("<H", payload, file_header, 0x14C)
    struct.pack_into("<H", payload, file_header + 2, 1)
    struct.pack_into("<H", payload, file_header + 16, 0xE0)
    optional = file_header + 20
    struct.pack_into("<H", payload, optional, 0x10B)
    struct.pack_into("<I", payload, optional + 16, 0x1000)
    struct.pack_into("<I", payload, optional + 60, 0x400)
    section = optional + 0xE0
    payload[section : section + 5] = b".text"
    struct.pack_into("<I", payload, section + 8, 0x1000)
    struct.pack_into("<I", payload, section + 12, 0x1000)
    struct.pack_into("<I", payload, section + 16, 0x1000)
    struct.pack_into("<I", payload, section + 20, 0x400)
    struct.pack_into("<I", payload, section + 36, 0x60000020)
    payload[0x400:] = bytes([marker]) * (len(payload) - 0x400)
    return bytes(payload)


def _sources(tmp_path: Path, rows: int = 4) -> tuple[RegionSource, ...]:
    result = []
    for ordinal in range(rows):
        payload = _pe(ordinal + 1)
        path = tmp_path / f"sample-{ordinal}.exe"
        path.write_bytes(payload)
        result.append(
            RegionSource(
                ordinal=ordinal,
                path=path,
                source_sha256=hashlib.sha256(payload).hexdigest(),
                declared_size=len(payload),
                availability="supported",
                label=ordinal % 2,
            )
        )
    return tuple(result)


def test_resumable_builder_seals_identity_free_ragged_cache(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    staging = tmp_path / "staging"
    output = tmp_path / "cache.npz"
    result = build_region_cache(
        sources,
        output_cache=output,
        staging_directory=staging,
        block_rows=2,
        expected_rows=4,
    )
    assert result.attempted == result.supported == 4
    assert result.silent_drops == 0
    assert result.class_coverage == {"0": 1.0, "1": 1.0}
    assert result.blockers == ()
    assert result.decision == "full_train_region_cache_gate_pass_seed41_pilot_may_begin"
    cache = load_ragged_region_cache(output, expected_sha256=result.cache.sha256, expected_rows=4)
    assert np.diff(cache.row_region_offsets).tolist() == [16, 16, 16, 16]
    assert cache.token_values.size > 0


def test_progress_trims_uncommitted_token_and_partial_ledger_tails(tmp_path: Path) -> None:
    sources = _sources(tmp_path, rows=2)
    staging = tmp_path / "staging"
    output = tmp_path / "cache.npz"
    result = build_region_cache(
        sources,
        output_cache=output,
        staging_directory=staging,
        block_rows=1,
        expected_rows=2,
    )
    token_path = staging / "region_tokens.bin"
    ledger_path = staging / "region_progress.jsonl"
    committed_size = token_path.stat().st_size
    with token_path.open("ab") as handle:
        handle.write(b"uncommitted")
    with ledger_path.open("ab") as handle:
        handle.write(b'{"partial":')
    progress = load_region_cache_progress(ledger_path, token_path)
    assert len(progress.records) == 2
    assert token_path.stat().st_size == committed_size
    assert ledger_path.read_bytes().endswith(b"\n")
    assert result.final_record_sha256 == progress.final_record_sha256


def test_resume_rejects_a_different_source_plan(tmp_path: Path) -> None:
    sources = _sources(tmp_path, rows=2)
    staging = tmp_path / "staging"
    build_region_cache(
        sources,
        output_cache=tmp_path / "first.npz",
        staging_directory=staging,
        block_rows=1,
        expected_rows=2,
    )
    changed = list(sources)
    changed[1] = RegionSource(
        ordinal=1,
        path=changed[1].path,
        source_sha256="0" * 64,
        declared_size=changed[1].declared_size,
        availability=changed[1].availability,
        label=changed[1].label,
    )
    with pytest.raises(RegionCacheBuildError, match="scope commitment"):
        build_region_cache(
            tuple(changed),
            output_cache=tmp_path / "second.npz",
            staging_directory=staging,
            block_rows=1,
            expected_rows=2,
        )
