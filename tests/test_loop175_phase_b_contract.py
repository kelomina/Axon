from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop175.phase_b_contract import (  # noqa: E402
    PhaseBContractError,
    load_phase_b_protocol,
    validate_bound_evidence,
    validate_protocol_structure,
    write_exclusive_json,
)


def _payload() -> dict[str, object]:
    path = PROJECT_ROOT / "manifests/roadmap_9997/loop175_section_region_moe/phase_b_protocol.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_live_phase_b_protocol_and_evidence_bindings_are_valid() -> None:
    protocol = load_phase_b_protocol(PROJECT_ROOT)
    validate_bound_evidence(PROJECT_ROOT, protocol)
    assert protocol.sha256


def test_protocol_rejects_dimension_or_readiness_drift() -> None:
    payload = _payload()
    payload["inputs"]["b0_cache"]["shape"] = [20_000, 577]  # type: ignore[index]
    with pytest.raises(PhaseBContractError, match="B0"):
        validate_protocol_structure(payload)

    payload = _payload()
    payload["ready_for"]["seed41_outer_oof"] = True  # type: ignore[index]
    with pytest.raises(PhaseBContractError, match="readiness"):
        validate_protocol_structure(payload)


def test_exclusive_writer_refuses_to_replace_receipt(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    write_exclusive_json(output, {"status": "first"})
    with pytest.raises(PhaseBContractError, match="overwrite"):
        write_exclusive_json(output, {"status": "second"})

