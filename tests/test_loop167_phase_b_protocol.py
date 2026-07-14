from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_loop167_phase_b_protocol import (  # noqa: E402
    build_phase_b_protocol_payload,
    canonical_json_bytes,
)


def test_phase_b_protocol_freezes_required_dimensions_and_replay_semantics() -> None:
    payload = build_phase_b_protocol_payload()

    assert payload["schema"] == "axon_loop167_phase_b_protocol_v1"
    assert set(payload["phase_a_bindings"]) == {
        "proposal",
        "authorization",
        "semantic_delta_mapping",
        "frozen_deduplicated_baseline_allowlist",
        "source_semantics_addendum",
        "source_closure",
        "static_decision",
    }
    assert payload["feature_contract"]["b0"]["value_dimension"] == 571
    assert payload["feature_contract"]["b0"]["missing_indicator_dimension"] == 6
    assert payload["feature_contract"]["b1"]["value_dimension"] == 536
    assert payload["feature_contract"]["b1"]["missing_indicator_dimension"] == 4
    assert payload["feature_contract"]["novel"]["value_dimension"] == 292
    assert payload["fit_contract"]["maximum_total_fits"] == 75
    assert payload["fit_contract"]["seed_policy"] == "deterministic_replay_not_independent_robustness_trials"
    assert payload["ready_for"]["raw_access"] is False
    assert canonical_json_bytes(payload).endswith(b"\n")
