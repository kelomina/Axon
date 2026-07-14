from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from seal_loop167_phase_b_protocol_addendum import build_addendum_payload  # noqa: E402


def test_phase_b_protocol_addendum_makes_all_runs_identical_replays() -> None:
    payload = build_addendum_payload()
    replay = payload["deterministic_replay"]

    assert replay["run_labels"] == [41, 42, 43]
    assert replay["canonical_replay_seed"] == 41
    assert replay["counterfactual_permutation_seed"] == 41
    assert replay["all_arm_feature_matrix_hashes_must_match_across_run_labels"] is True
    assert replay["all_arm_prediction_hashes_must_match_across_run_labels"] is True
    assert replay["run_labels_are_not_independent_statistical_trials"] is True
