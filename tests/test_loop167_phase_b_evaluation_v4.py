from __future__ import annotations

import numpy as np

from src.loop167_phase_b.evaluation_v4 import binary_metrics, component_bootstrap_lower_bound


def test_binary_metrics_uses_fixed_binary_decisions() -> None:
    metrics = binary_metrics(
        np.array([0, 0, 1, 1], dtype=np.uint8),
        np.array([0, 1, 0, 1], dtype=np.uint8),
    )

    assert metrics.errors == 2
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert metrics.true_positive == 1
    assert metrics.true_negative == 1


def test_component_bootstrap_is_deterministic_and_separates_positive_signal() -> None:
    protocol_sha256 = "a" * 64
    components = np.array(["a", "b", "c", "d"], dtype=object)
    control_errors = np.array([1, 1, 1, 1], dtype=np.uint8)
    candidate_errors = np.array([0, 0, 0, 0], dtype=np.uint8)

    first = component_bootstrap_lower_bound(
        control_errors,
        candidate_errors,
        components,
        protocol_sha256=protocol_sha256,
        replay_seed=41,
        replicates=500,
    )
    repeated = component_bootstrap_lower_bound(
        control_errors,
        candidate_errors,
        components,
        protocol_sha256=protocol_sha256,
        replay_seed=41,
        replicates=500,
    )
    negative = component_bootstrap_lower_bound(
        candidate_errors,
        control_errors,
        components,
        protocol_sha256=protocol_sha256,
        replay_seed=41,
        replicates=500,
    )

    assert first == repeated == 4.0
    assert negative == -4.0
