from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import src.loop167_phase_b.fit_worker as fit_worker
from src.loop167_phase_b.arm_contract import ARM_NAMES, REPLAY_SEEDS, build_arm_matrices
from src.loop167_phase_b.fit_worker import (
    FROZEN_HGB_PARAMETERS,
    PhaseBFeatureCache,
    run_phase_b_fit,
    run_phase_b_fit_for_test,
    validate_phase_b_fit_input,
)
from src.loop167_phase_b.progress_ledger import FitLedger, FitLedgerError, validate_fit_ledger


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _cache(rows: int = 10) -> PhaseBFeatureCache:
    row_axis = np.arange(rows, dtype=np.float32)[:, None]
    return PhaseBFeatureCache(
        b0_values=np.broadcast_to(row_axis * 0.1, (rows, 571)).copy(),
        b0_missing_indicators=np.zeros((rows, 6), dtype=np.float32),
        b1_values=np.broadcast_to(row_axis * 0.2 + 1.0, (rows, 536)).copy(),
        b1_missing_indicators=np.zeros((rows, 4), dtype=np.float32),
        novel_values=np.broadcast_to(row_axis * 0.3 + 2.0, (rows, 292)).copy(),
        novel_complete=np.array([(index % 3) != 1 for index in range(rows)], dtype=bool),
    )


def _labels_and_folds(rows: int = 10) -> tuple[np.ndarray, np.ndarray]:
    folds = np.repeat(np.arange(5, dtype=np.int8), rows // 5)
    labels = np.resize(np.array([0, 1], dtype=np.uint8), rows)
    return labels, folds


def _run_synthetic(cache: PhaseBFeatureCache, labels: np.ndarray, folds: np.ndarray, path: Path):
    with FitLedger.create(path) as ledger:
        result = run_phase_b_fit_for_test(
            cache,
            labels,
            folds,
            ledger,
            fit_protocol_commitment_sha256=_sha("fit-protocol"),
            feature_rows_commitment_sha256=_sha("feature-rows"),
            raw_ledger_final_record_sha256=_sha("raw-ledger"),
            synthetic=True,
        )
    return result


def test_synthetic_worker_closes_all_75_cells_in_preregistered_arm_order(tmp_path: Path) -> None:
    cache = _cache()
    labels, folds = _labels_and_folds()
    ledger_path = tmp_path / "fit-ledger.jsonl"

    result = _run_synthetic(cache, labels, folds, ledger_path)

    validation = validate_fit_ledger(ledger_path)
    assert result.total_fit_units == 75
    assert validation.complete is True
    assert validation.completed_unit_count == 75
    assert validation.final_record_sha256 == result.fit_ledger_final_record_sha256
    assert result.primary_controls is None

    records = [json.loads(line) for line in ledger_path.read_text(encoding="ascii").splitlines()]
    units = [record for record in records if record["event"] == "fit_unit_completed"]
    assert [record["arm_ordinal"] for record in units] == [ordinal for ordinal in range(5) for _ in range(15)]
    assert [
        (record["arm_ordinal"], record["replay_ordinal"], record["fold_ordinal"])
        for record in units
    ] == [
        (arm_ordinal, replay_ordinal, fold_ordinal)
        for arm_ordinal in range(5)
        for replay_ordinal in range(3)
        for fold_ordinal in range(5)
    ]


def test_synthetic_worker_replays_canonical_seed_and_falls_back_bitwise(tmp_path: Path) -> None:
    cache = _cache()
    labels, folds = _labels_and_folds()
    result = _run_synthetic(cache, labels, folds, tmp_path / "replay-ledger.jsonl")

    assert len(result.matrix_replay_sha256) == 64
    assert len(result.evaluation_replay_sha256) == 64
    canonical = result.replay_evaluations[41]
    missing = ~cache.novel_complete
    for replay_seed in REPLAY_SEEDS:
        current = result.replay_evaluations[replay_seed]
        assert set(current) == set(ARM_NAMES)
        for arm in ARM_NAMES:
            assert current[arm].scores.tobytes() == canonical[arm].scores.tobytes()
            assert current[arm].hard_decisions.tobytes() == canonical[arm].hard_decisions.tobytes()
        for arm in ("M", "CF"):
            assert current[arm].scores[missing].tobytes() == current["B0"].scores[missing].tobytes()
            assert (
                current[arm].hard_decisions[missing].tobytes()
                == current["B0"].hard_decisions[missing].tobytes()
            )


def test_worker_uses_separate_fit_and_holdout_counterfactual_domains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _cache()
    labels, folds = _labels_and_folds()
    observed_roles: list[tuple[int, str, int]] = []
    original = fit_worker.build_arm_matrices

    def observed_build(*args: object, **kwargs: object):
        observed_roles.append((int(kwargs["outer_fold"]), str(kwargs["role"]), args[0].shape[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(fit_worker, "build_arm_matrices", observed_build)
    _run_synthetic(cache, labels, folds, tmp_path / "counterfactual-ledger.jsonl")

    for fold in range(5):
        assert (fold, "fit", 8) in observed_roles
        assert (fold, "holdout", 2) in observed_roles


def test_base_control_matrices_match_the_frozen_arm_contract() -> None:
    cache = _cache()
    expected = build_arm_matrices(
        cache.b0_values,
        cache.b0_missing_indicators,
        cache.b1_values,
        cache.b1_missing_indicators,
        cache.novel_values,
        cache.novel_complete,
        protocol_sha256=_sha("fit-protocol"),
        replay_seed=41,
        outer_fold=0,
        role="fit",
    )
    actual = fit_worker._build_base_matrices(cache)

    for arm in ("B0", "B1", "M", "A"):
        assert np.array_equal(actual.for_arm(arm), expected.for_arm(arm))


def test_production_api_has_no_synthetic_or_grid_override_and_requires_20k() -> None:
    parameters = inspect.signature(run_phase_b_fit).parameters
    for forbidden in ("synthetic", "rows", "arms", "seed", "threshold", "estimator"):
        assert forbidden not in parameters
    assert inspect.signature(run_phase_b_fit_for_test).parameters["synthetic"].default is inspect.Parameter.empty
    assert dict(FROZEN_HGB_PARAMETERS) == {
        "loss": "log_loss",
        "learning_rate": 0.06,
        "max_iter": 260,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_bins": 255,
        "early_stopping": False,
        "random_state": 41,
    }

    cache = _cache()
    labels, folds = _labels_and_folds()
    with pytest.raises(ValueError, match="exactly 20000"):
        validate_phase_b_fit_input(cache, labels, folds)


def test_production_input_validation_requires_20k_rows_and_five_4k_folds() -> None:
    rows = 20_000
    cache = PhaseBFeatureCache(
        b0_values=np.broadcast_to(np.zeros(571, dtype=np.float32), (rows, 571)),
        b0_missing_indicators=np.broadcast_to(np.zeros(6, dtype=np.float32), (rows, 6)),
        b1_values=np.broadcast_to(np.zeros(536, dtype=np.float32), (rows, 536)),
        b1_missing_indicators=np.broadcast_to(np.zeros(4, dtype=np.float32), (rows, 4)),
        novel_values=np.broadcast_to(np.zeros(292, dtype=np.float32), (rows, 292)),
        novel_complete=np.ones(rows, dtype=bool),
    )
    folds = np.repeat(np.arange(5, dtype=np.int8), 4_000)
    labels = np.resize(np.array([0, 1], dtype=np.uint8), rows)

    summary = validate_phase_b_fit_input(cache, labels, folds)

    assert summary.row_count == 20_000
    assert summary.rows_per_fold == 4_000
    assert summary.synthetic is False


def test_worker_rejects_prestarted_ledger_instead_of_resuming_or_duplicating(tmp_path: Path) -> None:
    cache = _cache()
    labels, folds = _labels_and_folds()
    ledger_path = tmp_path / "prestarted.jsonl"
    with FitLedger.create(ledger_path) as ledger:
        ledger.fit_started(
            fit_protocol_commitment_sha256=_sha("fit-protocol"),
            feature_rows_commitment_sha256=_sha("feature-rows"),
            raw_ledger_final_record_sha256=_sha("raw-ledger"),
        )
        with pytest.raises(FitLedgerError, match="first and only start"):
            run_phase_b_fit_for_test(
                cache,
                labels,
                folds,
                ledger,
                fit_protocol_commitment_sha256=_sha("fit-protocol"),
                feature_rows_commitment_sha256=_sha("feature-rows"),
                raw_ledger_final_record_sha256=_sha("raw-ledger"),
                synthetic=True,
            )


def test_worker_rejects_a_duplicate_fit_cell_from_its_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _cache()
    labels, folds = _labels_and_folds()
    ledger_path = tmp_path / "duplicate-cell.jsonl"
    with FitLedger.create(ledger_path) as ledger:
        original = ledger.fit_unit_completed
        duplicate_pending = True

        def duplicate_first_cell(**kwargs: int) -> str:
            nonlocal duplicate_pending
            record_sha256 = original(**kwargs)
            if duplicate_pending:
                duplicate_pending = False
                original(**kwargs)
            return record_sha256

        monkeypatch.setattr(ledger, "fit_unit_completed", duplicate_first_cell)
        with pytest.raises(FitLedgerError, match="exactly once"):
            run_phase_b_fit_for_test(
                cache,
                labels,
                folds,
                ledger,
                fit_protocol_commitment_sha256=_sha("fit-protocol"),
                feature_rows_commitment_sha256=_sha("feature-rows"),
                raw_ledger_final_record_sha256=_sha("raw-ledger"),
                synthetic=True,
            )
    assert validate_fit_ledger(ledger_path).completed_unit_count == 1


def test_test_entrypoint_rejects_an_implicit_or_false_synthetic_marker(tmp_path: Path) -> None:
    cache = _cache()
    labels, folds = _labels_and_folds()
    with FitLedger.create(tmp_path / "synthetic-marker.jsonl") as ledger:
        with pytest.raises(ValueError, match="explicit synthetic=True"):
            run_phase_b_fit_for_test(
                cache,
                labels,
                folds,
                ledger,
                fit_protocol_commitment_sha256=_sha("fit-protocol"),
                feature_rows_commitment_sha256=_sha("feature-rows"),
                raw_ledger_final_record_sha256=_sha("raw-ledger"),
                synthetic=False,
            )


def test_fit_worker_has_no_raw_path_or_checkpoint_import_surface() -> None:
    source = inspect.getsource(fit_worker)
    for forbidden in (
        "from pathlib",
        "import pathlib",
        "import os",
        ".open(",
        "np.load",
        "np.save",
        "raw_context",
        "one_pass_reader",
        "RawFeatureContext",
        "checkpoint",
    ):
        assert forbidden not in source
