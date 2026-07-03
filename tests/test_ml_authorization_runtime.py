import json
import shutil
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ml_authorization_runtime import (  # noqa: E402
    AuthorizationError,
    assert_operations_authorized,
    required_operations_for_eval,
    required_operations_for_train,
)


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _preflight(**decisions):
    default_decisions = {
        "train_val_allowed": False,
        "threshold_sweep_allowed": False,
        "test10k_allowed": False,
        "full_test_allowed": False,
    }
    default_decisions.update(decisions)
    blockers = {
        "train_val": ["train_blocked"],
        "threshold_sweep": ["threshold_blocked"],
        "test10k": ["test10k_blocked"],
        "full_test": ["full_test_blocked"],
    }
    return {
        "schema": "axon_ml_authorization_preflight_v2",
        "operation_authorization": {
            "decisions": default_decisions,
            "operation_blockers": blockers,
        },
    }


def test_train_operation_requirements_separate_train_and_full_test():
    assert required_operations_for_train(fast=True, skip_test_eval=False) == []
    assert required_operations_for_train(fast=False, skip_test_eval=True) == ["train_val"]
    assert required_operations_for_train(fast=False, skip_test_eval=False) == ["train_val", "full_test"]


def test_eval_operation_requirements_distinguish_test10k_fulltest_and_thresholds():
    assert required_operations_for_eval(
        split="val",
        max_eval_samples=None,
        sweep_thresholds=None,
        decision_threshold=None,
    ) == []
    assert required_operations_for_eval(
        split="test",
        max_eval_samples=10000,
        sweep_thresholds=None,
        decision_threshold=None,
    ) == ["test10k"]
    assert required_operations_for_eval(
        split="test",
        max_eval_samples=None,
        sweep_thresholds=None,
        decision_threshold=None,
    ) == ["full_test"]
    assert required_operations_for_eval(
        split="test",
        max_eval_samples=10000,
        sweep_thresholds="0.4,0.5",
        decision_threshold=None,
    ) == ["test10k", "threshold_sweep"]


def test_assert_operations_authorized_blocks_missing_decision():
    with pytest.raises(AuthorizationError) as excinfo:
        assert_operations_authorized(_preflight(), ["train_val"])

    assert "train_val" in str(excinfo.value)
    assert "train_blocked" in str(excinfo.value)


def test_assert_operations_authorized_allows_only_requested_operations():
    result = assert_operations_authorized(
        _preflight(train_val_allowed=True),
        ["train_val"],
    )

    assert result["authorized"] is True
    assert result["required_operations"] == ["train_val"]


def test_authorized_main_blocks_before_importing_heavy_work():
    with _case_dir("authorized_main_blocks") as tmp_path:
        preflight = tmp_path / "blocked_preflight.json"
        preflight.write_text(json.dumps(_preflight()), encoding="utf-8")

        completed = subprocess.run(
            [
                sys.executable,
                "scripts/authorized_main.py",
                "--ml-preflight",
                str(preflight),
                "--",
                "eval",
                "--checkpoint",
                "missing.pt",
                "--data-dir",
                "missing-data",
                "--split",
                "test",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

    assert completed.returncode == 2
    assert "[ML Authorization Blocked]" in completed.stderr
    assert "full_test" in completed.stderr
    assert "Checkpoint not found" not in completed.stdout
