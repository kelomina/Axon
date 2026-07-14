from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import loop166.windows_process_lineage as lineage  # noqa: E402
from loop166.windows_process_lineage import (  # noqa: E402
    ProcessLineageError,
    _validate_observed_lineage,
)


def test_direct_parent_lineage_requires_no_redirector():
    audit = _validate_observed_lineage(
        expected_parent_pid=10,
        current_pid=12,
        direct_parent_pid=10,
        process_parents={},
        process_images={},
        launcher_executable="unused",
        base_executable="unused",
        is_windows=False,
    )

    assert audit == {
        "mode": "direct_parent",
        "expected_parent_pid": 10,
        "current_pid": 12,
        "direct_parent_pid": 10,
        "redirector_pid": 0,
    }


def test_exactly_one_windows_venv_redirector_is_accepted(tmp_path: Path):
    launcher = tmp_path / "venv" / "python.exe"
    base = tmp_path / "pythoncore.exe"
    launcher.parent.mkdir()
    launcher.write_bytes(b"launcher")
    base.write_bytes(b"base")
    audit = _validate_observed_lineage(
        expected_parent_pid=10,
        current_pid=12,
        direct_parent_pid=11,
        process_parents={10: 1, 11: 10, 12: 11},
        process_images={10: str(base), 11: str(launcher), 12: str(base)},
        launcher_executable=launcher,
        base_executable=base,
        is_windows=True,
    )

    assert audit["mode"] == "windows_venv_redirector"
    assert audit["redirector_pid"] == 11


def test_windows_direct_parent_requires_frozen_base_images(tmp_path: Path):
    launcher = tmp_path / "venv-python.exe"
    base = tmp_path / "pythoncore.exe"
    other = tmp_path / "other.exe"
    for path in (launcher, base, other):
        path.write_bytes(path.name.encode("ascii"))
    common = {
        "expected_parent_pid": 10,
        "current_pid": 12,
        "direct_parent_pid": 10,
        "process_parents": {12: 10},
        "launcher_executable": launcher,
        "base_executable": base,
        "is_windows": True,
    }

    audit = _validate_observed_lineage(
        **common,
        process_images={10: str(base), 12: str(base)},
    )
    assert audit["mode"] == "direct_parent"
    assert Path(audit["current_image"]).resolve(strict=True) == base.resolve(strict=True)
    assert Path(audit["expected_parent_image"]).resolve(strict=True) == base.resolve(
        strict=True
    )

    for drifted_pid in (10, 12):
        images = {10: str(base), 12: str(base), drifted_pid: str(other)}
        with pytest.raises(ProcessLineageError, match=f"image drifted for PID {drifted_pid}"):
            _validate_observed_lineage(**common, process_images=images)

    with pytest.raises(ProcessLineageError, match="process snapshot"):
        _validate_observed_lineage(
            **{**common, "process_parents": {12: 9}},
            process_images={10: str(base), 12: str(base)},
        )


def test_windows_direct_parent_collects_live_process_evidence(
    tmp_path: Path,
    monkeypatch,
):
    launcher = tmp_path / "venv-python.exe"
    base = tmp_path / "pythoncore.exe"
    launcher.write_bytes(b"launcher")
    base.write_bytes(b"base")
    observed_pids = set()

    def fake_evidence(pids):
        observed_pids.update(pids)
        return {12: 10}, {10: str(base), 12: str(base)}

    monkeypatch.setattr(lineage.os, "getpid", lambda: 12)
    monkeypatch.setattr(lineage.os, "getppid", lambda: 10)
    monkeypatch.setattr(lineage.platform, "system", lambda: "Windows")
    monkeypatch.setattr(lineage, "_windows_process_evidence", fake_evidence)

    audit = lineage.validate_spawn_lineage(
        10,
        launcher_executable=launcher,
        base_executable=base,
    )

    assert audit["mode"] == "direct_parent"
    assert observed_pids == {10, 12}


def test_two_layers_or_image_drift_are_rejected(tmp_path: Path):
    launcher = tmp_path / "venv-python.exe"
    base = tmp_path / "pythoncore.exe"
    other = tmp_path / "other.exe"
    for path in (launcher, base, other):
        path.write_bytes(path.name.encode("ascii"))
    common = {
        "expected_parent_pid": 10,
        "current_pid": 13,
        "direct_parent_pid": 12,
        "launcher_executable": launcher,
        "base_executable": base,
        "is_windows": True,
    }

    with pytest.raises(ProcessLineageError, match="not exactly one"):
        _validate_observed_lineage(
            **common,
            process_parents={10: 1, 11: 10, 12: 11, 13: 12},
            process_images={10: str(base), 12: str(launcher), 13: str(base)},
        )

    with pytest.raises(ProcessLineageError, match="image drifted"):
        _validate_observed_lineage(
            **common,
            process_parents={10: 1, 12: 10, 13: 12},
            process_images={10: str(base), 12: str(other), 13: str(base)},
        )

    with pytest.raises(ProcessLineageError, match="active virtual environment"):
        _validate_observed_lineage(
            **{**common, "launcher_executable": base},
            process_parents={10: 1, 12: 10, 13: 12},
            process_images={10: str(base), 12: str(base), 13: str(base)},
        )


@pytest.mark.skipif(platform.system().casefold() != "windows", reason="Windows-only")
def test_windows_process_evidence_rejects_dead_process():
    process = subprocess.Popen([sys._base_executable, "-c", "pass"])
    try:
        process.wait(timeout=30)
        with pytest.raises(ProcessLineageError, match="no longer active"):
            lineage._windows_process_evidence({os.getpid(), process.pid})
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)


@pytest.mark.skipif(platform.system().casefold() != "windows", reason="Windows-only")
def test_real_python_venv_redirector_lineage_is_verified():
    environment = dict(os.environ)
    environment["AXON_EXPECTED_PARENT_PID"] = str(os.getpid())
    environment["AXON_FROZEN_LAUNCHER"] = sys.executable
    environment["AXON_FROZEN_BASE"] = sys._base_executable
    environment["PYTHONPATH"] = str(SRC_DIR)
    probe = (
        "import json, os; "
        "from loop166.windows_process_lineage import validate_spawn_lineage; "
        "print(json.dumps(validate_spawn_lineage("
        "int(os.environ['AXON_EXPECTED_PARENT_PID']), "
        "launcher_executable=os.environ['AXON_FROZEN_LAUNCHER'], "
        "base_executable=os.environ['AXON_FROZEN_BASE'])))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    audit = json.loads(completed.stdout)

    assert audit["mode"] == "windows_venv_redirector"
    assert audit["expected_parent_pid"] == os.getpid()
    assert Path(audit["redirector_image"]).resolve(strict=True) == Path(
        sys.executable
    ).resolve(strict=True)
