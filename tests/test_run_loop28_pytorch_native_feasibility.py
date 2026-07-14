from __future__ import annotations

import importlib.util
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_loop28_pytorch_native_feasibility.py"
SPEC = importlib.util.spec_from_file_location("run_loop28_pytorch_native_feasibility", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_tiny_input_is_fixed_contiguous_float32() -> None:
    first = runner.tiny_input()
    second = runner.tiny_input()

    assert first.shape == (2, 8)
    assert first.dtype == np.dtype("float32")
    assert first.flags.c_contiguous
    assert np.array_equal(first, second)
    assert runner._array_record(first)["sha256"] == (
        "caa371218bdbb95cb73bfe7ab65ec2f8f69222a747fca8f889b2bdc3e693d28b"
    )


def test_comparison_requires_float_tolerance_and_exact_indices() -> None:
    float_reference = np.asarray([0.0, 1.0], dtype=np.float32)
    float_candidate = np.asarray([0.0, 1.0 + 5.0e-7], dtype=np.float32)
    assert runner.compare_arrays(float_reference, float_candidate)["passed"] is True

    float_candidate[1] = np.float32(1.0 + 2.0e-6)
    assert runner.compare_arrays(float_reference, float_candidate)["passed"] is False

    indices = np.asarray([[1, 0]], dtype=np.int64)
    changed = np.asarray([[0, 1]], dtype=np.int64)
    assert runner.compare_arrays(indices, indices.copy())["passed"] is True
    assert runner.compare_arrays(indices, changed)["passed"] is False


def test_archive_inventory_rejects_duplicate_and_unsafe_members(tmp_path: Path) -> None:
    valid = tmp_path / "valid.pt2"
    with zipfile.ZipFile(valid, "w") as archive:
        archive.writestr("model/model.pyd", b"binary")
        archive.writestr("model/metadata.json", b"{}")
    assert [row["name"] for row in runner._archive_inventory(valid)] == [
        "model/model.pyd",
        "model/metadata.json",
    ]

    duplicate = tmp_path / "duplicate.pt2"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(duplicate, "w") as archive:
            archive.writestr("model/model.pyd", b"one")
            archive.writestr("model/model.pyd", b"two")
    with pytest.raises(runner.NativeFeasibilityError, match="duplicate archive"):
        runner._archive_inventory(duplicate)

    unsafe = tmp_path / "unsafe.pt2"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../escape.pyd", b"binary")
    with pytest.raises(runner.NativeFeasibilityError, match="path is unsafe"):
        runner._archive_inventory(unsafe)


def test_cpp_contract_is_lean_and_temp_contained() -> None:
    cmake = (PROJECT_ROOT / "tools/axon_tiny_pytorch_native/CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    aoti = (PROJECT_ROOT / "tools/axon_tiny_pytorch_native/src/aoti_probe.cpp").read_text(
        encoding="utf-8"
    )
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    aoti_target = cmake.split("add_executable(axon_tiny_aoti_probe", 1)[1].split(
        "add_executable(axon_tiny_libtorch_probe", 1
    )[0]
    assert "torch_python" not in aoti_target
    assert "python314" not in aoti_target
    assert "torch_cpu.lib" in cmake
    assert "AOTI_DEVICE_KEY" in aoti
    assert "AXON_AOTI_DISPOSABLE_TEMP" in source
    assert '"TEMP": str(temp_root)' in source
    assert "external_safety_sentinel" in source
