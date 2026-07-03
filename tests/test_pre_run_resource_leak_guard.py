from __future__ import annotations

import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pre_run_resource_leak_guard import MemorySnapshot, evaluate_guard  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _gpu_ok() -> dict:
    return {
        "available": True,
        "memory_used_mb": 100,
        "memory_total_mb": 1000,
        "memory_used_pct": 10.0,
        "utilization_pct": 0,
        "compute_app_count": 0,
        "python_compute_app_count": 0,
        "python_compute_apps": [],
    }


def test_guard_passes_low_risk_script():
    with _case_dir("loop77_guard_pass") as tmp_path:
        script = tmp_path / "safe.py"
        script.write_text("print('metadata only')\n", encoding="utf-8")
        payload = evaluate_guard(
            target_scripts=[script],
            memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=600),
            python_processes=[],
            gpu_summary=_gpu_ok(),
        )

    assert payload["guard_ready"] is True
    assert payload["decision"] == "pass"
    assert payload["static_scan"]["finding_count"] == 0


def test_guard_blocks_high_system_memory():
    with _case_dir("loop77_memory_fail") as tmp_path:
        script = tmp_path / "safe.py"
        script.write_text("print('ok')\n", encoding="utf-8")
        payload = evaluate_guard(
            target_scripts=[script],
            max_system_used_pct=90.0,
            memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=50),
            python_processes=[],
            gpu_summary=_gpu_ok(),
        )

    assert payload["guard_ready"] is False
    assert "system_memory_used_pct_exceeds_limit" in payload["failures"]


def test_guard_blocks_heavy_python_process():
    with _case_dir("loop77_python_fail") as tmp_path:
        script = tmp_path / "safe.py"
        script.write_text("print('ok')\n", encoding="utf-8")
        payload = evaluate_guard(
            target_scripts=[script],
            max_python_rss_mb=1024,
            memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=600),
            python_processes=[{"pid": 123, "name": "python", "rss_mb": 4096.0, "cpu_seconds": 1.0}],
            gpu_summary=_gpu_ok(),
        )

    assert payload["guard_ready"] is False
    assert "python_process_rss_exceeds_limit" in payload["failures"]


def test_guard_blocks_static_risk_patterns():
    with _case_dir("loop77_static_fail") as tmp_path:
        script = tmp_path / "risky.py"
        script.write_text("import torch\nwhile True:\n    pass\n", encoding="utf-8")
        payload = evaluate_guard(
            target_scripts=[script],
            memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=600),
            python_processes=[],
            gpu_summary=_gpu_ok(),
        )

    assert payload["guard_ready"] is False
    assert "static_risk_patterns_detected" in payload["failures"]
    risk_ids = {item["risk_id"] for item in payload["static_scan"]["findings"]}
    assert {"torch_import", "infinite_loop"} <= risk_ids


def test_guard_ignores_risk_words_inside_string_literals():
    with _case_dir("loop77_string_literal") as tmp_path:
        script = tmp_path / "fixture_writer.py"
        script.write_text('sample = "import torch\\nwhile True:\\n    pass\\n"\nprint(sample)\n', encoding="utf-8")
        payload = evaluate_guard(
            target_scripts=[script],
            memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=600),
            python_processes=[],
            gpu_summary=_gpu_ok(),
        )

    assert payload["guard_ready"] is True
    assert payload["static_scan"]["finding_count"] == 0


def test_guard_allows_explicitly_allowed_static_risk():
    with _case_dir("loop77_static_allowed") as tmp_path:
        script = tmp_path / "uses_np_load.py"
        script.write_text("np.load('features.npz')\n", encoding="utf-8")
        payload = evaluate_guard(
            target_scripts=[script],
            allowed_risks={"npz_array_load"},
            memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=600),
            python_processes=[],
            gpu_summary=_gpu_ok(),
        )

    assert payload["guard_ready"] is True
    assert payload["static_scan"]["finding_count"] == 0
    assert payload["limits"]["allowed_static_risks"] == ["npz_array_load"]


def test_guard_blocks_missing_target_script():
    with _case_dir("loop77_missing") as tmp_path:
        payload = evaluate_guard(
            target_scripts=[tmp_path / "missing.py"],
            memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=600),
            python_processes=[],
            gpu_summary=_gpu_ok(),
        )

    assert payload["guard_ready"] is False
    assert "target_script_missing" in payload["failures"]
    assert len(payload["static_scan"]["missing_files"]) == 1
