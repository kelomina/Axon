from __future__ import annotations

import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pre_run_resource_leak_guard import MemorySnapshot, evaluate_guard, validate_guard_receipt  # noqa: E402


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
    assert payload["python_processes"]["total_rss_mb"] == 4096.0


def test_guard_blocks_too_many_python_processes():
    with _case_dir("loop77_python_count_fail") as tmp_path:
        script = tmp_path / "safe.py"
        script.write_text("print('ok')\n", encoding="utf-8")
        payload = evaluate_guard(
            target_scripts=[script],
            max_python_process_count=2,
            memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=600),
            python_processes=[
                {"pid": 1, "name": "python", "rss_mb": 100.0, "cpu_seconds": 0.0},
                {"pid": 2, "name": "python", "rss_mb": 100.0, "cpu_seconds": 0.0},
                {"pid": 3, "name": "python", "rss_mb": 100.0, "cpu_seconds": 0.0},
            ],
            gpu_summary=_gpu_ok(),
        )

    assert payload["guard_ready"] is False
    assert "python_process_count_exceeds_limit" in payload["failures"]


def test_guard_blocks_total_python_rss():
    with _case_dir("loop77_python_total_rss_fail") as tmp_path:
        script = tmp_path / "safe.py"
        script.write_text("print('ok')\n", encoding="utf-8")
        payload = evaluate_guard(
            target_scripts=[script],
            max_python_rss_mb=1024,
            max_total_python_rss_mb=2048,
            memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=600),
            python_processes=[
                {"pid": 1, "name": "python", "rss_mb": 900.0, "cpu_seconds": 0.0},
                {"pid": 2, "name": "python", "rss_mb": 900.0, "cpu_seconds": 0.0},
                {"pid": 3, "name": "python", "rss_mb": 900.0, "cpu_seconds": 0.0},
            ],
            gpu_summary=_gpu_ok(),
        )

    assert payload["guard_ready"] is False
    assert "python_total_rss_exceeds_limit" in payload["failures"]
    assert "python_process_rss_exceeds_limit" not in payload["failures"]


def test_guard_blocks_second_gpu_memory_pressure():
    with _case_dir("loop77_second_gpu_fail") as tmp_path:
        script = tmp_path / "safe.py"
        script.write_text("print('ok')\n", encoding="utf-8")
        gpu_summary = {
            "available": True,
            "memory_used_mb": 100,
            "memory_total_mb": 1000,
            "memory_used_pct": 10.0,
            "utilization_pct": 0,
            "devices": [
                {"index": 0, "memory_used_mb": 100, "memory_total_mb": 1000, "memory_used_pct": 10.0},
                {"index": 1, "memory_used_mb": 980, "memory_total_mb": 1000, "memory_used_pct": 98.0},
            ],
            "compute_app_count": 0,
            "python_compute_app_count": 0,
            "python_compute_apps": [],
        }
        payload = evaluate_guard(
            target_scripts=[script],
            max_gpu_memory_used_pct=95.0,
            memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=600),
            python_processes=[],
            gpu_summary=gpu_summary,
        )

    assert payload["guard_ready"] is False
    assert "gpu_memory_used_pct_exceeds_limit" in payload["failures"]


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


def test_guard_follows_local_imports_when_requested():
    with _case_dir("loop77_follow_imports") as tmp_path:
        wrapper = tmp_path / "wrapper.py"
        heavy = tmp_path / "heavy.py"
        wrapper.write_text("import heavy\nprint('wrapper only')\n", encoding="utf-8")
        heavy.write_text("import torch\n", encoding="utf-8")
        common_kwargs = {
            "target_scripts": [wrapper],
            "memory_snapshot": MemorySnapshot(total_mb=1000, available_mb=600),
            "python_processes": [],
            "gpu_summary": _gpu_ok(),
        }

        without_follow = evaluate_guard(**common_kwargs)
        with_follow = evaluate_guard(**common_kwargs, follow_local_imports=True)

    assert without_follow["guard_ready"] is True
    assert with_follow["guard_ready"] is False
    assert "static_risk_patterns_detected" in with_follow["failures"]
    assert str(heavy.resolve(strict=False)) in with_follow["static_scan"]["scanned_files"]
    risk_ids = {item["risk_id"] for item in with_follow["static_scan"]["findings"]}
    assert "torch_import" in risk_ids


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
            allowed_risks={"npz_array_load", "array_or_object_load"},
            memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=600),
            python_processes=[],
            gpu_summary=_gpu_ok(),
        )

    assert payload["guard_ready"] is True
    assert payload["static_scan"]["finding_count"] == 0
    assert payload["limits"]["allowed_static_risks"] == ["array_or_object_load", "npz_array_load"]


def test_guard_blocks_ast_whole_file_reads():
    with _case_dir("loop77_ast_whole_file") as tmp_path:
        script = tmp_path / "whole_file.py"
        script.write_text(
            "\n".join(
                [
                    "from pathlib import Path",
                    "payload = Path('x.txt').read_text()",
                    "with open('y.bin', 'rb') as handle:",
                    "    payload2 = handle.readlines()",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = evaluate_guard(
            target_scripts=[script],
            memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=600),
            python_processes=[],
            gpu_summary=_gpu_ok(),
        )

    assert payload["guard_ready"] is False
    risk_ids = {item["risk_id"] for item in payload["static_scan"]["findings"]}
    assert "whole_file_read" in risk_ids


def test_guard_blocks_ast_reader_and_directory_materialization():
    with _case_dir("loop77_ast_materialization") as tmp_path:
        script = tmp_path / "materialize.py"
        script.write_text(
            "\n".join(
                [
                    "import csv",
                    "from pathlib import Path",
                    "with open('rows.csv', newline='') as handle:",
                    "    rows = list(csv.DictReader(handle))",
                    "root = Path('.')",
                    "files = list(root.rglob('*.npz'))",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = evaluate_guard(
            target_scripts=[script],
            memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=600),
            python_processes=[],
            gpu_summary=_gpu_ok(),
        )

    assert payload["guard_ready"] is False
    risk_ids = {item["risk_id"] for item in payload["static_scan"]["findings"]}
    assert "reader_materialization" in risk_ids
    assert "directory_materialization" in risk_ids


def test_guard_blocks_ast_executor_map_patterns():
    with _case_dir("loop77_ast_executor_map") as tmp_path:
        script = tmp_path / "executor_map.py"
        script.write_text(
            "\n".join(
                [
                    "from concurrent.futures import ThreadPoolExecutor",
                    "def fn(row):",
                    "    return row",
                    "rows = range(100)",
                    "with ThreadPoolExecutor() as executor:",
                    "    results = list(executor.map(fn, rows))",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = evaluate_guard(
            target_scripts=[script],
            allowed_risks={"thread_pool"},
            memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=600),
            python_processes=[],
            gpu_summary=_gpu_ok(),
        )

    assert payload["guard_ready"] is False
    risk_ids = {item["risk_id"] for item in payload["static_scan"]["findings"]}
    assert "executor_map_unbounded" in risk_ids


def test_guard_ast_ignores_streaming_safe_patterns():
    with _case_dir("loop77_ast_streaming_safe") as tmp_path:
        script = tmp_path / "streaming_safe.py"
        script.write_text(
            "\n".join(
                [
                    "import csv",
                    "import itertools",
                    "from pathlib import Path",
                    "with open('rows.csv', newline='') as handle:",
                    "    reader = csv.DictReader(handle)",
                    "    for row in reader:",
                    "        pass",
                    "with open('sample.bin', 'rb') as handle:",
                    "    chunk = handle.read(4096)",
                    "with open('rows.csv', newline='') as handle:",
                    "    reader = csv.DictReader(handle)",
                    "    preview = list(itertools.islice(reader, 10))",
                    "root = Path('.')",
                    "for path in root.rglob('*.npz'):",
                    "    pass",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = evaluate_guard(
            target_scripts=[script],
            memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=600),
            python_processes=[],
            gpu_summary=_gpu_ok(),
        )

    assert payload["guard_ready"] is True
    assert payload["static_scan"]["finding_count"] == 0


def test_guard_allows_ast_risk_when_allowed():
    with _case_dir("loop77_ast_allowed") as tmp_path:
        script = tmp_path / "allowed_read_text.py"
        script.write_text("from pathlib import Path\npayload = Path('x').read_text()\n", encoding="utf-8")
        payload = evaluate_guard(
            target_scripts=[script],
            allowed_risks={"whole_file_read"},
            memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=600),
            python_processes=[],
            gpu_summary=_gpu_ok(),
        )

    assert payload["guard_ready"] is True
    assert payload["static_scan"]["finding_count"] == 0


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


def test_guard_receipt_validates_target_hash_command_and_cwd():
    with _case_dir("loop77_receipt_valid") as tmp_path:
        script = tmp_path / "safe.py"
        script.write_text("print('ok')\n", encoding="utf-8")
        command = ["python", str(script)]
        payload = evaluate_guard(
            target_scripts=[script],
            command=command,
            created_at=1000.0,
            memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=600),
            python_processes=[],
            gpu_summary=_gpu_ok(),
        )

        valid = validate_guard_receipt(
            payload,
            expected_target_scripts=[script],
            expected_command=command,
            expected_cwd=Path.cwd(),
            now=1005.0,
        )
        wrong_command = validate_guard_receipt(
            payload,
            expected_target_scripts=[script],
            expected_command=["python", "other.py"],
            expected_cwd=Path.cwd(),
            now=1005.0,
        )
        script.write_text("print('changed')\n", encoding="utf-8")
        changed_target = validate_guard_receipt(
            payload,
            expected_target_scripts=[script],
            expected_command=command,
            expected_cwd=Path.cwd(),
            now=1005.0,
        )

    assert valid["valid"] is True
    assert wrong_command["valid"] is False
    assert "receipt_command_mismatch" in wrong_command["failures"]
    assert changed_target["valid"] is False
    assert "receipt_target_hash_mismatch" in changed_target["failures"]


def test_guard_receipt_rejects_stale_payload():
    with _case_dir("loop77_receipt_stale") as tmp_path:
        script = tmp_path / "safe.py"
        script.write_text("print('ok')\n", encoding="utf-8")
        payload = evaluate_guard(
            target_scripts=[script],
            created_at=1000.0,
            memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=600),
            python_processes=[],
            gpu_summary=_gpu_ok(),
        )
        validation = validate_guard_receipt(
            payload,
            expected_target_scripts=[script],
            max_age_seconds=60.0,
            now=1100.0,
        )

    assert validation["valid"] is False
    assert "receipt_expired" in validation["failures"]
