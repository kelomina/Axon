from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_loop28_pytorch_native_negative_closure.py"
    )
    spec = importlib.util.spec_from_file_location("loop28_native_negative_closure", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_negative_chain_verifies_current_immutable_evidence() -> None:
    module = _load_module()
    root = Path(__file__).resolve().parents[1]
    chain = module.verify_negative_chain(root)
    assert chain["attempt_002"]["budget_reconciliation"]["aoti_compile_and_package_calls"] == 1
    assert chain["attempt_002"]["partial_artifact_audit"]["aoti_package_created"] is False
    assert chain["controller_failure"]["package_load_count"] == 0


def test_failure_manifest_reports_negative_not_incomplete() -> None:
    module = _load_module()
    root = Path(__file__).resolve().parents[1]
    payload = module.build_failure_manifest(root, generated_at_utc="2026-07-12T00:10:00Z")
    assert payload["decision"] == (
        "tiny_aoti_package_generation_closed_utf8_cp936_compiler_probe_failure_no_load"
    )
    assert payload["outcome"]["aoti_compile_and_package_call_count"] == 1
    assert payload["outcome"]["aoti_package_created"] is False
    assert payload["outcome"]["native_runtime_execution_count"] == 0
    assert payload["outcome"]["quality_metric_count"] == 0
    assert payload["integrity"]["positive_package_artifacts_absent"] is True


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "duplicate.json"
    path.write_text('{"decision": "one", "decision": "two"}', encoding="utf-8")
    with pytest.raises(module.NegativeClosureError, match="Duplicate JSON key"):
        module.load_json_strict(path)


@pytest.mark.parametrize(
    "timestamp",
    ["", "2026-07-12T00:10:00", "not-a-timestampZ", "2026-07-12T08:10:00+08:00Z"],
)
def test_timestamp_validation_fails_closed(timestamp: str) -> None:
    module = _load_module()
    with pytest.raises(module.NegativeClosureError):
        module._validate_timestamp(timestamp)
