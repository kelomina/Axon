from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_loop28_pytorch_native_feasibility_manifest.py"
SPEC = importlib.util.spec_from_file_location(
    "build_loop28_pytorch_native_feasibility_manifest", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_implementation_manifest_binds_lean_safe_runtime_sources() -> None:
    payload = builder.build_implementation_manifest(
        PROJECT_ROOT,
        generated_at_utc="2026-07-11T23:10:00Z",
    )

    assert payload["decision"] == (
        "tiny_native_feasibility_source_frozen_build_requires_authorization"
    )
    assert payload["integrity"]["artifact_count"] == 14
    assert payload["build_contract"]["aoti_link_libraries"] == [
        "torch_cpu.lib",
        "c10.lib",
    ]
    assert payload["contract"]["direct_aten_is_toolchain_control"] is True
    assert payload["contract"]["normal_user_temp_for_aoti_forbidden"] is True


def test_design_amendment_rejects_python_and_cuda_dependencies() -> None:
    chain = builder._verify_chain(PROJECT_ROOT)
    forbidden = chain["amendment"]["package_safety_contract"]["forbidden_cpp_dependencies"]

    assert "torch_python.dll" in forbidden
    assert "python314.dll" in forbidden
    assert "torch_cuda.dll" in forbidden
    assert chain["amendment"]["runtime_contract"]["fresh_process_count_per_native_lane"] == 3


def test_post_builder_waits_for_build_receipt() -> None:
    if (PROJECT_ROOT / builder.BUILD_RECEIPT).is_file():
        pytest.skip("build receipt exists after the one-shot build phase")

    with pytest.raises(builder.NativeManifestError, match="Required artifact"):
        builder.build_post_manifest(
            PROJECT_ROOT,
            generated_at_utc="2026-07-11T23:15:00Z",
        )


def test_strict_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    artifact = tmp_path / "duplicate.json"
    artifact.write_text('{"schema": "a", "schema": "b"}', encoding="utf-8")

    with pytest.raises(builder.NativeManifestError, match="Duplicate JSON key"):
        builder.load_json_strict(artifact)
