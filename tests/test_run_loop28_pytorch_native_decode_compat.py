from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import zipfile
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_loop28_pytorch_native_decode_compat.py"
    )
    spec = importlib.util.spec_from_file_location("loop28_native_decode_compat", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_manifest_builder():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_loop28_pytorch_native_decode_compat_manifest.py"
    )
    spec = importlib.util.spec_from_file_location("loop28_native_decode_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_successor_input_is_fresh_and_hash_bound() -> None:
    module = _load_module()
    array, payload = module._new_input_array()
    assert array.shape == (2, 8)
    assert len(payload) == 64
    assert module._sha256_bytes(payload) == module.EXPECTED_INPUT_SHA256
    assert module.EXPECTED_INPUT_SHA256 != module.PARENT_PARTIAL_INPUT_SHA256


def test_worker_script_sets_environment_before_default_locale_worker(tmp_path: Path) -> None:
    module = _load_module()
    payload = module._worker_script_bytes(
        stage="preflight",
        ownership_token="a" * 64,
    ).decode("ascii")
    assert "-X utf8=0" in payload
    assert "chcp" not in payload.casefold()
    assert str(module.PYTHON_EXE.resolve()) in payload
    assert 'set "TORCHINDUCTOR_COMPILE_THREADS=1"' in payload
    assert payload.index('if not exist "job_assigned.flag"') < payload.index("vcvars64.bat")
    assert payload.index('set "TEMP=') < payload.index("-X utf8=0")
    assert payload.index('set "TORCHINDUCTOR_CACHE_DIR=') < payload.index("-X utf8=0")


def test_windows_job_object_assigns_before_release_gate(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows Job Objects are Windows-only")
    module = _load_module()
    script = tmp_path / "wait_for_job.cmd"
    script.write_text(
        "@echo off\r\n:wait\r\nif not exist release.flag goto wait\r\nexit /b 0\r\n",
        encoding="ascii",
    )
    job = module._WindowsKillOnCloseJob()
    process = subprocess.Popen(
        ["cmd.exe", "/d", "/c", script.name],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        job.assign(process)
        (tmp_path / "release.flag").write_bytes(b"release\n")
        process.communicate(timeout=10)
        assert process.returncode == 0
        assert job.wait_empty(module.time.perf_counter() + 10) == 0
    finally:
        if process.poll() is None:
            job.terminate(module.time.perf_counter() + 10)
            process.wait(timeout=10)
        job.close()


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema": "one", "schema": "two"}', encoding="utf-8")
    with pytest.raises(module.DecodeCompatError, match="Duplicate JSON key"):
        module.load_json_strict(path)


def test_strict_json_rejects_nonfinite_numbers(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "nonfinite.json"
    path.write_text('{"budget": NaN}', encoding="utf-8")
    with pytest.raises(module.DecodeCompatError, match="Non-finite"):
        module.load_json_strict(path)


def test_archive_accepts_one_regular_precompiled_pyd(tmp_path: Path) -> None:
    module = _load_module()
    package = tmp_path / "safe.pt2"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model/model.pyd", b"precompiled")
        archive.writestr("model/metadata.json", json.dumps({"device": "cpu"}))
    audit = module.audit_package_archive(package)
    assert audit["precompiled_pyd_count"] == 1
    assert audit["member_count"] == 2
    assert audit["special_members"] == 0


@pytest.mark.parametrize(
    "member",
    [
        "../model.pyd",
        "/model.pyd",
        "C:/model.pyd",
        "model/stream:ads.pyd",
        "model/NUL.pyd",
        "model/trailing./model.pyd",
        "model/trailing /model.pyd",
    ],
)
def test_archive_rejects_windows_unsafe_paths(tmp_path: Path, member: str) -> None:
    module = _load_module()
    package = tmp_path / "unsafe.pt2"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(member, b"precompiled")
    with pytest.raises(module.DecodeCompatError, match="archive"):
        module.audit_package_archive(package)


def test_archive_rejects_backslash_and_casefold_collision(tmp_path: Path) -> None:
    module = _load_module()
    backslash = tmp_path / "backslash.pt2"
    with zipfile.ZipFile(backslash, "w") as archive:
        archive.writestr("dir/model.pyd", b"precompiled")
    backslash.write_bytes(backslash.read_bytes().replace(b"dir/model.pyd", b"dir\\model.pyd"))
    with pytest.raises(module.DecodeCompatError, match="unsafe raw path"):
        module.audit_package_archive(backslash)

    collision = tmp_path / "collision.pt2"
    with zipfile.ZipFile(collision, "w") as archive:
        archive.writestr("model/Kernel.pyd", b"one")
        archive.writestr("MODEL/kernel.PYD", b"two")
    with pytest.raises(module.DecodeCompatError, match="collision"):
        module.audit_package_archive(collision)


def test_archive_rejects_special_and_multiple_pyd_members(tmp_path: Path) -> None:
    module = _load_module()
    special = tmp_path / "special.pt2"
    link = zipfile.ZipInfo("model/model.pyd")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(special, "w") as archive:
        archive.writestr(link, b"target")
    with pytest.raises(module.DecodeCompatError, match="special member"):
        module.audit_package_archive(special)

    multiple = tmp_path / "multiple.pt2"
    with zipfile.ZipFile(multiple, "w") as archive:
        archive.writestr("model/one.pyd", b"one")
        archive.writestr("model/two.pyd", b"two")
    with pytest.raises(module.DecodeCompatError, match="exactly one"):
        module.audit_package_archive(multiple)


@pytest.mark.parametrize(
    "name",
    ["python314.dll", "torch_python.dll", "torch.dll", "torch_cuda.dll", "cudart64_13.dll"],
)
def test_dependency_policy_rejects_python_torch_and_cuda(name: str) -> None:
    module = _load_module()
    assert module._forbidden_dependency(name) is True


def test_lease_consumption_is_exclusive_and_records_time(tmp_path: Path) -> None:
    module = _load_module()
    authorization_path = Path("authorization.json")
    ready_path = Path("lease.json")
    final_path = Path("lease.final.json")
    authorization = {
        "schema": "test_authorization_v1",
        "decision": "authorize_test",
        "source_artifacts": [
            {
                "name": "source",
                "path": "source.py",
                "sha256": "",
                "size_bytes": 0,
            }
        ],
    }
    source = tmp_path / "source.py"
    source.write_text("pass\n", encoding="utf-8")
    authorization["source_artifacts"][0]["sha256"] = module.sha256_file(source)
    authorization["source_artifacts"][0]["size_bytes"] = source.stat().st_size
    authorization_file = tmp_path / authorization_path
    authorization_file.write_text(json.dumps(authorization), encoding="utf-8")
    authorization_sha = module.sha256_file(authorization_file)
    ready = {
        "schema": "test_lease_v1",
        "status": "ready",
        "single_use": True,
        "authorization_sha256": authorization_sha,
    }
    (tmp_path / ready_path).write_text(json.dumps(ready), encoding="utf-8")
    consumed = module._consume_lease(
        tmp_path,
        authorization_path=authorization_path,
        ready_path=ready_path,
        final_path=final_path,
        authorization_schema="test_authorization_v1",
        authorization_decision="authorize_test",
        lease_schema="test_lease_v1",
    )
    assert consumed["status"] == "consumed_before_execution"
    assert not (tmp_path / ready_path).exists()
    final = module.load_json_strict(tmp_path / final_path)
    assert final["consumed_at_utc"].endswith("Z")
    assert final["source_artifacts"] == authorization["source_artifacts"]
    with pytest.raises(module.DecodeCompatError):
        module._consume_lease(
            tmp_path,
            authorization_path=authorization_path,
            ready_path=ready_path,
            final_path=final_path,
            authorization_schema="test_authorization_v1",
            authorization_decision="authorize_test",
            lease_schema="test_lease_v1",
        )


@pytest.mark.parametrize(
    ("counters", "expected_class", "expected_decision"),
    [
        (
            {},
            "administrative",
            "administrative_failure_no_protected_package_call",
        ),
        (
            {"torch_export_calls": 1},
            "pre_export",
            "decode_compat_pre_export_failure_no_package",
        ),
        (
            {"aoti_compile_and_package_calls": 1},
            "protected_call",
            "decode_compat_applied_aoti_compile_or_package_still_unsupported",
        ),
        (
            {
                "aoti_compile_and_package_calls": 1,
                "aoti_compile_and_package_completed": 1,
            },
            "dependency",
            "decode_compat_package_dependency_leakage_no_load",
        ),
        (
            {
                "aoti_compile_and_package_calls": 1,
                "aoti_compile_and_package_completed": 1,
            },
            "static_audit",
            "decode_compat_package_static_audit_failed_no_load",
        ),
        (
            {"aoti_compile_and_package_calls": 1},
            "budget",
            "budget_exhausted_no_claim",
        ),
    ],
)
def test_package_failure_classification(
    counters: dict[str, int], expected_class: str, expected_decision: str
) -> None:
    module = _load_module()
    error = {
        "dependency": "Dependency closure is unsafe: ambiguous=x",
        "budget": "Stage wall-clock budget expired",
    }.get(expected_class, "x")
    actual = module._package_failure_class({"counters": counters}, error)
    assert actual == (expected_class, expected_decision)


def test_package_timeout_is_always_budget_failure() -> None:
    module = _load_module()
    actual = module._package_failure_class(
        {"counters": {"torch_imports": 1}},
        "package worker failed",
        {"timed_out": True},
    )
    assert actual == ("budget", "budget_exhausted_no_claim")


def test_worker_cleanup_rejects_reparse_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    lease = {"sha256": "b" * 64}
    token = module._worker_ownership_token("preflight", lease)
    paths = module._prepare_worker_root(tmp_path, "preflight", token, lease)
    target = tmp_path / "outside"
    target.mkdir()
    link = paths["work_root"] / "escape"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(module.DecodeCompatError, match="reparse"):
        module._remove_owned_worker_root(tmp_path, "preflight", token)
    assert target.is_dir()


def test_dependency_ambiguity_is_path_based_even_for_equal_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    torch_lib = tmp_path / "torch_lib"
    system32 = tmp_path / "Windows" / "System32"
    torch_lib.mkdir()
    system32.mkdir(parents=True)
    (torch_lib / "same.dll").write_bytes(b"same")
    (system32 / "same.dll").write_bytes(b"same")
    start = tmp_path / "model.pyd"
    start.write_bytes(b"start")
    monkeypatch.setattr(module, "TORCH_LIB", torch_lib)
    monkeypatch.setenv("SystemRoot", str(tmp_path / "Windows"))
    monkeypatch.setattr(
        module,
        "_dumpbin_dependency_inventory",
        lambda _path, **_kwargs: {
            "dependencies": ["same.dll"],
            "invocations": [],
            "regular_and_delay_imports_audited": True,
        },
    )
    with pytest.raises(module.DecodeCompatError, match="ambiguous"):
        module.build_dependency_closure({"generated_pyd": start})


def test_consumed_stage_lease_round_trips_through_manifest_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_module()
    builder = _load_manifest_builder()
    authorization_path = Path("manifests/preflight_authorization.json")
    ready_path = Path("manifests/preflight_lease.json")
    final_path = Path("manifests/preflight_lease.final.json")
    artifact_root = Path("artifacts/tiny_v2/package_attempt_001")
    work_root = Path("reports/work/decode_probe_attempt_001")
    terminal_outputs = [Path("reports/success.json"), Path("reports/failure.json")]
    source = tmp_path / "source.py"
    source.write_text("pass\n", encoding="utf-8")
    command = [
        "E:/Project/python/Axon_v2.6Exp/vnev/Scripts/python.exe",
        runner.RUNNER.as_posix(),
        "preflight",
    ]
    budget = dict(builder.PREFLIGHT_BUDGET)
    authorization = {
        "schema": builder.PREFLIGHT_AUTHORIZATION_SCHEMA,
        "loop_id": builder.LOOP_ID,
        "attempt_id": builder.PREFLIGHT_ATTEMPT_ID,
        "decision": builder.PREFLIGHT_AUTHORIZATION_DECISION,
        "canonical_command": command,
        "artifact_root": artifact_root.as_posix(),
        "work_root": work_root.as_posix(),
        "terminal_outputs": [path.as_posix() for path in terminal_outputs],
        "budget": budget,
        "source_artifacts": [
            {
                "path": "source.py",
                "sha256": runner.sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        ],
    }
    authorization_file = tmp_path / authorization_path
    authorization_file.parent.mkdir(parents=True)
    authorization_file.write_text(json.dumps(authorization), encoding="utf-8")
    ready = {
        "schema": builder.PREFLIGHT_LEASE_SCHEMA,
        "loop_id": builder.LOOP_ID,
        "lease_id": builder.PREFLIGHT_ATTEMPT_ID,
        "attempt_id": builder.PREFLIGHT_ATTEMPT_ID,
        "status": "ready",
        "single_use": True,
        "authorization_path": authorization_path.as_posix(),
        "authorization_sha256": runner.sha256_file(authorization_file),
        "canonical_command": command,
        "artifact_root": artifact_root.as_posix(),
        "work_root": work_root.as_posix(),
        "terminal_outputs": [path.as_posix() for path in terminal_outputs],
        "budget_sha256": runner._canonical_json_sha256(budget),
        "consumed_path": final_path.as_posix(),
    }
    ready_file = tmp_path / ready_path
    ready_file.write_text(json.dumps(ready), encoding="utf-8")
    runner._consume_lease(
        tmp_path,
        authorization_path=authorization_path,
        ready_path=ready_path,
        final_path=final_path,
        authorization_schema=builder.PREFLIGHT_AUTHORIZATION_SCHEMA,
        authorization_decision=builder.PREFLIGHT_AUTHORIZATION_DECISION,
        lease_schema=builder.PREFLIGHT_LEASE_SCHEMA,
    )
    monkeypatch.setattr(builder, "ARTIFACT_ROOT", artifact_root)
    parsed_authorization, parsed_lease = builder._validate_stage_chain(
        tmp_path,
        authorization_path=authorization_path,
        final_lease_path=final_path,
        authorization_schema=builder.PREFLIGHT_AUTHORIZATION_SCHEMA,
        authorization_decision=builder.PREFLIGHT_AUTHORIZATION_DECISION,
        lease_schema=builder.PREFLIGHT_LEASE_SCHEMA,
        attempt_id=builder.PREFLIGHT_ATTEMPT_ID,
        mode="preflight",
        work_root=work_root,
        output_paths=terminal_outputs,
    )
    assert parsed_authorization == authorization
    assert parsed_lease["status"] == "consumed_before_execution"


def test_runner_and_manifest_builder_share_frozen_stage_budgets() -> None:
    runner = _load_module()
    builder = _load_manifest_builder()
    assert runner.PREFLIGHT_BUDGET == builder.PREFLIGHT_BUDGET
    assert runner.PACKAGE_BUDGET == builder.PACKAGE_BUDGET


def test_dependency_closure_stops_before_exceeding_dumpbin_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    torch_lib = tmp_path / "torch_lib"
    system32 = tmp_path / "Windows" / "System32"
    torch_lib.mkdir()
    system32.mkdir(parents=True)
    starts: dict[str, Path] = {}
    for index in range(33):
        path = tmp_path / f"node_{index}.pyd"
        path.write_bytes(str(index).encode("ascii"))
        starts[f"node_{index}"] = path
    calls = 0

    def fake_inventory(_path: Path, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "dependencies": [],
            "invocations": [{}, {}],
            "regular_and_delay_imports_audited": True,
        }

    monkeypatch.setattr(module, "TORCH_LIB", torch_lib)
    monkeypatch.setenv("SystemRoot", str(tmp_path / "Windows"))
    monkeypatch.setattr(module, "_dumpbin_dependency_inventory", fake_inventory)
    with pytest.raises(module.DecodeCompatError, match="dumpbin process budget"):
        module.build_dependency_closure(starts)
    assert calls == 32
