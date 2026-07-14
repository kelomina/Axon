from __future__ import annotations

import hashlib
import hmac
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import diagnose_loop28_parity as diagnostic  # noqa: E402


def _trace() -> diagnostic.PythonDiagnosticTrace:
    probabilities = np.asarray([0.75, 0.25], dtype=np.float32)
    return diagnostic.PythonDiagnosticTrace(
        components={
            "byte_seq": np.asarray([77, 90, 1, 2], dtype=np.int64),
            "pe_features": np.asarray([1.25, -2.5, 3.75], dtype=np.float32),
            "stat_features": np.asarray([0.5, 1.5], dtype=np.float32),
            "base_logits": np.asarray([2.0, 0.0], dtype=np.float32),
            "base_probabilities": probabilities,
            "stage2_features": np.asarray([0.25, 0.0625, 0.25, -1.0], dtype=np.float32),
        },
        prediction={
            "prediction": 0,
            "prob_benign": 0.9,
            "prob_malicious": 0.1,
            "base_model": {
                "prediction": 0,
                "prob_benign": float(probabilities[0]),
                "prob_malicious": float(probabilities[1]),
            },
            "stage2": {"prob_malicious": 0.1},
        },
        stage2_boundaries=(
            diagnostic.FeatureBoundary("base_probability_transforms", 0, 3),
            diagnostic.FeatureBoundary("pe_features", 3, 1),
        ),
    )


def _native_prediction(
    trace: diagnostic.PythonDiagnosticTrace,
    key: bytes,
    native_arrays: dict[str, np.ndarray],
    *,
    component: str | None = None,
    block_elements: int | None = None,
    stage2_probability: float = 0.1,
) -> dict:
    names = [component] if component is not None else list(diagnostic.COMPONENT_ORDER)
    components = {}
    for name in names:
        array = native_arrays[name]
        dtype = diagnostic.EXPECTED_COMPONENT_DTYPES[name]
        record = {
            "dtype": dtype,
            "shape": list(array.shape),
            "digest": diagnostic.tensor_hmac(
                key,
                name=name,
                dtype=dtype,
                array=array,
            ),
        }
        if component is not None:
            assert block_elements == 1
            record["blocks"] = [
                {
                    "start": index,
                    "count": 1,
                    "digest": diagnostic.tensor_hmac(
                        key,
                        name=name,
                        dtype=dtype,
                        array=array,
                        start=index,
                        count=1,
                    ),
                }
                for index in range(array.size)
            ]
        components[name] = record
    return {
        "ok": True,
        "prediction": 0,
        "prob_benign": 1.0 - stage2_probability,
        "prob_malicious": stage2_probability,
        "base_model": {
            "prediction": 0,
            "prob_benign": 0.75,
            "prob_malicious": 0.25,
        },
        "stage2": {"enabled": True, "prob_malicious": stage2_probability},
        "diagnostics": {
            "schema": diagnostic.NATIVE_DIAGNOSTIC_SCHEMA,
            "encoding": diagnostic.TENSOR_ENCODING,
            "digest": "hmac-sha256",
            "components": components,
        },
    }


def test_tensor_hmac_uses_exact_canonical_little_endian_message():
    key = bytes(range(32))
    values = np.asarray([0x0102, -7], dtype=">i8")
    raw = np.asarray(values, dtype="<i8").tobytes()
    message = (
        b"axon_tensor_le_v1\0" + b"byte_seq\0" + b"i64le\0" + struct.pack("<QQQ", 2, 0, 2) + raw
    )
    expected = hmac.new(key, message, hashlib.sha256).hexdigest()

    assert (
        diagnostic.tensor_hmac(
            key,
            name="byte_seq",
            dtype="i64le",
            array=values,
        )
        == expected
    )


def test_diagnose_trace_localizes_each_mismatched_feature_once():
    trace = _trace()
    native_arrays = {name: value.copy() for name, value in trace.components.items()}
    native_arrays["pe_features"][1] = np.float32(-2.25)
    native_arrays["stage2_features"][3] = np.float32(-0.75)
    calls = []

    def runner(key: bytes, component: str | None, block_elements: int | None):
        calls.append((len(key), component, block_elements))
        return _native_prediction(
            trace,
            key,
            native_arrays,
            component=component,
            block_elements=block_elements,
        )

    result = diagnostic.diagnose_trace(trace, native_runner=runner)
    by_name = {row["name"]: row for row in result["component_results"]}

    assert calls == [(32, None, None), (32, "pe_features", 1), (32, "stage2_features", 1)]
    assert by_name["pe_features"]["mismatch_indices"] == [1]
    assert by_name["stage2_features"]["mismatch_indices"] == [3]
    assert by_name["stat_features"]["drilldown_executions"] == 0
    assert result["mismatched_components"] == ["pe_features", "stage2_features"]
    assert result["stage_boundaries"]["first_mismatch_stage"] == "feature_extraction"
    assert result["stage2_feature_boundaries"][1]["mismatch_count"] == 1
    assert result["execution_counts"] == {"python": 1, "native": 3, "crossfeed": 3}
    assert result["decision"] == "first_divergence_localized"


def test_non_drilldown_byte_mismatch_is_localized_to_stage_boundary():
    trace = _trace()
    native_arrays = {name: value.copy() for name, value in trace.components.items()}
    native_arrays["byte_seq"][0] = 88
    calls = []

    def runner(key: bytes, component: str | None, block_elements: int | None):
        calls.append((component, block_elements))
        return _native_prediction(trace, key, native_arrays)

    result = diagnostic.diagnose_trace(trace, native_runner=runner)

    assert calls == [(None, None)]
    assert result["component_results"][0]["mismatch_count"] is None
    assert result["stage_boundaries"]["first_mismatch_stage"] == "feature_extraction"
    assert result["decision"] == "first_divergence_localized"


def test_non_drilldown_base_mismatch_is_localized_to_base_boundary():
    trace = _trace()
    native_arrays = {name: value.copy() for name, value in trace.components.items()}
    native_arrays["base_logits"][1] = np.float32(0.125)

    def runner(key: bytes, component: str | None, block_elements: int | None):
        return _native_prediction(trace, key, native_arrays)

    result = diagnostic.diagnose_trace(trace, native_runner=runner)

    assert result["stage_boundaries"]["first_mismatch_stage"] == "base_inference"
    assert result["decision"] == "first_divergence_localized"


def test_stage2_probability_drift_is_localized_when_all_component_hmacs_match():
    trace = _trace()

    def runner(key: bytes, component: str | None, block_elements: int | None):
        return _native_prediction(
            trace,
            key,
            dict(trace.components),
            stage2_probability=0.10001,
        )

    result = diagnostic.diagnose_trace(trace, native_runner=runner)

    assert result["mismatched_component_count"] == 0
    assert result["stage2_inference"]["probability_within_tolerance"] is False
    assert result["stage_boundaries"]["first_mismatch_stage"] == "stage2_inference"
    assert result["decision"] == "first_divergence_localized"


def test_native_execution_duration_fails_closed_after_120_seconds():
    trace = _trace()
    clock_values = iter([10.0, 130.000001])

    def runner(key: bytes, component: str | None, block_elements: int | None):
        return _native_prediction(trace, key, dict(trace.components))

    with pytest.raises(diagnostic.DiagnosticContractError, match="native execution"):
        diagnostic.diagnose_trace(
            trace,
            native_runner=runner,
            clock=lambda: next(clock_values),
            per_execution_limit_seconds=120.0,
        )


def test_native_runner_passes_ephemeral_key_only_on_stdin(monkeypatch, tmp_path: Path):
    trace = _trace()
    key = b"k" * 32
    payload = _native_prediction(trace, key, dict(trace.components))
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        kwargs["stdout"].write((json.dumps(payload) + "\n").encode("utf-8"))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(diagnostic.subprocess, "run", fake_run)
    paths = {}
    for name in ("sample", "selftest", "dll", "onnx", "stage2"):
        paths[name] = tmp_path / name

    output_sizes = []
    result = diagnostic.run_native_diagnostics(
        sample_path=paths["sample"],
        allowed_raw_root=tmp_path,
        selftest_path=paths["selftest"],
        dll_path=paths["dll"],
        onnx_path=paths["onnx"],
        stage2_path=paths["stage2"],
        key=key,
        timeout_seconds=120,
        max_output_bytes=1024 * 1024,
        output_sizes=output_sizes,
    )

    assert result["diagnostics"]["schema"] == diagnostic.NATIVE_DIAGNOSTIC_SCHEMA
    assert captured["kwargs"]["input"] == (key.hex() + "\n").encode("ascii")
    assert "--parity_diagnostics" in captured["command"]
    assert key.hex() not in captured["command"]
    assert "logits" not in result["base_model"]
    assert "probabilities" not in result["base_model"]
    assert len(output_sizes) == 1 and output_sizes[0] > 0
    with pytest.raises(diagnostic.DiagnosticContractError, match="aggregate output budget"):
        diagnostic.run_native_diagnostics(
            sample_path=paths["sample"],
            allowed_raw_root=tmp_path,
            selftest_path=paths["selftest"],
            dll_path=paths["dll"],
            onnx_path=paths["onnx"],
            stage2_path=paths["stage2"],
            key=key,
            timeout_seconds=120,
            max_output_bytes=output_sizes[0],
            output_sizes=output_sizes,
        )


def test_diagnostic_analysis_contains_no_key_digest_path_or_raw_arrays():
    trace = _trace()
    native_arrays = {name: value.copy() for name, value in trace.components.items()}
    native_arrays["stat_features"][0] = np.float32(0.75)
    issued_keys = [bytes([index]) * 32 for index in (17, 29)]

    def key_factory(_count: int) -> bytes:
        return issued_keys.pop(0)

    def runner(key: bytes, component: str | None, block_elements: int | None):
        return _native_prediction(
            trace,
            key,
            native_arrays,
            component=component,
            block_elements=block_elements,
        )

    result = diagnostic.diagnose_trace(
        trace,
        native_runner=runner,
        key_factory=key_factory,
    )
    encoded = json.dumps(result, sort_keys=True)

    assert '"digest"' not in encoded
    assert '"path"' not in encoded
    assert "raw_arrays" not in encoded
    assert (bytes([17]) * 32).hex() not in encoded
    assert (bytes([29]) * 32).hex() not in encoded


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _artifact_records(root: Path, paths: dict[str, Path]) -> dict:
    return {
        name: {
            "path": path.as_posix(),
            "sha256": diagnostic.replay.file_sha256(root / path),
        }
        for name, path in paths.items()
    }


def test_server_owned_authorizations_bind_parent_and_final_artifacts(monkeypatch, tmp_path: Path):
    parent_paths = {"truth_manifest": Path("parent.json")}
    implementation_paths = {"python_diagnostic": Path("diagnostic.py")}
    runtime_paths = {"checkpoint": Path("checkpoint.bin")}
    implementation_manifest_path = Path("implementation_manifest.json")
    for path in (
        *parent_paths.values(),
        *implementation_paths.values(),
        *runtime_paths.values(),
        implementation_manifest_path,
    ):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.as_posix().encode())
    monkeypatch.setattr(diagnostic, "PARENT_EVIDENCE_PATHS", parent_paths)
    monkeypatch.setattr(diagnostic, "IMPLEMENTATION_ARTIFACT_PATHS", implementation_paths)
    monkeypatch.setattr(diagnostic, "RUNTIME_ARTIFACT_PATHS", runtime_paths)
    monkeypatch.setattr(diagnostic, "DEFAULT_AUTHORIZATION", Path("authorization.json"))
    monkeypatch.setattr(diagnostic, "DEFAULT_RUN_AUTHORIZATION", Path("run_authorization.json"))
    monkeypatch.setattr(
        diagnostic,
        "DEFAULT_IMPLEMENTATION_MANIFEST",
        implementation_manifest_path,
    )

    prereg_payload = {
        "schema": diagnostic.AUTHORIZATION_SCHEMA,
        "loop_id": "p0_loop28_parity_diagnostic_001",
        "authorization_level": "A1_scoped_diagnostic",
        "parent_evidence": _artifact_records(tmp_path, parent_paths),
        "frozen_sample": {
            "split": diagnostic.FIXED_SPLIT,
            "sample_index": diagnostic.FIXED_SAMPLE_INDEX,
            "source_sha256": diagnostic.FIXED_SAMPLE_SHA256,
            "size_bytes": diagnostic.FIXED_SAMPLE_SIZE_BYTES,
        },
        "budget": diagnostic.EXPECTED_BUDGET,
        "timeout_enforcement": diagnostic.EXPECTED_TIMEOUT_ENFORCEMENT,
        "frozen_tolerance": diagnostic.replay.DEFAULT_TOLERANCE,
        "allowed_splits": [diagnostic.FIXED_SPLIT],
        "allowed_logical_raw_root": diagnostic.FIXED_LOGICAL_RAW_ROOT,
        "allowed_resolved_raw_roots": diagnostic.FIXED_RESOLVED_RAW_ROOTS,
        "execution_requires_separate_run_authorization": True,
        "success_gate": "first_divergence_localized",
        "decision": "allow_implementation_and_bounded_train_only_diagnostic_after_run_authorization",
    }
    prereg_path = tmp_path / diagnostic.DEFAULT_AUTHORIZATION
    _write_json(prereg_path, prereg_payload)
    prereg = diagnostic.verify_diagnostic_authorization(tmp_path)
    manifest_sha256 = diagnostic.replay.file_sha256(tmp_path / implementation_manifest_path)
    monkeypatch.setattr(
        diagnostic.implementation_manifest,
        "verify_implementation_manifest",
        lambda _root, _path: {
            "implementation_manifest_sha256": manifest_sha256,
            "parent_evidence_sha256": prereg["parent_evidence_sha256"],
        },
    )

    run_payload = {
        "schema": diagnostic.RUN_AUTHORIZATION_SCHEMA,
        "loop_id": "p0_loop28_parity_diagnostic_001",
        "prereg_authorization_sha256": prereg["authorization_sha256"],
        "parent_evidence": prereg["parent_evidence_sha256"],
        "implementation_artifacts": _artifact_records(tmp_path, implementation_paths),
        "runtime_artifacts": _artifact_records(tmp_path, runtime_paths),
        "implementation_manifest": {
            "path": implementation_manifest_path.as_posix(),
            "sha256": manifest_sha256,
        },
        "frozen_sample": prereg_payload["frozen_sample"],
        "budget": diagnostic.EXPECTED_BUDGET,
        "timeout_enforcement": diagnostic.EXPECTED_TIMEOUT_ENFORCEMENT,
        "frozen_tolerance": diagnostic.replay.DEFAULT_TOLERANCE,
        "attempt_id": diagnostic.FIXED_ATTEMPT_ID,
        "attempt_lease_path": diagnostic.DEFAULT_ATTEMPT_LEASE.as_posix(),
        "generation": "final",
        "output_path": diagnostic.OUTPUT_PATHS_BY_GENERATION["final"].as_posix(),
        "decision": "allow_bounded_loop28_parity_diagnostic_run",
    }
    run_path = tmp_path / diagnostic.DEFAULT_RUN_AUTHORIZATION
    _write_json(run_path, run_payload)

    verified = diagnostic.verify_run_authorization(
        tmp_path,
        diagnostic.DEFAULT_RUN_AUTHORIZATION,
        diagnostic.OUTPUT_PATHS_BY_GENERATION["final"],
        prereg,
    )

    assert verified["status"] == "bounded_run_authorized"
    assert verified["generation"] == "final"
    run_payload["runtime_artifacts"]["checkpoint"]["sha256"] = "0" * 64
    _write_json(run_path, run_payload)
    with pytest.raises(diagnostic.DiagnosticContractError, match="SHA-256 mismatch"):
        diagnostic.verify_run_authorization(
            tmp_path,
            diagnostic.DEFAULT_RUN_AUTHORIZATION,
            diagnostic.OUTPUT_PATHS_BY_GENERATION["final"],
            prereg,
        )


def test_budget_audit_records_every_execution_class_and_fails_closed():
    trace = _trace()

    def runner(key: bytes, component: str | None, block_elements: int | None):
        return _native_prediction(trace, key, dict(trace.components))

    result = diagnostic.diagnose_trace(trace, native_runner=runner)
    audit = diagnostic._build_budget_audit(
        snapshot_duration=0.25,
        python_duration=1.5,
        diagnostic=result,
        total_duration=2.0,
        native_output_sizes=[512],
        require_native_output_accounting=True,
    )

    assert audit["generation"]["count"] == 1
    assert audit["verified_raw_snapshot"]["count"] == 1
    assert audit["python"]["durations_seconds"] == [1.5]
    assert audit["native"]["count"] == 1
    assert audit["crossfeed"]["count"] == 1
    assert audit["total_wall_clock"]["limit_seconds"] == 1200.0
    assert audit["output"]["native_total_bytes"] == 512
    assert audit["within_budget"] is True

    audit["python"]["within_budget"] = False
    audit["within_budget"] = False
    with pytest.raises(diagnostic.DiagnosticContractError, match="exceeded"):
        diagnostic._enforce_budget_audit(audit)


def test_generation_receipt_writer_never_overwrites_existing_file(tmp_path: Path):
    output = tmp_path / "diagnostic_receipt.final.json"
    diagnostic._write_receipt_exclusive(output, {"generation": "final"})

    with pytest.raises(diagnostic.DiagnosticContractError, match="already exists"):
        diagnostic._write_receipt_exclusive(output, {"generation": "final"})


def test_attempt_lease_consumes_run_authorization_before_raw_access(
    monkeypatch,
    tmp_path: Path,
):
    lease_path = Path("attempt.final.json")
    monkeypatch.setattr(diagnostic, "DEFAULT_ATTEMPT_LEASE", lease_path)
    run_authorization = {
        "generation": "final",
        "authorization_sha256": "a" * 64,
    }

    consumed = diagnostic._consume_attempt_lease(tmp_path, run_authorization)
    verified = diagnostic._verify_attempt_lease(tmp_path, consumed)

    assert verified["status"] == "authorization_consumed_before_raw_access"
    with pytest.raises(diagnostic.DiagnosticContractError, match="already consumed"):
        diagnostic._consume_attempt_lease(tmp_path, run_authorization)


def test_run_diagnostic_rejects_before_truth_model_or_raw_access(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        diagnostic,
        "verify_diagnostic_authorization",
        lambda _root: (_ for _ in ()).throw(
            diagnostic.DiagnosticContractError("no prereg authorization")
        ),
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("authorization failure must precede truth, model, and raw access")

    monkeypatch.setattr(
        diagnostic.implementation_manifest,
        "verify_implementation_manifest",
        forbidden,
    )
    monkeypatch.setattr(diagnostic.replay, "guard_pickle_before_load", forbidden)
    monkeypatch.setattr(diagnostic.replay, "snapshot_verified_sample", forbidden)

    with pytest.raises(diagnostic.DiagnosticContractError, match="no prereg"):
        diagnostic.run_diagnostic(
            tmp_path,
            run_authorization_path=diagnostic.DEFAULT_RUN_AUTHORIZATION,
            output_path=diagnostic.OUTPUT_PATHS_BY_GENERATION["final"],
        )


def test_fixed_sample_contract_rejects_any_other_split_identity():
    sample = diagnostic.replay.SampleIdentity(
        source_path="not-recorded",
        source_sha256="f" * 64,
        sample_index=1,
        label=0,
        split="train",
    )
    with pytest.raises(diagnostic.DiagnosticContractError, match="frozen diagnostic sample"):
        diagnostic._assert_fixed_sample(sample)
