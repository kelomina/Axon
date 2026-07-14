from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diagnose_loop28_onnx_fidelity as fidelity  # noqa: E402


def test_frozen_graph_probe_inventory_is_stable() -> None:
    model = onnx.load(ROOT / fidelity.DEFAULT_ONNX, load_external_data=False)

    macro = fidelity.build_probe_specs(model, "macro")
    routing = fidelity.build_probe_specs(model, "routing")

    assert len(macro) == 73
    assert len(routing) == 186
    assert {spec.key for spec in macro if spec.source == "layer1_out_proj"} == {
        "layer1_out_proj/15"
    }
    assert [spec.key for spec in routing[-3:]] == [
        "topk/62/scores",
        "topk/62/values",
        "topk/62/indices",
    ]
    assert all(spec.key not in {"topk/61/values", "topk/61/indices"} for spec in routing)


def test_compare_arrays_reports_float_and_index_drift() -> None:
    reference = np.array([0.0, 1.0, -2.0], dtype=np.float32)
    close = reference.copy()
    close[1] = np.nextafter(close[1], np.float32(2.0))
    far = reference.copy()
    far[2] += np.float32(1.0e-3)

    close_result = fidelity.compare_arrays(reference, close)
    far_result = fidelity.compare_arrays(reference, far)
    index_result = fidelity.compare_arrays(
        np.array([1, 2, 3], dtype=np.int64),
        np.array([1, 4, 3], dtype=np.int64),
    )

    assert close_result["passed"]
    assert close_result["exact_mismatch_count"] == 1
    assert close_result["max_ulp_delta"] == 1
    assert not far_result["passed"]
    assert far_result["above_tolerance_count"] == 1
    assert not index_result["passed"]
    assert index_result["first_mismatch_flat_indices"] == [1]


def test_routing_summary_detects_support_flip_and_small_margin() -> None:
    specs = (
        fidelity.ProbeSpec("topk/00/scores", "scores", 7, "float32", (1, 4), "topk_scores", 0),
        fidelity.ProbeSpec("topk/00/values", "values", 7, "float32", (1, 2), "topk_values", 0),
        fidelity.ProbeSpec("topk/00/indices", "indices", 7, "int64", (1, 2), "topk_indices", 0),
    )
    pytorch_capture = {
        "topk/00/scores": np.array([[0.0, 0.1, 0.1000001, 0.9]], dtype=np.float32),
        "topk/00/values": np.array([[0.9, 0.1000001]], dtype=np.float32),
        "topk/00/indices": np.array([[3, 2]], dtype=np.int64),
    }
    ort_capture = {
        "scores": np.array([[0.0, 0.1000002, 0.1000001, 0.9]], dtype=np.float32),
        "values": np.array([[0.9, 0.1000002]], dtype=np.float32),
        "indices": np.array([[3, 1]], dtype=np.int64),
    }

    summary = fidelity._routing_stability_summary(specs, pytorch_capture, ort_capture)

    assert summary["index_mismatch_route_count"] == 1
    assert summary["margin_guard_violation_route_count"] == 1
    assert summary["first_index_mismatch"]["occurrence"] == 0


def test_instrumentation_only_appends_selected_output(tmp_path: Path) -> None:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    hidden_info = helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, 2])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    nodes = [
        helper.make_node("Identity", ["input"], ["hidden"], name="hidden_node"),
        helper.make_node("Identity", ["hidden"], ["output"], name="output_node"),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "probe", [input_info], [output_info], value_info=[hidden_info])
    )
    source = tmp_path / "source.onnx"
    destination = tmp_path / "instrumented.onnx"
    onnx.save(model, source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    spec = fidelity.ProbeSpec("hidden/00", "hidden", 0, "float32", (1, 2), "hidden", 0)

    fidelity.instrument_onnx_model(source, destination, [spec])

    instrumented = onnx.load(destination, load_external_data=False)
    assert [value.name for value in instrumented.graph.output] == ["output", "hidden"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a": 1, "a": 2}', encoding="utf-8")

    with pytest.raises(fidelity.FidelityContractError, match="Duplicate JSON key"):
        fidelity.load_json_strict(path)


def test_project_path_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()

    with pytest.raises(fidelity.FidelityContractError, match="project-relative"):
        fidelity._resolve_within(root, Path("../outside"), must_exist=False)


def test_lease_is_consumed_before_execution(tmp_path: Path) -> None:
    authorization_path = tmp_path / fidelity.DEFAULT_RUN_AUTHORIZATION
    authorization_path.parent.mkdir(parents=True)
    authorization_path.write_text("{}\n", encoding="utf-8")
    authorization_sha = fidelity.sha256_file(authorization_path)
    lease_path = tmp_path / fidelity.DEFAULT_LEASE
    lease_path.write_text(
        json.dumps(
            {
                "schema": "axon_loop28_onnx_fidelity_localization_lease_v1",
                "loop_id": fidelity.LOOP_ID,
                "status": "ready",
                "localization_authorization": {"sha256": authorization_sha},
            }
        ),
        encoding="utf-8",
    )

    consumed = fidelity.consume_lease(tmp_path, authorization_sha)

    assert consumed["status"] == "consumed_before_execution"
    assert not lease_path.exists()
    final_path = tmp_path / fidelity.DEFAULT_FINAL_LEASE
    assert final_path.exists()
    assert (
        fidelity.load_json_strict(final_path)["original_lease_sha256"]
        == consumed["original_lease_sha256"]
    )
    with pytest.raises(FileNotFoundError):
        fidelity.consume_lease(tmp_path, authorization_sha)


def test_native_output_reader_rejects_path_escape(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    manifest = {
        "schema": "axon_onnx_fidelity_probe_output_v1",
        "runs": [
            {
                "index": index,
                "outputs": [
                    {
                        "name": "logits",
                        "dtype": "float32",
                        "shape": [1, 2],
                        "file": "../escape.bin",
                        "nbytes": 8,
                    }
                ],
            }
            for index in range(fidelity.REPEATS)
        ],
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(fidelity.FidelityContractError, match="path is invalid"):
        fidelity._read_native_probe_outputs(manifest_path, output_root)
