#!/usr/bin/env python3
"""Localize frozen Loop28 PyTorch/ONNX drift on synthetic inputs only."""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AxonExperimentConfig  # noqa: E402
from model import AxonMalwareModel  # noqa: E402
from predict_api import _extract_features  # noqa: E402
from security import load_safe_checkpoint  # noqa: E402

LOOP_ID = "p0_loop28_onnx_fidelity_001"
TOLERANCE = 1.0e-6
REPEATS = 3

DEFAULT_PROPOSAL = Path("manifests/roadmap_9997/p0_loop28_onnx_fidelity/proposal.json")
DEFAULT_AUTHORIZATION = Path("manifests/roadmap_9997/p0_loop28_onnx_fidelity/authorization.json")
DEFAULT_PREFLIGHT = Path("manifests/roadmap_9997/p0_loop28_onnx_fidelity/preflight.json")
DEFAULT_IMPLEMENTATION = Path(
    "manifests/roadmap_9997/p0_loop28_onnx_fidelity/implementation_manifest.json"
)
DEFAULT_RUN_AUTHORIZATION = Path(
    "manifests/roadmap_9997/p0_loop28_onnx_fidelity/localization_authorization.json"
)
DEFAULT_LEASE = Path("manifests/roadmap_9997/p0_loop28_onnx_fidelity/localization_lease.json")
DEFAULT_FINAL_LEASE = Path(
    "manifests/roadmap_9997/p0_loop28_onnx_fidelity/localization_lease.final.json"
)
DEFAULT_EVIDENCE = Path(
    "reports/roadmap_9997/p0_loop28_onnx_fidelity/localization_evidence.final.json"
)
DEFAULT_WORK_ROOT = Path("reports/roadmap_9997/p0_loop28_onnx_fidelity/work")

DEFAULT_CHECKPOINT = Path("models/random_20w_8192/best_model.pt")
DEFAULT_ONNX = Path("models/random_20w_8192/axon_loop28_base.onnx")
DEFAULT_ONNX_DATA = Path("models/random_20w_8192/axon_loop28_base.onnx.data")
DEFAULT_FIXTURE_CONTRACT = Path("tests/test_native_loop28_parity_source.py")
DEFAULT_PROBE = Path("tools/axon_onnx_fidelity/build/bin/Release/axon_onnx_fidelity_probe.exe")

EXPECTED_BASELINE_HASHES = {
    "checkpoint": "96a1b1ece41dd7dd9142a0f7f4330da3a7938a26cca8b01e0e7c7a1074e5e3a4",
    "onnx_graph": "3199b158fc8f7e3a53a516b2681aef8b5d5aa4a210baf66152fded72a3ff07f4",
    "onnx_data": "4865d52d861d780627ca9aea4b16f83d8c2df62dd5b2136217d1e42547b8c7fa",
    "fixture_contract": "e9360ac16ef5c5ea384788ce9f58d99e967d34c6ee8c53f79e96c9c240661d0c",
}

FIXTURES = (
    {
        "name": "pe32_numeric_resource_tls_callbacks",
        "pe_plus": False,
        "named_resource": False,
        "tls_callbacks": True,
        "expected_control": "fail",
    },
    {
        "name": "pe32_named_resource_tls_callbacks",
        "pe_plus": False,
        "named_resource": True,
        "tls_callbacks": True,
        "expected_control": "fail",
    },
    {
        "name": "pe32_numeric_resource_zero_tls_callbacks",
        "pe_plus": False,
        "named_resource": False,
        "tls_callbacks": False,
        "expected_control": "fail",
    },
    {
        "name": "pe32_plus_named_resource_zero_tls_callbacks",
        "pe_plus": True,
        "named_resource": True,
        "tls_callbacks": False,
        "expected_control": "pass",
    },
)

EXPECTED_MACRO_COUNTS = {
    "input_proj": 16,
    "layer0_qkv": 16,
    "layer0_out_proj": 16,
    "layer1_qkv": 16,
    "layer1_out_proj": 1,
    "byte_repr": 1,
    "pe_repr": 1,
    "stat_repr": 1,
    "fused_features": 1,
    "classifier_norm": 1,
    "classifier_linear": 1,
    "classifier_gelu": 1,
    "logits": 1,
}
EXPECTED_TOPK_COUNT = 62


class FidelityContractError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class ProbeSpec:
    key: str
    onnx_value: str
    node_index: int
    dtype: str
    shape: tuple[int | str, ...]
    source: str
    occurrence: int

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise FidelityContractError(f"Duplicate JSON key is forbidden: {key}")
        payload[key] = value
    return payload


def load_json_strict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FidelityContractError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise FidelityContractError(f"JSON artifact must be an object: {path}")
    return payload


def _resolve_within(project_root: Path, relative: Path, *, must_exist: bool = True) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise FidelityContractError(f"Path must be project-relative: {relative}")
    root = project_root.resolve(strict=True)
    resolved = (root / relative).resolve(strict=must_exist)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise FidelityContractError(f"Path escapes project root: {relative}") from exc
    return resolved


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise FidelityContractError(f"Output already exists: {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _onnx_type_info(value_info) -> tuple[str, tuple[int | str, ...]]:
    import onnx  # noqa: PLC0415

    elem_type = value_info.type.tensor_type.elem_type
    dtype_by_type = {
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.DOUBLE: "float64",
        onnx.TensorProto.INT64: "int64",
        onnx.TensorProto.INT32: "int32",
        onnx.TensorProto.BOOL: "bool",
    }
    if elem_type not in dtype_by_type:
        raise FidelityContractError(
            f"Unsupported ONNX probe dtype {elem_type} for {value_info.name}"
        )
    shape: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            shape.append(str(dimension.dim_param))
        else:
            shape.append("unknown")
    return dtype_by_type[elem_type], tuple(shape)


def _node_metadata(node) -> dict[str, str]:
    return {entry.key: entry.value for entry in node.metadata_props}


def _logical_fx_name(node) -> Optional[str]:
    fx_node = _node_metadata(node).get("pkg.torch.onnx.fx_node", "")
    match = re.match(r"%([A-Za-z0-9_]+)\s*:", fx_node)
    return match.group(1) if match else None


def _logical_output(node) -> Optional[str]:
    fx_name = _logical_fx_name(node)
    if fx_name and fx_name in node.output:
        return fx_name
    if "logits" in node.output:
        return "logits"
    return None


def _macro_group(node, output_name: str) -> Optional[str]:
    scope = _node_metadata(node).get("pkg.torch.onnx.name_scopes", "")
    if ".input_proj.1" in scope and output_name.startswith("gelu"):
        return "input_proj"
    if ".dsra_layers.0.qkv" in scope:
        return "layer0_qkv"
    if ".dsra_layers.0.out_proj" in scope:
        return "layer0_out_proj"
    if ".dsra_layers.1.qkv" in scope:
        return "layer1_qkv"
    if ".dsra_layers.1.out_proj" in scope:
        return "layer1_out_proj"
    if output_name == "mean_192":
        return "byte_repr"
    if ".pe_projector.projector.4" in scope:
        return "pe_repr"
    if ".stat_projector.2" in scope and output_name.startswith("gelu"):
        return "stat_repr"
    if output_name == "cat_444":
        return "fused_features"
    if ".classifier.0" in scope and output_name.startswith("layer_norm"):
        return "classifier_norm"
    if ".classifier.2" in scope:
        return "classifier_linear"
    if ".classifier.3" in scope and output_name.startswith("gelu"):
        return "classifier_gelu"
    if output_name == "logits":
        return "logits"
    return None


def build_probe_specs(onnx_model, profile: str) -> tuple[ProbeSpec, ...]:
    values = {
        value.name: value
        for value in (
            list(onnx_model.graph.value_info)
            + list(onnx_model.graph.output)
            + list(onnx_model.graph.input)
        )
    }
    specs: list[ProbeSpec] = []
    counters: dict[str, int] = {}

    if profile == "macro":
        for node_index, node in enumerate(onnx_model.graph.node):
            output_name = _logical_output(node)
            if output_name is None:
                continue
            group = _macro_group(node, output_name)
            if group is None:
                continue
            graph_occurrence = counters.get(group, 0)
            counters[group] = graph_occurrence + 1
            # layer-1 的前 15 个 out_proj 不影响最终 last-chunk pooling，导出器会删掉；
            # 冻结图唯一保留的输出对应 eager 的第 16 次调用。
            occurrence = 15 if group == "layer1_out_proj" else graph_occurrence
            value_info = values.get(output_name)
            if value_info is None:
                raise FidelityContractError(f"Missing ONNX value metadata: {output_name}")
            dtype, shape = _onnx_type_info(value_info)
            specs.append(
                ProbeSpec(
                    key=f"{group}/{occurrence:02d}",
                    onnx_value=output_name,
                    node_index=node_index,
                    dtype=dtype,
                    shape=shape,
                    source=group,
                    occurrence=occurrence,
                )
            )
        if counters != EXPECTED_MACRO_COUNTS:
            raise FidelityContractError(f"Frozen ONNX macro probe inventory drifted: {counters}")
    elif profile == "routing":
        topk_occurrence = 0
        for node_index, node in enumerate(onnx_model.graph.node):
            if node.op_type != "TopK" or len(node.output) != 2:
                continue
            # 最后一个 chunk 的两个 write TopK 仅更新未被消费的 final state，
            # 因而冻结图删除了 eager call 61/63；最后一个 graph read 对应 eager call 62。
            eager_occurrence = topk_occurrence + 1 if topk_occurrence == 61 else topk_occurrence
            probe_values = (
                ("scores", node.input[0]),
                *zip(("values", "indices"), node.output, strict=True),
            )
            for output_kind, output_name in probe_values:
                value_info = values.get(output_name)
                if value_info is None:
                    raise FidelityContractError(f"Missing ONNX value metadata: {output_name}")
                dtype, shape = _onnx_type_info(value_info)
                specs.append(
                    ProbeSpec(
                        key=f"topk/{eager_occurrence:02d}/{output_kind}",
                        onnx_value=output_name,
                        node_index=node_index,
                        dtype=dtype,
                        shape=shape,
                        source=f"topk_{output_kind}",
                        occurrence=eager_occurrence,
                    )
                )
            topk_occurrence += 1
        if topk_occurrence != EXPECTED_TOPK_COUNT:
            raise FidelityContractError(f"Frozen ONNX TopK inventory drifted: {topk_occurrence}")
    else:
        raise FidelityContractError(f"Unsupported probe profile: {profile}")

    if len({spec.key for spec in specs}) != len(specs):
        raise FidelityContractError(f"Duplicate semantic probe key in {profile}")
    if len({spec.onnx_value for spec in specs}) != len(specs):
        raise FidelityContractError(f"Duplicate ONNX probe value in {profile}")
    return tuple(specs)


def instrument_onnx_model(source: Path, destination: Path, specs: Sequence[ProbeSpec]) -> None:
    import onnx  # noqa: PLC0415

    model = onnx.load(source, load_external_data=False)
    values = {value.name: value for value in model.graph.value_info}
    existing_outputs = {value.name for value in model.graph.output}
    for spec in specs:
        if spec.onnx_value in existing_outputs:
            continue
        value_info = values.get(spec.onnx_value)
        if value_info is None:
            raise FidelityContractError(f"Cannot expose missing ONNX value: {spec.onnx_value}")
        model.graph.output.append(copy.deepcopy(value_info))
        existing_outputs.add(spec.onnx_value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, destination)


def _array_record(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "nbytes": int(contiguous.nbytes),
        "sha256": _sha256_bytes(contiguous.tobytes(order="C")),
    }


def _ordered_float32_bits(array: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
    sign = np.uint32(0x80000000)
    return np.where((bits & sign) != 0, np.bitwise_not(bits), bits | sign).astype(np.uint64)


def compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.ascontiguousarray(reference)
    candidate = np.ascontiguousarray(candidate)
    result: dict[str, Any] = {
        "reference": _array_record(reference),
        "candidate": _array_record(candidate),
        "shape_match": reference.shape == candidate.shape,
        "dtype_match": reference.dtype == candidate.dtype,
    }
    if not result["shape_match"] or not result["dtype_match"]:
        result.update({"passed": False, "reason": "shape_or_dtype_mismatch"})
        return result

    if np.issubdtype(reference.dtype, np.floating):
        reference64 = reference.astype(np.float64)
        candidate64 = candidate.astype(np.float64)
        finite_pair = np.isfinite(reference64) & np.isfinite(candidate64)
        nonfinite_match = bool(np.array_equal(np.isfinite(reference64), np.isfinite(candidate64)))
        difference = np.abs(reference64 - candidate64)
        finite_difference = difference[finite_pair]
        max_abs = float(finite_difference.max(initial=0.0))
        denominator = np.maximum(np.abs(reference64[finite_pair]), 1.0e-30)
        max_relative = float((finite_difference / denominator).max(initial=0.0))
        mismatch_count = int(np.count_nonzero(finite_difference > TOLERANCE))
        element_bytes_match = np.all(
            reference.view(np.uint8).reshape(reference.size, reference.itemsize)
            == candidate.view(np.uint8).reshape(candidate.size, candidate.itemsize),
            axis=1,
        )
        exact_mismatch_count = int(reference.size - np.count_nonzero(element_bytes_match))
        result.update(
            {
                "nonfinite_reference_count": int(np.count_nonzero(~np.isfinite(reference64))),
                "nonfinite_candidate_count": int(np.count_nonzero(~np.isfinite(candidate64))),
                "nonfinite_pattern_match": nonfinite_match,
                "max_absolute_delta": max_abs,
                "max_relative_delta": max_relative,
                "above_tolerance_count": mismatch_count,
                "exact_mismatch_count": exact_mismatch_count,
            }
        )
        if reference.dtype == np.float32:
            reference_bits = _ordered_float32_bits(reference).astype(np.int64)
            candidate_bits = _ordered_float32_bits(candidate).astype(np.int64)
            result["max_ulp_delta"] = int(np.abs(reference_bits - candidate_bits).max(initial=0))
        result["passed"] = nonfinite_match and max_abs <= TOLERANCE
        return result

    mismatch = np.flatnonzero(reference.reshape(-1) != candidate.reshape(-1))
    result.update(
        {
            "mismatch_count": int(mismatch.size),
            "first_mismatch_flat_indices": [int(value) for value in mismatch[:16]],
            "passed": mismatch.size == 0,
        }
    )
    return result


def _capture_hook(captures: dict[str, np.ndarray], counters: dict[str, int], group: str):
    def hook(_module, _inputs, output) -> None:
        if not hasattr(output, "detach"):
            raise FidelityContractError(f"Hook output is not a tensor: {group}")
        occurrence = counters.get(group, 0)
        counters[group] = occurrence + 1
        captures[f"{group}/{occurrence:02d}"] = output.detach().cpu().contiguous().numpy().copy()

    return hook


def capture_pytorch(
    model: AxonMalwareModel, inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    import torch  # noqa: PLC0415

    captures: dict[str, np.ndarray] = {}
    counters: dict[str, int] = {}
    layers = model.dsra_encoder.dsra_encoder.dsra_layers
    if layers is None or len(layers) != 2:
        raise FidelityContractError("Frozen checkpoint must expose exactly two DSRA layers")
    modules = (
        ("input_proj", model.dsra_encoder.input_proj[1]),
        ("layer0_qkv", layers[0].qkv),
        ("layer0_out_proj", layers[0].out_proj),
        ("layer1_qkv", layers[1].qkv),
        ("layer1_out_proj", layers[1].out_proj),
        ("pe_repr", model.dsra_encoder.pe_projector.projector[4]),
        ("stat_repr", model.stat_projector[2]),
        ("classifier_norm", model.classifier[0]),
        ("classifier_linear", model.classifier[2]),
        ("classifier_gelu", model.classifier[3]),
        ("logits", model.classifier[5]),
    )
    handles = [
        module.register_forward_hook(_capture_hook(captures, counters, group))
        for group, module in modules
    ]
    original_topk = torch.topk

    def traced_topk(input_tensor, k, *args, **kwargs):
        result = original_topk(input_tensor, k, *args, **kwargs)
        occurrence = counters.get("topk", 0)
        counters["topk"] = occurrence + 1
        captures[f"topk/{occurrence:02d}/values"] = (
            result.values.detach().cpu().contiguous().numpy().copy()
        )
        captures[f"topk/{occurrence:02d}/indices"] = (
            result.indices.detach().cpu().contiguous().numpy().copy()
        )
        captures[f"topk/{occurrence:02d}/scores"] = (
            input_tensor.detach().cpu().contiguous().numpy().copy()
        )
        return result

    byte_tensor = torch.from_numpy(inputs["byte_seq"]).long().unsqueeze(0)
    pe_tensor = torch.from_numpy(inputs["pe_features"]).float().unsqueeze(0)
    stat_tensor = torch.from_numpy(inputs["stat_features"]).float().unsqueeze(0)
    try:
        torch.topk = traced_topk
        with torch.inference_mode():
            output = model(
                byte_tensor,
                pe_tensor,
                stat_features=stat_tensor,
                return_features=True,
            )
        captures["byte_repr/00"] = output["byte_repr"].detach().cpu().numpy().copy()
        captures["fused_features/00"] = output["features"].detach().cpu().numpy().copy()
    finally:
        torch.topk = original_topk
        for handle in handles:
            handle.remove()
    return captures


def _load_fixture_contract(path: Path):
    spec = importlib.util.spec_from_file_location("loop28_frozen_fixture_contract", path)
    if spec is None or spec.loader is None:
        raise FidelityContractError("Unable to load frozen synthetic fixture contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = getattr(module, "_synthetic_pe_bytes", None)
    if not callable(builder):
        raise FidelityContractError("Frozen synthetic fixture builder is missing")
    return module


def _load_model(checkpoint_path: Path) -> tuple[AxonMalwareModel, AxonExperimentConfig]:
    checkpoint = load_safe_checkpoint(checkpoint_path, map_location="cpu")
    config = AxonExperimentConfig.from_dict(dict(checkpoint["config"]))
    model = AxonMalwareModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def _extract_fixture_inputs(
    config: AxonExperimentConfig, fixture_path: Path
) -> dict[str, np.ndarray]:
    features = _extract_features(config, fixture_path)
    if features is None:
        raise FidelityContractError(f"Synthetic feature extraction failed: {fixture_path.name}")
    inputs = {
        "byte_seq": np.ascontiguousarray(features.byte_seq, dtype=np.int64),
        "pe_features": np.ascontiguousarray(features.pe_features, dtype=np.float32),
        "stat_features": np.ascontiguousarray(features.stat_features, dtype=np.float32),
    }
    expected_shapes = {"byte_seq": (8192,), "pe_features": (256,), "stat_features": (49,)}
    for name, expected_shape in expected_shapes.items():
        if inputs[name].shape != expected_shape:
            raise FidelityContractError(
                f"Synthetic input shape drifted for {name}: {inputs[name].shape}"
            )
    return inputs


def _write_probe_inputs(directory: Path, inputs: dict[str, np.ndarray]) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, array in inputs.items():
        path = directory / f"{name}.bin"
        np.ascontiguousarray(array).tofile(path)
        paths[name] = path
    return paths


def _prepare_external_data(source_data: Path, destination: Path) -> str:
    try:
        os.link(source_data, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source_data, destination)
        return "copy"


def _numpy_dtype(name: str) -> np.dtype:
    mapping = {
        "float32": np.dtype("<f4"),
        "float64": np.dtype("<f8"),
        "int64": np.dtype("<i8"),
        "int32": np.dtype("<i4"),
        "bool": np.dtype("?"),
    }
    if name not in mapping:
        raise FidelityContractError(f"Unsupported native probe dtype: {name}")
    return mapping[name]


def _read_native_probe_outputs(
    manifest_path: Path, output_root: Path
) -> list[dict[str, np.ndarray]]:
    manifest = load_json_strict(manifest_path)
    if manifest.get("schema") != "axon_onnx_fidelity_probe_output_v1":
        raise FidelityContractError("Native probe output schema mismatch")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != REPEATS:
        raise FidelityContractError("Native probe repeat count mismatch")
    decoded_runs: list[dict[str, np.ndarray]] = []
    root = output_root.resolve(strict=True)
    for expected_index, run in enumerate(runs):
        if not isinstance(run, dict) or run.get("index") != expected_index:
            raise FidelityContractError("Native probe run index mismatch")
        rows = run.get("outputs")
        if not isinstance(rows, list):
            raise FidelityContractError("Native probe outputs must be a list")
        decoded: dict[str, np.ndarray] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise FidelityContractError("Native probe output row must be an object")
            name = str(row.get("name", ""))
            relative = Path(str(row.get("file", "")))
            if not name or relative.is_absolute() or ".." in relative.parts:
                raise FidelityContractError("Native probe output path is invalid")
            path = (root / relative).resolve(strict=True)
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise FidelityContractError("Native probe output escapes work root") from exc
            dtype = _numpy_dtype(str(row.get("dtype", "")))
            shape_value = row.get("shape")
            if not isinstance(shape_value, list) or any(
                not isinstance(v, int) for v in shape_value
            ):
                raise FidelityContractError("Native probe output shape is invalid")
            shape = tuple(int(value) for value in shape_value)
            expected_count = int(np.prod(shape, dtype=np.int64))
            array = np.fromfile(path, dtype=dtype)
            if array.size != expected_count or path.stat().st_size != int(row.get("nbytes", -1)):
                raise FidelityContractError(f"Native probe output size mismatch: {name}")
            if name in decoded:
                raise FidelityContractError(f"Duplicate native probe output: {name}")
            decoded[name] = array.reshape(shape)
        decoded_runs.append(decoded)
    return decoded_runs


def run_native_probe(
    *,
    executable: Path,
    onnx_path: Path,
    input_paths: dict[str, Path],
    output_root: Path,
    timeout_seconds: int,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = output_root / "probe_manifest.json"
    command = [
        str(executable),
        "--onnx",
        str(onnx_path),
        "--byte",
        str(input_paths["byte_seq"]),
        "--pe",
        str(input_paths["pe_features"]),
        "--stat",
        str(input_paths["stat_features"]),
        "--output-dir",
        str(output_root),
        "--manifest",
        str(manifest_path),
        "--repeat",
        str(REPEATS),
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise FidelityContractError(
            f"Native ONNX probe failed ({completed.returncode}): {completed.stderr[-2000:]}"
        )
    runs = _read_native_probe_outputs(manifest_path, output_root)
    output_bytes = sum(path.stat().st_size for path in output_root.rglob("*") if path.is_file())
    execution = {
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "stdout_sha256": _sha256_bytes(completed.stdout.encode()),
        "stderr_sha256": _sha256_bytes(completed.stderr.encode()),
        "output_manifest_sha256": sha256_file(manifest_path),
        "temporary_output_bytes": output_bytes,
    }
    shutil.rmtree(output_root)
    return runs, execution


def _determinism(records: Sequence[dict[str, np.ndarray]]) -> dict[str, Any]:
    if not records:
        raise FidelityContractError("Determinism check requires at least one run")
    baseline = records[0]
    key_sets_match = all(set(record) == set(baseline) for record in records[1:])
    mismatches: list[str] = []
    if key_sets_match:
        for key, reference in baseline.items():
            if any(
                not np.array_equal(reference, record[key], equal_nan=True) for record in records[1:]
            ):
                mismatches.append(key)
    return {
        "repeat_count": len(records),
        "key_sets_match": key_sets_match,
        "bit_exact": key_sets_match and not mismatches,
        "mismatched_keys": mismatches,
    }


def _profile_comparison(
    specs: Sequence[ProbeSpec],
    pytorch_capture: dict[str, np.ndarray],
    ort_capture: dict[str, np.ndarray],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.key not in pytorch_capture:
            raise FidelityContractError(f"PyTorch capture is missing {spec.key}")
        if spec.onnx_value not in ort_capture:
            raise FidelityContractError(f"ORT capture is missing {spec.onnx_value}")
        comparison = compare_arrays(pytorch_capture[spec.key], ort_capture[spec.onnx_value])
        rows.append({"probe": spec.as_dict(), "comparison": comparison})
    failed = [row for row in rows if not row["comparison"]["passed"]]
    first = (
        min(failed, key=lambda row: (row["probe"]["node_index"], row["probe"]["key"]))
        if failed
        else None
    )
    return {
        "probe_count": len(rows),
        "passed_count": len(rows) - len(failed),
        "failed_count": len(failed),
        "first_divergence": first,
        "rows": rows,
    }


def _routing_stability_summary(
    specs: Sequence[ProbeSpec],
    pytorch_capture: dict[str, np.ndarray],
    ort_capture: dict[str, np.ndarray],
) -> dict[str, Any]:
    by_key = {spec.key: spec for spec in specs}
    occurrences = sorted(spec.occurrence for spec in specs if spec.source == "topk_scores")
    rows: list[dict[str, Any]] = []
    for occurrence in occurrences:
        scores_key = f"topk/{occurrence:02d}/scores"
        values_key = f"topk/{occurrence:02d}/values"
        indices_key = f"topk/{occurrence:02d}/indices"
        score_spec = by_key[scores_key]
        values_spec = by_key[values_key]
        indices_spec = by_key[indices_key]
        pytorch_scores = pytorch_capture[scores_key].astype(np.float64)
        ort_scores = ort_capture[score_spec.onnx_value].astype(np.float64)
        pytorch_values = pytorch_capture[values_key]
        pytorch_indices = pytorch_capture[indices_key]
        ort_indices = ort_capture[indices_spec.onnx_value]
        k = int(pytorch_values.shape[-1])
        if k >= pytorch_scores.shape[-1]:
            raise FidelityContractError(
                "TopK support margin requires at least one unselected score"
            )
        sorted_scores = np.sort(pytorch_scores, axis=-1)
        support_margin = sorted_scores[..., -k] - sorted_scores[..., -k - 1]
        route_score_delta = np.max(np.abs(pytorch_scores - ort_scores), axis=-1)
        guard_violation = support_margin <= 8.0 * route_score_delta
        index_mismatch_count = int(np.count_nonzero(pytorch_indices != ort_indices))
        rows.append(
            {
                "occurrence": occurrence,
                "node_index": score_spec.node_index,
                "k": k,
                "query_count": int(support_margin.size),
                "minimum_pytorch_support_margin": float(support_margin.min(initial=np.inf)),
                "maximum_route_score_delta": float(route_score_delta.max(initial=0.0)),
                "margin_guard_multiplier": 8.0,
                "margin_guard_violation_count": int(np.count_nonzero(guard_violation)),
                "exact_support_tie_count": int(np.count_nonzero(support_margin == 0.0)),
                "index_mismatch_count": index_mismatch_count,
                "values_onnx_value": values_spec.onnx_value,
                "indices_onnx_value": indices_spec.onnx_value,
            }
        )
    index_failures = [row for row in rows if row["index_mismatch_count"] > 0]
    margin_failures = [row for row in rows if row["margin_guard_violation_count"] > 0]
    return {
        "route_count": len(rows),
        "index_mismatch_route_count": len(index_failures),
        "margin_guard_violation_route_count": len(margin_failures),
        "first_index_mismatch": index_failures[0] if index_failures else None,
        "first_margin_guard_violation": margin_failures[0] if margin_failures else None,
        "rows": rows,
    }


def _softmax_malicious(logits: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size != 2 or not np.all(np.isfinite(values)):
        raise FidelityContractError("Base logits must contain two finite values")
    shifted = values - values.max()
    exponentials = np.exp(shifted)
    return float(exponentials[1] / exponentials.sum())


def _validate_hashes(paths: dict[str, Path]) -> dict[str, str]:
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    for name, expected in EXPECTED_BASELINE_HASHES.items():
        if hashes.get(name) != expected:
            raise FidelityContractError(
                f"Baseline hash mismatch for {name}: {hashes.get(name)} != {expected}"
            )
    return hashes


def _verify_execution_governance(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import build_loop28_onnx_fidelity_manifest as manifest_builder  # noqa: PLC0415

    implementation_path = _resolve_within(project_root, DEFAULT_IMPLEMENTATION)
    run_authorization_path = _resolve_within(project_root, DEFAULT_RUN_AUTHORIZATION)
    implementation = manifest_builder.verify_implementation_manifest(
        project_root, DEFAULT_IMPLEMENTATION
    )
    run_authorization = load_json_strict(run_authorization_path)
    if implementation.get("schema") != "axon_loop28_onnx_fidelity_implementation_manifest_v1":
        raise FidelityContractError("Implementation manifest schema mismatch")
    if implementation.get("decision") != "implementation_manifest_complete":
        raise FidelityContractError("Implementation manifest is not complete")
    if run_authorization.get("schema") != "axon_loop28_onnx_fidelity_localization_authorization_v1":
        raise FidelityContractError("Localization authorization schema mismatch")
    if run_authorization.get("loop_id") != LOOP_ID:
        raise FidelityContractError("Localization authorization loop mismatch")
    binding = run_authorization.get("implementation_manifest")
    if not isinstance(binding, dict) or binding.get("sha256") != sha256_file(implementation_path):
        raise FidelityContractError("Localization authorization implementation binding mismatch")
    proposal = load_json_strict(_resolve_within(project_root, DEFAULT_PROPOSAL))
    authorization = load_json_strict(_resolve_within(project_root, DEFAULT_AUTHORIZATION))
    preflight = load_json_strict(_resolve_within(project_root, DEFAULT_PREFLIGHT))
    parent_path = _resolve_within(
        project_root,
        Path("manifests/roadmap_9997/p0_loop28_parity_remediation/post_remediation_manifest.json"),
    )
    expected_fields = {
        "proposal_sha256": sha256_file(_resolve_within(project_root, DEFAULT_PROPOSAL)),
        "authorization_sha256": sha256_file(_resolve_within(project_root, DEFAULT_AUTHORIZATION)),
        "preflight_sha256": sha256_file(_resolve_within(project_root, DEFAULT_PREFLIGHT)),
        "parent_closure_sha256": sha256_file(parent_path),
        "ready_lease_path": DEFAULT_LEASE.as_posix(),
        "consumed_lease_path": DEFAULT_FINAL_LEASE.as_posix(),
        "evidence_path": DEFAULT_EVIDENCE.as_posix(),
        "fixture_names": [row["name"] for row in proposal["fixture_matrix"]],
        "baseline_artifacts": authorization["baseline_artifacts"],
        "budget": proposal["budget"],
        "claim_scope": {
            "synthetic_only": True,
            "raw_split_heldout_access_allowed": False,
            "training_or_fitting_allowed": False,
            "quality_metric_allowed": False,
            "quality_claim_allowed": False,
            "certification_claim_allowed": False,
        },
    }
    for field, expected in expected_fields.items():
        if run_authorization.get(field) != expected:
            raise FidelityContractError(f"Localization authorization binding mismatch: {field}")
    if preflight.get("decision") != "synthetic_fidelity_tooling_implementation_ready":
        raise FidelityContractError("Localization preflight is not implementation-ready")
    if run_authorization.get("decision") != "authorize_synthetic_localization_run":
        raise FidelityContractError("Localization run is not authorized")
    attempt_id = run_authorization.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id.startswith(f"{LOOP_ID}_attempt_"):
        raise FidelityContractError("Localization attempt identity is invalid")
    return implementation, run_authorization


def consume_lease(project_root: Path, run_authorization_sha256: str) -> dict[str, Any]:
    lease_path = _resolve_within(project_root, DEFAULT_LEASE)
    final_path = _resolve_within(project_root, DEFAULT_FINAL_LEASE, must_exist=False)
    lease = load_json_strict(lease_path)
    original_sha256 = sha256_file(lease_path)
    if lease.get("schema") != "axon_loop28_onnx_fidelity_localization_lease_v1":
        raise FidelityContractError("Localization lease schema mismatch")
    if lease.get("loop_id") != LOOP_ID or lease.get("status") != "ready":
        raise FidelityContractError("Localization lease is not ready")
    binding = lease.get("localization_authorization")
    if not isinstance(binding, dict) or binding.get("sha256") != run_authorization_sha256:
        raise FidelityContractError("Localization lease authorization binding mismatch")

    # 在任何模型载入前先独占写入 final lease；进程崩溃也不能重用预算。
    consumed = dict(lease)
    consumed.update(
        {
            "status": "consumed_before_execution",
            "consumed_at_utc": _utc_now(),
            "original_lease_sha256": original_sha256,
        }
    )
    _write_json_exclusive(final_path, consumed)
    lease_path.unlink()
    return {
        "path": DEFAULT_FINAL_LEASE.as_posix(),
        "sha256": sha256_file(final_path),
        "original_lease_sha256": original_sha256,
        "status": "consumed_before_execution",
    }


def _validate_authorized_baselines(
    project_root: Path, baseline_artifacts: dict[str, Any]
) -> dict[str, str]:
    if not isinstance(baseline_artifacts, dict) or not baseline_artifacts:
        raise FidelityContractError("Authorized baseline inventory is missing")
    hashes: dict[str, str] = {}
    for name, record in baseline_artifacts.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise FidelityContractError(f"Authorized baseline record is invalid: {name}")
        path = _resolve_within(project_root, Path(str(record["path"])))
        digest = sha256_file(path)
        if digest != record["sha256"]:
            raise FidelityContractError(f"Authorized baseline hash drifted: {name}")
        hashes[name] = digest
    return hashes


def _fixture_bytes(fixture_contract, fixture: dict[str, Any]) -> bytes:
    return fixture_contract._synthetic_pe_bytes(
        pe_plus=fixture["pe_plus"],
        named_resource=fixture["named_resource"],
        tls_callbacks=fixture["tls_callbacks"],
    )


def run_localization(project_root: Path) -> dict[str, Any]:
    import onnx  # noqa: PLC0415
    import torch  # noqa: PLC0415

    root = project_root.resolve(strict=True)
    _, run_authorization = _verify_execution_governance(root)
    run_authorization_path = _resolve_within(root, DEFAULT_RUN_AUTHORIZATION)
    run_authorization_sha256 = sha256_file(run_authorization_path)
    lease = consume_lease(root, run_authorization_sha256)
    dependency_hashes_before = _validate_authorized_baselines(
        root, run_authorization["baseline_artifacts"]
    )

    baseline_paths = {
        "checkpoint": _resolve_within(root, DEFAULT_CHECKPOINT),
        "onnx_graph": _resolve_within(root, DEFAULT_ONNX),
        "onnx_data": _resolve_within(root, DEFAULT_ONNX_DATA),
        "fixture_contract": _resolve_within(root, DEFAULT_FIXTURE_CONTRACT),
    }
    probe_path = _resolve_within(root, DEFAULT_PROBE)
    before_hashes = _validate_hashes(baseline_paths)
    probe_sha256 = sha256_file(probe_path)

    frozen_graph = onnx.load(baseline_paths["onnx_graph"], load_external_data=False)
    macro_specs = build_probe_specs(frozen_graph, "macro")
    routing_specs = build_probe_specs(frozen_graph, "routing")

    torch.set_num_threads(1)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model, config = _load_model(baseline_paths["checkpoint"])
    fixture_contract = _load_fixture_contract(baseline_paths["fixture_contract"])

    evidence_path = _resolve_within(root, DEFAULT_EVIDENCE, must_exist=False)
    if evidence_path.exists():
        raise FidelityContractError(f"Evidence already exists: {evidence_path}")
    work_root = _resolve_within(root, DEFAULT_WORK_ROOT, must_exist=False)
    work_root.mkdir(parents=True, exist_ok=True)

    fixture_evidence: list[dict[str, Any]] = []
    started = time.monotonic()
    native_subprocesses = 0
    with tempfile.TemporaryDirectory(prefix="run_", dir=work_root) as temporary:
        temporary_root = Path(temporary)
        external_mode = _prepare_external_data(
            baseline_paths["onnx_data"],
            temporary_root / baseline_paths["onnx_data"].name,
        )
        instrumented_models: dict[str, Path] = {}
        for profile, specs in (("macro", macro_specs), ("routing", routing_specs)):
            destination = temporary_root / f"axon_loop28_base.{profile}.onnx"
            instrument_onnx_model(baseline_paths["onnx_graph"], destination, specs)
            instrumented_models[profile] = destination

        for fixture in FIXTURES:
            fixture_dir = temporary_root / fixture["name"]
            fixture_dir.mkdir()
            fixture_path = fixture_dir / f"{fixture['name']}.exe"
            payload = _fixture_bytes(fixture_contract, fixture)
            fixture_path.write_bytes(payload)
            inputs = _extract_fixture_inputs(config, fixture_path)
            input_paths = _write_probe_inputs(fixture_dir / "inputs", inputs)

            pytorch_runs = [capture_pytorch(model, inputs) for _ in range(REPEATS)]
            pytorch_determinism = _determinism(pytorch_runs)
            pytorch_capture = pytorch_runs[0]
            profiles: dict[str, Any] = {}
            logits_capture: Optional[np.ndarray] = None
            executions: list[dict[str, Any]] = []
            for profile, specs in (("macro", macro_specs), ("routing", routing_specs)):
                native_subprocesses += 1
                ort_runs, execution = run_native_probe(
                    executable=probe_path,
                    onnx_path=instrumented_models[profile],
                    input_paths=input_paths,
                    output_root=fixture_dir / f"ort_{profile}",
                    timeout_seconds=180,
                )
                ort_determinism = _determinism(ort_runs)
                comparison = _profile_comparison(specs, pytorch_capture, ort_runs[0])
                profiles[profile] = {
                    "ort_determinism": ort_determinism,
                    "comparison": comparison,
                }
                if profile == "routing":
                    profiles[profile]["stability"] = _routing_stability_summary(
                        specs,
                        pytorch_capture,
                        ort_runs[0],
                    )
                executions.append({"profile": profile, **execution})
                if profile == "macro":
                    logits_capture = ort_runs[0].get("logits")
            if logits_capture is None:
                raise FidelityContractError("Macro probe did not return logits")

            pytorch_probability = _softmax_malicious(pytorch_capture["logits/00"])
            ort_probability = _softmax_malicious(logits_capture)
            probability_delta = abs(pytorch_probability - ort_probability)
            observed_control = "pass" if probability_delta <= TOLERANCE else "fail"
            fixture_evidence.append(
                {
                    "fixture": dict(fixture),
                    "fixture_bytes": {
                        "size": len(payload),
                        "sha256": _sha256_bytes(payload),
                    },
                    "inputs": {name: _array_record(value) for name, value in inputs.items()},
                    "pytorch_determinism": pytorch_determinism,
                    "profiles": profiles,
                    "base_probability": {
                        "pytorch": pytorch_probability,
                        "onnxruntime": ort_probability,
                        "absolute_delta": probability_delta,
                        "tolerance": TOLERANCE,
                        "expected_control": fixture["expected_control"],
                        "observed_control": observed_control,
                        "control_reproduced": observed_control == fixture["expected_control"],
                    },
                    "native_executions": executions,
                }
            )

    elapsed = time.monotonic() - started
    after_hashes = _validate_hashes(baseline_paths)
    dependency_hashes_after = _validate_authorized_baselines(
        root, run_authorization["baseline_artifacts"]
    )
    controls_reproduced = all(
        row["base_probability"]["control_reproduced"] for row in fixture_evidence
    )
    deterministic = all(
        row["pytorch_determinism"]["bit_exact"]
        and all(profile["ort_determinism"]["bit_exact"] for profile in row["profiles"].values())
        for row in fixture_evidence
    )
    first_divergences = [
        {
            "fixture": row["fixture"]["name"],
            "macro": row["profiles"]["macro"]["comparison"]["first_divergence"],
            "routing": row["profiles"]["routing"]["comparison"]["first_divergence"],
        }
        for row in fixture_evidence
    ]
    localized = any(
        row["macro"] is not None or row["routing"] is not None for row in first_divergences
    )
    if not controls_reproduced:
        decision = "invalid_positive_control_or_lineage_drift"
    elif elapsed > 2400 or native_subprocesses > 12:
        decision = "budget_exhausted_no_claim"
    elif localized:
        decision = "localized_negative_no_raw"
    else:
        decision = "synthetic_fidelity_verified_raw_still_requires_new_authorization"

    payload = {
        "schema": "axon_loop28_onnx_fidelity_localization_evidence_v1",
        "loop_id": LOOP_ID,
        "generated_at_utc": _utc_now(),
        "governance": {
            "proposal_sha256": sha256_file(_resolve_within(root, DEFAULT_PROPOSAL)),
            "authorization_sha256": sha256_file(_resolve_within(root, DEFAULT_AUTHORIZATION)),
            "preflight_sha256": sha256_file(_resolve_within(root, DEFAULT_PREFLIGHT)),
            "implementation_manifest_sha256": sha256_file(
                _resolve_within(root, DEFAULT_IMPLEMENTATION)
            ),
            "localization_authorization_sha256": run_authorization_sha256,
            "consumed_lease": lease,
        },
        "scope": {
            "synthetic_only": True,
            "dataset_raw_accessed": False,
            "split_metadata_accessed": False,
            "cache_rows_accessed": False,
            "heldout_accessed": False,
            "training_or_fitting_performed": False,
            "quality_metric_computed": False,
            "f1_computed": False,
        },
        "runtime_contract": {
            "cpu_only": True,
            "graph_optimization": "ORT_DISABLE_ALL",
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "execution_mode": "ORT_SEQUENTIAL",
            "repeats": REPEATS,
            "probe_sha256": probe_sha256,
            "temporary_external_data_mode": external_mode,
        },
        "graph": {
            "onnx_graph_sha256": before_hashes["onnx_graph"],
            "onnx_data_sha256": before_hashes["onnx_data"],
            "macro_probe_count": len(macro_specs),
            "routing_probe_count": len(routing_specs),
            "topk_node_count": EXPECTED_TOPK_COUNT,
        },
        "fixtures": fixture_evidence,
        "first_divergences": first_divergences,
        "determinism_all_passed": deterministic,
        "positive_controls_reproduced": controls_reproduced,
        "baseline_integrity": {
            "before": before_hashes,
            "after": after_hashes,
            "stable": before_hashes == after_hashes,
        },
        "dependency_integrity": {
            "before": dependency_hashes_before,
            "after": dependency_hashes_after,
            "stable": dependency_hashes_before == dependency_hashes_after,
        },
        "budget": {
            "fixture_count": len(fixture_evidence),
            "profile_count": 2,
            "native_subprocess_count": native_subprocesses,
            "wall_clock_seconds": elapsed,
            "retained_probe_output_bytes": 0,
        },
        "claim_boundary": {
            "synthetic_localization_claim_allowed": controls_reproduced and localized,
            "population_parity_claim_allowed": False,
            "quality_claim_allowed": False,
            "native_loop28_ready_claim_allowed": False,
            "native_loop151_ready_claim_allowed": False,
            "raw_rerun_allowed": False,
            "certification_claim_allowed": False,
        },
        "decision": decision,
    }
    _write_json_exclusive(evidence_path, payload)
    return payload


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Run the governed synthetic-only Loop28 ONNX activation localization."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = run_localization(args.project_root)
    except Exception as exc:  # noqa: BLE001 - fail closed with one concise CLI error.
        print(f"[Error] {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "positive_controls_reproduced": result["positive_controls_reproduced"],
                "determinism_all_passed": result["determinism_all_passed"],
                "evidence": DEFAULT_EVIDENCE.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0 if result["decision"] == "localized_negative_no_raw" else 2


if __name__ == "__main__":
    raise SystemExit(main())
