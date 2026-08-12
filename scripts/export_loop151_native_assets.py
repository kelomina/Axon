#!/usr/bin/env python3
"""Export frozen Loop151 sklearn payloads to primitive native assets.

The exporter deliberately keeps model inference semantics explicit.  It does
not serialize sklearn objects or Python bytecode: trees, forest leaf
probabilities, scalers, and binary logistic coefficients are represented as
JSON primitives so a later native loader can consume the files without
Python/sklearn.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pickle
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class NativeAssetExportError(ValueError):
    """Raised when a frozen model is outside the supported native subset."""


class _FeatureConfigProxy:
    """Allow old ``__main__.FeatureConfig`` pickles without importing a script."""

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(dict(state))


class _PayloadUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # noqa: D102
        if module == "__main__" and name == "FeatureConfig":
            return _FeatureConfigProxy
        return super().find_class(module, name)


def load_frozen_payload(path: Path) -> tuple[dict[str, Any], str]:
    """Load one frozen payload and return it with its source SHA-256."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise NativeAssetExportError(f"Cannot read frozen model: {path}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = _PayloadUnpickler(io.BytesIO(raw)).load()
    except Exception as exc:  # pragma: no cover - exact sklearn errors vary by version
        raise NativeAssetExportError(f"Cannot deserialize frozen model: {path}") from exc
    if not isinstance(payload, dict):
        raise NativeAssetExportError(f"Frozen payload is not a mapping: {path}")
    return payload, digest


def _finite_float(value: Any, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise NativeAssetExportError(f"Non-finite value in {field}")
    return result


def _primitive(value: Any, *, field: str = "value") -> Any:
    """Convert metadata to JSON primitives while rejecting executable objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _finite_float(value, field=field)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _finite_float(value, field=field)
    if isinstance(value, np.ndarray):
        return [_primitive(item, field=field) for item in value.tolist()]
    if is_dataclass(value):
        return _primitive(asdict(value), field=field)
    if isinstance(value, Mapping):
        return {str(key): _primitive(item, field=f"{field}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(item, field=field) for item in value]
    if hasattr(value, "__dict__"):
        return _primitive(vars(value), field=field)
    raise NativeAssetExportError(f"Metadata contains unsupported value at {field}: {type(value)!r}")


def _binary_classes(model: Any) -> list[int]:
    classes = [int(value) for value in np.asarray(getattr(model, "classes_", [])).reshape(-1)]
    if classes != [0, 1]:
        raise NativeAssetExportError(f"Only binary classes [0, 1] are supported, got {classes}")
    return classes


def _export_hist_gradient_boosting(model: Any) -> dict[str, Any]:
    if type(model).__name__ != "HistGradientBoostingClassifier":
        raise NativeAssetExportError("Expected HistGradientBoostingClassifier")
    if int(getattr(model, "n_trees_per_iteration_", 0)) != 1:
        raise NativeAssetExportError("Only binary HGB models with one tree per iteration are supported")
    node_values: list[float] = []
    node_feature_idx: list[int] = []
    node_thresholds: list[float] = []
    node_missing_left: list[int] = []
    node_left: list[int] = []
    node_right: list[int] = []
    node_is_leaf: list[int] = []
    tree_offsets = [0]
    tree_count = 0
    for iteration in model._predictors:
        if len(iteration) != 1:
            raise NativeAssetExportError("HGB model has more than one tree per iteration")
        tree = iteration[0]
        for node in tree.nodes:
            node_values.append(_finite_float(node["value"], field="hgb.node.value"))
            node_feature_idx.append(int(node["feature_idx"]))
            node_thresholds.append(_finite_float(node["num_threshold"], field="hgb.node.threshold"))
            node_missing_left.append(int(bool(node["missing_go_to_left"])))
            node_left.append(int(node["left"]))
            node_right.append(int(node["right"]))
            node_is_leaf.append(int(bool(node["is_leaf"])))
        tree_offsets.append(len(node_feature_idx))
        tree_count += 1
    baseline = np.asarray(model._baseline_prediction, dtype=np.float64).reshape(-1)
    if baseline.size != 1:
        raise NativeAssetExportError("HGB baseline prediction is not scalar")
    return {
        "kind": "hist_gradient_boosting",
        "classes": _binary_classes(model),
        "n_features": int(getattr(model, "n_features_in_", 0)),
        "tree_count": tree_count,
        "baseline_prediction": _finite_float(baseline[0], field="hgb.baseline_prediction"),
        "tree_node_offsets": tree_offsets,
        "node_values": node_values,
        "node_feature_idx": node_feature_idx,
        "node_num_thresholds": node_thresholds,
        "node_missing_go_to_left": node_missing_left,
        "node_left": node_left,
        "node_right": node_right,
        "node_is_leaf": node_is_leaf,
    }


def _export_forest(model: Any) -> dict[str, Any]:
    model_name = type(model).__name__
    if model_name not in {"ExtraTreesClassifier", "RandomForestClassifier"}:
        raise NativeAssetExportError(f"Unsupported forest type: {model_name}")
    tree_offsets = [0]
    node_values: list[float] = []
    node_feature_idx: list[int] = []
    node_thresholds: list[float] = []
    node_missing_left: list[int] = []
    node_left: list[int] = []
    node_right: list[int] = []
    node_is_leaf: list[int] = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        values = np.asarray(tree.value, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (1, 2):
            raise NativeAssetExportError("Only binary single-output forests are supported")
        missing_left = getattr(tree, "missing_go_to_left", np.zeros(tree.node_count, dtype=np.uint8))
        for index in range(int(tree.node_count)):
            node_values.append(_finite_float(values[index, 0, 1], field="forest.node.value[1]"))
            node_feature_idx.append(int(tree.feature[index]))
            node_thresholds.append(_finite_float(tree.threshold[index], field="forest.node.threshold"))
            node_missing_left.append(int(bool(missing_left[index])))
            node_left.append(int(tree.children_left[index]))
            node_right.append(int(tree.children_right[index]))
            node_is_leaf.append(int(tree.children_left[index] == tree.children_right[index]))
        tree_offsets.append(len(node_feature_idx))
    return {
        "kind": "extra_trees" if model_name == "ExtraTreesClassifier" else "random_forest",
        "forest_type": model_name,
        "classes": _binary_classes(model),
        "n_features": int(getattr(model, "n_features_in_", 0)),
        "tree_count": len(model.estimators_),
        "n_estimators": int(getattr(model, "n_estimators", len(model.estimators_))),
        "probability_leaf": True,
        "tree_node_offsets": tree_offsets,
        "node_values": node_values,
        "node_feature_idx": node_feature_idx,
        "node_num_thresholds": node_thresholds,
        "node_missing_go_to_left": node_missing_left,
        "node_left": node_left,
        "node_right": node_right,
        "node_is_leaf": node_is_leaf,
    }


def _export_scaler(model: Any) -> dict[str, Any]:
    if type(model).__name__ != "StandardScaler":
        raise NativeAssetExportError("Expected StandardScaler")
    mean = np.asarray(model.mean_, dtype=np.float64).reshape(-1)
    scale = np.asarray(model.scale_, dtype=np.float64).reshape(-1)
    if mean.shape != scale.shape:
        raise NativeAssetExportError("StandardScaler mean/scale dimensions differ")
    return {
        "kind": "standard_scaler",
        "n_features": int(mean.size),
        "mean": [_finite_float(value, field="scaler.mean") for value in mean],
        "scale": [_finite_float(value, field="scaler.scale") for value in scale],
    }


def _export_logistic(model: Any) -> dict[str, Any]:
    if type(model).__name__ != "LogisticRegression":
        raise NativeAssetExportError("Expected LogisticRegression")
    classes = _binary_classes(model)
    coefficients = np.asarray(model.coef_, dtype=np.float64)
    intercept = np.asarray(model.intercept_, dtype=np.float64).reshape(-1)
    if coefficients.shape != (1, int(getattr(model, "n_features_in_", coefficients.shape[-1]))):
        raise NativeAssetExportError("Only binary one-row logistic coefficients are supported")
    if intercept.shape != (1,):
        raise NativeAssetExportError("Only binary one-row logistic intercept is supported")
    return {
        "kind": "logistic_regression",
        "classes": classes,
        "n_features": int(coefficients.shape[1]),
        "coef": [[_finite_float(value, field="logreg.coef") for value in coefficients[0]]],
        "intercept": [_finite_float(intercept[0], field="logreg.intercept")],
    }


def export_model(model: Any) -> dict[str, Any]:
    """Export one fitted estimator/pipeline into a primitive model tree."""
    # 只导出推理所需的数值参数，禁止把 sklearn/Python 对象带入部署资产。
    model_name = type(model).__name__
    if model_name == "Pipeline":
        steps = []
        for name, step in model.steps:
            steps.append([str(name), export_model(step)])
        return {"kind": "pipeline", "steps": steps}
    if model_name == "StandardScaler":
        return _export_scaler(model)
    if model_name == "LogisticRegression":
        return _export_logistic(model)
    if model_name == "HistGradientBoostingClassifier":
        return _export_hist_gradient_boosting(model)
    if model_name in {"ExtraTreesClassifier", "RandomForestClassifier"}:
        return _export_forest(model)
    raise NativeAssetExportError(f"Unsupported estimator type: {model_name}")


def _feature_config(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    value = payload.get("feature_config")
    return None if value is None else _primitive(value, field="feature_config")


def _asset_metadata(payload: Mapping[str, Any], source_path: Path, source_sha256: str) -> dict[str, Any]:
    return {
        "source": {
            "filename": source_path.name,
            "sha256": source_sha256,
            "bytes": int(source_path.stat().st_size),
        },
        "checkpoint_config": _primitive(payload.get("checkpoint_config"), field="checkpoint_config"),
        "feature_config": _feature_config(payload),
        "identity_feature_policy": _primitive(payload.get("identity_feature_policy"), field="identity_feature_policy"),
        "protocol": _primitive(payload.get("protocol"), field="protocol"),
    }


def export_payload(payload: Mapping[str, Any], source_path: Path, source_sha256: str) -> dict[str, Any]:
    """Export a Stage-2, content-cross, or selector payload."""
    if "base_models" in payload and "meta_model" in payload:
        asset = {
            "schema": "axon_loop151_native_stage2_asset_v1",
            "asset_type": "stage2_stacker",
            **_asset_metadata(payload, source_path, source_sha256),
            "threshold": _finite_float(payload.get("threshold", 0.5), field="threshold"),
            "drop_base_prob_features": bool(payload.get("drop_base_prob_features", False)),
            "dropped_feature_count": int(payload.get("dropped_feature_count", 0)),
            "base_specs": _primitive(payload.get("base_specs", []), field="base_specs"),
            "feature_name_groups": _primitive(payload.get("feature_name_groups", {}), field="feature_name_groups"),
            "stack_feature_names": _primitive(payload.get("stack_feature_names", []), field="stack_feature_names"),
            "selected": _primitive(payload.get("selected"), field="selected"),
            "base_models": [export_model(model) for model in payload["base_models"]],
            "meta_model": export_model(payload["meta_model"]),
        }
    elif "model" in payload and "content_cross_feature_names" in payload:
        asset = {
            "schema": "axon_loop151_native_content_cross_asset_v1",
            "asset_type": "content_cross_classifier",
            **_asset_metadata(payload, source_path, source_sha256),
            "threshold": _finite_float(payload.get("threshold", 0.5), field="threshold"),
            "content_cross_feature_names": _primitive(payload["content_cross_feature_names"], field="content_cross_feature_names"),
            "selected": _primitive(payload.get("selected"), field="selected"),
            "model": export_model(payload["model"]),
        }
    elif "model" in payload and "feature_names" in payload:
        asset = {
            "schema": "axon_loop151_native_selector_asset_v1",
            "asset_type": "pairwise_selector",
            **_asset_metadata(payload, source_path, source_sha256),
            "feature_names": _primitive(payload["feature_names"], field="feature_names"),
            "key_columns": _primitive(payload.get("key_columns"), field="key_columns"),
            "baseline_score_column": _primitive(payload.get("baseline_score_column"), field="baseline_score_column"),
            "candidate_score_column": _primitive(payload.get("candidate_score_column"), field="candidate_score_column"),
            "selected": _primitive(payload.get("selected"), field="selected"),
            "model": export_model(payload["model"]),
        }
    else:
        raise NativeAssetExportError(f"Unsupported frozen payload schema: {sorted(payload)}")
    _assert_primitives(asset)
    return asset


def _assert_primitives(value: Any, *, path: str = "asset") -> None:
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_primitives(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise NativeAssetExportError(f"Non-string JSON key at {path}")
            _assert_primitives(item, path=f"{path}.{key}")
        return
    raise NativeAssetExportError(f"Non-primitive value at {path}: {type(value)!r}")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _score_tree_forest(asset: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != int(asset["n_features"]):
        raise NativeAssetExportError("Forest input dimension mismatch")
    offsets = asset["tree_node_offsets"]
    values = np.asarray(asset["node_values"], dtype=np.float64)
    feature_idx = asset["node_feature_idx"]
    thresholds = asset["node_num_thresholds"]
    missing_left = asset["node_missing_go_to_left"]
    left = asset["node_left"]
    right = asset["node_right"]
    is_leaf = asset["node_is_leaf"]
    scores = np.zeros(matrix.shape[0], dtype=np.float64)
    for tree_index in range(int(asset["tree_count"])):
        start, stop = int(offsets[tree_index]), int(offsets[tree_index + 1])
        for row_index, row in enumerate(matrix):
            node = start
            while True:
                node_offset = node - start
                if int(is_leaf[node]):
                    scores[row_index] += float(values[node])
                    break
                feature = int(feature_idx[node])
                value = row[feature]
                go_left = bool(missing_left[node]) if math.isnan(value) else value <= float(thresholds[node])
                node = start + (int(left[node]) if go_left else int(right[node]))
                if node < start or node >= stop or node_offset > stop - start:
                    raise NativeAssetExportError("Invalid forest child offset")
    return scores / max(1, int(asset["tree_count"]))


def _score_hgb(asset: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != int(asset["n_features"]):
        raise NativeAssetExportError("HGB input dimension mismatch")
    offsets = asset["tree_node_offsets"]
    score = np.full(matrix.shape[0], float(asset["baseline_prediction"]), dtype=np.float64)
    for tree_index in range(int(asset["tree_count"])):
        start, stop = int(offsets[tree_index]), int(offsets[tree_index + 1])
        for row_index, row in enumerate(matrix):
            node = start
            while True:
                index = node
                if int(asset["node_is_leaf"][index]):
                    score[row_index] += float(asset["node_values"][index])
                    break
                feature = int(asset["node_feature_idx"][index])
                value = row[feature]
                go_left = bool(asset["node_missing_go_to_left"][index]) if math.isnan(value) else value <= float(asset["node_num_thresholds"][index])
                node = start + (int(asset["node_left"][index]) if go_left else int(asset["node_right"][index]))
                if node < start or node >= stop:
                    raise NativeAssetExportError("Invalid HGB child offset")
    return _sigmoid(score)


def score_native_model(asset: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
    """Score an exported model using only its primitive representation."""
    kind = asset.get("kind")
    matrix = np.asarray(matrix)
    if kind == "pipeline":
        transformed = matrix
        for step in asset["steps"]:
            step_model = step[1] if isinstance(step, list) else step["model"]
            transformed = score_native_transform(step_model, transformed)
        return transformed
    if kind == "hist_gradient_boosting":
        return _score_hgb(asset, matrix)
    if kind in {"tree_forest", "extra_trees", "random_forest"}:
        return _score_tree_forest(asset, matrix)
    if kind == "logistic_regression":
        if matrix.ndim != 2 or matrix.shape[1] != int(asset["n_features"]):
            raise NativeAssetExportError("Logistic input dimension mismatch")
        raw = matrix.astype(np.float64, copy=False) @ np.asarray(asset["coef"], dtype=np.float64)[0] + float(asset["intercept"][0])
        return _sigmoid(raw)
    raise NativeAssetExportError(f"Unsupported native model kind: {kind}")


def score_native_transform(asset: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
    if asset.get("kind") == "standard_scaler":
        values = np.asarray(matrix)
        if values.ndim != 2 or values.shape[1] != int(asset["n_features"]):
            raise NativeAssetExportError("Scaler input dimension mismatch")
        scale = np.asarray(asset["scale"], dtype=np.float64)
        scale = np.where(np.abs(scale) < 1.0e-12, 1.0, scale)
        transformed = (values.astype(np.float64, copy=False) - np.asarray(asset["mean"], dtype=np.float64)) / scale
        return transformed.astype(values.dtype, copy=False)
    if asset.get("kind") in {"hist_gradient_boosting", "tree_forest", "extra_trees", "random_forest", "logistic_regression", "pipeline"}:
        return score_native_model(asset, matrix)
    raise NativeAssetExportError(f"Unsupported pipeline step: {asset.get('kind')}")


def _stack_features(base_scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(base_scores, dtype=np.float32), 1.0e-6, 1.0 - 1.0e-6)
    summary = np.column_stack(
        [
            clipped.mean(axis=1),
            clipped.std(axis=1),
            clipped.min(axis=1),
            clipped.max(axis=1),
            clipped.max(axis=1) - clipped.min(axis=1),
            np.median(clipped, axis=1),
            np.log(clipped / (1.0 - clipped)).mean(axis=1),
        ]
    )
    return np.column_stack([clipped, summary]).astype(np.float32, copy=False)


def score_native_asset(asset: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
    """Score a complete exported asset (stacker/classifier/selector)."""
    # Stage-2 的 base 概率和统计汇总必须按训练脚本的固定顺序重建。
    asset_type = asset.get("asset_type")
    if asset_type == "stage2_stacker":
        values = np.asarray(matrix, dtype=np.float64)
        scoring = values[:, 6:] if asset.get("drop_base_prob_features") else values
        base_scores = np.column_stack([score_native_model(model, scoring) for model in asset["base_models"]])
        return score_native_model(asset["meta_model"], _stack_features(base_scores))
    if asset_type in {"content_cross_classifier", "pairwise_selector"}:
        return score_native_model(asset["model"], matrix)
    raise NativeAssetExportError(f"Unsupported native asset type: {asset_type}")


def export_file(input_path: Path, output_path: Path) -> dict[str, Any]:
    payload, source_sha256 = load_frozen_payload(input_path)
    asset = export_payload(payload, Path(input_path), source_sha256)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asset, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    output_bytes = output_path.read_bytes()
    return {
        "output": str(output_path),
        "asset_type": asset["asset_type"],
        "schema": asset["schema"],
        "source_sha256": source_sha256,
        "output_bytes": len(output_bytes),
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the frozen Loop151 model DAG to native JSON assets.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--conservative", type=Path, required=True)
    parser.add_argument("--content-cross", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "primary": args.primary,
        "conservative": args.conservative,
        "content_cross": args.content_cross,
        "noise": args.noise,
        "selector": args.selector,
    }
    results = []
    for name, input_path in inputs.items():
        results.append(export_file(input_path, args.output_dir / f"{name}.native.json"))
    manifest = {
        "schema": "axon_loop151_native_asset_manifest_v1",
        "loop_id": "Loop151",
        "assets": results,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "assets": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
