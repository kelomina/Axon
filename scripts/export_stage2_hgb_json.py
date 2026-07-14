#!/usr/bin/env python3
"""Export a frozen sklearn HistGradientBoostingClassifier to JSON.

The JSON produced here is meant for the native C++ scanner DLL. It contains
only primitive arrays, so runtime inference does not need Python or sklearn.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    prefix_len: int
    chunk_count: int
    include_pe: bool
    include_stat: bool
    include_lightweight: bool
    include_byte_summary: bool
    include_content_pe: bool = False
    content_cache_dir: Optional[str] = None
    include_content_pe_v2: bool = False
    content_pe_v2_cache_dir: Optional[str] = None
    content_pe_v2_groups: tuple[str, ...] = ("all",)
    include_content_string: bool = False
    content_string_cache_dir: Optional[str] = None
    include_content_cert: bool = False
    content_cert_cache_dir: Optional[str] = None


def _feature_config_to_dict(value) -> dict:
    if hasattr(value, "__dict__"):
        value = dict(value.__dict__)
    if not isinstance(value, dict):
        raise ValueError("feature_config must be a mapping-like object")
    out = {}
    for field in FeatureConfig.__dataclass_fields__:
        if field in value:
            item = value[field]
            if isinstance(item, tuple):
                item = list(item)
            out[field] = item
    return out


def _tree_to_payload(tree) -> dict:
    nodes = tree.nodes
    out_nodes = []
    for node in nodes:
        out_nodes.append(
            {
                "value": float(node["value"]),
                "feature_idx": int(node["feature_idx"]),
                "num_threshold": float(node["num_threshold"]),
                "missing_go_to_left": bool(node["missing_go_to_left"]),
                "left": int(node["left"]),
                "right": int(node["right"]),
                "is_leaf": bool(node["is_leaf"]),
            }
        )
    return {"nodes": out_nodes}


def export_stage2(input_path: Path, output_path: Path) -> dict:
    with Path(input_path).open("rb") as handle:
        payload = pickle.load(handle)

    model = payload["model"]
    if model.__class__.__name__ != "HistGradientBoostingClassifier":
        raise ValueError(f"Unsupported stage2 model type: {model.__class__.__name__}")
    if int(getattr(model, "n_trees_per_iteration_", 0)) != 1:
        raise ValueError("Only binary HistGradientBoostingClassifier payloads are supported")

    trees = []
    tree_node_offsets = [0]
    node_values = []
    node_feature_idx = []
    node_num_thresholds = []
    node_missing_go_to_left = []
    node_left = []
    node_right = []
    node_is_leaf = []
    for iteration in model._predictors:
        if len(iteration) != 1:
            raise ValueError("Only one tree per boosting iteration is supported")
        tree_payload = _tree_to_payload(iteration[0])
        trees.append(tree_payload)
        for node in tree_payload["nodes"]:
            node_values.append(node["value"])
            node_feature_idx.append(node["feature_idx"])
            node_num_thresholds.append(node["num_threshold"])
            node_missing_go_to_left.append(1 if node["missing_go_to_left"] else 0)
            node_left.append(node["left"])
            node_right.append(node["right"])
            node_is_leaf.append(1 if node["is_leaf"] else 0)
        tree_node_offsets.append(len(node_values))

    baseline = np.asarray(model._baseline_prediction, dtype=np.float64).reshape(-1)
    if baseline.size != 1:
        raise ValueError("Stage2 baseline prediction must contain one scalar")

    feature_config = _feature_config_to_dict(payload["feature_config"])
    out = {
        "schema": "axon_stage2_hgb_json_v1",
        "source": str(Path(input_path).resolve()),
        "model_type": "HistGradientBoostingClassifier",
        "classes": [int(value) for value in np.asarray(model.classes_).reshape(-1)],
        "baseline_prediction": float(baseline[0]),
        "threshold": float(payload.get("threshold", 0.5)),
        "feature_config": feature_config,
        "checkpoint_config": dict(payload["checkpoint_config"]),
        "selected": payload.get("selected"),
        "n_features": int(getattr(model, "n_features_in_", 0)),
        "tree_count": int(len(trees)),
        "tree_node_offsets": tree_node_offsets,
        "node_values": node_values,
        "node_feature_idx": node_feature_idx,
        "node_num_thresholds": node_num_thresholds,
        "node_missing_go_to_left": node_missing_go_to_left,
        "node_left": node_left,
        "node_right": node_right,
        "node_is_leaf": node_is_leaf,
        "trees": trees,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {
        "output": str(output_path.resolve()),
        "tree_count": out["tree_count"],
        "n_features": out["n_features"],
        "threshold": out["threshold"],
    }


def _predict_json_model(model_payload: dict, matrix: np.ndarray) -> np.ndarray:
    scores = np.full(matrix.shape[0], float(model_payload["baseline_prediction"]), dtype=np.float64)
    for tree in model_payload["trees"]:
        nodes = tree["nodes"]
        for row_index, row in enumerate(matrix):
            node_index = 0
            while True:
                node = nodes[node_index]
                if node["is_leaf"]:
                    scores[row_index] += float(node["value"])
                    break
                feature_value = float(row[int(node["feature_idx"])])
                if math.isnan(feature_value):
                    go_left = bool(node["missing_go_to_left"])
                else:
                    go_left = feature_value <= float(node["num_threshold"])
                node_index = int(node["left"] if go_left else node["right"])
    return 1.0 / (1.0 + np.exp(-scores))


def verify_export(input_path: Path, output_path: Path) -> dict:
    with Path(input_path).open("rb") as handle:
        original = pickle.load(handle)["model"]
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(16, int(payload["n_features"]))).astype(np.float32)
    expected = original.predict_proba(matrix)[:, 1]
    actual = _predict_json_model(payload, matrix)
    return {
        "max_abs_diff": float(np.max(np.abs(expected - actual))),
        "mean_abs_diff": float(np.mean(np.abs(expected - actual))),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage-2 HGB model to native JSON")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = export_stage2(args.input, args.output)
    if args.verify:
        summary["verify"] = verify_export(args.input, args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
