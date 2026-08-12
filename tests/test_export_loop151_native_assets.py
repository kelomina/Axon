from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from export_loop151_native_assets import (  # noqa: E402
    _assert_primitives,
    _stack_features,
    export_model,
    export_payload,
    score_native_asset,
    score_native_model,
)


def _dataset(seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(160, 7)).astype(np.float64)
    labels = (features[:, 0] - 0.5 * features[:, 1] + 0.2 * features[:, 2] > 0).astype(np.int64)
    return features, labels


def _json_round_trip(value):
    return json.loads(json.dumps(value, allow_nan=False))


def test_hgb_native_asset_matches_sklearn() -> None:
    features, labels = _dataset()
    model = HistGradientBoostingClassifier(
        max_iter=12,
        max_leaf_nodes=7,
        learning_rate=0.08,
        random_state=11,
    ).fit(features, labels)
    asset = _json_round_trip(export_model(model))
    expected = model.predict_proba(features)[:, 1]
    actual = score_native_model(asset, features)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)
    _assert_primitives(asset)


def test_extra_trees_native_asset_matches_sklearn() -> None:
    features, labels = _dataset()
    model = ExtraTreesClassifier(n_estimators=13, random_state=12, n_jobs=1).fit(features, labels)
    asset = _json_round_trip(export_model(model))
    expected = model.predict_proba(features)[:, 1]
    actual = score_native_model(asset, features)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)
    _assert_primitives(asset)


def test_scaled_logistic_native_asset_matches_sklearn() -> None:
    features, labels = _dataset()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.4, solver="liblinear", max_iter=2000),
    ).fit(features, labels)
    asset = _json_round_trip(export_model(model))
    expected = model.predict_proba(features)[:, 1]
    actual = score_native_model(asset, features)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)
    _assert_primitives(asset)


def test_stage2_stacker_native_asset_matches_sklearn_components(tmp_path: Path) -> None:
    features, labels = _dataset()
    base_models = [
        HistGradientBoostingClassifier(max_iter=10, max_leaf_nodes=7, random_state=21).fit(features, labels),
        ExtraTreesClassifier(n_estimators=9, random_state=22, n_jobs=1).fit(features, labels),
        make_pipeline(StandardScaler(), LogisticRegression(C=0.7, solver="liblinear", max_iter=2000)).fit(features, labels),
    ]
    base_scores = np.column_stack([model.predict_proba(features)[:, 1] for model in base_models])
    stack_features = _stack_features(base_scores)
    meta_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.2, solver="liblinear", max_iter=2000),
    ).fit(stack_features, labels)
    payload = {
        "base_models": base_models,
        "meta_model": meta_model,
        "threshold": 0.37,
        "drop_base_prob_features": False,
        "dropped_feature_count": 0,
        "base_specs": [{"name": "synthetic"}],
        "feature_name_groups": {},
        "stack_feature_names": [],
        "selected": {"name": "synthetic"},
        "checkpoint_config": {"max_byte_length": 7},
        "feature_config": {"prefix_len": 0, "chunk_count": 1},
    }
    source = tmp_path / "synthetic.pkl"
    source.write_bytes(b"synthetic frozen payload")
    asset = _json_round_trip(export_payload(payload, source, "0" * 64))
    expected = meta_model.predict_proba(
        _stack_features(np.column_stack([model.predict_proba(features)[:, 1] for model in base_models]))
    )[:, 1]
    actual = score_native_asset(asset, features)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-8)
    _assert_primitives(asset)


def test_stage2_drop_base_probability_features_is_preserved() -> None:
    features, labels = _dataset()
    base_input = features[:, 6:]
    base_model = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000).fit(base_input, labels)
    base_scores = base_model.predict_proba(base_input)[:, 1].reshape(-1, 1)
    meta_model = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000).fit(
        _stack_features(base_scores), labels
    )
    payload = {
        "base_models": [base_model],
        "meta_model": meta_model,
        "drop_base_prob_features": True,
        "dropped_feature_count": 6,
        "threshold": 0.5,
        "base_specs": [],
        "feature_name_groups": {},
        "stack_feature_names": [],
        "checkpoint_config": {},
        "feature_config": {},
    }
    source = Path(__file__)
    asset = _json_round_trip(export_payload(payload, source, "1" * 64))
    actual = score_native_asset(asset, features)
    expected = meta_model.predict_proba(_stack_features(base_scores))[:, 1]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)
