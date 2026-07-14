import argparse

import numpy as np

from scripts import evaluate_stage2_cache_model as eval_stage2
from scripts.train_stage2_cache_matrix import FeatureConfig


def _feature_config(content_cache_dir: str) -> FeatureConfig:
    return FeatureConfig(
        prefix_len=0,
        chunk_count=1,
        include_pe=False,
        include_stat=False,
        include_lightweight=False,
        include_byte_summary=False,
        content_cache_dir=content_cache_dir,
    )


def test_loop43_payload_appends_content_cross_features(monkeypatch, tmp_path):
    base = np.ones((2, 3), dtype=np.float32)
    rows = [{"source_sha256": "a"}, {"source_sha256": "b"}]
    payload = {
        "schema": "axon_loop43_content_cross_payload_v1",
        "content_cross_feature_names": ["cross_a", "cross_b"],
    }
    feature_config = _feature_config(str(tmp_path / "v1"))
    args = argparse.Namespace(
        content_pe_cache_dir=None,
        content_pe_v2_cache_dir=tmp_path / "v2",
        allow_content_sidecar_build=True,
    )

    captured = {}

    def fake_build_content_cross_matrix(kept_rows, config):
        captured["rows"] = kept_rows
        captured["config"] = config
        return np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)

    monkeypatch.setattr(eval_stage2, "build_content_cross_matrix", fake_build_content_cross_matrix)

    matrix, names = eval_stage2.append_payload_extra_features(base, rows, payload, feature_config, args)

    assert matrix.shape == (2, 5)
    assert names == ["cross_a", "cross_b"]
    assert captured["rows"] == rows
    assert captured["config"].content_pe_cache_dir.endswith("v1")
    assert captured["config"].content_pe_v2_cache_dir.endswith("v2")
    np.testing.assert_allclose(matrix[:, -2:], [[2.0, 3.0], [4.0, 5.0]])


def test_loop43_payload_requires_v2_cache_dir(tmp_path):
    base = np.ones((1, 3), dtype=np.float32)
    payload = {"schema": "axon_loop43_content_cross_payload_v1"}
    feature_config = _feature_config(str(tmp_path / "v1"))
    args = argparse.Namespace(content_pe_cache_dir=None, content_pe_v2_cache_dir=None, allow_content_sidecar_build=False)

    try:
        eval_stage2.append_payload_extra_features(base, [{}], payload, feature_config, args)
    except ValueError as exc:
        assert "content-cross payload requires" in str(exc)
    else:
        raise AssertionError("expected missing v2 cache dir to fail")


def test_cache_only_content_cross_requires_existing_sidecars(tmp_path):
    row = {"source_sha256": "a" * 64}
    try:
        eval_stage2.build_content_cross_matrix_from_sidecars(
            [row],
            content_pe_cache_dir=tmp_path / "v1",
            content_pe_v2_cache_dir=tmp_path / "v2",
            progress_interval=0,
        )
    except FileNotFoundError as exc:
        assert "content_pe_v1_sidecar_missing" in str(exc)
    else:
        raise AssertionError("expected cache-only eval to fail on missing sidecars")


def test_feature_dim_guard_fails_on_mismatch():
    class Model:
        n_features_in_ = 4

    try:
        eval_stage2.assert_expected_feature_dim(Model(), np.zeros((2, 3), dtype=np.float32))
    except ValueError as exc:
        assert "feature dimension mismatch" in str(exc)
    else:
        raise AssertionError("expected feature dimension mismatch to fail")


def test_predict_scores_chunked_preserves_order(monkeypatch):
    calls = []

    def fake_predict_scores(_model, matrix):
        calls.append(matrix.copy())
        return matrix[:, 0].astype(np.float32, copy=False)

    monkeypatch.setattr(eval_stage2, "predict_scores", fake_predict_scores)
    matrix = np.arange(6, dtype=np.float32).reshape(6, 1)
    scores = eval_stage2.predict_scores_chunked(object(), matrix, 2)

    np.testing.assert_allclose(scores, np.arange(6, dtype=np.float32))
    assert [call.shape[0] for call in calls] == [2, 2, 2]
