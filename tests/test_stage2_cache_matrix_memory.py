from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_stage2_cache_matrix as stage2  # noqa: E402


class _ExplodingCsvHandle:
    def __init__(self):
        self.lines = iter(
            [
                "source_path,label,prob_malicious\n",
                "a.exe,0,0.1\n",
                "b.exe,1,0.9\n",
            ]
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.lines)
        if line.startswith("b.exe"):
            raise AssertionError("read_prediction_rows read past max_rows")
        return line


class _FakeCsvPath:
    def open(self, *_args, **_kwargs):
        return _ExplodingCsvHandle()


def test_read_prediction_rows_max_rows_stops_without_extra_read(monkeypatch):
    monkeypatch.setattr(stage2, "resolve_path", lambda _path: _FakeCsvPath())

    rows = stage2.read_prediction_rows(Path("ignored.csv"), max_rows=1)

    assert rows == [{"source_path": "a.exe", "label": "0", "prob_malicious": "0.1"}]


def test_fresh_model_candidate_returns_unfitted_clone():
    template = LogisticRegression(C=0.25, max_iter=123)

    first = stage2.fresh_model_candidate(template)
    second = stage2.fresh_model_candidate(template)

    assert first is not template
    assert second is not template
    assert first is not second
    assert first.C == 0.25
    assert second.max_iter == 123


def test_append_feature_columns_preserves_existing_and_extra_values():
    matrix = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    extra = np.asarray([[10.0], [20.0]], dtype=np.float32)

    combined = stage2.append_feature_columns(matrix, extra)

    assert combined.dtype == np.float32
    np.testing.assert_allclose(
        combined, np.asarray([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]], dtype=np.float32)
    )


def test_resolve_knn_batch_size_clamps_dense_similarity_budget():
    batch_size = stage2.resolve_knn_batch_size(
        2048,
        query_count=5000,
        memory_count=200_000,
        dtype=np.float32,
        max_similarity_mib=4,
    )

    assert batch_size == 5


def test_resolve_knn_batch_size_rejects_impossible_budget():
    try:
        stage2.resolve_knn_batch_size(
            2048,
            query_count=10,
            memory_count=200_000,
            dtype=np.float32,
            max_similarity_mib=0.5,
        )
    except MemoryError as exc:
        assert "too small for one query row" in str(exc)
    else:
        raise AssertionError("Expected MemoryError for impossible kNN similarity budget")


def test_stage2_knn_reference_sidecar_roundtrip(tmp_path):
    model_path = tmp_path / "stage2_selected_model.pkl"
    reference_path = stage2.stage2_knn_reference_path(model_path)
    assert reference_path.name == "stage2_selected_model.knn_reference.npz"
    assert stage2.stage2_metadata_path(model_path).name == "stage2_selected_model.metadata.json"

    reference = {
        "mean": np.asarray([1.0, 2.0], dtype=np.float64),
        "std": np.asarray([0.5, 1.5], dtype=np.float64),
        "memory_norm": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        "memory_labels": np.asarray([0, 1], dtype=np.int32),
    }

    stage2.save_stage2_knn_reference_npz(reference_path, reference)
    loaded = stage2.load_stage2_knn_reference_npz(reference_path)

    assert loaded["mean"].dtype == np.float32
    assert loaded["std"].dtype == np.float32
    assert loaded["memory_norm"].dtype == np.float32
    assert loaded["memory_labels"].dtype == np.int64
    np.testing.assert_allclose(loaded["memory_norm"], reference["memory_norm"])
    assert loaded["memory_labels"].tolist() == [0, 1]


def test_stage2_knn_payload_loads_sidecar_without_in_pickle_reference(tmp_path):
    model_path = tmp_path / "stage2_selected_model.pkl"
    reference_path = stage2.stage2_knn_reference_path(model_path)
    reference = {
        "mean": np.asarray([0.0], dtype=np.float32),
        "std": np.asarray([1.0], dtype=np.float32),
        "memory_norm": np.asarray([[1.0], [-1.0]], dtype=np.float32),
        "memory_labels": np.asarray([1, 0], dtype=np.int64),
    }
    stage2.save_stage2_knn_reference_npz(reference_path, reference)

    payload = {
        "enabled": True,
        "reference_storage": "npz_sidecar",
        "reference_path": reference_path.name,
        "top_ks": [1],
        "feature_names": ["knn1_mal_ratio"],
    }
    loaded = stage2.load_stage2_knn_reference_from_payload(model_path, payload)

    assert "reference" not in payload
    np.testing.assert_allclose(loaded["mean"], np.asarray([0.0], dtype=np.float32))
    assert loaded["memory_labels"].tolist() == [1, 0]


def test_stage2_model_metadata_excludes_large_knn_reference(tmp_path):
    model_path = tmp_path / "stage2_selected_model.pkl"
    feature_config = stage2.FeatureConfig(
        prefix_len=0,
        chunk_count=1,
        include_pe=False,
        include_stat=False,
        include_lightweight=False,
        include_byte_summary=False,
    )
    payload = {
        "feature_config": feature_config,
        "checkpoint_config": {"max_byte_length": 8192, "pe_feature_dim": 256},
        "threshold": np.float32(0.75),
        "knn": {
            "enabled": True,
            "top_ks": [5, 10],
            "batch_size": 32,
            "similarity_memory_mib": 16.0,
            "feature_names": ["knn5_mal_ratio"],
            "reference_storage": "npz_sidecar",
            "reference_path": "stage2_selected_model.knn_reference.npz",
        },
    }

    metadata = stage2.stage2_model_metadata_from_payload(payload, model_path)

    assert metadata["schema"] == "axon_stage2_model_metadata_v1"
    assert metadata["knn"]["enabled"] is True
    assert metadata["knn"]["reference_storage"] == "npz_sidecar"
    assert metadata["knn"]["reference_path"] == "stage2_selected_model.knn_reference.npz"
    assert "reference" not in metadata["knn"]


def test_build_matrix_skips_missing_cache_and_returns_trimmed_arrays(monkeypatch, tmp_path):
    missing_cache = tmp_path / "missing.npz"
    first_cache = tmp_path / "first.npz"
    second_cache = tmp_path / "second.npz"
    first_cache.write_bytes(b"placeholder")
    second_cache.write_bytes(b"placeholder")
    rows = [
        {"cache_path": str(missing_cache), "label": "0", "prob_malicious": "0.1"},
        {"cache_path": str(first_cache), "label": "0", "prob_malicious": "0.2"},
        {"cache_path": str(second_cache), "label": "1", "prob_malicious": "0.8"},
    ]
    config = SimpleNamespace(
        max_byte_length=4,
        pe_feature_dim=2,
        stat_feature_dim=2,
        lightweight_feature_dim=0,
    )
    feature_config = stage2.FeatureConfig(
        prefix_len=0,
        chunk_count=1,
        include_pe=True,
        include_stat=True,
        include_lightweight=False,
        include_byte_summary=False,
    )

    def fake_load_cached_feature_npz(_cache_path, *_args, expected_label=None, **_kwargs):
        label = int(expected_label)
        return (
            np.zeros(4, dtype=np.uint8),
            np.asarray([label + 1.0, label + 2.0], dtype=np.float32),
            np.asarray([label + 3.0, label + 4.0], dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            label,
        )

    monkeypatch.setattr(stage2, "_load_cached_feature_npz", fake_load_cached_feature_npz)

    matrix, labels, base_probs, kept_rows, counts = stage2.build_matrix(
        rows, config, feature_config
    )

    assert matrix.shape == (2, 10)
    assert labels.tolist() == [0, 1]
    np.testing.assert_allclose(base_probs, np.asarray([0.2, 0.8], dtype=np.float32))
    assert kept_rows == rows[1:]
    assert counts == {"total": 3, "kept": 2, "skipped_missing_cache": 1}
