import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from predict_api import (
    MAX_STAGE2_PICKLE_BYTES,
    NESTED_PREDICTION_RESPONSE_LIMIT,
    NESTED_SCAN_ENTRY_RESPONSE_LIMIT,
    FamilyClassifier,
    PredictRequest,
    Stage2FeatureConfig,
    Stage2ModelBundle,
    _predict_nested,
    _stage2_feature_vector,
    _stage2_metadata_path,
    _trusted_stage2_binding,
)


class DummyStage2Model:
    def predict_proba(self, matrix):
        assert matrix.shape == (1, 6)
        return np.asarray([[0.25, 0.75]], dtype=np.float32)


def test_stage2_payload_unpickler_loads_compat_feature_config(tmp_path: Path):
    path = tmp_path / "stage2.pkl"
    config = Stage2FeatureConfig(
        prefix_len=0,
        chunk_count=1,
        include_pe=False,
        include_stat=False,
        include_lightweight=False,
        include_byte_summary=False,
    )
    payload = {
        "model": DummyStage2Model(),
        "feature_config": config,
        "threshold": 0.6,
        "checkpoint_config": {
            "max_byte_length": 8192,
            "pe_feature_dim": 256,
            "stat_feature_dim": 49,
            "lightweight_feature_dim": 256,
            "dsra_dim": 160,
            "dsra_heads": 4,
            "num_classes": 2,
        },
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    _stage2_metadata_path(path).write_text(
        json.dumps(
            {
                "schema": "axon_stage2_model_metadata_v1",
                "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "knn": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )

    bundle = Stage2ModelBundle.load(path)

    assert bundle.threshold == 0.6
    assert bundle.feature_config.prefix_len == 0
    assert bundle.predict_probability(np.zeros(6, dtype=np.float32)) == 0.75


def test_stage2_load_rejects_metadata_knn_before_unpickle(monkeypatch, tmp_path: Path):
    path = tmp_path / "stage2.pkl"
    path.write_bytes(b"this payload must not be opened")
    _stage2_metadata_path(path).write_text(
        json.dumps(
            {
                "schema": "axon_stage2_model_metadata_v1",
                "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "knn": {
                    "enabled": True,
                    "reference_storage": "npz_sidecar",
                    "reference_path": "stage2.knn_reference.npz",
                },
            }
        ),
        encoding="utf-8",
    )

    def fail_if_unpickled(_self):
        raise AssertionError("Stage2 pickle was unpickled before metadata rejection")

    monkeypatch.setattr("predict_api._Stage2PayloadUnpickler.load", fail_if_unpickled)

    with pytest.raises(ValueError, match="frozen kNN memory"):
        Stage2ModelBundle.load(path)


def test_stage2_load_requires_metadata_before_unpickle(monkeypatch, tmp_path: Path):
    path = tmp_path / "stage2.pkl"
    path.write_bytes(b"this payload must not be opened")

    def fail_if_unpickled(_self):
        raise AssertionError("Stage2 pickle was unpickled without metadata")

    monkeypatch.setattr("predict_api._Stage2PayloadUnpickler.load", fail_if_unpickled)

    with pytest.raises(ValueError, match="metadata sidecar is required"):
        Stage2ModelBundle.load(path)


def test_stage2_load_requires_metadata_bound_model_sha_before_unpickle(
    monkeypatch,
    tmp_path: Path,
):
    path = tmp_path / "stage2.pkl"
    path.write_bytes(b"this payload must not be opened")
    _stage2_metadata_path(path).write_text(
        json.dumps({"schema": "axon_stage2_model_metadata_v1", "knn": {"enabled": False}}),
        encoding="utf-8",
    )

    def fail_if_unpickled(_self):
        raise AssertionError("Stage2 pickle was unpickled without a bound model SHA")

    monkeypatch.setattr("predict_api._Stage2PayloadUnpickler.load", fail_if_unpickled)

    with pytest.raises(ValueError, match="immutable model_sha256"):
        Stage2ModelBundle.load(path)


def test_stage2_request_path_must_exist_in_server_trust_manifest(
    monkeypatch,
    tmp_path: Path,
):
    model = tmp_path / "attacker.pkl"
    model.write_bytes(b"self-reported digest is not trust")
    trust_manifest = tmp_path / "trust.json"
    trust_manifest.write_text(
        json.dumps({"schema": "axon_pickle_sha256_allowlist_v1", "entries": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr("predict_api.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("predict_api.DEFAULT_STAGE2_TRUST_MANIFEST", trust_manifest)

    with pytest.raises(ValueError, match="not present in the server trust manifest"):
        _trusted_stage2_binding(model)


def test_stage2_load_accepts_external_sha_bound_metadata(tmp_path: Path):
    path = tmp_path / "stage2.pkl"
    payload = {
        "model": DummyStage2Model(),
        "feature_config": {
            "prefix_len": 0,
            "chunk_count": 1,
            "include_pe": False,
            "include_stat": False,
            "include_lightweight": False,
            "include_byte_summary": False,
        },
        "threshold": 0.6,
        "checkpoint_config": {
            "max_byte_length": 8192,
            "pe_feature_dim": 256,
            "stat_feature_dim": 49,
            "lightweight_feature_dim": 256,
            "dsra_dim": 160,
            "dsra_heads": 4,
            "num_classes": 2,
        },
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    model_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path = tmp_path / "frozen-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "axon_stage2_model_metadata_v1",
                "model_sha256": model_sha,
                "knn": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )

    bundle = Stage2ModelBundle.load(
        path,
        metadata_path=metadata_path,
        expected_model_sha256=model_sha,
    )

    assert bundle.threshold == 0.6


def test_stage2_sha_mismatch_is_rejected_before_unpickle(monkeypatch, tmp_path: Path):
    path = tmp_path / "stage2.pkl"
    path.write_bytes(b"this payload must not be opened")
    metadata_path = tmp_path / "frozen-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "axon_stage2_model_metadata_v1",
                "model_sha256": "0" * 64,
                "knn": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )

    def fail_if_unpickled(_self):
        raise AssertionError("Stage2 pickle was unpickled before SHA rejection")

    monkeypatch.setattr("predict_api._Stage2PayloadUnpickler.load", fail_if_unpickled)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        Stage2ModelBundle.load(
            path,
            metadata_path=metadata_path,
            expected_model_sha256="0" * 64,
        )


def test_stage2_size_cap_rejects_before_model_bytes_are_read(monkeypatch, tmp_path: Path):
    path = tmp_path / "oversized.pkl"
    path.write_bytes(b"not actually oversized")
    metadata_path = tmp_path / "frozen-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "axon_stage2_model_metadata_v1",
                "model_sha256": "0" * 64,
                "knn": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    original_stat = Path.stat
    original_read_bytes = Path.read_bytes

    def fake_stat(candidate: Path, *args, **kwargs):
        if candidate == path:
            return SimpleNamespace(st_size=MAX_STAGE2_PICKLE_BYTES + 1)
        return original_stat(candidate, *args, **kwargs)

    def guarded_read_bytes(candidate: Path):
        if candidate == path:
            raise AssertionError("Oversized Stage2 model bytes were read before size rejection")
        return original_read_bytes(candidate)

    monkeypatch.setattr(Path, "stat", fake_stat)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    with pytest.raises(ValueError, match="exceeds"):
        Stage2ModelBundle.load(
            path,
            metadata_path=metadata_path,
            expected_model_sha256="0" * 64,
        )


def test_stage2_load_rejects_oversized_byte_summary_config(tmp_path: Path):
    path = tmp_path / "stage2.pkl"
    payload = {
        "model": DummyStage2Model(),
        "feature_config": {
            "prefix_len": 8193,
            "chunk_count": 4097,
            "include_pe": False,
            "include_stat": False,
            "include_lightweight": False,
            "include_byte_summary": True,
        },
        "checkpoint_config": {
            "max_byte_length": 8192,
            "pe_feature_dim": 256,
            "stat_feature_dim": 49,
            "lightweight_feature_dim": 256,
            "dsra_dim": 160,
            "dsra_heads": 4,
            "num_classes": 2,
        },
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    _stage2_metadata_path(path).write_text(
        json.dumps(
            {
                "schema": "axon_stage2_model_metadata_v1",
                "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "knn": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prefix_len"):
        Stage2ModelBundle.load(path)


def test_stage2_feature_vector_matches_loop28_dimension(tmp_path: Path):
    class Features:
        byte_seq = np.arange(8192, dtype=np.uint8)
        pe_features = np.zeros(256, dtype=np.float32)
        stat_features = np.zeros(49, dtype=np.float32)
        lightweight_features = np.zeros(256, dtype=np.float32)

    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"MZ" + bytes(range(64)))
    config = Stage2FeatureConfig(
        prefix_len=256,
        chunk_count=16,
        include_pe=True,
        include_stat=True,
        include_lightweight=True,
        include_byte_summary=True,
        include_content_pe=True,
    )

    vector = _stage2_feature_vector(sample, Features(), 0.8, config)

    assert vector.shape == (1520,)
    assert np.isfinite(vector).all()


def test_family_classifier_predicts_scaled_nearest_centroid(tmp_path: Path):
    path = tmp_path / "family.json"
    path.write_text(
        """
        {
          "schema": "axon_family_classifier_v1",
          "feature_dim": 4,
          "pe_feature_dim": 2,
          "stat_feature_dim": 2,
          "cluster_ids": [7, 9],
          "family_names": ["axon_group_7", "axon_group_9"],
          "thresholds": [1.0, 1.0],
          "centroids": [[0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0]],
          "scaler_mean": [0.0, 0.0, 0.0, 0.0],
          "scaler_scale": [1.0, 1.0, 1.0, 1.0]
        }
        """,
        encoding="utf-8",
    )

    classifier = FamilyClassifier.load(path)
    result = classifier.predict(
        np.asarray([0.2, 0.1], dtype=np.float32),
        np.asarray([0.0, 0.1], dtype=np.float32),
    )

    assert result["cluster_id"] == 7
    assert result["family_name"] == "axon_group_7"
    assert result["is_new_family"] is False


def _nested_entry(index: int, *, kind: str = "other", extracted_path: str | None = None) -> dict:
    return {
        "id": index,
        "parent_id": 0 if index else None,
        "depth": 1 if index else 0,
        "logical_path": f"outer.zip/item_{index}.bin",
        "extracted_path": extracted_path,
        "kind": kind,
        "candidate_for_axon": kind == "pe",
        "archive": False,
        "status": "candidate",
    }


def _nested_report(entries: list[dict], temp_dir: str | None = None) -> dict:
    return {
        "version": 1,
        "input": "outer.zip",
        "temp_dir": temp_dir,
        "limits": {},
        "summary": {"total_entries": len(entries)},
        "entries": entries,
    }


def test_nested_prediction_truncates_scan_entries_without_loading_model(
    monkeypatch, tmp_path: Path
):
    entries = [_nested_entry(index) for index in range(NESTED_SCAN_ENTRY_RESPONSE_LIMIT + 3)]
    monkeypatch.setattr(
        "predict_api.run_archive_scan", lambda _file_path, _options: _nested_report(entries)
    )

    def fail_load_context(*_args, **_kwargs):
        raise AssertionError("No-PE nested scan must not load the model")

    monkeypatch.setattr("predict_api._load_prediction_context", fail_load_context)

    request = PredictRequest(
        file=str(tmp_path / "outer.zip"), checkpoint=str(tmp_path / "model.pt"), scan_nested=True
    )
    result = _predict_nested(
        request, tmp_path / "model.pt", "cpu", tmp_path / "outer.zip", None, None
    )

    assert result["ok"] is True
    assert result["pe_prediction_count"] == 0
    assert result["scan_entry_count"] == NESTED_SCAN_ENTRY_RESPONSE_LIMIT + 3
    assert result["scan_entries_truncated"] is True
    assert len(result["scan_entries"]) == NESTED_SCAN_ENTRY_RESPONSE_LIMIT
    assert result["archive_cleanup"]["reason"] == "no_temp_dir"


def test_nested_prediction_truncates_prediction_response_but_counts_all(
    monkeypatch, tmp_path: Path
):
    entries = [
        _nested_entry(index, kind="pe", extracted_path=str(tmp_path / f"inner_{index}.exe"))
        for index in range(NESTED_PREDICTION_RESPONSE_LIMIT + 5)
    ]
    monkeypatch.setattr(
        "predict_api.run_archive_scan", lambda _file_path, _options: _nested_report(entries)
    )
    context = SimpleNamespace(
        checkpoint_path=tmp_path / "model.pt", device="cpu", stage2=None, family_classifier=None
    )
    monkeypatch.setattr("predict_api._load_prediction_context", lambda *_args, **_kwargs: context)

    def fake_predict(_context, inner_path):
        index = int(Path(inner_path).stem.rsplit("_", 1)[1])
        return {
            "status": "predicted",
            "prediction": 1 if index % 2 == 0 else 0,
            "label": "malicious" if index % 2 == 0 else "benign",
            "confidence": 0.9,
            "prob_malicious": 0.9 if index % 2 == 0 else 0.1,
        }

    monkeypatch.setattr("predict_api._predict_pe_file", fake_predict)

    request = PredictRequest(
        file=str(tmp_path / "outer.zip"), checkpoint=str(tmp_path / "model.pt"), scan_nested=True
    )
    result = _predict_nested(
        request, tmp_path / "model.pt", "cpu", tmp_path / "outer.zip", None, None
    )

    expected_total = NESTED_PREDICTION_RESPONSE_LIMIT + 5
    assert result["ok"] is True
    assert result["pe_prediction_count"] == expected_total
    assert result["malicious_inner_count"] == (expected_total + 1) // 2
    assert result["parent_verdict"] == "malicious"
    assert result["predictions_truncated"] is True
    assert len(result["predictions"]) == NESTED_PREDICTION_RESPONSE_LIMIT


def test_nested_prediction_cleans_temp_dir_when_inner_prediction_raises(
    monkeypatch, tmp_path: Path
):
    temp_dir = tmp_path / "axon-archive-scanner-raises"
    temp_dir.mkdir()
    inner = temp_dir / "inner.exe"
    inner.write_bytes(b"MZ")
    entries = [_nested_entry(1, kind="pe", extracted_path=str(inner))]
    monkeypatch.setattr("archive_scanner.tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        "predict_api.run_archive_scan",
        lambda _file_path, _options: _nested_report(entries, temp_dir=str(temp_dir)),
    )
    context = SimpleNamespace(
        checkpoint_path=tmp_path / "model.pt", device="cpu", stage2=None, family_classifier=None
    )
    monkeypatch.setattr("predict_api._load_prediction_context", lambda *_args, **_kwargs: context)

    def raise_prediction_error(_context, _inner_path):
        raise RuntimeError("boom")

    monkeypatch.setattr("predict_api._predict_pe_file", raise_prediction_error)
    request = PredictRequest(
        file=str(tmp_path / "outer.zip"), checkpoint=str(tmp_path / "model.pt"), scan_nested=True
    )

    with pytest.raises(RuntimeError, match="boom"):
        _predict_nested(request, tmp_path / "model.pt", "cpu", tmp_path / "outer.zip", None, None)

    assert not temp_dir.exists()
