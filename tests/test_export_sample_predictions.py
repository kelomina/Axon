import csv
import inspect
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import export_sample_predictions  # noqa: E402
from export_sample_predictions import export_predictions, load_checkpoint_config, load_sample_records_from_csv, lookup_cache_sample, map_cache_samples_by_source  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_load_sample_records_from_group_members_csv():
    with _case_dir("export_sample_records") as tmp_path:
        samples_path = tmp_path / "group_members.csv"
        with samples_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "group_id",
                    "group_source",
                    "group_size",
                    "is_rare_group",
                    "is_singleton",
                    "sample_index",
                    "source_path",
                    "label",
                    "split",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "group_id": 1,
                "group_source": "singleton",
                "group_size": 1,
                "is_rare_group": "True",
                "is_singleton": "True",
                "sample_index": 42,
                "source_path": "data/sample-a.exe",
                "label": 1,
                "split": "test",
            })

        records = load_sample_records_from_csv(samples_path)

    assert len(records) == 1
    assert records[0].index == 42
    assert records[0].source_path == "data/sample-a.exe"
    assert records[0].label == 1
    assert records[0].split == "test"
    assert records[0].metadata["group_id"] == "1"
    assert records[0].metadata["group_size"] == "1"


def test_manifest_relative_path_matches_absolute_group_member_path():
    absolute_source = Path.cwd() / "data" / "待拉黑" / "sample-a.exe"
    manifest_samples = [{
        "source_path": "data/待拉黑/sample-a.exe",
        "cache_path": "data/.cache/sample-a.npz",
        "label": 1,
    }]

    cache_by_source = map_cache_samples_by_source(manifest_samples)
    matched = lookup_cache_sample(cache_by_source, str(absolute_source))

    assert matched is manifest_samples[0]


def test_load_checkpoint_config_falls_back_for_legacy_training_config(monkeypatch):
    import export_sample_predictions

    checkpoint_path = Path("legacy.pt")

    monkeypatch.setattr(
        export_sample_predictions,
        "load_safe_checkpoint",
        lambda *_args, **_kwargs: {
            "config": {},
            "train_config": {
                "batch_size": 99,
                "decision_threshold": 0.53,
                "max_epochs": 5,
                "swa_start_epoch": -10,
            },
        },
    )

    _checkpoint, _config, train_config = load_checkpoint_config(checkpoint_path, batch_size=16)

    assert train_config.batch_size == 16
    assert train_config.decision_threshold == 0.53


def test_export_predictions_records_feature_mask_summary(monkeypatch):
    with _case_dir("export_feature_mask_summary") as tmp_path:
        checkpoint_path = tmp_path / "model.pt"
        config_path = tmp_path / "config.toml"
        data_dir = tmp_path / "data"
        output_path = tmp_path / "predictions.csv"
        mask_path = tmp_path / "mask.json"
        checkpoint_path.write_bytes(b"checkpoint")
        config_path.write_text("", encoding="utf-8")
        mask_path.write_text("{}", encoding="utf-8")
        data_dir.mkdir()

        class DummyModel:
            def load_state_dict(self, _state):
                return None

            def to(self, _device):
                return self

            def eval(self):
                return self

        calls = {"mask_loaded": False}

        monkeypatch.setattr(
            export_sample_predictions,
            "load_checkpoint_config",
            lambda *_args, **_kwargs: (
                {"model_state_dict": {}},
                type("Config", (), {"max_byte_length": 1, "pe_feature_dim": 2, "stat_feature_dim": 1, "lightweight_feature_dim": 1})(),
                type("TrainConfig", (), {"decision_threshold": 0.5, "batch_size": 1, "num_workers": 0})(),
            ),
        )
        monkeypatch.setattr(
            export_sample_predictions,
            "resolve_config",
            lambda *_args, **_kwargs: ({}, type("CurrentConfig", (), {"data_dir": str(data_dir)})(), None),
        )
        monkeypatch.setattr(export_sample_predictions, "load_sample_records_from_csv", lambda _path: [])
        monkeypatch.setattr(export_sample_predictions, "load_manifest_samples", lambda *_args: ([], "manifest"))
        monkeypatch.setattr(export_sample_predictions, "AxonMalwareModel", lambda _config: DummyModel())

        def fake_load_feature_mask_tensors(path, _config, _device):
            calls["mask_loaded"] = path == mask_path.resolve()
            return object(), object(), {"kept_total": 3, "kept_pe": 2, "kept_stat": 1}

        monkeypatch.setattr(export_sample_predictions, "load_feature_mask_tensors", fake_load_feature_mask_tensors)

        summary = export_predictions(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            data_dir=data_dir,
            output_path=output_path,
            batch_size=1,
            device_name="cpu",
            samples_path=tmp_path / "samples.csv",
            feature_mask_path=mask_path,
        )

    assert calls["mask_loaded"]
    assert summary["feature_mask"] == str(mask_path.resolve())
    assert summary["feature_mask_summary"] == "kept_total=3, kept_pe=2, kept_stat=1"


def test_export_predictions_streams_predictions_and_missing_rows(monkeypatch):
    with _case_dir("export_streaming_rows") as tmp_path:
        checkpoint_path = tmp_path / "model.pt"
        config_path = tmp_path / "config.toml"
        data_dir = tmp_path / "data"
        output_path = tmp_path / "predictions.csv"
        samples_path = tmp_path / "samples.csv"
        checkpoint_path.write_bytes(b"checkpoint")
        config_path.write_text("", encoding="utf-8")
        data_dir.mkdir()
        cache_dir = data_dir / ".cache"
        cache_dir.mkdir()
        (cache_dir / "a.npz").write_bytes(b"placeholder")
        (cache_dir / "b.npz").write_bytes(b"placeholder")
        with samples_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["source_path", "label", "split", "sample_index", "group_id"],
            )
            writer.writeheader()
            writer.writerow({"source_path": "a.exe", "label": 0, "split": "val", "sample_index": 1, "group_id": "g1"})
            writer.writerow({"source_path": "b.exe", "label": 1, "split": "val", "sample_index": 2, "group_id": "g2"})
            writer.writerow({"source_path": "missing.exe", "label": 1, "split": "val", "sample_index": 3, "group_id": "g3"})

        class DummyModel:
            def load_state_dict(self, _state):
                return None

            def to(self, _device):
                return self

            def eval(self):
                return self

            def __call__(self, byte_seq, pe_features, stat_features=None):
                logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float32)[: byte_seq.shape[0]]
                return {"logits": logits}

        config = type(
            "Config",
            (),
            {"max_byte_length": 2, "pe_feature_dim": 2, "stat_feature_dim": 1, "lightweight_feature_dim": 1},
        )()
        train_config = type("TrainConfig", (), {"decision_threshold": 0.5, "batch_size": 2, "num_workers": 0})()
        monkeypatch.setattr(
            export_sample_predictions,
            "load_checkpoint_config",
            lambda *_args, **_kwargs: ({"model_state_dict": {}}, config, train_config),
        )
        monkeypatch.setattr(
            export_sample_predictions,
            "resolve_config",
            lambda *_args, **_kwargs: ({}, type("CurrentConfig", (), {"data_dir": str(data_dir)})(), None),
        )
        monkeypatch.setattr(
            export_sample_predictions,
            "load_manifest_samples",
            lambda *_args: (
                [
                    {"source_path": "a.exe", "cache_path": "a.npz", "label": 0, "source_sha256": "a" * 64},
                    {"source_path": "b.exe", "cache_path": "b.npz", "label": 1, "source_sha256": "b" * 64},
                ],
                "manifest",
            ),
        )
        monkeypatch.setattr(export_sample_predictions, "AxonMalwareModel", lambda _config: DummyModel())

        def fake_load_cached_feature_npz(
            cache_path,
            _max_byte_length,
            _pe_feature_dim,
            _stat_feature_dim,
            _lightweight_feature_dim,
            *,
            expected_label,
            expected_source_sha256,
        ):
            assert expected_source_sha256 in {"a" * 64, "b" * 64}
            return (
                np.array([77, 90], dtype=np.uint8),
                np.ones(2, dtype=np.float32),
                np.ones(1, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                int(expected_label),
            )

        monkeypatch.setattr(export_sample_predictions, "_load_cached_feature_npz", fake_load_cached_feature_npz)

        summary = export_predictions(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            data_dir=data_dir,
            output_path=output_path,
            batch_size=2,
            device_name="cpu",
            samples_path=samples_path,
        )
        prediction_rows = list(csv.DictReader(output_path.open("r", encoding="utf-8-sig", newline="")))
        missing_rows = list(csv.DictReader((tmp_path / "predictions_missing_cache.csv").open("r", encoding="utf-8-sig", newline="")))

    assert summary["raw_samples"] == 3
    assert summary["predicted_samples"] == 2
    assert summary["missing_cache_samples"] == 1
    assert [row["source_path"] for row in prediction_rows] == ["a.exe", "b.exe"]
    assert [row["prediction"] for row in prediction_rows] == ["0", "1"]
    assert missing_rows == [{"source_path": "missing.exe", "label": "1", "split": "val"}]


def test_export_predictions_does_not_accumulate_output_rows():
    source = inspect.getsource(export_sample_predictions.export_predictions)

    assert "rows = []" not in source
    assert "missing_cache = []" not in source
    assert ".writerows(" not in source
    assert "predicted_count" in source
    assert "missing_count" in source
