import csv
import inspect
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_split_from_cache  # noqa: E402
from evaluate_split_from_cache import (  # noqa: E402
    cache_eval_num_workers,
    evaluate_from_cache,
    iter_split_rows,
    load_manifest_lookup,
    lookup_manifest_sample,
    write_prediction_rows,
    write_missing_cache_rows,
)


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(row)
        return rows


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_lookup_manifest_sample_matches_original_source_path_for_worktree_rows():
    with _case_dir("split_cache_lookup_original") as tmp_path:
        original = tmp_path / "data" / "待拉黑" / "sample.exe"
        materialized = tmp_path / "data" / "random_20w_worktree" / "待拉黑" / "sample.exe"
        manifest_path = tmp_path / "manifest_38672ba0.json"
        sample = {
            "source_path": str(original),
            "cache_path": str(tmp_path / "cache.npz"),
            "label": 1,
            "source_sha256": "a" * 64,
        }
        manifest_path.write_text(json.dumps({"samples": [sample]}), encoding="utf-8")

        by_source, by_sha = load_manifest_lookup(manifest_path)
        matched, reason = lookup_manifest_sample(
            {
                "source_path": str(materialized),
                "original_source_path": str(original),
                "label": "1",
            },
            by_source,
            by_sha,
        )

    assert matched == sample
    assert reason == "source_path"


def test_cache_eval_num_workers_rejects_windows_worker_copies():
    with pytest.raises(ValueError, match="num_workers > 0"):
        cache_eval_num_workers(1)


def test_lookup_manifest_sample_falls_back_to_sha_from_path_stem():
    with _case_dir("split_cache_lookup_sha") as tmp_path:
        source_sha = "b" * 64
        manifest_path = tmp_path / "manifest_38672ba0.json"
        sample = {
            "source_path": str(tmp_path / "other" / "sample.exe"),
            "cache_path": str(tmp_path / "cache.npz"),
            "label": 0,
            "source_sha256": source_sha,
        }
        manifest_path.write_text(json.dumps({"samples": [sample]}), encoding="utf-8")

        by_source, by_sha = load_manifest_lookup(manifest_path)
        matched, reason = lookup_manifest_sample(
            {"source_path": str(tmp_path / f"{source_sha}.exe"), "label": "0"},
            by_source,
            by_sha,
        )

    assert matched == sample
    assert reason == "source_sha256_from_path"


def test_write_missing_cache_rows_preserves_original_source_path_header():
    with _case_dir("split_cache_missing_rows") as tmp_path:
        output_path = tmp_path / "missing.csv"

        write_missing_cache_rows(
            output_path,
            [
                {
                    "source_path": "worktree.exe",
                    "original_source_path": "original.exe",
                    "label": "1",
                    "split": "test",
                    "sample_index": "7",
                    "reason": "missing",
                }
            ],
        )

        rows = _read_csv_rows(output_path)

    assert set(rows[0]) == {"source_path", "original_source_path", "label", "split", "sample_index", "reason"}
    assert rows[0]["original_source_path"] == "original.exe"


def test_write_prediction_rows_computes_prediction_and_correct_flag():
    with _case_dir("split_cache_prediction_rows") as tmp_path:
        output_path = tmp_path / "predictions.csv"

        write_prediction_rows(
            output_path,
            [
                {
                    "source_path": "sample.exe",
                    "original_source_path": "",
                    "cache_path": "sample.npz",
                    "source_sha256": "c" * 64,
                    "label": 1,
                    "split": "val",
                    "sample_index": "3",
                    "prob_malicious": 0.7,
                }
            ],
            threshold=0.6,
        )

        rows = _read_csv_rows(output_path)

    assert rows[0]["prob_malicious"] == "0.7"
    assert rows[0]["prediction"] == "1"
    assert rows[0]["correct"] == "True"


def test_iter_split_rows_filters_and_stops_at_max_rows():
    with _case_dir("split_cache_iter_rows") as tmp_path:
        split_csv = tmp_path / "split.csv"
        with split_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source_path", "label", "split", "sample_index"])
            writer.writeheader()
            for index in range(5):
                writer.writerow(
                    {
                        "source_path": f"sample-{index}.exe",
                        "label": index % 2,
                        "split": "val" if index != 1 else "train",
                        "sample_index": index,
                    }
                )

        rows = list(iter_split_rows(split_csv, split="val", max_rows=2))

    assert [row["source_path"] for row in rows] == ["sample-0.exe", "sample-2.exe"]


def test_evaluate_from_cache_streams_prediction_and_missing_outputs(monkeypatch):
    with _case_dir("split_cache_streaming_eval") as tmp_path:
        checkpoint = tmp_path / "model.pt"
        config = tmp_path / "config.toml"
        manifest = tmp_path / "manifest.json"
        split_csv = tmp_path / "split.csv"
        output_json = tmp_path / "report.json"
        predictions_csv = tmp_path / "predictions.csv"
        missing_csv = tmp_path / "missing.csv"
        checkpoint.write_bytes(b"checkpoint")
        config.write_text("", encoding="utf-8")
        cache_a = tmp_path / "a.npz"
        cache_b = tmp_path / "b.npz"
        cache_a.write_bytes(b"cache-a")
        cache_b.write_bytes(b"cache-b")
        manifest.write_text(
            json.dumps(
                {
                    "samples": [
                        {"source_path": "a.exe", "cache_path": str(cache_a), "label": 0, "source_sha256": "a" * 64},
                        {"source_path": "b.exe", "cache_path": str(cache_b), "label": 1, "source_sha256": "b" * 64},
                    ]
                }
            ),
            encoding="utf-8",
        )
        with split_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["source_path", "original_source_path", "label", "split", "sample_index"],
            )
            writer.writeheader()
            writer.writerow({"source_path": "a.exe", "original_source_path": "", "label": 0, "split": "val", "sample_index": 1})
            writer.writerow({"source_path": "b.exe", "original_source_path": "", "label": 1, "split": "val", "sample_index": 2})
            writer.writerow({"source_path": "missing.exe", "original_source_path": "", "label": 1, "split": "val", "sample_index": 3})

        class DummyModel:
            def load_state_dict(self, _state):
                return None

            def to(self, _device):
                return self

            def eval(self):
                return self

            def __call__(self, byte_seq, pe_features, stat_features=None):
                return {"logits": torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float32)[: byte_seq.shape[0]]}

        monkeypatch.setattr(evaluate_split_from_cache, "load_safe_checkpoint", lambda *_args, **_kwargs: {"model_state_dict": {}})
        monkeypatch.setattr(evaluate_split_from_cache, "AxonMalwareModel", lambda _config: DummyModel())

        def fake_load_cached_feature_npz(
            _cache_path,
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

        monkeypatch.setattr(evaluate_split_from_cache, "_load_cached_feature_npz", fake_load_cached_feature_npz)

        payload = evaluate_from_cache(
            checkpoint_path=checkpoint,
            config_path=config,
            split_csv=split_csv,
            manifest_path=manifest,
            output_json=output_json,
            split="val",
            threshold=0.5,
            sweep_thresholds=None,
            batch_size=2,
            num_workers=0,
            max_rows=None,
            device_name="cpu",
            missing_cache_output=missing_csv,
            output_predictions_csv=predictions_csv,
        )
        prediction_rows = _read_csv_rows(predictions_csv)
        missing_rows = _read_csv_rows(missing_csv)

    assert payload["raw_rows"] == 3
    assert payload["predicted_samples"] == 2
    assert payload["missing_cache_samples"] == 1
    assert [row["source_path"] for row in prediction_rows] == ["a.exe", "b.exe"]
    assert [row["prediction"] for row in prediction_rows] == ["0", "1"]
    assert missing_rows[0]["source_path"] == "missing.exe"
    assert missing_rows[0]["reason"] == "missing"


def test_evaluate_from_cache_avoids_full_split_and_output_row_lists():
    source = inspect.getsource(evaluate_split_from_cache.evaluate_from_cache)
    iterator_source = inspect.getsource(evaluate_split_from_cache.iter_split_rows)

    assert "list(csv.DictReader" not in iterator_source
    assert "raw_rows = iter_split_rows" not in source
    assert "prediction_rows" not in source
    assert "missing_cache: list" not in source
    assert "missing_cache_count" in source
    assert "raw_row_count" in source


def test_evaluate_from_cache_uses_restricted_checkpoint_loader():
    source = inspect.getsource(evaluate_split_from_cache.evaluate_from_cache)

    assert "load_safe_checkpoint(resolve_path(checkpoint_path), map_location=\"cpu\")" in source
    assert "weights_only=False" not in source
