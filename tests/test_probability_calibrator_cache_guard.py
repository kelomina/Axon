import csv
import inspect
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_probability_calibrator as eval_calibrator  # noqa: E402
import train_probability_calibrator as train_calibrator  # noqa: E402
from evaluate_probability_calibrator import (  # noqa: E402
    _load_prediction_features,
    _metrics,
    _slice_metrics,
    _write_calibrated_prediction_rows,
)
from train_probability_calibrator import _load_prediction_features as _load_train_prediction_features  # noqa: E402


SHA_A = "a" * 64
SHA_B = "b" * 64


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        import shutil

        shutil.rmtree(path, ignore_errors=True)


def _write_cache_npz(path: Path, label: int, source_sha256: str = SHA_A) -> None:
    np.savez(
        path,
        byte_sequence=np.zeros(8, dtype=np.uint8),
        pe_features=np.ones(4, dtype=np.float32),
        stat_features=np.ones(2, dtype=np.float32),
        label=np.asarray(label, dtype=np.int64),
        source_sha256=np.asarray(source_sha256),
    )


def _write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["cache_path", "label", "prob_malicious", "source_path", "source_sha256", "split", "sample_index"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def test_calibrator_feature_loader_fails_when_cache_is_missing_by_default():
    with _case_dir("calibrator_missing_cache") as tmp_path:
        existing_cache = tmp_path / "existing.npz"
        missing_cache = tmp_path / "missing.npz"
        predictions = tmp_path / "predictions.csv"
        _write_cache_npz(existing_cache, label=1)
        _write_predictions(
            predictions,
            [
                {"cache_path": str(existing_cache), "label": "1", "prob_malicious": "0.8", "source_sha256": SHA_A},
                {"cache_path": str(missing_cache), "label": "0", "prob_malicious": "0.2", "source_sha256": SHA_B},
            ],
        )

        with pytest.raises(FileNotFoundError, match="missing cache files"):
            _load_prediction_features(predictions)


def test_calibrator_feature_loader_can_allow_diagnostic_subset_runs():
    with _case_dir("calibrator_allow_missing_cache") as tmp_path:
        existing_cache = tmp_path / "existing.npz"
        missing_cache = tmp_path / "missing.npz"
        predictions = tmp_path / "predictions.csv"
        _write_cache_npz(existing_cache, label=1)
        _write_predictions(
            predictions,
            [
                {"cache_path": str(existing_cache), "label": "1", "prob_malicious": "0.8", "source_sha256": SHA_A},
                {"cache_path": str(missing_cache), "label": "0", "prob_malicious": "0.2", "source_sha256": SHA_B},
            ],
        )

        features, labels, probabilities, kept_rows, counts = _load_prediction_features(
            predictions,
            allow_missing_cache=True,
            missing_cache_output=tmp_path / "missing_cache.csv",
        )
        missing_rows = _read_csv_rows(tmp_path / "missing_cache.csv")

    assert features.shape == (1, 10)
    assert labels.tolist() == [1]
    assert np.allclose(probabilities, [0.8])
    assert kept_rows[0]["label"] == "1"
    assert counts["total"] == 2
    assert counts["kept"] == 1
    assert counts["skipped_missing_cache"] == 1
    assert len(counts["missing_cache_examples"]) == 1
    assert counts["missing_cache_output"].endswith("missing_cache.csv")
    assert missing_rows[0]["cache_path"] == str(missing_cache)
    assert missing_rows[0]["label"] == "0"


def test_calibrator_missing_cache_output_preserves_bom_source_path_header():
    with _case_dir("calibrator_bom_source_path") as tmp_path:
        missing_cache = tmp_path / "missing.npz"
        predictions = tmp_path / "predictions.csv"
        with predictions.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["source_path", "source_sha256", "cache_path", "label", "prob_malicious", "split", "sample_index"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "source_path": "data/raw.exe",
                    "source_sha256": SHA_A,
                    "cache_path": str(missing_cache),
                    "label": "1",
                    "prob_malicious": "0.8",
                    "split": "test",
                    "sample_index": "7",
                }
            )

        with pytest.raises(ValueError, match="No usable prediction rows"):
            _load_prediction_features(
                predictions,
                allow_missing_cache=True,
                missing_cache_output=tmp_path / "missing_cache.csv",
            )
        missing_rows = _read_csv_rows(tmp_path / "missing_cache.csv")

    assert missing_rows[0]["source_path"] == "data/raw.exe"
    assert missing_rows[0]["source_sha256"] == SHA_A
    assert missing_rows[0]["sample_index"] == "7"


def test_calibrator_feature_loader_rejects_invalid_source_sha256():
    with _case_dir("calibrator_invalid_sha") as tmp_path:
        existing_cache = tmp_path / "existing.npz"
        predictions = tmp_path / "predictions.csv"
        _write_cache_npz(existing_cache, label=1, source_sha256=SHA_A)
        _write_predictions(
            predictions,
            [
                {"cache_path": str(existing_cache), "label": "1", "prob_malicious": "0.8", "source_sha256": "not-a-sha"},
            ],
        )

        with pytest.raises(ValueError, match="invalid source_sha256"):
            _load_prediction_features(predictions)


def test_calibrator_feature_loader_rejects_cache_sha_mismatch():
    with _case_dir("calibrator_cache_sha_mismatch") as tmp_path:
        existing_cache = tmp_path / "existing.npz"
        predictions = tmp_path / "predictions.csv"
        _write_cache_npz(existing_cache, label=1, source_sha256=SHA_A)
        _write_predictions(
            predictions,
            [
                {"cache_path": str(existing_cache), "label": "1", "prob_malicious": "0.8", "source_sha256": SHA_B},
            ],
        )

        with pytest.raises(ValueError, match="source_sha256 mismatch"):
            _load_prediction_features(predictions)


def test_calibrator_feature_loader_rejects_cache_label_mismatch():
    with _case_dir("calibrator_cache_label_mismatch") as tmp_path:
        existing_cache = tmp_path / "existing.npz"
        predictions = tmp_path / "predictions.csv"
        _write_cache_npz(existing_cache, label=0, source_sha256=SHA_A)
        _write_predictions(
            predictions,
            [
                {"cache_path": str(existing_cache), "label": "1", "prob_malicious": "0.8", "source_sha256": SHA_A},
            ],
        )

        with pytest.raises(ValueError, match="Cache label mismatch"):
            _load_prediction_features(predictions)


def test_train_calibrator_feature_loader_uses_same_hash_label_guards():
    with _case_dir("train_calibrator_cache_guards") as tmp_path:
        existing_cache = tmp_path / "existing.npz"
        predictions = tmp_path / "predictions.csv"
        _write_cache_npz(existing_cache, label=1, source_sha256=SHA_A)
        _write_predictions(
            predictions,
            [
                {"cache_path": str(existing_cache), "label": "1", "prob_malicious": "0.8", "source_sha256": SHA_A},
            ],
        )

        features, labels, probabilities, counts = _load_train_prediction_features(predictions)

    assert features.shape == (1, 10)
    assert labels.tolist() == [1]
    assert np.allclose(probabilities, [0.8])
    assert counts["kept"] == 1


def test_train_calibrator_feature_loader_rejects_wrong_split():
    with _case_dir("train_calibrator_wrong_split") as tmp_path:
        existing_cache = tmp_path / "existing.npz"
        predictions = tmp_path / "predictions.csv"
        _write_cache_npz(existing_cache, label=1, source_sha256=SHA_A)
        _write_predictions(
            predictions,
            [
                {
                    "cache_path": str(existing_cache),
                    "label": "1",
                    "prob_malicious": "0.8",
                    "source_sha256": SHA_A,
                    "split": "val",
                },
            ],
        )

        with pytest.raises(ValueError, match="split mismatch"):
            _load_train_prediction_features(predictions, expected_split="train")


def test_eval_calibrator_feature_loader_can_skip_kept_rows():
    with _case_dir("eval_calibrator_skip_kept_rows") as tmp_path:
        existing_cache = tmp_path / "existing.npz"
        predictions = tmp_path / "predictions.csv"
        _write_cache_npz(existing_cache, label=1, source_sha256=SHA_A)
        _write_predictions(
            predictions,
            [
                {"cache_path": str(existing_cache), "label": "1", "prob_malicious": "0.8", "source_sha256": SHA_A},
            ],
        )

        features, labels, probabilities, kept_rows, counts = _load_prediction_features(
            predictions,
            include_kept_rows=False,
        )

    assert features.shape == (1, 10)
    assert labels.tolist() == [1]
    assert np.allclose(probabilities, [0.8])
    assert kept_rows == []
    assert counts["kept"] == 1


def test_probability_calibrator_loaders_avoid_full_csv_rows_and_vstack():
    train_loader_source = inspect.getsource(train_calibrator._load_prediction_features)
    eval_loader_source = inspect.getsource(eval_calibrator._load_prediction_features)
    train_main_source = inspect.getsource(train_calibrator.main)

    assert "list(csv.DictReader" not in train_loader_source
    assert "list(csv.DictReader" not in eval_loader_source
    assert "np.vstack" not in train_loader_source
    assert "np.vstack" not in eval_loader_source
    assert "np.empty" in train_loader_source
    assert "np.empty" in eval_loader_source
    assert "missing_cache_rows = []" not in eval_loader_source
    assert "missing_writer.writerow" in eval_loader_source
    assert "fitted_candidates" not in train_main_source
    assert "best_model" in train_main_source


def test_calibrator_metrics_allow_single_class_hard_holdouts():
    metrics = _metrics(
        scores=np.asarray([0.2, 0.8, 0.9], dtype=np.float32),
        labels=np.asarray([1, 1, 1], dtype=np.int64),
        threshold=0.5,
    )

    assert metrics["auc"] is None
    assert metrics["true_positive"] == 2
    assert metrics["false_negative"] == 1
    assert metrics["errors"] == 1


def test_calibrator_slice_metrics_do_not_use_path_name_groups():
    rows = [
        {"source_path": "data/待加入白名单/a.exe"},
        {"source_path": "data/待拉黑/b.exe"},
    ]
    labels = np.asarray([0, 1], dtype=np.int64)
    baseline_scores = np.asarray([0.8, 0.2], dtype=np.float32)
    calibrator_scores = np.asarray([0.1, 0.9], dtype=np.float32)

    slices = _slice_metrics(
        rows=rows,
        labels=labels,
        baseline_scores=baseline_scores,
        calibrator_scores=calibrator_scores,
        baseline_threshold=0.5,
        calibrator_threshold=0.5,
    )

    assert "whitelist_benign_label_0" not in slices
    assert slices["benign_label_0"]["baseline"]["false_positive"] == 1
    assert slices["benign_label_0"]["calibrator_metrics"]["false_positive"] == 0
    assert slices["malicious_label_1"]["baseline"]["false_negative"] == 1
    assert slices["malicious_label_1"]["calibrator_metrics"]["false_negative"] == 0
    assert "baseline_near_threshold_0.40_0.60" not in slices


def test_write_calibrated_prediction_rows_records_transitions_without_path_decisions():
    with _case_dir("calibrated_prediction_rows") as tmp_path:
        output = tmp_path / "calibrated.csv"
        rows = [
            {"source_path": "looks_bad.exe", "source_sha256": SHA_A, "cache_path": "a.npz", "split": "val", "sample_index": "1"},
            {"source_path": "looks_good.exe", "source_sha256": SHA_B, "cache_path": "b.npz", "split": "val", "sample_index": "2"},
        ]

        _write_calibrated_prediction_rows(
            output,
            rows=rows,
            labels=np.asarray([0, 1], dtype=np.int64),
            baseline_scores=np.asarray([0.9, 0.2], dtype=np.float32),
            calibrator_model_scores=np.asarray([0.1, 0.8], dtype=np.float32),
            calibrator_scores=np.asarray([0.1, 0.8], dtype=np.float32),
            baseline_threshold=0.5,
            calibrator_threshold=0.5,
            blend_model_weight=1.0,
        )
        written = _read_csv_rows(output)

    assert [row["error_transition"] for row in written] == ["fixed_by_calibrator", "fixed_by_calibrator"]
    assert "filename" not in written[0]
    assert "directory" not in written[0]
    assert "extension" not in written[0]
    assert written[0]["baseline_threshold"] == "0.5"
    assert written[0]["calibrated_threshold"] == "0.5"
    assert written[0]["blend_model_weight"] == "1.0"
    assert float(written[0]["calibrated_minus_baseline"]) < 0.0
    assert written[0]["source_path"] == "looks_bad.exe"
    assert written[0]["calibrated_prediction"] == "0"
    assert written[1]["calibrated_prediction"] == "1"
