from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_loop146_loop136_allrow_calibrator import full_feature_matrix, main, metrics  # noqa: E402


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "source_path",
        "cache_path",
        "source_sha256",
        "label",
        "split",
        "sample_index",
        "stage2_prob_malicious",
        "prediction",
        "baseline_prob_malicious",
        "candidate_prob_malicious",
        "selector_score",
        "selector_accept_candidate",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _rows(split: str, offset: int = 0) -> list[dict[str, object]]:
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    probs = [0.02, 0.12, 0.28, 0.42, 0.58, 0.72, 0.88, 0.98]
    rows = []
    for index, (label, prob) in enumerate(zip(labels, probs)):
        rows.append(
            {
                "source_path": f"sample-{split}-{index}",
                "cache_path": f"cache-{split}-{index}.npz",
                "source_sha256": f"{offset + index:064x}",
                "label": label,
                "split": split,
                "sample_index": offset + index,
                "stage2_prob_malicious": prob,
                "prediction": int(prob >= 0.5),
                "baseline_prob_malicious": prob,
                "candidate_prob_malicious": min(max(prob + (0.04 if label else -0.04), 0.001), 0.999),
                "selector_score": 0.0,
                "selector_accept_candidate": 0,
            }
        )
    return rows


def test_feature_matrix_excludes_identity_fields():
    rows = _rows("val")
    matrix, labels, predictions, names = full_feature_matrix(rows)

    assert matrix.shape == (8, len(names))
    assert labels.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert predictions.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert not any("source" in name or "path" in name or "sha" in name for name in names)


def test_train_and_eval_roundtrip(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _write_predictions(train_csv, _rows("train"))
    _write_predictions(val_csv, _rows("val", offset=100))

    out_dir = tmp_path / "out"
    assert (
        main(
            [
                "train",
                "--train-predictions",
                str(train_csv),
                "--val-predictions",
                str(val_csv),
                "--output-dir",
                str(out_dir),
                "--thresholds",
                "0.5",
            ]
        )
        == 0
    )

    report = json.loads((out_dir / "loop146_allrow_calibrator_report.json").read_text(encoding="utf-8"))
    assert report["records"] == {"train": 8, "val": 8}
    assert report["feature_dim"] == 20

    eval_json = tmp_path / "eval.json"
    eval_csv = tmp_path / "eval.csv"
    assert (
        main(
            [
                "eval",
                "--model",
                str(out_dir / "loop146_allrow_calibrator.pkl"),
                "--predictions",
                str(val_csv),
                "--output-json",
                str(eval_json),
                "--output-predictions-csv",
                str(eval_csv),
            ]
        )
        == 0
    )
    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    assert payload["records"] == {"total": 8, "kept": 8}
    assert "metrics" in payload


def test_metrics_counts_false_positive_and_false_negative():
    result = metrics(
        labels=__import__("numpy").asarray([0, 0, 1, 1]),
        predictions=__import__("numpy").asarray([0, 1, 0, 1]),
    )
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["errors"] == 2
