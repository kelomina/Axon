import pytest

from scripts.benchmark_onnx_dll import binary_metrics, percentile, select_balanced_rows


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_select_balanced_rows_interleaves_labels() -> None:
    rows = [
        {"split": "train", "label": "0", "id": "b0"},
        {"split": "train", "label": "0", "id": "b1"},
        {"split": "train", "label": "1", "id": "m0"},
        {"split": "train", "label": "1", "id": "m1"},
    ]
    assert [row["id"] for row in select_balanced_rows(rows, "train", 4)] == ["b0", "m0", "b1", "m1"]


def test_binary_metrics_reports_confusion_matrix() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0, 1, 1, 0])
    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["f1"] == 0.5
