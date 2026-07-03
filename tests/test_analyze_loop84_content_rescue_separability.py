import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_loop84_content_rescue_separability import (  # noqa: E402
    REGRESSION_GROUP,
    RESCUE_GROUP,
    build_focus_rows,
    classification_metrics,
)


def test_build_focus_rows_maps_only_rescue_and_regression_groups_by_sha():
    overlap_rows = [
        {"join_key": "sha-rescue", "overlap_group": RESCUE_GROUP},
        {"join_key": "sha-regression", "overlap_group": REGRESSION_GROUP},
        {"join_key": "sha-both", "overlap_group": "both_correct"},
        {"join_key": "sha-missing", "overlap_group": RESCUE_GROUP},
    ]
    base_rows = {
        "sha-rescue": {"source_sha256": "sha-rescue", "label": "1", "cache_path": "a.npz"},
        "sha-regression": {"source_sha256": "sha-regression", "label": "0", "cache_path": "b.npz"},
        "sha-both": {"source_sha256": "sha-both", "label": "1", "cache_path": "c.npz"},
    }

    focus_rows, selector_labels, summary = build_focus_rows(
        overlap_rows=overlap_rows,
        base_predictions_by_sha=base_rows,
    )

    assert [row["_loop82_join_key"] for row in focus_rows] == ["sha-rescue", "sha-regression"]
    assert selector_labels.tolist() == [1, 0]
    assert summary["focus_rows"] == 2
    assert summary["selector_labels"] == {"1": 1, "0": 1}
    assert summary["skipped"] == {"missing_base_prediction": 1}


def test_classification_metrics_reports_confusion_counts():
    labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
    scores = np.asarray([0.9, 0.2, 0.7, 0.1], dtype=np.float32)

    metrics = classification_metrics(labels, scores, threshold=0.5)

    assert metrics["true_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["errors"] == 2
