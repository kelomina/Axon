from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_loop70_nested_oof_meta as loop70  # noqa: E402


def test_read_oof_rows_stops_at_max_rows(tmp_path):
    path = tmp_path / "oof.csv"
    path.write_text(
        "sample_index,label,base_oof_prob_malicious,candidate_oof_prob_malicious,allow_oof_prob\n"
        "0,0,0.1,0.2,0.3\n"
        "1,1,0.4,0.5,0.6\n",
        encoding="utf-8",
    )

    rows = loop70.read_oof_rows(path, max_rows=1)

    assert len(rows) == 1
    assert rows[0]["sample_index"] == "0"


def test_read_oof_rows_requires_explicit_bound(tmp_path):
    path = tmp_path / "oof.csv"
    path.write_text(
        "sample_index,label,base_oof_prob_malicious,candidate_oof_prob_malicious,allow_oof_prob\n"
        "0,0,0.1,0.2,0.3\n",
        encoding="utf-8",
    )

    try:
        loop70.read_oof_rows(path)
    except ValueError as exc:
        assert "unbounded CSV reads" in str(exc)
    else:
        raise AssertionError("Expected read_oof_rows to reject unbounded reads")


def test_read_oof_rows_validates_expected_count(tmp_path):
    path = tmp_path / "oof.csv"
    path.write_text(
        "sample_index,label,base_oof_prob_malicious,candidate_oof_prob_malicious,allow_oof_prob\n"
        "0,0,0.1,0.2,0.3\n",
        encoding="utf-8",
    )

    try:
        loop70.read_oof_rows(path, expected_rows=2)
    except ValueError as exc:
        assert "OOF row count mismatch" in str(exc)
    else:
        raise AssertionError("Expected OOF row mismatch")


def test_build_meta_score_features_uses_score_only_columns():
    features, names = loop70.build_meta_score_features(
        base_scores=np.asarray([0.1, 0.9], dtype=np.float32),
        candidate_scores=np.asarray([0.2, 0.8], dtype=np.float32),
        allow_scores=np.asarray([0.3, 0.7], dtype=np.float32),
        final_scores=np.asarray([0.4, 0.6], dtype=np.float32),
        final_predictions=np.asarray([0, 1], dtype=np.int64),
        override_mask=np.asarray([False, True]),
        possible_mask=np.asarray([True, True]),
    )

    assert features.shape == (2, len(names))
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert all("path" not in name and "sha" not in name and "sample" not in name for name in names)
