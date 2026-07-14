import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_feature_mask_holdout_summary import build_summary  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_summary(path: Path, *, total: int, fp: int, fn: int) -> Path:
    path.write_text(
        json.dumps(
            {
                "predictions": str(path.with_suffix(".csv")),
                "threshold": 0.5,
                "total_predictions": total,
                "false_positive_count": fp,
                "false_negative_count": fn,
                "error_count": fp + fn,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_feature_mask_holdout_summary_computes_deltas():
    with _case_dir("feature_mask_holdout_summary") as tmp_path:
        hard_fn_full = _write_summary(tmp_path / "hard_fn_full.json", total=39, fp=0, fn=12)
        hard_fn_mask = _write_summary(tmp_path / "hard_fn_mask.json", total=39, fp=0, fn=11)
        hard_error_full = _write_summary(tmp_path / "hard_error_full.json", total=155, fp=83, fn=60)
        hard_error_mask = _write_summary(tmp_path / "hard_error_mask.json", total=155, fp=83, fn=60)

        summary = build_summary(
            feature_mask=Path("mask.json"),
            hard_fn_full=hard_fn_full,
            hard_fn_mask=hard_fn_mask,
            hard_error_full=hard_error_full,
            hard_error_mask=hard_error_mask,
        )

    assert summary["schema"] == "axon_feature_mask_holdout_summary_v1"
    assert summary["sections"]["hard_fn_current_subset"]["delta_mask_minus_full"] == {
        "false_positive": 0,
        "false_negative": -1,
        "errors": -1,
    }
    assert summary["sections"]["hard_error_current_subset"]["delta_mask_minus_full"] == {
        "false_positive": 0,
        "false_negative": 0,
        "errors": 0,
    }


def test_build_feature_mask_holdout_summary_rejects_mismatched_counts():
    with _case_dir("feature_mask_holdout_mismatch") as tmp_path:
        hard_fn_full = _write_summary(tmp_path / "hard_fn_full.json", total=39, fp=0, fn=12)
        hard_fn_mask = _write_summary(tmp_path / "hard_fn_mask.json", total=38, fp=0, fn=11)
        hard_error_full = _write_summary(tmp_path / "hard_error_full.json", total=155, fp=83, fn=60)
        hard_error_mask = _write_summary(tmp_path / "hard_error_mask.json", total=155, fp=83, fn=60)

        try:
            build_summary(
                feature_mask=Path("mask.json"),
                hard_fn_full=hard_fn_full,
                hard_fn_mask=hard_fn_mask,
                hard_error_full=hard_error_full,
                hard_error_mask=hard_error_mask,
            )
        except ValueError as exc:
            message = str(exc)
        else:
            message = ""

    assert "sample counts differ" in message
