import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_loop80_calibrator_fulltest_summary import build_summary, render_markdown  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _metrics(*, f1: float, fp: int, fn: int, errors: int) -> dict:
    return {
        "samples": 160000,
        "threshold": 0.5,
        "accuracy": 0.9,
        "precision": 0.9,
        "recall": 0.9,
        "f1": f1,
        "auc": 0.99,
        "false_positive": fp,
        "false_negative": fn,
        "errors": errors,
    }


def test_loop80_summary_blocks_when_calibrator_does_not_meet_target_or_loop57():
    with _case_dir("loop80_summary_block") as tmp_path:
        fulltest_eval = _write_json(
            tmp_path / "fulltest.json",
            {
                "rows": {"total": 160000, "kept": 160000, "skipped_missing_cache": 0},
                "calibrator": {
                    "model": "cal.pkl",
                    "features": "probability+stat_features+pe_features only",
                    "threshold": 0.44,
                },
                "baseline": _metrics(f1=0.92, fp=4000, fn=6000, errors=10000),
                "calibrator_metrics": _metrics(f1=0.968, fp=2900, fn=2100, errors=5000),
            },
        )
        loop57_eval = _write_json(
            tmp_path / "loop57.json",
            {"metrics": _metrics(f1=0.988, fp=1200, fn=700, errors=1900)},
        )
        baseline_eval = _write_json(
            tmp_path / "baseline.json",
            {"metrics": _metrics(f1=0.92, fp=4000, fn=6000, errors=10000)},
        )

        report = build_summary(
            fulltest_eval=fulltest_eval,
            loop57_eval=loop57_eval,
            baseline_eval=baseline_eval,
        )
        markdown = render_markdown(report)

    assert report["decision"] == "not_final_candidate"
    assert report["rows"]["skipped_missing_cache"] == 0
    assert "calibrator full-test F1 is below target" in report["blockers"]
    assert "calibrator does not beat current Loop57 full-test best" in report["blockers"]
    assert report["deltas"]["calibrator_vs_8192_baseline"]["errors"] == -5000
    assert "Loop80 Calibrator Full-Test Summary" in markdown


def test_loop80_summary_can_mark_target_met_when_every_gate_passes():
    with _case_dir("loop80_summary_pass") as tmp_path:
        fulltest_eval = _write_json(
            tmp_path / "fulltest.json",
            {
                "rows": {"total": 160000, "kept": 160000, "skipped_missing_cache": 0},
                "calibrator": {
                    "model": "cal.pkl",
                    "features": "probability+stat_features+pe_features only",
                    "threshold": 0.44,
                },
                "baseline": _metrics(f1=0.92, fp=4000, fn=6000, errors=10000),
                "calibrator_metrics": _metrics(f1=0.9992, fp=100, fn=0, errors=100),
            },
        )
        loop57_eval = _write_json(
            tmp_path / "loop57.json",
            {"metrics": _metrics(f1=0.988, fp=1200, fn=700, errors=1900)},
        )
        baseline_eval = _write_json(
            tmp_path / "baseline.json",
            {"metrics": _metrics(f1=0.92, fp=4000, fn=6000, errors=10000)},
        )

        report = build_summary(
            fulltest_eval=fulltest_eval,
            loop57_eval=loop57_eval,
            baseline_eval=baseline_eval,
        )

    assert report["decision"] == "target_met"
    assert report["blockers"] == []
    assert report["target_gap"]["errors_to_remove_best_case"] == 0
