import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_loop85_noise_strategy_gate import build_summary  # noqa: E402


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


def _fixture_files(tmp_path: Path, *, health_issues: int = 0, cross_label_groups: int = 0):
    loop57 = _write_json(
        tmp_path / "loop57.json",
        {
            "records": {"kept": 160000},
            "metrics": {"f1": 0.98, "errors": 10, "false_positive": 6, "false_negative": 4},
        },
    )
    loop63 = _write_json(
        tmp_path / "loop63.json",
        {
            "loop57_error_rows": 10,
            "review_lane_counts": {"A": 3},
            "error_type_counts": {"FP": 6, "FN": 4},
            "manual_label_verdict_blank_count": 10,
            "recommended_action_blank_count": 10,
            "outputs": {"queue_csv": "queue.csv"},
        },
    )
    health = _write_json(
        tmp_path / "health.json",
        {
            "rows": 3,
            "error_type_counts": {"FP": 2, "FN": 1},
            "objective_issue_row_count": health_issues,
            "issue_counts": {},
        },
    )
    duplicate = _write_json(
        tmp_path / "duplicate.json",
        {
            "duplicate_groups": 1,
            "duplicate_detail_rows": 2,
            "cross_label_groups": cross_label_groups,
            "cross_split_groups": 0,
            "focus_duplicate_groups": 1,
            "focus_duplicate_detail_rows": 2,
        },
    )
    loop65 = _write_json(
        tmp_path / "loop65.json",
        {
            "selected_rows": 2,
            "category_counts": {"a": 1, "b": 1},
            "error_type_counts": {"FP": 1, "FN": 1},
            "manual_fields_blank_output": True,
            "outputs": {"review_csv": "review.csv"},
        },
    )
    loop82 = _write_json(
        tmp_path / "loop82.json",
        {
            "ready_for_val_fusion_probe": True,
            "overlap_counts": {"calibrator_only_correct": 2, "loop57_only_correct": 5},
        },
    )
    loop83 = _write_json(
        tmp_path / "loop83.json",
        {
            "rule_scan": {"improves_loop57": False, "best": {"errors": 12}},
        },
    )
    loop84 = _write_json(
        tmp_path / "loop84.json",
        {
            "interpretation": {"is_promising": False},
            "selector_cv": {"best": {"auc": 0.6}},
        },
    )
    return loop57, loop63, health, duplicate, loop65, loop82, loop83, loop84


def test_loop85_strategy_gate_prefers_review_when_existing_evidence_has_no_objective_issue():
    with _case_dir("loop85_gate") as tmp_path:
        files = _fixture_files(tmp_path)
        report = build_summary(
            loop57_full_eval=files[0],
            loop63_queue_summary=files[1],
            loop63_health_summary=files[2],
            loop64_duplicate_summary=files[3],
            loop65_review_summary=files[4],
            loop82_complementarity=files[5],
            loop83_rescue_profile=files[6],
            loop84_content_rescue=files[7],
        )

    assert report["blockers"] == []
    assert report["decisions"]["automatic_replacement_allowed"] is False
    assert report["decisions"]["test10k_allowed_for_current_calibrator_fusion"] is False
    assert "locked-manifest original-label pool" in report["decisions"]["replacement_rule"]
    assert "filename/path/directory similarity" in report["decisions"]["replacement_rule"]
    assert report["fusion_evidence"]["stop_current_calibrator_fusion"] is True
    assert report["next_actions"][0]["action"] == "manual_or_external_evidence_review"
    assert "bootstrap method" in report["identity_feature_policy"]["label_source_boundary"]
    assert "locked manifest label pool" in report["identity_feature_policy"]["redraw_boundary"]


def test_loop85_strategy_gate_blocks_when_health_or_cross_label_duplicate_issue_exists():
    with _case_dir("loop85_gate_block") as tmp_path:
        files = _fixture_files(tmp_path, health_issues=1, cross_label_groups=1)
        report = build_summary(
            loop57_full_eval=files[0],
            loop63_queue_summary=files[1],
            loop63_health_summary=files[2],
            loop64_duplicate_summary=files[3],
            loop65_review_summary=files[4],
            loop82_complementarity=files[5],
            loop83_rescue_profile=files[6],
            loop84_content_rescue=files[7],
        )

    assert "A-lane health audit found objective cache/source issues" in report["blockers"]
    assert "Manifest duplicate audit found cross-label duplicate groups" in report["blockers"]
