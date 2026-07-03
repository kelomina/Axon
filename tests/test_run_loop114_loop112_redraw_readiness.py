from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_loop114_loop112_redraw_readiness import run_loop112_redraw_readiness  # noqa: E402


LOOP87_FIELDS = [
    "review_batch_rank",
    "source_path",
    "source_sha256",
    "sample_index",
    "split",
    "label",
    "loop57_error_type",
    "manual_label_verdict",
    "manual_verdict_note",
    "recommended_action",
    "loop87_status",
    "loop87_reason",
    "loop87_issue_flags",
    "loop87_plan_action",
    "loop87_replacement_required",
    "loop87_replacement_label",
    "loop87_training_policy_allowed",
]


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _sha(seed: int) -> str:
    return f"{seed:064x}"[-64:]


def _write_exact_20w_split(path: Path, special_rows: list[dict] | None = None) -> Path:
    special_rows = special_rows or []
    required = {
        ("train", "0"): 10000,
        ("train", "1"): 10000,
        ("val", "0"): 10000,
        ("val", "1"): 10000,
        ("test", "0"): 80000,
        ("test", "1"): 80000,
    }
    for row in special_rows:
        key = (row["split"], row["label"])
        required[key] -= 1

    path.parent.mkdir(parents=True, exist_ok=True)
    used_sample_indices = {str(row["sample_index"]) for row in special_rows}
    fieldnames = ["source_path", "source_sha256", "sample_index", "split", "label"]
    next_index = 1_000_000
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in special_rows:
            writer.writerow(row)
        for (split, label), count in required.items():
            for _ in range(count):
                while str(next_index) in used_sample_indices:
                    next_index += 1
                writer.writerow(
                    {
                        "source_path": f"data/{split}/{label}/{next_index}.exe",
                        "source_sha256": _sha(next_index),
                        "sample_index": str(next_index),
                        "split": split,
                        "label": label,
                    }
                )
                next_index += 1
    return path


def _loop87_row(**overrides: str) -> dict:
    row = {
        "review_batch_rank": "1",
        "source_path": "data/test/1/777.exe",
        "source_sha256": _sha(777),
        "sample_index": "777",
        "split": "test",
        "label": "1",
        "loop57_error_type": "FN",
        "manual_label_verdict": "",
        "manual_verdict_note": "",
        "recommended_action": "",
        "loop87_status": "no_decision",
        "loop87_reason": "blank_manual_fields",
        "loop87_issue_flags": "",
        "loop87_plan_action": "no_action",
        "loop87_replacement_required": "false",
        "loop87_replacement_label": "",
        "loop87_training_policy_allowed": "false",
    }
    row.update(overrides)
    return row


def _loop87_summary(*, rows: int, replacement_required: int = 0, ready: bool = True, manual_quality: dict | None = None) -> dict:
    return {
        "schema": "axon_loop87_review_evidence_verdict_import_v1",
        "decision": "ready_for_redraw_plan_review_only" if replacement_required else "ready_noop_no_actionable_verdicts",
        "import_ready": ready,
        "rows": rows,
        "expected_rows": rows,
        "invalid_rows": 0 if ready else 1,
        "training_policy_rows": 0,
        "blocking_issues": [] if ready else ["invalid_verdict_rows"],
        "duplicate_sample_index_rows": 0,
        "actionable_rows": replacement_required,
        "replacement_required_rows": replacement_required,
        "replacement_counts_by_original_label": {"1": replacement_required} if replacement_required else {},
        "manual_quality": manual_quality
        if manual_quality is not None
        else {
            "blank_verdict_rows": rows - replacement_required,
            "actionable_verdict_missing_note_rows": 0,
            "evidence_note_missing_content_or_external_rows": 0,
            "evidence_note_identity_or_score_only_rows": 0,
        },
    }


def _pipeline_case(tmp_path: Path, loop87_rows: list[dict], *, loop87_summary: dict | None = None, loop112_blockers=None) -> dict:
    out_dir = tmp_path / "pipeline"
    loop87_validated_csv = _write_csv(out_dir / "loop87_validated.csv", loop87_rows, LOOP87_FIELDS)
    loop87_json = _write_json(
        out_dir / "loop87.json",
        loop87_summary if loop87_summary is not None else _loop87_summary(
            rows=len(loop87_rows),
            replacement_required=sum(row["loop87_replacement_required"] == "true" for row in loop87_rows),
        ),
    )
    loop110_json = _write_json(
        out_dir / "loop110.json",
        {
            "schema": "axon_loop110_focus_verdict_pipeline_v1",
            "decision": "ready_for_redraw_preflight_review_only"
            if any(row["loop87_replacement_required"] == "true" for row in loop87_rows)
            else "ready_noop_no_actionable_verdicts",
            "blockers": [],
            "counts": {
                "loop87_rows": len(loop87_rows),
                "loop87_actionable_rows": sum(row["loop87_replacement_required"] == "true" for row in loop87_rows),
                "loop87_replacement_required_rows": sum(row["loop87_replacement_required"] == "true" for row in loop87_rows),
                "loop87_training_policy_rows": 0,
            },
            "outputs": {
                "loop87_validated_csv": str(loop87_validated_csv),
                "loop87_json": str(loop87_json),
            },
        },
    )
    blockers = loop112_blockers or []
    loop112_json = _write_json(
        tmp_path / "loop112.json",
        {
            "schema": "axon_loop112_external_focus_verdict_pipeline_v1",
            "decision": "blocked_before_redraw_preflight" if blockers else "ready_noop_no_actionable_verdicts",
            "blockers": blockers,
            "stages": {
                "loop110_focus_verdict_pipeline": {
                    "ran": not blockers,
                    "passed": not blockers,
                    "decision": None if blockers else "ready_noop_no_actionable_verdicts",
                    "blockers": [],
                }
            },
            "counts": {
                "loop87_rows": len(loop87_rows),
                "loop87_actionable_rows": sum(row["loop87_replacement_required"] == "true" for row in loop87_rows),
                "loop87_replacement_required_rows": sum(row["loop87_replacement_required"] == "true" for row in loop87_rows),
                "loop87_training_policy_rows": 0,
            },
            "outputs": {"loop110_json": str(loop110_json)},
        },
    )
    return {
        "loop112_json": loop112_json,
        "loop110_json": loop110_json,
        "loop87_json": loop87_json,
        "loop87_validated_csv": loop87_validated_csv,
    }


def _run(tmp_path: Path, loop87_rows: list[dict], *, loop87_summary: dict | None = None, loop112_blockers=None) -> dict:
    split_csv = _write_exact_20w_split(
        tmp_path / "split.csv",
        [
            {
                "source_path": "data/test/1/777.exe",
                "source_sha256": _sha(777),
                "sample_index": "777",
                "split": "test",
                "label": "1",
            }
        ],
    )
    case = _pipeline_case(
        tmp_path,
        loop87_rows,
        loop87_summary=loop87_summary,
        loop112_blockers=loop112_blockers,
    )
    return run_loop112_redraw_readiness(
        loop112_summary_json=case["loop112_json"],
        split_csv=split_csv,
        output_dir=tmp_path / "loop114",
        output_json=tmp_path / "loop114_summary.json",
        output_md=tmp_path / "loop114.md",
        manifest_json=tmp_path / "manifest.json",
        data_dir=tmp_path / "data",
    )


def test_loop114_noop_waits_for_external_verdicts_without_training_authorization(tmp_path: Path):
    summary = _run(tmp_path, [_loop87_row()])

    assert summary["decision"] == "await_external_verdicts"
    assert summary["counts"]["replacement_required"] == 0
    assert summary["ready_for"]["test10k"] is False
    assert summary["ready_for"]["full_test"] is False
    assert summary["decisions"]["training_allowed"] is False
    assert Path(summary["outputs"]["adjustment_plan_csv"]).exists()


def test_loop114_feature_broken_becomes_fresh_same_label_replacement_request(tmp_path: Path):
    summary = _run(
        tmp_path,
        [
            _loop87_row(
                manual_label_verdict="feature_broken",
                manual_verdict_note="PE parse evidence and npz feature mismatch confirm broken feature extraction",
                recommended_action="replace_with_fresh_same_label_candidate",
                loop87_status="exclude_replace",
                loop87_reason="manual_bad_row_replacement_required",
                loop87_plan_action="quarantine_and_fresh_redraw",
                loop87_replacement_required="true",
                loop87_replacement_label="1",
            )
        ],
    )
    plan_rows = list(csv.DictReader(Path(summary["outputs"]["adjustment_plan_csv"]).open("r", encoding="utf-8-sig", newline="")))

    assert summary["decision"] == "needs_replacement_candidate_pool"
    assert summary["counts"]["replacement_required"] == 1
    assert summary["decisions"]["fresh_redraw_allowed"] is True
    assert summary["decisions"]["training_allowed"] is False
    assert plan_rows[0]["plan_action"] == "exclude_and_replace"
    assert plan_rows[0]["replacement_required"] == "true"
    assert plan_rows[0]["replacement_label"] == "1"
    assert plan_rows[0]["planned_label"] == "1"
    assert plan_rows[0]["usable_for_training_policy"] == "false"


def test_loop114_label_wrong_is_not_direct_relabel(tmp_path: Path):
    summary = _run(
        tmp_path,
        [
            _loop87_row(
                manual_label_verdict="label_wrong",
                manual_verdict_note="External VT and PE content evidence show the sample needs quarantine review",
                recommended_action="replace_with_fresh_same_label_candidate",
                loop87_status="label_wrong_replace",
                loop87_reason="manual_bad_row_replacement_required",
                loop87_plan_action="quarantine_and_fresh_redraw",
                loop87_replacement_required="true",
                loop87_replacement_label="1",
            )
        ],
    )
    plan_rows = list(csv.DictReader(Path(summary["outputs"]["adjustment_plan_csv"]).open("r", encoding="utf-8-sig", newline="")))

    assert summary["decision"] == "needs_replacement_candidate_pool"
    assert plan_rows[0]["manual_label_verdict"] == "label_wrong"
    assert plan_rows[0]["plan_action"] == "exclude_and_replace"
    assert plan_rows[0]["planned_label"] == plan_rows[0]["original_label"] == "1"
    assert "relabel" not in {row["plan_action"] for row in plan_rows}


def test_loop114_blocks_when_loop112_was_blocked_by_forbidden_external_fields(tmp_path: Path):
    summary = _run(
        tmp_path,
        [_loop87_row()],
        loop112_blockers=["loop111_import_not_ready"],
    )

    assert summary["decision"] == "blocked_upstream_loop112"
    assert "loop112:loop111_import_not_ready" in summary["blockers"]
    assert summary["ready_for"]["fresh_redraw"] is False
    assert summary["decisions"]["test10k_allowed"] is False


def test_loop114_blocks_identity_or_model_score_only_loop87_evidence(tmp_path: Path):
    summary = _run(
        tmp_path,
        [
            _loop87_row(
                manual_label_verdict="label_correct",
                manual_verdict_note="filename and loop57 probability prove it",
                recommended_action="model_blindspot",
                loop87_status="invalid",
                loop87_issue_flags="manual_verdict_note_identity_or_score_only",
            )
        ],
        loop87_summary=_loop87_summary(
            rows=1,
            ready=False,
            manual_quality={
                "blank_verdict_rows": 0,
                "actionable_verdict_missing_note_rows": 0,
                "evidence_note_missing_content_or_external_rows": 1,
                "evidence_note_identity_or_score_only_rows": 1,
            },
        ),
    )

    assert summary["decision"] == "blocked_upstream_loop87"
    assert "loop87:identity_or_model_score_only_evidence" in summary["blockers"]
    assert summary["decisions"]["training_allowed"] is False
