from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_loop152_loop150_val_focus_redraw_readiness import run_loop150_val_focus_redraw_readiness  # noqa: E402


PREFLIGHT_FIELDS = [
    "review_focus_id",
    "current_label",
    "manual_label_verdict",
    "manual_verdict_note",
    "recommended_action",
    "loop126_status",
    "loop126_issue_flags",
    "loop126_plan_action",
    "loop126_replacement_required",
    "loop126_replacement_label",
    "loop126_training_policy_allowed",
]
PRIVATE_FIELDS = ["review_focus_id", "source_path", "source_sha256", "cache_path", "label"]


def _sha(seed: int) -> str:
    return f"{seed:064x}"[-64:]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


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
        required[(row["split"], row["label"])] -= 1

    fieldnames = ["source_path", "source_sha256", "sample_index", "split", "label"]
    used_sample_indices = {str(row["sample_index"]) for row in special_rows}
    next_index = 1_000_000
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _preflight_row(**overrides: str) -> dict:
    row = {
        "review_focus_id": "loop150_val_focus_000001",
        "current_label": "1",
        "manual_label_verdict": "",
        "manual_verdict_note": "",
        "recommended_action": "",
        "loop126_status": "no_decision",
        "loop126_issue_flags": "",
        "loop126_plan_action": "no_action",
        "loop126_replacement_required": "false",
        "loop126_replacement_label": "",
        "loop126_training_policy_allowed": "false",
    }
    row.update(overrides)
    return row


def _case(
    tmp_path: Path,
    preflight_rows: list[dict],
    *,
    replacement_required_rows: int = 0,
) -> tuple[Path, Path, Path, Path]:
    special = {
        "source_path": "data/val/1/777.exe",
        "source_sha256": _sha(777),
        "sample_index": "777",
        "split": "val",
        "label": "1",
    }
    split_csv = _write_exact_20w_split(tmp_path / "split.csv", [special])
    private_map_csv = _write_csv(
        tmp_path / "private.csv",
        [
            {
                "review_focus_id": "loop150_val_focus_000001",
                "source_path": special["source_path"],
                "source_sha256": special["source_sha256"],
                "cache_path": "data/.cache/777.npz",
                "label": "1",
            }
        ],
        PRIVATE_FIELDS,
    )
    preflight_csv = _write_csv(tmp_path / "preflight.csv", preflight_rows, PREFLIGHT_FIELDS)
    status_counts = {}
    for row in preflight_rows:
        status_counts[row["loop126_status"]] = status_counts.get(row["loop126_status"], 0) + 1
    preflight_json = _write_json(
        tmp_path / "preflight.json",
        {
            "schema": "axon_loop126_review_annotation_preflight_v1",
            "rows": len(preflight_rows),
            "expected_rows": len(preflight_rows),
            "annotated_rows": replacement_required_rows,
            "actionable_rows": replacement_required_rows,
            "replacement_required_rows": replacement_required_rows,
            "ready_for_private_mapping": bool(replacement_required_rows),
            "blockers": [],
            "status_counts": status_counts,
            "manual_quality": {
                "blank_verdict_rows": len(preflight_rows) - replacement_required_rows,
                "actionable_verdict_missing_note_rows": 0,
                "evidence_note_missing_content_or_external_rows": 0,
                "evidence_note_identity_or_score_only_rows": 0,
            },
        },
    )
    return preflight_csv, preflight_json, private_map_csv, split_csv


def test_loop152_noop_waits_for_external_verdicts(tmp_path: Path):
    preflight_csv, preflight_json, private_map_csv, split_csv = _case(tmp_path, [_preflight_row()])

    summary = run_loop150_val_focus_redraw_readiness(
        preflight_csv=preflight_csv,
        preflight_json=preflight_json,
        private_map_csv=private_map_csv,
        split_csv=split_csv,
        output_dir=tmp_path / "loop152",
        output_json=tmp_path / "loop152_summary.json",
        output_md=tmp_path / "loop152.md",
        manifest_json=tmp_path / "manifest.json",
        data_dir=tmp_path / "data",
    )

    assert summary["decision"] == "await_external_verdicts"
    assert summary["counts"]["replacement_required"] == 0
    assert summary["ready_for"]["test10k"] is False
    assert summary["ready_for"]["full_test"] is False
    assert summary["decisions"]["training_allowed"] is False
    assert Path(summary["outputs"]["adjustment_plan_csv"]).exists()


def test_loop152_label_wrong_becomes_fresh_same_label_replacement_request(tmp_path: Path):
    preflight_csv, preflight_json, private_map_csv, split_csv = _case(
        tmp_path,
        [
            _preflight_row(
                manual_label_verdict="label_wrong",
                manual_verdict_note="PE header and external VT evidence confirm this row is not a valid training example",
                recommended_action="replace_with_fresh_same_label_candidate",
                loop126_status="bad_row_replacement_required",
                loop126_reason="manual_bad_row_replacement_required",
                loop126_plan_action="quarantine_and_replace_fresh_same_original_label",
                loop126_replacement_required="true",
                loop126_replacement_label="1",
            )
        ],
        replacement_required_rows=1,
    )

    summary = run_loop150_val_focus_redraw_readiness(
        preflight_csv=preflight_csv,
        preflight_json=preflight_json,
        private_map_csv=private_map_csv,
        split_csv=split_csv,
        output_dir=tmp_path / "loop152",
        output_json=tmp_path / "loop152_summary.json",
        manifest_json=tmp_path / "manifest.json",
        data_dir=tmp_path / "data",
    )

    assert summary["decision"] == "needs_replacement_candidate_pool"
    assert summary["counts"]["replacement_required"] == 1
    assert summary["ready_for"]["fresh_redraw"] is True
    assert summary["ready_for"]["test10k"] is False
    plan_rows = list(csv.DictReader(Path(summary["outputs"]["adjustment_plan_csv"]).open(encoding="utf-8-sig")))
    assert plan_rows[0]["plan_action"] == "exclude_and_replace"
    assert plan_rows[0]["replacement_required"] == "true"
    assert plan_rows[0]["replacement_label"] == "1"
    assert plan_rows[0]["usable_for_training_policy"] == "false"
