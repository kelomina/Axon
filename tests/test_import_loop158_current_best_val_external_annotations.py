from __future__ import annotations

import csv
from pathlib import Path

import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from import_loop158_current_best_val_external_annotations import (  # noqa: E402
    ANNOTATION_FIELDS,
    import_loop158_external_annotations,
)


CONTEXT_FIELDS = ["review_focus_id", "current_label", "error_type", "content_tags"]
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


def _write_exact_20w_split(path: Path, special_rows: list[dict]) -> Path:
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
    next_index = 1_000_000
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in special_rows:
            writer.writerow(row)
        for (split, label), count in required.items():
            for _ in range(count):
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


def _base_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    context_csv = _write_csv(
        tmp_path / "loop157_context.csv",
        [
            {
                "review_focus_id": "loop156_val_error_000001",
                "current_label": "0",
                "error_type": "fp",
                "content_tags": "resource_rich|version_resource_present",
            },
            {
                "review_focus_id": "loop156_val_error_000002",
                "current_label": "1",
                "error_type": "fn",
                "content_tags": "script_exec_present|overlay_high_entropy",
            },
        ],
        CONTEXT_FIELDS,
    )
    special_rows = [
        {
            "source_path": "data/val/0/101.exe",
            "source_sha256": _sha(101),
            "sample_index": "101",
            "split": "val",
            "label": "0",
        },
        {
            "source_path": "data/val/1/202.exe",
            "source_sha256": _sha(202),
            "sample_index": "202",
            "split": "val",
            "label": "1",
        },
    ]
    private_map_csv = _write_csv(
        tmp_path / "private.csv",
        [
            {
                "review_focus_id": "loop156_val_error_000001",
                "source_path": special_rows[0]["source_path"],
                "source_sha256": special_rows[0]["source_sha256"],
                "cache_path": "data/.cache/101.npz",
                "label": "0",
            },
            {
                "review_focus_id": "loop156_val_error_000002",
                "source_path": special_rows[1]["source_path"],
                "source_sha256": special_rows[1]["source_sha256"],
                "cache_path": "data/.cache/202.npz",
                "label": "1",
            },
        ],
        PRIVATE_FIELDS,
    )
    split_csv = _write_exact_20w_split(tmp_path / "split.csv", special_rows)
    return context_csv, private_map_csv, split_csv


def _run(tmp_path: Path, annotation_rows: list[dict], *, fieldnames: list[str] | None = None):
    context_csv, private_map_csv, split_csv = _base_files(tmp_path)
    returned_csv = _write_csv(tmp_path / "returned.csv", annotation_rows, fieldnames or ANNOTATION_FIELDS)
    return import_loop158_external_annotations(
        returned_annotations_csv=returned_csv,
        context_csv=context_csv,
        private_map_csv=private_map_csv,
        split_csv=split_csv,
        output_dir=tmp_path / "loop158",
        output_json=tmp_path / "loop158_summary.json",
        output_md=tmp_path / "loop158_summary.md",
        expected_rows=2,
        manifest_json=tmp_path / "manifest.json",
        data_dir=tmp_path / "data",
    )


def test_loop158_header_only_external_return_is_safe_noop(tmp_path: Path):
    summary = _run(tmp_path, [])

    assert summary["decision"] == "ready_noop_no_external_annotations"
    assert summary["external_annotation_audit"]["rows"] == 0
    assert summary["private_join_performed"] is False
    assert summary["ready_for"]["train_val_only"] is False
    assert summary["ready_for"]["test10k"] is False
    assert summary["ready_for"]["full_test"] is False


def test_loop158_label_correct_import_runs_preflight_without_redraw(tmp_path: Path):
    summary = _run(
        tmp_path,
        [
            {
                "review_focus_id": "loop156_val_error_000001",
                "manual_label_verdict": "label_correct",
                "manual_verdict_note": "PE header and external VT evidence support the locked benign label",
                "recommended_action": "model_blindspot",
            }
        ],
    )

    assert summary["decision"] == "await_external_verdicts"
    assert summary["private_join_performed"] is True
    assert summary["preflight"]["status_counts"] == {"label_correct_model_blindspot": 1}
    assert summary["preflight"]["replacement_required_rows"] == 0
    assert summary["ready_for"]["fresh_redraw"] is False
    assert summary["ready_for"]["test10k"] is False


def test_loop158_bad_row_import_only_allows_same_original_label_redraw_readiness(tmp_path: Path):
    summary = _run(
        tmp_path,
        [
            {
                "review_focus_id": "loop156_val_error_000002",
                "manual_label_verdict": "feature_broken",
                "manual_verdict_note": "PE header import evidence and external VT evidence confirm corrupt feature extraction",
                "recommended_action": "replace_with_fresh_same_label_candidate",
            }
        ],
    )

    assert summary["decision"] == "needs_replacement_candidate_pool"
    assert summary["private_join_performed"] is True
    assert summary["preflight"]["replacement_required_rows"] == 1
    assert summary["ready_for"]["fresh_redraw"] is True
    assert summary["ready_for"]["train_val_only"] is False
    assert summary["ready_for"]["test10k"] is False
    plan_rows = list(csv.DictReader(Path(summary["outputs"]["redraw_adjustment_plan_csv"]).open(encoding="utf-8-sig")))
    assert plan_rows[0]["plan_action"] == "exclude_and_replace"
    assert plan_rows[0]["replacement_label"] == "1"
    assert plan_rows[0]["usable_for_training_policy"] == "false"


def test_loop158_blocks_extra_identity_or_model_columns(tmp_path: Path):
    row = {
        "review_focus_id": "loop156_val_error_000001",
        "manual_label_verdict": "label_correct",
        "manual_verdict_note": "PE header and external VT evidence support the locked benign label",
        "recommended_action": "model_blindspot",
        "source_sha256": "a" * 64,
        "prob_malicious": "0.99",
    }
    summary = _run(tmp_path, [row], fieldnames=[*ANNOTATION_FIELDS, "source_sha256", "prob_malicious"])

    assert summary["decision"] == "blocked_external_annotation_import"
    assert "external_annotations_must_contain_exact_four_columns" in summary["blockers"]
    assert "external_annotations_contain_identity_or_model_columns" in summary["blockers"]
    assert summary["private_join_performed"] is False


def test_loop158_blocks_blank_manual_rows(tmp_path: Path):
    summary = _run(
        tmp_path,
        [
            {
                "review_focus_id": "loop156_val_error_000001",
                "manual_label_verdict": "",
                "manual_verdict_note": "",
                "recommended_action": "",
            }
        ],
    )

    assert summary["decision"] == "blocked_external_annotation_import"
    assert "external_annotations_blank_manual_rows" in summary["blockers"]
    assert summary["private_join_performed"] is False


def test_loop158_blocks_identity_terms_inside_notes_even_with_content_evidence(tmp_path: Path):
    summary = _run(
        tmp_path,
        [
            {
                "review_focus_id": "loop156_val_error_000001",
                "manual_label_verdict": "label_correct",
                "manual_verdict_note": "PE evidence supports benign, and source_sha256 also matched a known file",
                "recommended_action": "model_blindspot",
            }
        ],
    )

    assert summary["decision"] == "blocked_external_annotation_import"
    assert "external_annotation_notes_reference_identity_or_model_terms" in summary["blockers"]
    assert summary["external_annotation_audit"]["note_identity_or_model_term_rows"] == 1
