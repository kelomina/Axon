import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_strict_metadata_redraw_status import build_status  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _payloads(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "enrichment_json": _write(tmp_path / "enrichment.json", {"rows": 200000, "row_issue_count": 8, "shape_failures": []}),
        "metadata_plan_json": _write(tmp_path / "metadata_plan.json", {"plan_ready": True, "plan_rows": 8}),
        "metadata_corrected_json": _write(
            tmp_path / "metadata_corrected.json",
            {"excluded_rows": 8, "replacement_summary": {"selected_replacements": 8}},
        ),
        "metadata_replacement_audit_json": _write(
            tmp_path / "metadata_replacement.json",
            {"replacement_integrity_ok": True, "label_balance_enforced": True},
        ),
        "first_cache_recovery_json": _write(
            tmp_path / "first_recovery.json",
            {"status_counts": {"extracted": 7, "feature_extract_failed": 1}},
        ),
        "cache_failure_plan_json": _write(tmp_path / "cache_failure_plan.json", {"plan_ready": True, "plan_rows": 1}),
        "cache_failure_corrected_json": _write(
            tmp_path / "cache_failure_corrected.json",
            {"excluded_rows": 1, "replacement_summary": {"selected_replacements": 1}},
        ),
        "cache_failure_replacement_audit_json": _write(
            tmp_path / "cache_failure_replacement.json",
            {"replacement_integrity_ok": True, "label_balance_enforced": True},
        ),
        "final_cache_ready_json": _write(
            tmp_path / "final_cache.json",
            {
                "cache_ready": True,
                "cache_metadata_validation_enabled": True,
                "total_rows": 200000,
                "covered_rows": 200000,
                "missing_rows": 0,
                "metadata_failure_rows": 0,
                "label_balance_enforced": True,
                "shape_failures": [],
                "manifest_match_counts": {"source_sha256": 200000},
                "split_summary": {"rows": 200000},
            },
        ),
        "final_split_metadata_json": _write(
            tmp_path / "final_meta.json",
            {
                "audit_ready": True,
                "validate_npz": True,
                "expect_20w": True,
                "rows": 200000,
                "row_issue_count": 0,
                "metadata_issue_counts": {},
                "shape_failures": [],
                "match_counts": {"source_sha256": 200000},
            },
        ),
    }
    return paths


def test_status_allows_only_val_first_when_all_final_audits_pass():
    with _case_dir("metadata_redraw_status_ready") as tmp_path:
        paths = _payloads(tmp_path)

        payload = build_status(**paths)

    assert payload["decision"] == "ready_for_val_first_reverification"
    assert payload["ready_for"] == {"train_val_only": True, "test10k": False, "full_test": False}
    assert payload["counts"]["final_cache_covered_rows"] == 200000
    assert "no identity field is model evidence" in payload["identity_feature_policy"]


def test_status_blocks_when_final_cache_is_short():
    with _case_dir("metadata_redraw_status_blocked") as tmp_path:
        paths = _payloads(tmp_path)
        _write(
            paths["final_cache_ready_json"],
            {
                "cache_ready": False,
                "cache_metadata_validation_enabled": True,
                "total_rows": 200000,
                "covered_rows": 199999,
                "missing_rows": 1,
                "metadata_failure_rows": 0,
                "label_balance_enforced": True,
                "shape_failures": [],
            },
        )

        payload = build_status(**paths)

    assert payload["decision"] == "blocked_strict_metadata_redraw"
    assert "final_cache_not_ready" in payload["blockers"]
    assert "final_cache_coverage_not_200000" in payload["blockers"]
    assert payload["ready_for"]["train_val_only"] is False
