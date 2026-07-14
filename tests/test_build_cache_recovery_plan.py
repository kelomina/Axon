import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_cache_recovery_plan import build_recovery_plan, write_markdown  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_missing_csv(path: Path) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_path", "label", "split"])
        writer.writeheader()
        writer.writerow({"source_path": r"E:\data\bad\a.exe", "label": "1", "split": "test"})
        writer.writerow({"source_path": r"E:\data\bad\b.dll", "label": "1", "split": "test"})
    return path


def test_build_cache_recovery_plan_summarizes_missing_csv_and_guardrails():
    with _case_dir("cache_recovery_plan") as tmp_path:
        missing_csv = _write_missing_csv(tmp_path / "hard_fn_missing.csv")
        audit_path = tmp_path / "audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "schema": "axon_cache_coverage_audit_v1",
                    "checks": [
                        {
                            "name": "official_test_current_cache_subset",
                            "total": 100,
                            "covered": 80,
                            "missing": 20,
                            "coverage_ratio": 0.8,
                            "missing_examples": [r"E:\cache\a.npz"],
                            "missing_output": str(missing_csv),
                            "blocked_recommendations": ["probability_calibration"],
                        },
                        {
                            "name": "hard_fn_holdout_current_cache_subset",
                            "total": 10,
                            "covered": 8,
                            "missing": 2,
                            "coverage_ratio": 0.8,
                            "missing_output": str(missing_csv),
                            "blocked_recommendations": ["probability_calibration", "ga_feature_mask"],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        plan = build_recovery_plan(audit_path)
        markdown_path = tmp_path / "plan.md"
        write_markdown(plan, markdown_path)
        markdown_text = markdown_path.read_text(encoding="utf-8")

    by_name = {row["name"]: row for row in plan["targets"]}
    assert plan["blocked_recommendations"] == ["ga_feature_mask", "probability_calibration"]
    assert "Use the official-test missing-cache CSV" in by_name["official_test_current_cache_subset"]["recommended_recovery_action"]
    assert by_name["hard_fn_holdout_current_cache_subset"]["missing_csv_summary"]["rows"] == 2
    assert by_name["hard_fn_holdout_current_cache_subset"]["missing_csv_summary"]["suffix_counts"] == {
        ".dll": 1,
        ".exe": 1,
    }
    assert "Do not delete or clear data/.cache." in plan["guardrails"]
    assert "Recovery Targets" in markdown_text
