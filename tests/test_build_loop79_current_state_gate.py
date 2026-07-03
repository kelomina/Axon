import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_loop79_current_state_gate import build_gate, render_markdown  # noqa: E402


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


def _write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_index",
                "split",
                "label",
                "old_source_path",
                "new_source_path",
                "new_cache_path",
                "source_sha256",
                "selection_status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _replacement_rows(count: int = 130) -> list[dict]:
    rows = []
    for index in range(count):
        rows.append(
            {
                "sample_index": str(index),
                "split": "train",
                "label": "0",
                "old_source_path": f"old-{index}",
                "new_source_path": f"new-{index}",
                "new_cache_path": f"cache-{index}.npz",
                "source_sha256": f"{index:064x}"[-64:],
                "selection_status": "strict_extracted",
            }
        )
    return rows


def _write_good_artifacts(tmp_path: Path) -> dict[str, Path]:
    rows = _replacement_rows()
    replacement_report = _write_json(
        tmp_path / "replacement.json",
        {
            "replacement_rows": 130,
            "replacement_demand_by_split_label": {"train:0": 130},
            "replacement_selection_status_counts": {"strict_extracted": 130},
            "manifest_added": 130,
            "manifest_samples_after": 200000,
            "split_rows_after": 200000,
            "split_counts_after": {
                "train:0": 10000,
                "train:1": 10000,
                "val:0": 10000,
                "val:1": 10000,
                "test:0": 80000,
                "test:1": 80000,
            },
            "cache_storage_format": "uncompressed",
            "strict_pe_replacements_only": True,
        },
    )
    replacement_csv = _write_csv(tmp_path / "replacement.csv", rows)
    replaced_coverage = _write_json(
        tmp_path / "replaced_coverage.json",
        {"total_rows": 200000, "covered_rows": 200000, "missing_rows": 0},
    )
    current_ready = _write_json(
        tmp_path / "current_ready.json",
        {
            "cache_ready": True,
            "total_rows": 200000,
            "covered_rows": 200000,
            "missing_rows": 0,
            "label_balance_enforced": True,
            "shape_failures": [],
        },
    )
    current_coverage = _write_json(
        tmp_path / "current_coverage.json",
        {"covered_rows": 200000, "missing_rows": 0},
    )
    sample_integrity = _write_json(
        tmp_path / "sample_integrity.json",
        {
            "audit_ready": True,
            "sampled_rows": 2000,
            "failed_rows": 0,
            "shape_failures": [],
            "sampled_split_counts": {"train": 200, "val": 200, "test": 1600},
            "sampled_label_counts": {"0": 1000, "1": 1000},
        },
    )
    ab_report = _write_json(
        tmp_path / "ab.json",
        {
            "probability_calibration": {
                "all_strict_rows_kept": True,
                "no_test_used_for_training": True,
                "strict_evaluations": [
                    {"name": "official", "rows": {"total": 10, "kept": 10, "skipped_missing_cache": 0}}
                ],
            },
            "ga_feature_mask": {
                "feature_mask_20k": {
                    "source": "mask20k.json",
                    "mask_lowest_errors": {"delta_errors": -3},
                },
                "hard_holdouts": {
                    "sections": {
                        "hard_fn": {
                            "full": {"total_predictions": 5},
                            "mask": {"total_predictions": 5},
                        }
                    }
                },
                "high_value_benign": {
                    "delta_mask_minus_baseline": {"false_positive": 2}
                },
            },
            "conclusion": {
                "probability_calibration": "strictly_reverified_useful",
                "ga_feature_mask": "strictly_reverified_high_security_candidate_not_default",
            },
        },
    )
    train_val = _write_json(
        tmp_path / "train_val.json",
        {
            "protocol": "train split trains calibrator; val split selects threshold; no path/group metadata; no test used",
            "train_rows": {"kept": 20000},
            "val_rows": {"kept": 20000},
            "selected": {"delta_val_f1_vs_baseline": 0.01},
        },
    )
    test10k = _write_json(
        tmp_path / "test10k.json",
        {
            "rows": {"kept": 10000, "skipped_missing_cache": 0},
            "delta_f1_vs_baseline": 0.02,
            "delta_errors_vs_baseline": -100,
        },
    )
    return {
        "replacement_report": replacement_report,
        "replacement_csv": replacement_csv,
        "replaced_coverage": replaced_coverage,
        "current_cache_ready": current_ready,
        "current_coverage": current_coverage,
        "sample_integrity": sample_integrity,
        "ab_report": ab_report,
        "replaced_calibrator_train_val": train_val,
        "replaced_calibrator_test10k": test10k,
    }


def test_loop79_gate_passes_when_all_current_state_evidence_is_strict():
    with _case_dir("loop79_pass") as tmp_path:
        paths = _write_good_artifacts(tmp_path)
        report = build_gate(**paths)
        markdown = render_markdown(report)

    assert report["decision"] == "pass"
    assert report["sections"]["fixed_v2_replacement_130"]["replacement_rows"] == 130
    assert report["sections"]["current_split_cache"]["covered_rows"] == 200000
    assert report["sections"]["probability_calibration"]["current_test10k_delta_errors"] == -100
    assert report["sections"]["ga_feature_mask"]["operational_verdict"] == "not_default_because_high_value_benign_fp_increases"
    assert "Loop79 Current State Gate" in markdown


def test_loop79_gate_blocks_bad_self_replacement_and_missing_current_cache():
    with _case_dir("loop79_block") as tmp_path:
        paths = _write_good_artifacts(tmp_path)
        rows = _replacement_rows()
        rows[0]["new_source_path"] = rows[0]["old_source_path"]
        _write_csv(paths["replacement_csv"], rows)
        current_ready = json.loads(paths["current_cache_ready"].read_text(encoding="utf-8"))
        current_ready["missing_rows"] = 1
        current_ready["covered_rows"] = 199999
        current_ready["cache_ready"] = False
        _write_json(paths["current_cache_ready"], current_ready)

        report = build_gate(**paths)

    assert report["decision"] == "block"
    assert "at least one replacement reuses its old source path" in report["blockers"]["fixed_v2_replacement_130"]
    assert "current corrected split cache_ready is false" in report["blockers"]["current_split_cache"]
