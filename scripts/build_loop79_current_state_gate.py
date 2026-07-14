#!/usr/bin/env python3
"""Build a read-only gate for the current 20w fixed-v2 state.

The gate consolidates already produced evidence. It does not train, evaluate,
load model checkpoints, open NPZ arrays, mutate cache files, or scan raw data.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence


EXPECTED_TOTAL = 200000
EXPECTED_REPLACEMENTS = 130
EXPECTED_LABEL_SPLIT_COUNTS = {
    "train:0": 10000,
    "train:1": 10000,
    "val:0": 10000,
    "val:1": 10000,
    "test:0": 80000,
    "test:1": 80000,
}
FORBIDDEN_IDENTITY_EVIDENCE = [
    "filename",
    "path",
    "extension",
    "directory",
    "hash",
    "source_sha256",
    "sample_index",
    "split",
    "row_order",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _add_block(blockers: list[str], condition: bool, message: str) -> None:
    if condition:
        blockers.append(message)


def _int_field(payload: dict[str, Any], key: str, default: int = -1) -> int:
    value = payload.get(key, default)
    if value is None or value == "":
        return default
    return int(value)


def _split_label_counts_from_replacement_csv(rows: Sequence[dict[str, str]]) -> dict[str, int]:
    counts = Counter(f"{row.get('split', '')}:{row.get('label', '')}" for row in rows)
    return dict(sorted(counts.items()))


def verify_replacement_130(
    *,
    replacement_report_path: Path,
    replacement_csv_path: Path,
    replaced_coverage_path: Path,
) -> dict[str, Any]:
    report = load_json(replacement_report_path)
    replacement_rows = read_csv_rows(replacement_csv_path)
    coverage = load_json(replaced_coverage_path)
    blockers: list[str] = []

    status_counts = Counter(row.get("selection_status", "") for row in replacement_rows)
    old_paths = [row.get("old_source_path", "") for row in replacement_rows]
    new_paths = [row.get("new_source_path", "") for row in replacement_rows]
    self_replacements = sum(1 for old, new in zip(old_paths, new_paths) if old and old == new)
    demand_counts = report.get("replacement_demand_by_split_label") or {}
    csv_counts = _split_label_counts_from_replacement_csv(replacement_rows)

    _add_block(blockers, int(report.get("replacement_rows", -1)) != EXPECTED_REPLACEMENTS, "replacement report is not exactly 130 rows")
    _add_block(blockers, len(replacement_rows) != EXPECTED_REPLACEMENTS, "replacement CSV is not exactly 130 rows")
    _add_block(blockers, dict(status_counts) != {"strict_extracted": EXPECTED_REPLACEMENTS}, "not every replacement row is strict_extracted")
    _add_block(blockers, bool(self_replacements), "at least one replacement reuses its old source path")
    _add_block(blockers, csv_counts != demand_counts, "replacement CSV split/label counts do not match demand")
    _add_block(blockers, int(report.get("manifest_added", -1)) != EXPECTED_REPLACEMENTS, "manifest_added is not 130")
    _add_block(blockers, int(report.get("manifest_samples_after", -1)) != EXPECTED_TOTAL, "manifest after replacement is not 200000")
    _add_block(blockers, int(report.get("split_rows_after", -1)) != EXPECTED_TOTAL, "split after replacement is not 200000")
    _add_block(blockers, report.get("split_counts_after") != EXPECTED_LABEL_SPLIT_COUNTS, "post replacement split/label balance is wrong")
    _add_block(blockers, report.get("cache_storage_format") != "uncompressed", "replacement cache storage is not uncompressed")
    _add_block(blockers, report.get("strict_pe_replacements_only") is not True, "replacement report does not enforce strict PE replacements only")
    _add_block(blockers, int(coverage.get("total_rows", -1)) != EXPECTED_TOTAL, "replaced coverage audit is not 200000 rows")
    _add_block(blockers, int(coverage.get("covered_rows", -1)) != EXPECTED_TOTAL, "replaced coverage audit is not fully covered")
    _add_block(blockers, int(coverage.get("missing_rows", -1)) != 0, "replaced coverage audit has missing rows")

    return {
        "status": "pass" if not blockers else "block",
        "blockers": blockers,
        "replacement_report": str(replacement_report_path),
        "replacement_csv": str(replacement_csv_path),
        "replaced_coverage": str(replaced_coverage_path),
        "replacement_rows": len(replacement_rows),
        "replacement_demand_by_split_label": demand_counts,
        "replacement_csv_counts_by_split_label": csv_counts,
        "selection_status_counts": dict(sorted(status_counts.items())),
        "self_replacements": self_replacements,
        "manifest_samples_after": report.get("manifest_samples_after"),
        "split_rows_after": report.get("split_rows_after"),
        "covered_rows_after": coverage.get("covered_rows"),
        "missing_rows_after": coverage.get("missing_rows"),
    }


def verify_current_split(
    *,
    current_cache_ready_path: Path,
    current_coverage_path: Path,
    sample_integrity_path: Path,
) -> dict[str, Any]:
    ready = load_json(current_cache_ready_path)
    coverage = load_json(current_coverage_path)
    sample = load_json(sample_integrity_path)
    blockers: list[str] = []

    _add_block(blockers, ready.get("cache_ready") is not True, "current corrected split cache_ready is false")
    _add_block(blockers, int(ready.get("total_rows", -1)) != EXPECTED_TOTAL, "current corrected split is not 200000 rows")
    _add_block(blockers, int(ready.get("covered_rows", -1)) != EXPECTED_TOTAL, "current corrected split is not fully covered")
    _add_block(blockers, int(ready.get("missing_rows", -1)) != 0, "current corrected split has missing cache rows")
    _add_block(blockers, ready.get("label_balance_enforced") is not True, "current corrected split did not enforce label balance")
    _add_block(blockers, bool(ready.get("shape_failures")), "current corrected split has shape failures")
    _add_block(blockers, ready.get("cache_metadata_validation_enabled") is not True, "current corrected split cache metadata validation is not enabled")
    _add_block(blockers, int(ready.get("metadata_checked_rows", -1)) != EXPECTED_TOTAL, "current corrected split metadata check did not cover 200000 rows")
    _add_block(blockers, int(ready.get("metadata_failure_rows", -1)) != 0, "current corrected split cache metadata has failures")
    _add_block(blockers, int(coverage.get("covered_rows", -1)) != EXPECTED_TOTAL, "current coverage reaudit is not fully covered")
    _add_block(blockers, int(coverage.get("missing_rows", -1)) != 0, "current coverage reaudit has missing rows")
    _add_block(blockers, sample.get("audit_ready") is not True, "Loop78 sample integrity audit is not ready")
    _add_block(blockers, int(sample.get("sampled_rows", -1)) < 2000, "Loop78 sample integrity audit sampled fewer than 2000 rows")
    _add_block(blockers, int(sample.get("failed_rows", -1)) != 0, "Loop78 sample integrity audit has failed rows")
    _add_block(blockers, bool(sample.get("shape_failures")), "Loop78 sample integrity audit has shape failures")

    return {
        "status": "pass" if not blockers else "block",
        "blockers": blockers,
        "cache_ready_report": str(current_cache_ready_path),
        "coverage_reaudit": str(current_coverage_path),
        "sample_integrity_report": str(sample_integrity_path),
        "total_rows": ready.get("total_rows"),
        "covered_rows": ready.get("covered_rows"),
        "missing_rows": ready.get("missing_rows"),
        "label_balance_enforced": ready.get("label_balance_enforced"),
        "cache_metadata_validation_enabled": ready.get("cache_metadata_validation_enabled"),
        "metadata_checked_rows": ready.get("metadata_checked_rows"),
        "metadata_failure_rows": ready.get("metadata_failure_rows"),
        "metadata_issue_counts": ready.get("metadata_issue_counts"),
        "sampled_rows": sample.get("sampled_rows"),
        "sample_failed_rows": sample.get("failed_rows"),
        "sampled_split_counts": sample.get("sampled_split_counts"),
        "sampled_label_counts": sample.get("sampled_label_counts"),
    }


def verify_probability_calibration(
    *,
    ab_report_path: Path,
    replaced_train_val_path: Path,
    replaced_test10k_path: Path,
) -> dict[str, Any]:
    ab = load_json(ab_report_path)
    train_val = load_json(replaced_train_val_path)
    test10k = load_json(replaced_test10k_path)
    probability = ab.get("probability_calibration") or {}
    blockers: list[str] = []

    strict_evaluations = probability.get("strict_evaluations") or []
    all_strict_rows_kept = bool(probability.get("all_strict_rows_kept"))
    strict_missing = [
        row.get("name")
        for row in strict_evaluations
        if int((row.get("rows") or {}).get("skipped_missing_cache") or 0) != 0
        or (row.get("rows") or {}).get("total") != (row.get("rows") or {}).get("kept")
    ]
    protocol_text = str(train_val.get("protocol", "")).casefold()

    _add_block(blockers, probability.get("no_test_used_for_training") is not True, "A/B calibrator report does not prove no-test training")
    _add_block(blockers, not all_strict_rows_kept, "A/B calibrator strict evaluations did not keep every row")
    _add_block(blockers, bool(strict_missing), "A/B calibrator strict evaluations have missing cache rows")
    _add_block(blockers, "no test used" not in protocol_text, "current replacement calibrator protocol does not state no test used")
    _add_block(blockers, "no path/group metadata" not in protocol_text, "current replacement calibrator protocol does not exclude path/group metadata")
    _add_block(blockers, _int_field(train_val.get("train_rows") or {}, "kept") != 20000, "current replacement calibrator did not keep all 20000 train rows")
    _add_block(blockers, _int_field(train_val.get("val_rows") or {}, "kept") != 20000, "current replacement calibrator did not keep all 20000 val rows")
    _add_block(blockers, float((train_val.get("selected") or {}).get("delta_val_f1_vs_baseline") or 0.0) <= 0.0, "current replacement calibrator did not improve Val F1")
    _add_block(blockers, _int_field(test10k.get("rows") or {}, "kept") != 10000, "current replacement calibrator Test-10k did not keep all 10000 rows")
    _add_block(blockers, _int_field(test10k.get("rows") or {}, "skipped_missing_cache") != 0, "current replacement calibrator Test-10k skipped missing cache rows")
    _add_block(blockers, float(test10k.get("delta_f1_vs_baseline") or 0.0) <= 0.0, "current replacement calibrator did not improve Test-10k F1")
    _add_block(blockers, int(test10k.get("delta_errors_vs_baseline") or 0) >= 0, "current replacement calibrator did not reduce Test-10k errors")

    return {
        "status": "pass" if not blockers else "block",
        "blockers": blockers,
        "ab_report": str(ab_report_path),
        "current_train_val_report": str(replaced_train_val_path),
        "current_test10k_report": str(replaced_test10k_path),
        "ab_conclusion": (ab.get("conclusion") or {}).get("probability_calibration"),
        "ab_all_strict_rows_kept": all_strict_rows_kept,
        "ab_no_test_used_for_training": probability.get("no_test_used_for_training"),
        "current_val_delta_f1": (train_val.get("selected") or {}).get("delta_val_f1_vs_baseline"),
        "current_test10k_delta_f1": test10k.get("delta_f1_vs_baseline"),
        "current_test10k_delta_errors": test10k.get("delta_errors_vs_baseline"),
        "strict_missing_evaluations": strict_missing,
    }


def verify_ga_feature_mask(*, ab_report_path: Path) -> dict[str, Any]:
    ab = load_json(ab_report_path)
    ga = ab.get("ga_feature_mask") or {}
    conclusion = (ab.get("conclusion") or {}).get("ga_feature_mask")
    blockers: list[str] = []
    hard_sections = ((ga.get("hard_holdouts") or {}).get("sections") or {})
    hard_mismatches = [
        name
        for name, section in hard_sections.items()
        if (section.get("full") or {}).get("total_predictions") != (section.get("mask") or {}).get("total_predictions")
    ]
    mask_20k = ga.get("feature_mask_20k") or {}
    high_value = ga.get("high_value_benign") or {}
    high_value_delta = (high_value.get("delta_mask_minus_baseline") or {}).get("false_positive")

    _add_block(blockers, conclusion != "strictly_reverified_high_security_candidate_not_default", "GA mask conclusion is not the expected non-default candidate verdict")
    _add_block(blockers, bool(hard_mismatches), "GA hard-holdout full/mask row counts differ")
    _add_block(blockers, int((mask_20k.get("mask_lowest_errors") or {}).get("delta_errors") or 0) >= 0, "GA mask did not reduce 20k validation errors")
    _add_block(blockers, high_value_delta is None, "GA high-value benign FP delta is missing")

    return {
        "status": "pass" if not blockers else "block",
        "blockers": blockers,
        "ab_report": str(ab_report_path),
        "conclusion": conclusion,
        "feature_mask_source": mask_20k.get("source"),
        "validation_20k_delta_errors": (mask_20k.get("mask_lowest_errors") or {}).get("delta_errors"),
        "hard_holdout_sections": sorted(hard_sections.keys()),
        "hard_holdout_count_mismatches": hard_mismatches,
        "high_value_benign_delta_false_positive": high_value_delta,
        "operational_verdict": (
            "not_default_because_high_value_benign_fp_increases"
            if isinstance(high_value_delta, int) and high_value_delta > 0
            else "candidate_requires_additional_review"
        ),
    }


def build_gate(
    *,
    replacement_report: Path,
    replacement_csv: Path,
    replaced_coverage: Path,
    current_cache_ready: Path,
    current_coverage: Path,
    sample_integrity: Path,
    ab_report: Path,
    replaced_calibrator_train_val: Path,
    replaced_calibrator_test10k: Path,
) -> dict[str, Any]:
    sections = {
        "fixed_v2_replacement_130": verify_replacement_130(
            replacement_report_path=replacement_report,
            replacement_csv_path=replacement_csv,
            replaced_coverage_path=replaced_coverage,
        ),
        "current_split_cache": verify_current_split(
            current_cache_ready_path=current_cache_ready,
            current_coverage_path=current_coverage,
            sample_integrity_path=sample_integrity,
        ),
        "probability_calibration": verify_probability_calibration(
            ab_report_path=ab_report,
            replaced_train_val_path=replaced_calibrator_train_val,
            replaced_test10k_path=replaced_calibrator_test10k,
        ),
        "ga_feature_mask": verify_ga_feature_mask(ab_report_path=ab_report),
    }
    blockers = {
        name: section["blockers"]
        for name, section in sections.items()
        if section.get("blockers")
    }
    return {
        "schema": "axon_loop79_current_state_gate_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": "read-only current-state gate; no training, no model loading, no NPZ array loading, no cache mutation, no raw data scan",
        "decision": "pass" if not blockers else "block",
        "blockers": blockers,
        "sections": sections,
        "identity_evidence_policy": {
            "forbidden_as_model_evidence": FORBIDDEN_IDENTITY_EVIDENCE,
            "allowed_uses": [
                "loading",
                "cache alignment",
                "manifest audit",
                "duplicate detection",
                "manual review indexing",
            ],
            "model_evidence_allowed": [
                "byte_sequence",
                "pe_features",
                "stat_features",
                "lightweight_features when explicitly audited",
                "validated model probability as calibration input only",
            ],
        },
        "memory_leak_profile": {
            "loads_model": False,
            "uses_cuda": False,
            "opens_npz_files": False,
            "scans_raw_data": False,
            "writes_cache": False,
        },
        "next_allowed_step": (
            "Val-first model or calibrator work may continue from the current corrected 20w split; GA feature mask remains non-default."
            if not blockers
            else "Resolve blockers before any train, Test-10k, or full-test confirmation."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Loop79 Current State Gate",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Generated at: `{report['generated_at']}`",
        f"- Protocol: {report['protocol']}",
        "",
        "## Sections",
        "",
        "| section | status | key evidence |",
        "|---|---|---|",
    ]
    for name, section in report["sections"].items():
        if name == "fixed_v2_replacement_130":
            evidence = f"{section['replacement_rows']} replacements, missing after={section['missing_rows_after']}"
        elif name == "current_split_cache":
            evidence = (
                f"{section['covered_rows']}/{section['total_rows']} covered, "
                f"metadata failures={section['metadata_failure_rows']}, sampled={section['sampled_rows']}"
            )
        elif name == "probability_calibration":
            evidence = (
                f"Val delta F1={section['current_val_delta_f1']}, "
                f"Test-10k delta errors={section['current_test10k_delta_errors']}"
            )
        else:
            evidence = (
                f"20k delta errors={section['validation_20k_delta_errors']}, "
                f"high-value FP delta={section['high_value_benign_delta_false_positive']}"
            )
        lines.append(f"| {name} | `{section['status']}` | {evidence} |")

    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        for name, blockers in report["blockers"].items():
            lines.append(f"### {name}")
            lines.append("")
            for blocker in blockers:
                lines.append(f"- {blocker}")
            lines.append("")
    else:
        lines.append("None.")
        lines.append("")

    lines.extend(
        [
            "## Identity Evidence Policy",
            "",
            "Identity fields remain alignment and audit metadata only. They are forbidden as model, threshold, GA, or noise-cleaning evidence.",
            "",
            "Forbidden fields:",
        ]
    )
    for field in report["identity_evidence_policy"]["forbidden_as_model_evidence"]:
        lines.append(f"- `{field}`")
    lines.extend(["", f"Next allowed step: {report['next_allowed_step']}", ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Loop79 current-state gate report.")
    parser.add_argument("--replacement-report", type=Path, default=Path("reports/random_20w_split/random_20w_8192_replace_130_bad_features.json"))
    parser.add_argument("--replacement-csv", type=Path, default=Path("reports/random_20w_split/random_20w_8192_replacement_130_strict.csv"))
    parser.add_argument("--replaced-coverage", type=Path, default=Path("reports/random_20w_split/random_20w_8192_uncompressed_cache_coverage_audit_replaced_130.json"))
    parser.add_argument("--current-cache-ready", type=Path, default=Path("reports/random_20w_split/loop100_cache_ready_metadata.json"))
    parser.add_argument("--current-coverage", type=Path, default=Path("reports/random_20w_split/current_split_cache_coverage_reaudit.json"))
    parser.add_argument("--sample-integrity", type=Path, default=Path("reports/random_20w_split/loop78_cache_sample_integrity_1pct.json"))
    parser.add_argument("--ab-report", type=Path, default=Path("reports/model_review/final_model_selection/ab_strict_reverification_report.json"))
    parser.add_argument("--replaced-calibrator-train-val", type=Path, default=Path("reports/random_20w_split/random_20w_8192_replaced_calibrator_train_val.json"))
    parser.add_argument("--replaced-calibrator-test10k", type=Path, default=Path("reports/random_20w_split/random_20w_8192_replaced_calibrator_test10k_eval.json"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_gate(
        replacement_report=args.replacement_report,
        replacement_csv=args.replacement_csv,
        replaced_coverage=args.replaced_coverage,
        current_cache_ready=args.current_cache_ready,
        current_coverage=args.current_coverage,
        sample_integrity=args.sample_integrity,
        ab_report=args.ab_report,
        replaced_calibrator_train_val=args.replaced_calibrator_train_val,
        replaced_calibrator_test10k=args.replaced_calibrator_test10k,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["decision"] == "pass" or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
