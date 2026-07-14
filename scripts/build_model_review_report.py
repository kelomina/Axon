#!/usr/bin/env python3
"""Build a standard model-review report from existing Axon evaluation artifacts.

The report is intentionally read-only: it does not train, predict, or mutate model
artifacts. It consolidates selection, val-threshold, error, and group evidence into
one Markdown/JSON package so candidate models are reviewed on the same checklist.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence


MAX_JSON_INPUT_BYTES = 8 * 1024 * 1024
MAX_REPORT_ROWS = 20
MAX_VAL_THRESHOLD_ENTRIES = 40
MAX_VAL_THRESHOLD_NODES = 5000
MAX_VAL_THRESHOLD_DEPTH = 8


def load_json(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    if path.stat().st_size > MAX_JSON_INPUT_BYTES:
        raise ValueError(f"JSON artifact is too large for review projection: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = handle.read(8388609)
    if len(raw.encode("utf-8")) > MAX_JSON_INPUT_BYTES:
        raise ValueError(f"JSON artifact exceeded bounded read limit: {path}")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)[:240]


def _project_scalar_dict(row: Any, keys: Optional[Sequence[str]] = None) -> dict:
    if not isinstance(row, dict):
        return {}
    selected_keys = list(keys) if keys is not None else list(row.keys())
    return {key: _scalar(row.get(key)) for key in selected_keys if key in row}


def _project_rows(rows: Any, keys: Optional[Sequence[str]] = None, *, limit: int = MAX_REPORT_ROWS) -> list[dict]:
    if not isinstance(rows, list):
        return []
    return [_project_scalar_dict(row, keys) for row in rows[:limit] if isinstance(row, dict)]


METRIC_KEYS = [
    "model",
    "name",
    "threshold",
    "f1",
    "precision",
    "recall",
    "accuracy",
    "auc",
    "fp",
    "fn",
    "false_positive",
    "false_negative",
    "errors",
    "true_positive",
    "true_negative",
    "total_predictions",
    "sample_count",
    "original_hard_family_test_f1",
    "original_hard_family_test_fp",
    "original_hard_family_test_fn",
    "original_hard_family_test_errors",
    "hard_error_holdout_f1",
    "hard_error_holdout_fp",
    "hard_error_holdout_fn",
    "hard_error_holdout_errors",
    "hard_fn_holdout_f1",
    "hard_fn_holdout_fp",
    "hard_fn_holdout_fn",
    "hard_fn_holdout_errors",
]


def _project_selection_report(payload: dict) -> dict:
    return {
        "decision_rule": _scalar(payload.get("decision_rule")),
        "baseline": _project_scalar_dict(payload.get("baseline"), METRIC_KEYS),
        "recommendation": _project_scalar_dict(payload.get("recommendation"), METRIC_KEYS),
        "candidate_summary": _project_rows(payload.get("candidate_summary"), METRIC_KEYS),
    }


def _project_error_summary(payload: Optional[dict]) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    return {
        **_project_scalar_dict(
            payload,
            [
                "threshold",
                "total_predictions",
                "error_count",
                "false_positive_count",
                "false_negative_count",
            ],
        ),
        "top_error_groups": _project_rows(
            payload.get("top_error_groups"),
            ["group_id", "error_count", "fp_count", "fn_count", "is_rare_group", "avg_prob_malicious"],
            limit=10,
        ),
    }


def _project_group_summary(payload: Optional[dict]) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    group_keys = ["groups", "predicted_samples", "error_count", "accuracy"]
    return {
        "overall": _project_scalar_dict(payload.get("overall"), group_keys),
        "rare_groups": _project_scalar_dict(payload.get("rare_groups"), group_keys),
        "singleton_groups": _project_scalar_dict(payload.get("singleton_groups"), group_keys),
        "worst_groups": _project_rows(payload.get("worst_groups"), limit=10),
    }


def _project_calibrator_evaluation(payload: Optional[dict]) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    result = {
        "predictions": _scalar(payload.get("predictions")),
        "rows": _project_scalar_dict(payload.get("rows")),
    }
    for key in [
        "baseline",
        "baseline_test_at_053_same_csv",
        "calibrator_metrics",
        "calibrator_test_at_val_selected_threshold",
    ]:
        value = _project_scalar_dict(payload.get(key), METRIC_KEYS)
        if value:
            result[key] = value
    slices = payload.get("slices")
    if isinstance(slices, dict):
        result["slices"] = {}
        for name, slice_payload in list(slices.items())[:MAX_REPORT_ROWS]:
            if not isinstance(slice_payload, dict):
                continue
            result["slices"][str(name)] = {
                "rows": _scalar(slice_payload.get("rows")),
                "baseline": _project_scalar_dict(slice_payload.get("baseline"), METRIC_KEYS),
                "calibrator_metrics": _project_scalar_dict(slice_payload.get("calibrator_metrics"), METRIC_KEYS),
            }
    return result


def _project_metric_bundle(bundle: Any) -> dict:
    if not isinstance(bundle, dict):
        return {}
    result = {}
    if isinstance(bundle.get("metrics"), dict):
        result["metrics"] = _project_scalar_dict(bundle.get("metrics"), METRIC_KEYS)
    if isinstance(bundle.get("delta_vs_baseline_full"), dict):
        result["delta_vs_baseline_full"] = _project_scalar_dict(bundle.get("delta_vs_baseline_full"))
    return result


def _project_feature_mask_evaluation(payload: Optional[dict]) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "feature_mask": _scalar(payload.get("feature_mask")),
        "feature_mask_summary": _project_scalar_dict(payload.get("feature_mask_summary")),
        "samples": _scalar(payload.get("samples")),
        "summary": {
            "baseline_full": _project_metric_bundle(summary.get("baseline_full")),
            "best_mask_f1": _project_metric_bundle(summary.get("best_mask_f1")),
            "best_mask_errors": _project_metric_bundle(summary.get("best_mask_errors")),
        },
    }


def _project_feature_mask_groups(payload: Optional[dict]) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    groups = []
    raw_groups = payload.get("groups")
    for row in raw_groups[:MAX_REPORT_ROWS] if isinstance(raw_groups, list) else []:
        if not isinstance(row, dict):
            continue
        groups.append(
            {
                "group": _scalar(row.get("group")),
                "full_050": _project_scalar_dict(row.get("full_050")),
                "delta_mask0525_minus_full050": _project_scalar_dict(row.get("delta_mask0525_minus_full050")),
            }
        )
    return {"groups": groups}


def _project_feature_mask_holdout(payload: Optional[dict]) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    sections = {}
    raw_sections = payload.get("sections")
    if isinstance(raw_sections, dict):
        for name, section in list(raw_sections.items())[:MAX_REPORT_ROWS]:
            if not isinstance(section, dict):
                continue
            sections[str(name)] = {
                "full": _project_scalar_dict(section.get("full"), METRIC_KEYS),
                "mask": _project_scalar_dict(section.get("mask"), METRIC_KEYS),
                "delta_mask_minus_full": _project_scalar_dict(section.get("delta_mask_minus_full")),
            }
    return {
        "comparison": _scalar(payload.get("comparison")),
        "sections": sections,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _metric_row(row: dict, prefix: str) -> dict:
    return {
        "f1": row.get(f"{prefix}_f1"),
        "fp": row.get(f"{prefix}_fp"),
        "fn": row.get(f"{prefix}_fn"),
        "errors": row.get(f"{prefix}_errors"),
    }


def _has_metrics(row: Optional[dict], prefix: str) -> bool:
    if not row:
        return False
    keys = [f"{prefix}_f1", f"{prefix}_fp", f"{prefix}_fn", f"{prefix}_errors"]
    return all(key in row for key in keys)


def _collect_val_threshold_entries(payload: Optional[dict]) -> list[dict]:
    if not payload:
        return []

    entries: list[dict] = []
    visited_nodes = 0

    def visit(node: Any, path: list[str], depth: int) -> None:
        nonlocal visited_nodes
        if len(entries) >= MAX_VAL_THRESHOLD_ENTRIES:
            return
        if depth > MAX_VAL_THRESHOLD_DEPTH:
            return
        visited_nodes += 1
        if visited_nodes > MAX_VAL_THRESHOLD_NODES:
            return
        if isinstance(node, dict):
            if isinstance(node.get("val_selected"), dict):
                entry = {
                    "name": ".".join(path) if path else "root",
                    "val_selected": _project_scalar_dict(node.get("val_selected"), METRIC_KEYS),
                    "test_at_val_selected": _project_scalar_dict(node.get("test_at_val_selected"), METRIC_KEYS),
                    "test_oracle_reference_only": _project_scalar_dict(
                        node.get("test_oracle_reference_only"), METRIC_KEYS
                    ),
                }
                entries.append(entry)
            for key, value in node.items():
                visit(value, [*path, str(key)], depth + 1)
        elif isinstance(node, list):
            for index, value in enumerate(node[:MAX_REPORT_ROWS]):
                visit(value, [*path, str(index)], depth + 1)

    visit(payload, [], 0)
    return entries


def _gate(name: str, passed: bool, evidence: str, severity: str = "required") -> dict:
    return {
        "name": name,
        "status": "PASS" if passed else "WARN",
        "severity": severity,
        "evidence": evidence if passed else f"Missing or incomplete: {evidence}",
    }


def build_review(
    *,
    title: str,
    selection_report_path: Path,
    val_threshold_report_path: Optional[Path] = None,
    error_summary_path: Optional[Path] = None,
    group_summary_path: Optional[Path] = None,
    calibrator_result_path: Optional[Path] = None,
    calibrator_evaluation_path: Optional[Path] = None,
    calibrator_holdout_evaluation_paths: Optional[Sequence[Path]] = None,
    calibrator_diagnostic_evaluation_paths: Optional[Sequence[Path]] = None,
    feature_mask_evaluation_path: Optional[Path] = None,
    feature_mask_groups_path: Optional[Path] = None,
    feature_mask_holdout_path: Optional[Path] = None,
) -> dict:
    raw_selection = load_json(selection_report_path)
    if raw_selection is None:
        raise ValueError("selection report is required")
    selection = _project_selection_report(raw_selection)
    val_threshold_payload = load_json(val_threshold_report_path)
    error_summary = _project_error_summary(load_json(error_summary_path))
    group_summary = _project_group_summary(load_json(group_summary_path))
    calibrator_evaluation = _project_calibrator_evaluation(load_json(calibrator_evaluation_path))
    calibrator_holdout_evaluations = [
        _project_calibrator_evaluation(load_json(path)) for path in (calibrator_holdout_evaluation_paths or [])
    ]
    calibrator_diagnostic_evaluations = [
        _project_calibrator_evaluation(load_json(path)) for path in (calibrator_diagnostic_evaluation_paths or [])
    ]
    feature_mask_evaluation = _project_feature_mask_evaluation(load_json(feature_mask_evaluation_path))
    feature_mask_groups = _project_feature_mask_groups(load_json(feature_mask_groups_path))
    feature_mask_holdout = _project_feature_mask_holdout(load_json(feature_mask_holdout_path))

    baseline = selection.get("baseline")
    recommendation = selection.get("recommendation")
    candidate_summary = selection.get("candidate_summary", [])
    val_entries = _collect_val_threshold_entries(val_threshold_payload)

    gates = [
        _gate(
            "overall original-test metrics",
            _has_metrics(recommendation, "original_hard_family_test"),
            "recommendation contains original_hard_family_test F1/FP/FN/errors",
        ),
        _gate(
            "hard-error holdout metrics",
            _has_metrics(recommendation, "hard_error_holdout"),
            "recommendation contains hard_error_holdout F1/FP/FN/errors",
        ),
        _gate(
            "hard-FN holdout metrics",
            _has_metrics(recommendation, "hard_fn_holdout"),
            "recommendation contains hard_fn_holdout F1/FP/FN/errors",
        ),
        _gate(
            "baseline comparison",
            isinstance(baseline, dict) and bool(baseline),
            "baseline row exists in selection report",
        ),
        _gate(
            "val-selected threshold evidence",
            bool(val_entries),
            "val threshold report contains val_selected and optional test_at_val_selected rows",
        ),
        _gate(
            "error analysis evidence",
            isinstance(error_summary, dict) and "false_positive_count" in error_summary and "false_negative_count" in error_summary,
            "prediction_error_summary.json contains FP/FN counts",
        ),
        _gate(
            "group evaluation evidence",
            isinstance(group_summary, dict) and "overall" in group_summary and "worst_groups" in group_summary,
            "group_evaluation_summary.json contains overall and worst_groups",
        ),
    ]
    if feature_mask_evaluation_path is not None:
        gates.append(
            _gate(
                "feature-mask threshold evidence",
                isinstance(feature_mask_evaluation, dict)
                and "summary" in feature_mask_evaluation
                and "best_mask_f1" in feature_mask_evaluation.get("summary", {})
                and "best_mask_errors" in feature_mask_evaluation.get("summary", {}),
                "feature-mask evaluation contains baseline and mask threshold summaries",
                severity="optional",
            )
        )
    if feature_mask_groups_path is not None:
        gates.append(
            _gate(
                "feature-mask source-group evidence",
                isinstance(feature_mask_groups, dict) and bool(feature_mask_groups.get("groups")),
                "feature-mask source-group evaluation contains per-source trade-off rows",
                severity="optional",
            )
        )

    passed_required = all(gate["status"] == "PASS" for gate in gates if gate["severity"] == "required")
    report = {
        "schema": "axon_model_review_v1",
        "title": title,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "selection_report": str(selection_report_path),
            "val_threshold_report": str(val_threshold_report_path) if val_threshold_report_path else None,
            "error_summary": str(error_summary_path) if error_summary_path else None,
            "group_summary": str(group_summary_path) if group_summary_path else None,
            "calibrator_result": str(calibrator_result_path) if calibrator_result_path else None,
            "calibrator_evaluation": str(calibrator_evaluation_path) if calibrator_evaluation_path else None,
            "calibrator_holdout_evaluations": [str(path) for path in (calibrator_holdout_evaluation_paths or [])],
            "calibrator_diagnostic_evaluations": [
                str(path) for path in (calibrator_diagnostic_evaluation_paths or [])
            ],
            "feature_mask_evaluation": str(feature_mask_evaluation_path) if feature_mask_evaluation_path else None,
            "feature_mask_groups": str(feature_mask_groups_path) if feature_mask_groups_path else None,
            "feature_mask_holdout": str(feature_mask_holdout_path) if feature_mask_holdout_path else None,
        },
        "review_status": "usable" if passed_required else "usable_with_warnings",
        "projection_limits": {
            "max_json_input_bytes": MAX_JSON_INPUT_BYTES,
            "max_report_rows": MAX_REPORT_ROWS,
            "max_val_threshold_entries": MAX_VAL_THRESHOLD_ENTRIES,
            "max_val_threshold_nodes": MAX_VAL_THRESHOLD_NODES,
            "max_val_threshold_depth": MAX_VAL_THRESHOLD_DEPTH,
        },
        "gates": gates,
        "decision_rule": selection.get("decision_rule"),
        "baseline": baseline,
        "recommendation": recommendation,
        "candidate_summary": candidate_summary,
        "val_threshold_entries": val_entries,
        "error_summary": error_summary,
        "group_summary": group_summary,
        "calibrator_evaluation": calibrator_evaluation,
        "calibrator_holdout_evaluations": calibrator_holdout_evaluations,
        "calibrator_diagnostic_evaluations": calibrator_diagnostic_evaluations,
        "feature_mask_evaluation": feature_mask_evaluation,
        "feature_mask_groups": feature_mask_groups,
        "feature_mask_holdout": feature_mask_holdout,
    }
    return report


def _append_metric_table(lines: list[str], rows: Sequence[dict]) -> None:
    lines.append("| model | threshold | original F1 | original FP | original FN | original errors | hard-error errors | hard-FN errors |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {model} | {threshold} | {f1} | {fp} | {fn} | {errors} | {he} | {hfn} |".format(
                model=row.get("model", ""),
                threshold=_fmt(row.get("threshold"), 3),
                f1=_fmt(row.get("original_hard_family_test_f1")),
                fp=_fmt(row.get("original_hard_family_test_fp")),
                fn=_fmt(row.get("original_hard_family_test_fn")),
                errors=_fmt(row.get("original_hard_family_test_errors")),
                he=_fmt(row.get("hard_error_holdout_errors")),
                hfn=_fmt(row.get("hard_fn_holdout_errors")),
            )
        )


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append(f"# {report['title']}")
    lines.append("")
    lines.append(f"- Generated at: `{report['generated_at']}`")
    lines.append(f"- Review status: `{report['review_status']}`")
    if report.get("decision_rule"):
        lines.append(f"- Decision rule: {report['decision_rule']}")
    lines.append("")

    lines.append("## Inputs")
    lines.append("")
    lines.append("| artifact | path |")
    lines.append("|---|---|")
    for name, path in report["inputs"].items():
        lines.append(f"| {name} | `{path or ''}` |")
    lines.append("")

    lines.append("## Gate Audit")
    lines.append("")
    lines.append("| check | status | evidence |")
    lines.append("|---|---|---|")
    for gate in report["gates"]:
        lines.append(f"| {gate['name']} | **{gate['status']}** | {gate['evidence']} |")
    lines.append("")

    lines.append("## Candidate Summary")
    lines.append("")
    _append_metric_table(lines, report.get("candidate_summary", []))
    lines.append("")

    recommendation = report.get("recommendation") or {}
    if recommendation:
        lines.append("## Current Recommendation")
        lines.append("")
        lines.append(f"- Model: `{recommendation.get('model', '')}`")
        lines.append(f"- Threshold: `{_fmt(recommendation.get('threshold'), 3)}`")
        lines.append(
            "- Original test: F1 `{f1}`, FP `{fp}`, FN `{fn}`, errors `{errors}`".format(
                f1=_fmt(recommendation.get("original_hard_family_test_f1")),
                fp=_fmt(recommendation.get("original_hard_family_test_fp")),
                fn=_fmt(recommendation.get("original_hard_family_test_fn")),
                errors=_fmt(recommendation.get("original_hard_family_test_errors")),
            )
        )
        lines.append(
            "- Hard holdouts: hard-error errors `{he}`, hard-FN errors `{hfn}`".format(
                he=_fmt(recommendation.get("hard_error_holdout_errors")),
                hfn=_fmt(recommendation.get("hard_fn_holdout_errors")),
            )
        )
        lines.append("")

    val_entries = report.get("val_threshold_entries", [])
    if val_entries:
        lines.append("## Val-Selected Threshold Evidence")
        lines.append("")
        lines.append("| source | val threshold | val F1 | test threshold | test F1 | test FP | test FN |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for entry in val_entries:
            val_row = entry.get("val_selected") or {}
            test_row = entry.get("test_at_val_selected") or {}
            lines.append(
                "| {name} | {vt} | {vf1} | {tt} | {tf1} | {tfp} | {tfn} |".format(
                    name=entry.get("name", ""),
                    vt=_fmt(val_row.get("threshold"), 3),
                    vf1=_fmt(val_row.get("f1")),
                    tt=_fmt(test_row.get("threshold"), 3),
                    tf1=_fmt(test_row.get("f1")),
                    tfp=_fmt(test_row.get("fp")),
                    tfn=_fmt(test_row.get("fn")),
                )
            )
        lines.append("")

    error_summary = report.get("error_summary") or {}
    if error_summary:
        lines.append("## Error Analysis")
        lines.append("")
        lines.append(
            "- Errors: `{errors}`; FP `{fp}`; FN `{fn}` at threshold `{threshold}` over `{total}` predictions.".format(
                errors=_fmt(error_summary.get("error_count")),
                fp=_fmt(error_summary.get("false_positive_count")),
                fn=_fmt(error_summary.get("false_negative_count")),
                threshold=_fmt(error_summary.get("threshold"), 3),
                total=_fmt(error_summary.get("total_predictions")),
            )
        )
        top_groups = error_summary.get("top_error_groups") or []
        if top_groups:
            lines.append("")
            lines.append("| top group | errors | FP | FN | rare | avg prob |")
            lines.append("|---|---:|---:|---:|---|---:|")
            for row in top_groups[:10]:
                lines.append(
                    "| {gid} | {errors} | {fp} | {fn} | {rare} | {prob} |".format(
                        gid=row.get("group_id", ""),
                        errors=_fmt(row.get("error_count")),
                        fp=_fmt(row.get("fp_count")),
                        fn=_fmt(row.get("fn_count")),
                        rare=row.get("is_rare_group", ""),
                        prob=_fmt(row.get("avg_prob_malicious")),
                    )
                )
        lines.append("")

    group_summary = report.get("group_summary") or {}
    if group_summary:
        lines.append("## Group Evaluation")
        lines.append("")
        overall = group_summary.get("overall") or {}
        rare = group_summary.get("rare_groups") or {}
        singleton = group_summary.get("singleton_groups") or {}
        lines.append("| slice | groups | samples | errors | accuracy |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, row in [("overall", overall), ("rare", rare), ("singleton", singleton)]:
            lines.append(
                "| {name} | {groups} | {samples} | {errors} | {acc} |".format(
                    name=name,
                    groups=_fmt(row.get("groups")),
                    samples=_fmt(row.get("predicted_samples")),
                    errors=_fmt(row.get("error_count")),
                    acc=_fmt(row.get("accuracy")),
                )
            )
        lines.append("")

    calibrator_sections = []
    if report.get("calibrator_evaluation"):
        calibrator_sections.append(("test", report.get("calibrator_evaluation") or {}))
    for index, evaluation in enumerate(report.get("calibrator_holdout_evaluations") or [], start=1):
        predictions = Path(str(evaluation.get("predictions", ""))).stem if evaluation else ""
        name = predictions or f"holdout_{index}"
        calibrator_sections.append((name, evaluation or {}))
    for index, evaluation in enumerate(report.get("calibrator_diagnostic_evaluations") or [], start=1):
        predictions = Path(str(evaluation.get("predictions", ""))).stem if evaluation else ""
        name = predictions or f"diagnostic_{index}"
        calibrator_sections.append((f"diagnostic:{name}", evaluation or {}))

    if calibrator_sections:
        lines.append("## Probability Calibrator")
        lines.append("")
        lines.append("| slice | model | rows kept/total | skipped cache | threshold | F1 | FP | FN | errors | AUC |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, calibrator_evaluation in calibrator_sections:
            rows = calibrator_evaluation.get("rows") or {}
            baseline = calibrator_evaluation.get("baseline") or calibrator_evaluation.get("baseline_test_at_053_same_csv") or {}
            calibrator = calibrator_evaluation.get("calibrator_metrics") or calibrator_evaluation.get("calibrator_test_at_val_selected_threshold") or {}
            if not (baseline and calibrator):
                continue
            row_count = "{kept}/{total}".format(
                kept=_fmt(rows.get("kept")),
                total=_fmt(rows.get("total") or rows.get("csv_total")),
            )
            skipped = _fmt(rows.get("skipped_missing_cache", 0))
            lines.append(
                "| {slice} | baseline | {rows} | {skipped} | {threshold} | {f1} | {fp} | {fn} | {errors} | {auc} |".format(
                    slice=name,
                    rows=row_count,
                    skipped=skipped,
                    threshold=_fmt(baseline.get("threshold"), 3),
                    f1=_fmt(baseline.get("f1")),
                    fp=_fmt(baseline.get("false_positive")),
                    fn=_fmt(baseline.get("false_negative")),
                    errors=_fmt(baseline.get("errors")),
                    auc=_fmt(baseline.get("auc")),
                )
            )
            lines.append(
                "| {slice} | calibrator | {rows} | {skipped} | {threshold} | {f1} | {fp} | {fn} | {errors} | {auc} |".format(
                    slice=name,
                    rows=row_count,
                    skipped=skipped,
                    threshold=_fmt(calibrator.get("threshold"), 3),
                    f1=_fmt(calibrator.get("f1")),
                    fp=_fmt(calibrator.get("false_positive")),
                    fn=_fmt(calibrator.get("false_negative")),
                    errors=_fmt(calibrator.get("errors")),
                    auc=_fmt(calibrator.get("auc")),
                )
            )
            slices = calibrator_evaluation.get("slices") or {}
            for slice_name in ["whitelist_benign_label_0", "benign_label_0", "malicious_label_1"]:
                slice_payload = slices.get(slice_name) or {}
                baseline_slice = slice_payload.get("baseline") or {}
                calibrator_slice = slice_payload.get("calibrator_metrics") or {}
                if not (baseline_slice and calibrator_slice):
                    continue
                slice_count = _fmt(slice_payload.get("rows"))
                lines.append(
                    "| {slice}::{subslice} | baseline | {rows} | {skipped} | {threshold} | {f1} | {fp} | {fn} | {errors} | {auc} |".format(
                        slice=name,
                        subslice=slice_name,
                        rows=slice_count,
                        skipped="",
                        threshold=_fmt(baseline_slice.get("threshold"), 3),
                        f1=_fmt(baseline_slice.get("f1")),
                        fp=_fmt(baseline_slice.get("false_positive")),
                        fn=_fmt(baseline_slice.get("false_negative")),
                        errors=_fmt(baseline_slice.get("errors")),
                        auc=_fmt(baseline_slice.get("auc")),
                    )
                )
                lines.append(
                    "| {slice}::{subslice} | calibrator | {rows} | {skipped} | {threshold} | {f1} | {fp} | {fn} | {errors} | {auc} |".format(
                        slice=name,
                        subslice=slice_name,
                        rows=slice_count,
                        skipped="",
                        threshold=_fmt(calibrator_slice.get("threshold"), 3),
                        f1=_fmt(calibrator_slice.get("f1")),
                        fp=_fmt(calibrator_slice.get("false_positive")),
                        fn=_fmt(calibrator_slice.get("false_negative")),
                        errors=_fmt(calibrator_slice.get("errors")),
                        auc=_fmt(calibrator_slice.get("auc")),
                    )
                )
        lines.append("")

    feature_mask_evaluation = report.get("feature_mask_evaluation") or {}
    if feature_mask_evaluation:
        summary = feature_mask_evaluation.get("summary") or {}
        baseline = summary.get("baseline_full") or {}
        best_mask_f1 = summary.get("best_mask_f1") or {}
        best_mask_errors = summary.get("best_mask_errors") or {}
        mask_summary = feature_mask_evaluation.get("feature_mask_summary") or {}

        def _mask_summary_row(name: str, row: dict) -> tuple[dict, dict]:
            return row.get("metrics") or {}, row.get("delta_vs_baseline_full") or {}

        baseline_metrics = baseline.get("metrics") or {}
        best_f1_metrics, best_f1_delta = _mask_summary_row("best_mask_f1", best_mask_f1)
        best_error_metrics, best_error_delta = _mask_summary_row("best_mask_errors", best_mask_errors)

        if baseline_metrics and best_f1_metrics and best_error_metrics:
            lines.append("## GA Feature Mask")
            lines.append("")
            lines.append(
                "- Mask: `{mask}`; kept features `{total}` (PE `{pe}`, stat `{stat}`); sample count `{samples}`.".format(
                    mask=feature_mask_evaluation.get("feature_mask", ""),
                    total=_fmt(mask_summary.get("kept_total")),
                    pe=_fmt(mask_summary.get("kept_pe")),
                    stat=_fmt(mask_summary.get("kept_stat")),
                    samples=_fmt(feature_mask_evaluation.get("samples")),
                )
            )
            lines.append("")
            lines.append("| row | threshold | F1 | FP | FN | errors | delta FP | delta FN | delta errors |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            lines.append(
                "| full baseline | {threshold} | {f1} | {fp} | {fn} | {errors} |  |  |  |".format(
                    threshold=_fmt(baseline_metrics.get("threshold"), 3),
                    f1=_fmt(baseline_metrics.get("f1")),
                    fp=_fmt(baseline_metrics.get("false_positive")),
                    fn=_fmt(baseline_metrics.get("false_negative")),
                    errors=_fmt(baseline_metrics.get("errors")),
                )
            )
            lines.append(
                "| mask best F1 | {threshold} | {f1} | {fp} | {fn} | {errors} | {dfp} | {dfn} | {derr} |".format(
                    threshold=_fmt(best_f1_metrics.get("threshold"), 3),
                    f1=_fmt(best_f1_metrics.get("f1")),
                    fp=_fmt(best_f1_metrics.get("false_positive")),
                    fn=_fmt(best_f1_metrics.get("false_negative")),
                    errors=_fmt(best_f1_metrics.get("errors")),
                    dfp=_fmt(best_f1_delta.get("false_positive")),
                    dfn=_fmt(best_f1_delta.get("false_negative")),
                    derr=_fmt(best_f1_delta.get("errors")),
                )
            )
            lines.append(
                "| mask lowest errors | {threshold} | {f1} | {fp} | {fn} | {errors} | {dfp} | {dfn} | {derr} |".format(
                    threshold=_fmt(best_error_metrics.get("threshold"), 3),
                    f1=_fmt(best_error_metrics.get("f1")),
                    fp=_fmt(best_error_metrics.get("false_positive")),
                    fn=_fmt(best_error_metrics.get("false_negative")),
                    errors=_fmt(best_error_metrics.get("errors")),
                    dfp=_fmt(best_error_delta.get("false_positive")),
                    dfn=_fmt(best_error_delta.get("false_negative")),
                    derr=_fmt(best_error_delta.get("errors")),
                )
            )
            lines.append("")

    feature_mask_groups = report.get("feature_mask_groups") or {}
    if feature_mask_groups.get("groups"):
        lines.append("### Feature Mask Source Groups")
        lines.append("")
        lines.append("| source group | samples | delta FP | delta FN | delta errors |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in feature_mask_groups.get("groups", []):
            full = row.get("full_050") or {}
            delta = row.get("delta_mask0525_minus_full050") or {}
            lines.append(
                "| {group} | {samples} | {dfp} | {dfn} | {derr} |".format(
                    group=row.get("group", ""),
                    samples=_fmt(full.get("sample_count")),
                    dfp=_fmt(delta.get("fp")),
                    dfn=_fmt(delta.get("fn")),
                    derr=_fmt(delta.get("errors")),
                )
            )
        lines.append("")

    feature_mask_holdout = report.get("feature_mask_holdout") or {}
    if feature_mask_holdout.get("sections"):
        lines.append("### Feature Mask Hard Holdouts")
        lines.append("")
        if feature_mask_holdout.get("comparison"):
            lines.append(f"- Comparison: {feature_mask_holdout['comparison']}")
            lines.append("")
        lines.append("| slice | variant | rows | threshold | FP | FN | errors | delta FP | delta FN | delta errors |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, section in feature_mask_holdout.get("sections", {}).items():
            full = section.get("full") or {}
            mask = section.get("mask") or {}
            delta = section.get("delta_mask_minus_full") or {}
            lines.append(
                "| {slice} | full | {rows} | {threshold} | {fp} | {fn} | {errors} |  |  |  |".format(
                    slice=name,
                    rows=_fmt(full.get("total_predictions")),
                    threshold=_fmt(full.get("threshold"), 3),
                    fp=_fmt(full.get("false_positive")),
                    fn=_fmt(full.get("false_negative")),
                    errors=_fmt(full.get("errors")),
                )
            )
            lines.append(
                "| {slice} | mask | {rows} | {threshold} | {fp} | {fn} | {errors} | {dfp} | {dfn} | {derr} |".format(
                    slice=name,
                    rows=_fmt(mask.get("total_predictions")),
                    threshold=_fmt(mask.get("threshold"), 3),
                    fp=_fmt(mask.get("false_positive")),
                    fn=_fmt(mask.get("false_negative")),
                    errors=_fmt(mask.get("errors")),
                    dfp=_fmt(delta.get("false_positive")),
                    dfn=_fmt(delta.get("false_negative")),
                    derr=_fmt(delta.get("errors")),
                )
            )
        lines.append("")

    lines.append("## Result")
    lines.append("")
    if report["review_status"] == "usable":
        lines.append("This review package satisfies the standard gate inputs and is usable as a model-selection artifact.")
    else:
        lines.append("This review package is usable for inspection, but one or more standard gate inputs are missing.")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a standard Axon model-review Markdown/JSON report.")
    parser.add_argument("--title", default="Axon Model Review")
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--val-threshold-report", type=Path, default=None)
    parser.add_argument("--error-summary", type=Path, default=None)
    parser.add_argument("--group-summary", type=Path, default=None)
    parser.add_argument("--calibrator-result", type=Path, default=None)
    parser.add_argument("--calibrator-evaluation", type=Path, default=None)
    parser.add_argument("--calibrator-holdout-evaluation", type=Path, action="append", default=[])
    parser.add_argument("--calibrator-diagnostic-evaluation", type=Path, action="append", default=[])
    parser.add_argument("--feature-mask-evaluation", type=Path, default=None)
    parser.add_argument("--feature-mask-groups", type=Path, default=None)
    parser.add_argument("--feature-mask-holdout", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_review(
        title=args.title,
        selection_report_path=args.selection_report,
        val_threshold_report_path=args.val_threshold_report,
        error_summary_path=args.error_summary,
        group_summary_path=args.group_summary,
        calibrator_result_path=args.calibrator_result,
        calibrator_evaluation_path=args.calibrator_evaluation,
        calibrator_holdout_evaluation_paths=args.calibrator_holdout_evaluation,
        calibrator_diagnostic_evaluation_paths=args.calibrator_diagnostic_evaluation,
        feature_mask_evaluation_path=args.feature_mask_evaluation,
        feature_mask_groups_path=args.feature_mask_groups,
        feature_mask_holdout_path=args.feature_mask_holdout,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "model_review_report.json"
    md_path = args.output_dir / "model_review_summary.md"
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Review status: {report['review_status']}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
