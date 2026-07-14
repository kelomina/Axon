#!/usr/bin/env python3
"""Analyze Val-only complementarity between Loop57 and the calibrator.

This is a read-only audit. It joins prediction CSVs by an explicit identity
key, uses labels/predictions/probabilities only for metric/error overlap, and
treats identity columns as alignment metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence


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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def group_by_key(
    rows: Sequence[dict[str, str]],
    *,
    name: str,
    join_key: str,
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        key_value = str(row.get(join_key, "")).strip()
        if not key_value:
            raise ValueError(f"{name} row missing {join_key}")
        grouped.setdefault(key_value, []).append(row)
    return grouped


def duplicate_key_summary(groups: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    duplicate_items = [(key, rows) for key, rows in groups.items() if len(rows) > 1]
    return {
        "duplicate_key_count": len(duplicate_items),
        "duplicate_row_count": sum(len(rows) for _, rows in duplicate_items),
        "duplicate_examples": [
            {
                "join_key": key,
                "count": len(rows),
                "labels": sorted({str(row.get("label", "")) for row in rows}),
                "sample_indices": [str(row.get("sample_index", "")) for row in rows[:5]],
            }
            for key, rows in duplicate_items[:5]
        ],
    }


def sort_join_keys(values: set[str], *, join_key: str) -> list[str]:
    if join_key == "sample_index":
        return sorted(values, key=lambda value: int(float(value)))
    return sorted(values)


def to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    return int(float(str(value)))


def to_float(value: Any) -> float:
    return float(str(value))


def metrics(labels: Sequence[int], predictions: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    tp = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "samples": len(labels),
        "accuracy": (tp + tn) / max(len(labels), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "errors": fp + fn,
    }


def build_joined_rows(
    *,
    loop57_predictions: Path,
    calibrator_predictions: Path,
    join_key: str = "sample_index",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    loop57_rows = read_rows(loop57_predictions)
    calibrator_rows = read_rows(calibrator_predictions)
    by_loop57 = group_by_key(loop57_rows, name="loop57", join_key=join_key)
    by_calibrator = group_by_key(calibrator_rows, name="calibrator", join_key=join_key)
    duplicate_loop57 = duplicate_key_summary(by_loop57)
    duplicate_calibrator = duplicate_key_summary(by_calibrator)
    common = sort_join_keys(set(by_loop57) & set(by_calibrator), join_key=join_key)
    missing_loop57 = sort_join_keys(set(by_calibrator) - set(by_loop57), join_key=join_key)
    missing_calibrator = sort_join_keys(set(by_loop57) - set(by_calibrator), join_key=join_key)

    joined = []
    label_mismatches = 0
    split_mismatches = 0
    ambiguous_common = []
    for key_value in common:
        loop57_group = by_loop57[key_value]
        calibrator_group = by_calibrator[key_value]
        if len(loop57_group) != 1 or len(calibrator_group) != 1:
            ambiguous_common.append(
                {
                    "join_key": key_value,
                    "loop57_count": len(loop57_group),
                    "calibrator_count": len(calibrator_group),
                    "loop57_labels": sorted({str(row.get("label", "")) for row in loop57_group}),
                    "calibrator_labels": sorted({str(row.get("label", "")) for row in calibrator_group}),
                    "loop57_sample_indices": [str(row.get("sample_index", "")) for row in loop57_group[:5]],
                    "calibrator_sample_indices": [str(row.get("sample_index", "")) for row in calibrator_group[:5]],
                }
            )
            continue
        row57 = loop57_group[0]
        rowc = calibrator_group[0]
        label57 = to_int(row57["label"])
        labelc = to_int(rowc["label"])
        if label57 != labelc:
            label_mismatches += 1
            continue
        if row57.get("split") != rowc.get("split"):
            split_mismatches += 1
        loop57_score = to_float(row57.get("final_prob_malicious", row57.get("prob_malicious", "")))
        calibrator_score = to_float(rowc["prob_malicious"])
        loop57_pred = to_int(row57["prediction"])
        calibrator_pred = to_int(rowc["prediction"])
        joined.append(
            {
                "join_key": key_value,
                "join_key_name": join_key,
                "sample_index": row57.get("sample_index") or rowc.get("sample_index", ""),
                "label": label57,
                "split": row57.get("split") or rowc.get("split", ""),
                "loop57_score": loop57_score,
                "loop57_prediction": loop57_pred,
                "loop57_correct": loop57_pred == label57,
                "calibrator_score": calibrator_score,
                "calibrator_prediction": calibrator_pred,
                "calibrator_correct": calibrator_pred == label57,
            }
        )
    summary = {
        "join_key": join_key,
        "loop57_rows": len(loop57_rows),
        "calibrator_rows": len(calibrator_rows),
        "loop57_unique_keys": len(by_loop57),
        "calibrator_unique_keys": len(by_calibrator),
        "common_keys": len(common),
        "common_rows": len(joined),
        "missing_loop57_rows": sum(len(by_calibrator[key]) for key in missing_loop57),
        "missing_calibrator_rows": sum(len(by_loop57[key]) for key in missing_calibrator),
        "label_mismatches": label_mismatches,
        "split_mismatches": split_mismatches,
        "ambiguous_common_keys": len(ambiguous_common),
        "ambiguous_common_examples": ambiguous_common[:5],
        "duplicate_keys": {
            "loop57": duplicate_loop57,
            "calibrator": duplicate_calibrator,
        },
    }
    return joined, summary


def build_summary(
    *,
    loop57_predictions: Path,
    calibrator_predictions: Path,
    join_key: str = "sample_index",
    output_overlap_csv: Optional[Path] = None,
) -> dict[str, Any]:
    joined, join_summary = build_joined_rows(
        loop57_predictions=loop57_predictions,
        calibrator_predictions=calibrator_predictions,
        join_key=join_key,
    )
    labels = [int(row["label"]) for row in joined]
    loop57_preds = [int(row["loop57_prediction"]) for row in joined]
    loop57_scores = [float(row["loop57_score"]) for row in joined]
    cal_preds = [int(row["calibrator_prediction"]) for row in joined]
    cal_scores = [float(row["calibrator_score"]) for row in joined]

    both_correct = []
    both_wrong = []
    loop57_only_correct = []
    calibrator_only_correct = []
    for row in joined:
        if row["loop57_correct"] and row["calibrator_correct"]:
            both_correct.append(row)
        elif (not row["loop57_correct"]) and (not row["calibrator_correct"]):
            both_wrong.append(row)
        elif row["loop57_correct"]:
            loop57_only_correct.append(row)
        else:
            calibrator_only_correct.append(row)

    oracle_preds = []
    oracle_scores = []
    for row in joined:
        if row["loop57_correct"] or row["calibrator_correct"]:
            oracle_preds.append(int(row["label"]))
            oracle_scores.append(1.0 if int(row["label"]) == 1 else 0.0)
        else:
            oracle_preds.append(int(row["loop57_prediction"]))
            oracle_scores.append(float(row["loop57_score"]))

    by_label_overlap = {
        str(label): {
            "both_wrong": sum(1 for row in both_wrong if row["label"] == label),
            "loop57_only_correct": sum(1 for row in loop57_only_correct if row["label"] == label),
            "calibrator_only_correct": sum(1 for row in calibrator_only_correct if row["label"] == label),
        }
        for label in [0, 1]
    }

    if output_overlap_csv is not None:
        output_overlap_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_overlap_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            fieldnames = [
                "join_key",
                "join_key_name",
                "sample_index",
                "label",
                "split",
                "loop57_score",
                "loop57_prediction",
                "loop57_correct",
                "calibrator_score",
                "calibrator_prediction",
                "calibrator_correct",
                "overlap_group",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in joined:
                if row["loop57_correct"] and row["calibrator_correct"]:
                    group = "both_correct"
                elif not row["loop57_correct"] and not row["calibrator_correct"]:
                    group = "both_wrong"
                elif row["loop57_correct"]:
                    group = "loop57_only_correct"
                else:
                    group = "calibrator_only_correct"
                writer.writerow({**row, "overlap_group": group})

    blockers = []
    if join_summary["common_rows"] != 20000:
        blockers.append("Val overlap is not exactly 20000 rows")
    if join_summary["duplicate_keys"]["loop57"]["duplicate_key_count"]:
        blockers.append(f"Loop57 prediction file has duplicate {join_key} values")
    if join_summary["duplicate_keys"]["calibrator"]["duplicate_key_count"]:
        blockers.append(f"Calibrator prediction file has duplicate {join_key} values")
    if join_summary["ambiguous_common_keys"]:
        blockers.append(f"Common {join_key} set contains ambiguous duplicate keys")
    if join_summary["label_mismatches"]:
        blockers.append("Joined rows have label mismatches")
    if join_summary["missing_loop57_rows"] or join_summary["missing_calibrator_rows"]:
        blockers.append(f"Prediction files do not cover the same {join_key} set")

    overlap_counts = Counter()
    overlap_counts["both_correct"] = len(both_correct)
    overlap_counts["both_wrong"] = len(both_wrong)
    overlap_counts["loop57_only_correct"] = len(loop57_only_correct)
    overlap_counts["calibrator_only_correct"] = len(calibrator_only_correct)

    return {
        "schema": "axon_loop81_val_complementarity_v1",
        "protocol": "Val-only prediction overlap audit; no training, no threshold search, no Test/Test-10k access",
        "loop57_predictions": str(loop57_predictions),
        "calibrator_predictions": str(calibrator_predictions),
        "join_key": join_key,
        "join_summary": join_summary,
        "blockers": blockers,
        "ready_for_val_fusion_probe": not blockers and len(calibrator_only_correct) >= 10,
        "metrics": {
            "loop57": metrics(labels, loop57_preds, loop57_scores),
            "calibrator": metrics(labels, cal_preds, cal_scores),
            "oracle_choose_correct_if_either": metrics(labels, oracle_preds, oracle_scores),
        },
        "overlap_counts": dict(overlap_counts),
        "by_label_overlap": by_label_overlap,
        "oracle_gain_vs_loop57_errors": len(calibrator_only_correct),
        "calibrator_regression_vs_loop57_errors": len(loop57_only_correct),
        "output_overlap_csv": str(output_overlap_csv) if output_overlap_csv is not None else None,
        "identity_feature_policy": {
            "sample_index": "alignment only",
            "source_sha256": "alignment/cache-audit only",
            "forbidden_as_model_evidence": FORBIDDEN_IDENTITY_EVIDENCE,
        },
        "next_step": (
            "Run a Val-only fusion probe only if it can learn to recover the "
            "calibrator-only-correct rows without importing identity fields or "
            "Test verdicts."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Val-only Loop57/calibrator complementarity.")
    parser.add_argument("--loop57-predictions", type=Path, required=True)
    parser.add_argument("--calibrator-predictions", type=Path, required=True)
    parser.add_argument(
        "--join-key",
        choices=["sample_index", "source_sha256"],
        default="sample_index",
        help="Identity key used only to align prediction rows for this audit.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-overlap-csv", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_summary(
        loop57_predictions=args.loop57_predictions,
        calibrator_predictions=args.calibrator_predictions,
        join_key=args.join_key,
        output_overlap_csv=args.output_overlap_csv,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not report["blockers"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
