#!/usr/bin/env python3
"""Build a source-aware manual adjudication queue without filling verdicts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FIELDNAMES = [
    "review_priority_rank",
    "review_lane",
    "source_group_key",
    "source_group_size",
    "source_group_error_type_counts",
    "source_group_priority_counts",
    "suspicion_level",
    "review_question_focus",
    "allowed_manual_label_verdicts",
    "allowed_recommended_actions",
    "replacement_rule",
    "review_rank",
    "priority",
    "reason",
    "error_type",
    "path_hint",
    "source_path",
    "source_sha256",
    "label",
    "prediction",
    "prob_malicious",
    "score_column",
    "stage2_prob_malicious",
    "base_prob_malicious",
    "top_k",
    "neighbor_label_counts",
    "same_label_count",
    "opposite_label_count",
    "opposite_label_ratio",
    "nearest_similarity",
    "top5_neighbor_labels",
    "top5_neighbor_similarities",
    "top5_neighbor_sha256",
    "top5_neighbor_paths",
    "manual_label_verdict",
    "manual_verdict_note",
    "recommended_action",
]

ALLOWED_VERDICTS = "label_correct|label_wrong|out_of_scope|feature_broken|uncertain"
ALLOWED_ACTIONS = "keep_label|relabel_train_only|replace_sample|quarantine_source_group|needs_more_evidence|model_blindspot"
REPLACEMENT_RULE = "feature_broken/out_of_scope rows must be replaced by fresh valid candidates; never self-fill."


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[dict]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _parts(path_text: str) -> list[str]:
    return [part for part in re.split(r"[\\/]+", str(path_text or "")) if part]


def _after_data_parts(path_text: str) -> list[str]:
    parts = _parts(path_text)
    lowered = [part.casefold() for part in parts]
    if "data" in lowered:
        return parts[lowered.index("data") + 1 :]
    return parts


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _probability(row: dict) -> float:
    for key in ("prob_malicious", "stage2_prob_malicious", "blend_prob_malicious"):
        value = row.get(key)
        if value not in (None, ""):
            return _safe_float(value)
    return 0.0


def _similarity(row: dict) -> float:
    return _safe_float(row.get("nearest_similarity"))


def _opposite_ratio(row: dict) -> float:
    return _safe_float(row.get("opposite_label_ratio"))


def _confidence(row: dict) -> float:
    prob = _probability(row)
    return 1.0 - prob if row.get("error_type") == "FN" else prob


def _month(path_text: str) -> str:
    match = re.search(r"20\d{2}-\d{2}", str(path_text or ""))
    return match.group(0) if match else "<none>"


def _date(path_text: str) -> str:
    match = re.search(r"20\d{2}-\d{2}-\d{2}", str(path_text or ""))
    return match.group(0) if match else "<none>"


def source_group_key(row: dict) -> str:
    parts = _after_data_parts(row.get("source_path", ""))
    if not parts:
        return "<unknown>"
    data_dir = parts[0]
    if data_dir in {"待加入白名单", "benign"} and _month(row.get("source_path", "")) == "<none>":
        return f"{data_dir}/<flat>"
    date = _date(row.get("source_path", ""))
    month = _month(row.get("source_path", ""))
    if date != "<none>" and month != "<none>":
        return f"{data_dir}/{month}/{date}"
    return "/".join(parts[: min(3, len(parts))])


def suspicion_level(row: dict) -> str:
    nearest = _similarity(row)
    opposite = _opposite_ratio(row)
    confidence = _confidence(row)
    if nearest >= 0.95 and opposite >= 0.80 and confidence >= 0.95:
        return "critical_label_conflict"
    if nearest >= 0.90 and opposite >= 0.80:
        return "strong_label_conflict"
    if opposite >= 0.80:
        return "moderate_label_conflict"
    return "review_required"


def review_lane(row: dict) -> str:
    error_type = row.get("error_type")
    level = suspicion_level(row)
    group = source_group_key(row)
    if error_type == "FP" and group.startswith("待加入白名单/") and level == "critical_label_conflict":
        return "A_whitelist_critical_fp"
    if error_type == "FP" and group.startswith("待加入白名单/") and level == "strong_label_conflict":
        return "B_whitelist_high_similarity_fp"
    if error_type == "FP" and group.startswith("待加入白名单/"):
        return "C_whitelist_remaining_fp"
    if error_type == "FN":
        return "D_malicious_batch_fn"
    return "Z_other"


def _compact_counter(counter: Counter[str]) -> str:
    return "|".join(f"{key}:{counter[key]}" for key in sorted(counter))


def _group_stats(rows: Sequence[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(source_group_key(row), []).append(row)
    stats = {}
    for key, items in grouped.items():
        stats[key] = {
            "size": len(items),
            "error_type_counts": _compact_counter(Counter(str(row.get("error_type", "")) for row in items)),
            "priority_counts": _compact_counter(Counter(str(row.get("priority", "")) for row in items)),
        }
    return stats


def review_question_focus(row: dict) -> str:
    lane = review_lane(row)
    if lane.startswith("A_") or lane.startswith("B_"):
        return "Check whether this benign-labeled file is actually in-scope benign, copied lineage, mislabeled malware, or source-list contamination."
    if lane.startswith("C_"):
        return "Check the white-list source policy and decide whether the current benign label is defensible."
    if lane.startswith("D_"):
        return "Check whether this malicious-labeled file is a true model blind spot, ambiguous source label, or broken/out-of-scope sample."
    return "Review label, scope, feature validity, and neighbor evidence."


def _lane_order(lane: str) -> int:
    order = {
        "A_whitelist_critical_fp": 0,
        "B_whitelist_high_similarity_fp": 1,
        "C_whitelist_remaining_fp": 2,
        "D_malicious_batch_fn": 3,
        "Z_other": 9,
    }
    return order.get(lane, 9)


def _sort_key(row: dict, stats: dict[str, dict]) -> tuple:
    lane = review_lane(row)
    group = source_group_key(row)
    group_size = int(stats[group]["size"])
    if lane.startswith("D_"):
        return (
            _lane_order(lane),
            -group_size,
            group,
            _safe_int(row.get("priority"), 999),
            -_opposite_ratio(row),
            -_confidence(row),
            row.get("source_path", ""),
        )
    return (
        _lane_order(lane),
        _safe_int(row.get("priority"), 999),
        -_similarity(row),
        -_opposite_ratio(row),
        -_confidence(row),
        row.get("source_path", ""),
    )


def build_queue(*, review_csv: Path, output_csv: Path, output_json: Path) -> dict:
    input_rows = read_rows(review_csv)
    stats = _group_stats(input_rows)
    rows = []
    for row in input_rows:
        group = source_group_key(row)
        item = dict(row)
        item["review_lane"] = review_lane(row)
        item["source_group_key"] = group
        item["source_group_size"] = stats[group]["size"]
        item["source_group_error_type_counts"] = stats[group]["error_type_counts"]
        item["source_group_priority_counts"] = stats[group]["priority_counts"]
        item["suspicion_level"] = suspicion_level(row)
        item["review_question_focus"] = review_question_focus(row)
        item["allowed_manual_label_verdicts"] = ALLOWED_VERDICTS
        item["allowed_recommended_actions"] = ALLOWED_ACTIONS
        item["replacement_rule"] = REPLACEMENT_RULE
        item["manual_label_verdict"] = ""
        item["manual_verdict_note"] = ""
        item["recommended_action"] = ""
        rows.append(item)

    rows.sort(key=lambda row: _sort_key(row, stats))
    for rank, row in enumerate(rows, start=1):
        row["review_priority_rank"] = rank

    write_rows(output_csv, rows)
    summary = {
        "schema": "axon_source_aware_adjudication_queue_v1",
        "review_csv": str(resolve_path(review_csv)),
        "rows": len(rows),
        "lane_counts": dict(sorted(Counter(row["review_lane"] for row in rows).items())),
        "suspicion_level_counts": dict(sorted(Counter(row["suspicion_level"] for row in rows).items())),
        "source_group_counts": dict(sorted(Counter(row["source_group_key"] for row in rows).items())),
        "manual_label_verdict_blank_count": sum(1 for row in rows if not row.get("manual_label_verdict")),
        "recommended_action_blank_count": sum(1 for row in rows if not row.get("recommended_action")),
        "outputs": {
            "queue_csv": str(resolve_path(output_csv)),
            "summary_json": str(resolve_path(output_json)),
        },
        "examples": rows[:20],
    }
    output_json = resolve_path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a source-aware manual adjudication queue.")
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_queue(
        review_csv=args.review_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
