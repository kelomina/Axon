#!/usr/bin/env python3
"""Build a manual adjudication queue for Loop28 high-confidence conflicts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEVERE_HIGH_BUCKETS = {
    "severe_fn_conflict_prob_le_0.01",
    "high_fn_conflict_prob_le_0.05",
    "severe_fp_conflict_prob_ge_0.99",
    "high_fp_conflict_prob_ge_0.95",
}

ALLOWED_VERDICTS = "label_correct|label_wrong|feature_broken|out_of_scope|uncertain"
ALLOWED_ACTIONS = "keep_sample|replace_with_fresh_same_label_candidate|quarantine_for_more_evidence|model_blindspot"
REPLACEMENT_RULE = (
    "If feature_broken/out_of_scope/label_wrong is confirmed, do not fill from this row. "
    "Re-sample one fresh valid candidate from the same label pool and preserve the exact 200000-row split."
)

FIELDNAMES = [
    "review_priority_rank",
    "review_lane",
    "conflict_bucket",
    "source_sha256_group_size",
    "duplicate_sha_group",
    "source_group_key",
    "source_group_size",
    "source_group_error_type_counts",
    "allowed_manual_label_verdicts",
    "allowed_recommended_actions",
    "replacement_rule",
    "source_path",
    "source_sha256",
    "sample_index",
    "label",
    "loop28_error_type",
    "loop28_score",
    "loop37_score",
    "byte_ngram_score",
    "loop26_blend_score",
    "corrected_by_loop37",
    "corrected_by_byte_ngram",
    "corrected_by_loop26_blend",
    "corrected_by_any_compared_model",
    "manual_label_verdict",
    "manual_verdict_note",
    "recommended_action",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[dict]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore", lineterminator="\n")
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


def _date_key(path_text: str) -> tuple[str, str]:
    month_match = re.search(r"20\d{2}-\d{2}", str(path_text or ""))
    date_match = re.search(r"20\d{2}-\d{2}-\d{2}", str(path_text or ""))
    return (
        month_match.group(0) if month_match else "<none>",
        date_match.group(0) if date_match else "<none>",
    )


def source_group_key(row: dict) -> str:
    parts = _after_data_parts(row.get("source_path", ""))
    if not parts:
        return "<unknown>"
    data_dir = parts[0]
    month, date = _date_key(row.get("source_path", ""))
    if date != "<none>":
        return f"{data_dir}/{month}/{date}"
    if data_dir in {"待加入白名单", "benign"}:
        return f"{data_dir}/<flat>"
    return "/".join(parts[: min(3, len(parts))])


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_text(value: object) -> bool:
    return str(value).strip().casefold() == "true"


def review_lane(row: dict) -> str:
    bucket = row.get("noise_bucket", "")
    corrected = _bool_text(row.get("corrected_by_any_compared_model"))
    if bucket.startswith("severe_") and not corrected:
        return "A_unfixed_severe_conflict"
    if bucket.startswith("severe_"):
        return "B_corrected_severe_conflict"
    if bucket.startswith("high_") and not corrected:
        return "C_unfixed_high_conflict"
    return "D_corrected_high_conflict"


def _lane_order(lane: str) -> int:
    order = {
        "A_unfixed_severe_conflict": 0,
        "B_corrected_severe_conflict": 1,
        "C_unfixed_high_conflict": 2,
        "D_corrected_high_conflict": 3,
    }
    return order.get(lane, 9)


def _confidence(row: dict) -> float:
    score = _safe_float(row.get("loop28_score"))
    if row.get("loop28_error_type") == "FN":
        return 1.0 - score
    return score


def _group_stats(rows: Sequence[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(source_group_key(row), []).append(row)
    stats = {}
    for key, items in grouped.items():
        stats[key] = {
            "size": len(items),
            "error_type_counts": "|".join(
                f"{name}:{count}" for name, count in sorted(Counter(row.get("loop28_error_type", "") for row in items).items())
            ),
        }
    return stats


def _sha_group_sizes(rows: Sequence[dict]) -> Counter[str]:
    return Counter(str(row.get("source_sha256") or "").casefold() for row in rows if row.get("source_sha256"))


def build_queue(*, residual_csv: Path, output_csv: Path, output_json: Path) -> dict:
    rows = [row for row in read_rows(residual_csv) if row.get("noise_bucket") in SEVERE_HIGH_BUCKETS]
    stats = _group_stats(rows)
    sha_group_sizes = _sha_group_sizes(rows)
    output_rows = []
    for row in rows:
        group = source_group_key(row)
        sha = str(row.get("source_sha256") or "").casefold()
        sha_group_size = int(sha_group_sizes.get(sha, 0)) if sha else 0
        item = dict(row)
        item["conflict_bucket"] = row.get("noise_bucket", "")
        item["source_sha256_group_size"] = sha_group_size
        item["duplicate_sha_group"] = sha_group_size > 1
        item["review_lane"] = review_lane(row)
        item["source_group_key"] = group
        item["source_group_size"] = stats[group]["size"]
        item["source_group_error_type_counts"] = stats[group]["error_type_counts"]
        item["allowed_manual_label_verdicts"] = ALLOWED_VERDICTS
        item["allowed_recommended_actions"] = ALLOWED_ACTIONS
        item["replacement_rule"] = REPLACEMENT_RULE
        item["manual_label_verdict"] = ""
        item["manual_verdict_note"] = ""
        item["recommended_action"] = ""
        output_rows.append(item)

    output_rows.sort(
        key=lambda row: (
            _lane_order(row["review_lane"]),
            -int(row["source_group_size"]),
            row["source_group_key"],
            -_confidence(row),
            row.get("source_sha256", ""),
        )
    )
    for rank, row in enumerate(output_rows, start=1):
        row["review_priority_rank"] = rank

    write_rows(output_csv, output_rows)
    summary = {
        "schema": "axon_loop28_conflict_adjudication_queue_v1",
        "residual_csv": str(resolve_path(residual_csv)),
        "rows": len(output_rows),
        "lane_counts": dict(sorted(Counter(row["review_lane"] for row in output_rows).items())),
        "conflict_bucket_counts": dict(sorted(Counter(row["conflict_bucket"] for row in output_rows).items())),
        "error_type_counts": dict(sorted(Counter(row.get("loop28_error_type", "") for row in output_rows).items())),
        "duplicate_sha_groups": sum(1 for count in sha_group_sizes.values() if count > 1),
        "duplicate_sha_extra_rows": sum(count - 1 for count in sha_group_sizes.values() if count > 1),
        "corrected_by_any_compared_model_counts": dict(
            sorted(Counter(str(_bool_text(row.get("corrected_by_any_compared_model"))) for row in output_rows).items())
        ),
        "manual_label_verdict_blank_count": sum(1 for row in output_rows if not row.get("manual_label_verdict")),
        "recommended_action_blank_count": sum(1 for row in output_rows if not row.get("recommended_action")),
        "replacement_rule": REPLACEMENT_RULE,
        "outputs": {
            "queue_csv": str(resolve_path(output_csv)),
            "summary_json": str(resolve_path(output_json)),
        },
        "examples": output_rows[:20],
    }
    resolved_json = resolve_path(output_json)
    resolved_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop28 high-confidence conflict adjudication queue.")
    parser.add_argument("--residual-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_queue(
        residual_csv=args.residual_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
