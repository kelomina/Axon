#!/usr/bin/env python3
"""Build a read-only human adjudication guide from a readiness CSV.

The output is intentionally advisory. It does not emit manual_label_verdict or
recommended_action columns, because those fields must be filled by a human or a
business review process in the original review CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GUIDE_FIELDNAMES = [
    "review_rank",
    "priority",
    "error_type",
    "source_path",
    "source_sha256",
    "current_label",
    "model_prediction",
    "prob_malicious",
    "base_prob_malicious",
    "nearest_similarity",
    "opposite_label_ratio",
    "neighbor_label_counts",
    "top5_neighbor_labels",
    "top5_neighbor_similarities",
    "top5_neighbor_paths",
    "source_exists",
    "source_sha256_ok",
    "cache_exists",
    "npz_shape_ok",
    "is_pe",
    "file_size",
    "machine",
    "number_of_sections",
    "section_names",
    "max_section_entropy",
    "overlay_size",
    "manual_review_ready",
    "readiness_reasons",
    "guide_suspicion_level",
    "guide_primary_evidence",
    "guide_review_questions",
    "allowed_manual_label_verdicts",
    "allowed_recommended_actions",
    "guide_replacement_rule",
]

ALLOWED_VERDICTS = "label_correct|label_wrong|out_of_scope|feature_broken|uncertain"
ALLOWED_ACTIONS = "keep_label|relabel_train_only|replace_sample|quarantine_source_group|needs_more_evidence|model_blindspot"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _truthy(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def suspicion_level(row: dict) -> str:
    nearest = _float(row, "nearest_similarity")
    opposite = _float(row, "opposite_label_ratio")
    prob = _float(row, "prob_malicious")
    error_type = str(row.get("error_type", ""))
    high_confidence = (error_type == "FP" and prob >= 0.95) or (error_type == "FN" and prob <= 0.05)
    if nearest >= 0.95 and opposite >= 0.8 and high_confidence:
        return "critical_label_conflict"
    if nearest >= 0.90 and opposite >= 0.8:
        return "strong_label_conflict"
    if opposite >= 0.6:
        return "moderate_label_conflict"
    return "review_required"


def primary_evidence(row: dict) -> str:
    pieces = [
        f"error_type={row.get('error_type', '')}",
        f"current_label={row.get('label', '')}",
        f"model_prediction={row.get('prediction', '')}",
        f"prob_malicious={row.get('prob_malicious', '')}",
        f"nearest_similarity={row.get('nearest_similarity', '')}",
        f"opposite_label_ratio={row.get('opposite_label_ratio', '')}",
        f"neighbor_label_counts={row.get('neighbor_label_counts', '')}",
        f"is_pe={row.get('is_pe', '')}",
        f"cache_ready={row.get('manual_review_ready', '')}",
    ]
    return "; ".join(pieces)


def review_questions(row: dict) -> str:
    label = str(row.get("label", ""))
    prediction = str(row.get("prediction", ""))
    questions = [
        "Does source/business evidence support the current label?",
        "Do top neighbors indicate same family or copied binary lineage?",
        "Is the file an in-scope valid PE sample for this experiment?",
    ]
    if label != prediction:
        questions.append("If the dataset label is wrong, should this row be relabeled or replaced?")
    questions.append("If source or features are broken, choose feature_broken plus replace_sample.")
    return " | ".join(questions)


def guide_row(row: dict) -> dict:
    return {
        "review_rank": row.get("review_rank", ""),
        "priority": row.get("priority", ""),
        "error_type": row.get("error_type", ""),
        "source_path": row.get("source_path", ""),
        "source_sha256": row.get("source_sha256", ""),
        "current_label": row.get("label", ""),
        "model_prediction": row.get("prediction", ""),
        "prob_malicious": row.get("prob_malicious", ""),
        "base_prob_malicious": row.get("base_prob_malicious", ""),
        "nearest_similarity": row.get("nearest_similarity", ""),
        "opposite_label_ratio": row.get("opposite_label_ratio", ""),
        "neighbor_label_counts": row.get("neighbor_label_counts", ""),
        "top5_neighbor_labels": row.get("top5_neighbor_labels", ""),
        "top5_neighbor_similarities": row.get("top5_neighbor_similarities", ""),
        "top5_neighbor_paths": row.get("top5_neighbor_paths", ""),
        "source_exists": row.get("source_exists", ""),
        "source_sha256_ok": row.get("source_sha256_ok", ""),
        "cache_exists": row.get("cache_exists", ""),
        "npz_shape_ok": row.get("npz_shape_ok", ""),
        "is_pe": row.get("is_pe", ""),
        "file_size": row.get("file_size", ""),
        "machine": row.get("machine", ""),
        "number_of_sections": row.get("number_of_sections", ""),
        "section_names": row.get("section_names", ""),
        "max_section_entropy": row.get("max_section_entropy", ""),
        "overlay_size": row.get("overlay_size", ""),
        "manual_review_ready": row.get("manual_review_ready", ""),
        "readiness_reasons": row.get("readiness_reasons", ""),
        "guide_suspicion_level": suspicion_level(row),
        "guide_primary_evidence": primary_evidence(row),
        "guide_review_questions": review_questions(row),
        "allowed_manual_label_verdicts": ALLOWED_VERDICTS,
        "allowed_recommended_actions": ALLOWED_ACTIONS,
        "guide_replacement_rule": "feature_broken/out_of_scope rows must be excluded and replaced with fresh valid same-label candidates; never self-fill.",
    }


def write_csv_rows(path: Path, rows: Sequence[dict]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GUIDE_FIELDNAMES, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict, rows: Sequence[dict], max_rows: int) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manual Review Adjudication Guide",
        "",
        "This guide is read-only evidence for human adjudication. It does not fill manual verdict fields.",
        "",
        "## Summary",
        "",
        f"- Source readiness CSV: `{summary['readiness_csv']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Manual-review ready rows: `{summary['manual_review_ready_rows']}`",
        f"- Suspicion levels: `{summary['suspicion_level_counts']}`",
        f"- Error types: `{summary['error_type_counts']}`",
        "",
        "## Fill Rules",
        "",
        f"- Allowed `manual_label_verdict`: `{ALLOWED_VERDICTS}`",
        f"- Allowed `recommended_action`: `{ALLOWED_ACTIONS}`",
        "- `label_wrong` pairs with `relabel_train_only`.",
        "- `feature_broken` or `out_of_scope` pairs with `replace_sample` or `quarantine_source_group`.",
        "- Bad or out-of-scope samples must be replaced with fresh valid candidates; do not reuse the same file as its replacement.",
        "",
        "## Top Rows",
        "",
    ]
    for row in rows[: max(0, int(max_rows))]:
        lines.extend(
            [
                f"### Rank {row['review_rank']}",
                "",
                f"- Suspicion: `{row['guide_suspicion_level']}`",
                f"- Source: `{row['source_path']}`",
                f"- Evidence: {row['guide_primary_evidence']}",
                f"- Questions: {row['guide_review_questions']}",
                "",
            ]
        )
    resolved.write_text("\n".join(lines), encoding="utf-8")


def build_guide(
    *,
    readiness_csv: Path,
    output_csv: Path,
    output_json: Path,
    output_md: Optional[Path] = None,
    markdown_rows: int = 20,
) -> dict:
    input_rows = read_csv_rows(readiness_csv)
    guide_rows = [guide_row(row) for row in input_rows]
    write_csv_rows(output_csv, guide_rows)

    suspicious = Counter(row["guide_suspicion_level"] for row in guide_rows)
    error_types = Counter(row["error_type"] for row in guide_rows)
    ready_rows = sum(1 for row in guide_rows if _truthy(row["manual_review_ready"]))
    summary = {
        "schema": "axon_manual_review_adjudication_guide_v1",
        "readiness_csv": str(resolve_path(readiness_csv)),
        "rows": len(guide_rows),
        "manual_review_ready_rows": ready_rows,
        "manual_review_not_ready_rows": len(guide_rows) - ready_rows,
        "suspicion_level_counts": dict(sorted(suspicious.items())),
        "error_type_counts": dict(sorted(error_types.items())),
        "output_csv": str(resolve_path(output_csv)),
        "output_json": str(resolve_path(output_json)),
        "output_md": str(resolve_path(output_md)) if output_md is not None else None,
        "notes": [
            "This guide is read-only and does not contain manual_label_verdict or recommended_action columns.",
            "Humans must fill verdict/action fields in the original manual review CSV, then rerun readiness.",
        ],
    }
    output_json = resolve_path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_md is not None:
        write_markdown(output_md, summary, guide_rows, markdown_rows)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a read-only adjudication guide from manual-review readiness CSV.")
    parser.add_argument("--readiness-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--markdown-rows", type=int, default=20)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_guide(
        readiness_csv=args.readiness_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
        output_md=args.output_md,
        markdown_rows=int(args.markdown_rows),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
