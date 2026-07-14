#!/usr/bin/env python3
"""Build a posthoc failure audit for Loop160 accepted rows.

This report explains why Loop160 failed full-test after a small Test-10k gain.
It is intentionally read-only and posthoc: full-test rows are diagnostic only
and must not become model-selection, threshold-selection, feature-mask, signer
term, or replacement-sampling evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_FEATURE_POLICY = (
    "source_path/cache_path/source_sha256/sample_index/split are retained only in the private diagnostic map for "
    "alignment and audit. Public rows use synthetic focus ids and aggregate score buckets. Full-test observations "
    "are posthoc diagnostics only and are forbidden for model, threshold, feature-mask, signer-term, replacement, "
    "or production-inference selection."
)
PUBLIC_FIELDS = [
    "loop162_focus_id",
    "evaluation_split",
    "accepted_quality",
    "label",
    "baseline_prediction",
    "candidate_prediction",
    "loop160_prediction",
    "gate_score_bucket",
    "auth_status",
    "trusted_signer_downgrade",
]
PRIVATE_FIELDS = [
    "loop162_focus_id",
    "evaluation_split",
    "source_path",
    "cache_path",
    "source_sha256",
    "sample_index",
    "label",
    "loop160_gate_score",
    "loop160_selected_threshold",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_rows(path: Path) -> list[dict[str, str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_bool(value: object) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y"}


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def score_bucket(score: float) -> str:
    if score <= 0.10:
        return "<=0.10"
    if score <= 0.20:
        return "(0.10,0.20]"
    if score <= 0.25:
        return "(0.20,0.25]"
    if score <= 0.30:
        return "(0.25,0.30]"
    return ">0.30"


def accepted_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if normalize_bool(row.get("loop160_accept_candidate"))]


def summarize_split(
    *,
    split_name: str,
    rows: Sequence[dict[str, str]],
    public_rows: list[dict[str, Any]],
    private_rows: list[dict[str, Any]],
    focus_prefix: str,
) -> dict[str, Any]:
    accepted = accepted_rows(rows)
    quality_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    wrong_bucket_counts: Counter[str] = Counter()
    accepted_score_sum = 0.0
    wrong_score_sum = 0.0
    wrong_rows = 0

    for index, row in enumerate(accepted, start=1):
        label = str(row.get("label") or "").strip()
        loop160_pred = _int(row.get("loop160_prediction"))
        quality = "accepted_correct" if normalize_bool(row.get("loop160_correct")) else "accepted_wrong"
        score = _float(row.get("loop160_gate_score"))
        bucket = score_bucket(score)
        focus_id = f"{focus_prefix}_{split_name}_{index:06d}"
        quality_counts[quality] += 1
        label_counts[label] += 1
        bucket_counts[bucket] += 1
        accepted_score_sum += score
        if quality == "accepted_wrong":
            wrong_rows += 1
            wrong_score_sum += score
            wrong_bucket_counts[bucket] += 1

        public_rows.append(
            {
                "loop162_focus_id": focus_id,
                "evaluation_split": split_name,
                "accepted_quality": quality,
                "label": label,
                "baseline_prediction": row.get("trusted_signer_guard_prediction", row.get("prediction", "")),
                "candidate_prediction": row.get("loop160_candidate_prediction", ""),
                "loop160_prediction": str(loop160_pred),
                "gate_score_bucket": bucket,
                "auth_status": row.get("auth_status", ""),
                "trusted_signer_downgrade": row.get("trusted_signer_guard_downgrade", ""),
            }
        )
        private_rows.append(
            {
                "loop162_focus_id": focus_id,
                "evaluation_split": split_name,
                "source_path": row.get("source_path", ""),
                "cache_path": row.get("cache_path", ""),
                "source_sha256": row.get("source_sha256", ""),
                "sample_index": row.get("sample_index", ""),
                "label": label,
                "loop160_gate_score": row.get("loop160_gate_score", ""),
                "loop160_selected_threshold": row.get("loop160_selected_threshold", ""),
            }
        )

    return {
        "rows": len(rows),
        "accepted_rows": len(accepted),
        "accepted_correct": int(quality_counts["accepted_correct"]),
        "accepted_wrong": int(quality_counts["accepted_wrong"]),
        "accepted_label_counts": dict(sorted(label_counts.items())),
        "score_bucket_counts": dict(sorted(bucket_counts.items())),
        "wrong_score_bucket_counts": dict(sorted(wrong_bucket_counts.items())),
        "mean_accepted_score": accepted_score_sum / len(accepted) if accepted else 0.0,
        "mean_wrong_score": wrong_score_sum / wrong_rows if wrong_rows else 0.0,
    }


def build_loop162_posthoc(
    *,
    val_predictions_csv: Path,
    test10k_predictions_csv: Path,
    full_predictions_csv: Path,
    output_json: Path,
    output_public_csv: Path,
    output_private_map_csv: Path,
    output_md: Optional[Path] = None,
    focus_prefix: str = "loop162_loop160_posthoc",
) -> dict[str, Any]:
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    split_summaries = {
        "val": summarize_split(
            split_name="val",
            rows=read_rows(val_predictions_csv),
            public_rows=public_rows,
            private_rows=private_rows,
            focus_prefix=focus_prefix,
        ),
        "test10k": summarize_split(
            split_name="test10k",
            rows=read_rows(test10k_predictions_csv),
            public_rows=public_rows,
            private_rows=private_rows,
            focus_prefix=focus_prefix,
        ),
        "full_test": summarize_split(
            split_name="full_test",
            rows=read_rows(full_predictions_csv),
            public_rows=public_rows,
            private_rows=private_rows,
            focus_prefix=focus_prefix,
        ),
    }
    full = split_summaries["full_test"]
    test10k = split_summaries["test10k"]
    val = split_summaries["val"]
    payload = {
        "schema": "axon_loop162_loop160_failure_posthoc_v1",
        "protocol": (
            "Posthoc accepted-row diagnostic for Loop160; full-test rows are explanatory only and forbidden for "
            "future model or threshold selection"
        ),
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "inputs": {
            "val_predictions_csv": str(resolve_path(val_predictions_csv)),
            "test10k_predictions_csv": str(resolve_path(test10k_predictions_csv)),
            "full_predictions_csv": str(resolve_path(full_predictions_csv)),
        },
        "split_summaries": split_summaries,
        "failure_review": {
            "test10k_accepted_rows": test10k["accepted_rows"],
            "test10k_wrong_rows": test10k["accepted_wrong"],
            "full_test_accepted_rows": full["accepted_rows"],
            "full_test_wrong_rows": full["accepted_wrong"],
            "full_test_wrong_minus_correct": full["accepted_wrong"] - full["accepted_correct"],
            "interpretation": (
                "Val/Test-10k accepted rows were too sparse to estimate full-test FP spillover; "
                "full-test posthoc shows wrong accepted rows outnumbered correct accepted rows."
            ),
        },
        "selection_policy": {
            "may_select_model_or_threshold_from_this_report": False,
            "may_expand_signer_terms_from_this_report": False,
            "may_sample_replacements_from_this_report": False,
            "full_test_rows_are_posthoc_only": True,
        },
        "outputs": {
            "public_csv": str(resolve_path(output_public_csv)),
            "private_map_csv": str(resolve_path(output_private_map_csv)),
            "output_json": str(resolve_path(output_json)),
            "output_md": str(resolve_path(output_md)) if output_md else "",
        },
        "decision": "posthoc_failure_record_only",
        "next_action": (
            "Do not repeat probability-only R11 rescue. If recall rescue continues, require independent Val-side "
            "content/external evidence before Test-10k promotion."
        ),
    }
    write_rows(output_public_csv, public_rows, PUBLIC_FIELDS)
    write_rows(output_private_map_csv, private_rows, PRIVATE_FIELDS)
    write_json(output_json, payload)
    if output_md is not None:
        write_markdown(output_md, payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Loop162 Loop160 Failure Posthoc",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Val accepted correct/wrong: `{payload['split_summaries']['val']['accepted_correct']}` / `{payload['split_summaries']['val']['accepted_wrong']}`",
        f"- Test-10k accepted correct/wrong: `{payload['split_summaries']['test10k']['accepted_correct']}` / `{payload['split_summaries']['test10k']['accepted_wrong']}`",
        f"- Full-test accepted correct/wrong: `{payload['split_summaries']['full_test']['accepted_correct']}` / `{payload['split_summaries']['full_test']['accepted_wrong']}`",
        "",
        "## Policy",
        "",
        IDENTITY_FEATURE_POLICY,
        "",
    ]
    resolved.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop162 Loop160 failure posthoc diagnostic.")
    parser.add_argument("--val-predictions-csv", type=Path, required=True)
    parser.add_argument("--test10k-predictions-csv", type=Path, required=True)
    parser.add_argument("--full-predictions-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-public-csv", type=Path, required=True)
    parser.add_argument("--output-private-map-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_loop162_posthoc(
        val_predictions_csv=args.val_predictions_csv,
        test10k_predictions_csv=args.test10k_predictions_csv,
        full_predictions_csv=args.full_predictions_csv,
        output_json=args.output_json,
        output_public_csv=args.output_public_csv,
        output_private_map_csv=args.output_private_map_csv,
        output_md=args.output_md,
    )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "val_accepted_wrong": payload["split_summaries"]["val"]["accepted_wrong"],
                "test10k_accepted_wrong": payload["split_summaries"]["test10k"]["accepted_wrong"],
                "full_test_accepted_wrong": payload["split_summaries"]["full_test"]["accepted_wrong"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
