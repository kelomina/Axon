#!/usr/bin/env python3
"""Build a single full-queue evidence intake CSV for Loop87 verdict import."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence


IDENTITY_FEATURE_POLICY = (
    "filename/path/extension/directory/source_sha256/cache_path/sample_index/split/row order are loading, "
    "alignment, cache-audit, duplicate-review, and manual-index fields only; they are not model evidence, "
    "verdict evidence, replacement sampling keys, or threshold/fusion inputs"
)
PROTOCOL = (
    "read-only full-queue verdict intake builder; no model fitting, no threshold selection, no automatic "
    "relabeling, no replacement sampling, no split/cache mutation"
)
REQUIRED_EVIDENCE_COLUMNS = [
    "sample_index",
    "split",
    "label",
    "loop57_error_type",
    "manual_label_verdict",
    "manual_verdict_note",
    "recommended_action",
]
OUTPUT_EXTRAS = ["loop95_wave_id", "loop95_wave_row_number", "loop95_intake_row_number"]


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_text(value: object) -> str:
    return str(value or "").strip()


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _parse_wave_spec(spec: str) -> tuple[int, Path]:
    parts = spec.split("=", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid --wave spec, expected WAVE_ID=evidence.csv: {spec}")
    wave_id = _int(parts[0], -1)
    if wave_id <= 0:
        raise ValueError(f"Invalid wave id in --wave spec: {spec}")
    return wave_id, Path(parts[1].strip())


def _merge_fieldnames(fieldnames: list[str], incoming: Sequence[str]) -> list[str]:
    seen = set(fieldnames)
    for field in incoming:
        if field not in seen:
            fieldnames.append(field)
            seen.add(field)
    return fieldnames


def _loop72_wave_index(loop72_rows: Sequence[dict[str, str]]) -> dict[int, set[str]]:
    index: dict[int, set[str]] = defaultdict(set)
    for row in loop72_rows:
        wave_id = _int(row.get("review_wave_id"), -1)
        sample_index = normalize_text(row.get("sample_index"))
        if wave_id > 0 and sample_index:
            index[wave_id].add(sample_index)
    return dict(index)


def _summary_wave_rows(multiwave_summary: dict[str, Any]) -> dict[int, int]:
    result: dict[int, int] = {}
    for row in multiwave_summary.get("wave_reports", []):
        wave_id = _int(row.get("wave_id"), -1)
        if wave_id > 0:
            result[wave_id] = _int(row.get("rows"))
    return result


def build_intake(
    *,
    loop72_plan_csv: Path,
    multiwave_summary_json: Path,
    waves: Sequence[tuple[int, Path]],
    output_csv: Path,
    output_json: Path,
    expected_rows: Optional[int] = None,
) -> dict[str, Any]:
    loop72_rows, loop72_fieldnames = read_csv_rows(loop72_plan_csv)
    loop72_by_wave = _loop72_wave_index(loop72_rows)
    multiwave = read_json(multiwave_summary_json)
    summary_wave_rows = _summary_wave_rows(multiwave)
    summary_combined_rows = _int(multiwave.get("combined", {}).get("rows"))
    queue_rows = _int(multiwave.get("combined", {}).get("queue_rows"))
    expected_rows = expected_rows if expected_rows is not None else summary_combined_rows or len(loop72_rows)

    blockers: list[str] = []
    warnings: list[str] = []
    if "review_wave_id" not in loop72_fieldnames:
        blockers.append("loop72_plan_missing_review_wave_id")
    if "sample_index" not in loop72_fieldnames:
        blockers.append("loop72_plan_missing_sample_index")

    wave_ids = [wave_id for wave_id, _path in waves]
    duplicated_wave_ids = sorted([wave_id for wave_id, count in Counter(wave_ids).items() if count > 1])
    if duplicated_wave_ids:
        blockers.extend(f"duplicate_wave_id:{wave_id}" for wave_id in duplicated_wave_ids)

    combined_rows: list[dict[str, Any]] = []
    output_fieldnames: list[str] = []
    sample_index_counts: Counter[str] = Counter()
    source_sha_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    error_type_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    missing_columns_by_wave: dict[str, list[str]] = {}
    wave_reports: list[dict[str, Any]] = []

    for wave_id, evidence_csv in sorted(waves, key=lambda item: item[0]):
        rows, fieldnames = read_csv_rows(evidence_csv)
        output_fieldnames = _merge_fieldnames(output_fieldnames, fieldnames)
        missing_columns = [field for field in REQUIRED_EVIDENCE_COLUMNS if field not in fieldnames]
        if missing_columns:
            missing_columns_by_wave[str(wave_id)] = missing_columns
            blockers.append(f"wave{wave_id}_missing_required_evidence_columns")

        expected_wave_indices = loop72_by_wave.get(wave_id, set())
        actual_wave_indices = {normalize_text(row.get("sample_index")) for row in rows if normalize_text(row.get("sample_index"))}
        missing_from_evidence = sorted(expected_wave_indices - actual_wave_indices)
        unexpected_in_evidence = sorted(actual_wave_indices - expected_wave_indices)
        if len(rows) != len(expected_wave_indices):
            blockers.append(f"wave{wave_id}_row_count_mismatch_loop72")
        if summary_wave_rows and len(rows) != summary_wave_rows.get(wave_id, len(rows)):
            blockers.append(f"wave{wave_id}_row_count_mismatch_multiwave_summary")
        if missing_from_evidence:
            blockers.append(f"wave{wave_id}_missing_loop72_sample_index")
        if unexpected_in_evidence:
            blockers.append(f"wave{wave_id}_unexpected_sample_index")

        for row_number, row in enumerate(rows, start=1):
            sample_index = normalize_text(row.get("sample_index"))
            source_sha = normalize_text(row.get("source_sha256")).lower()
            split = normalize_text(row.get("split")) or "blank"
            label = normalize_text(row.get("label")) or "blank"
            error_type = normalize_text(row.get("loop57_error_type")) or "blank"
            verdict = normalize_text(row.get("manual_label_verdict")) or "blank"
            action = normalize_text(row.get("recommended_action")) or "blank"

            if not sample_index:
                blockers.append(f"wave{wave_id}_missing_sample_index")
            else:
                sample_index_counts[sample_index] += 1
            if source_sha:
                source_sha_counts[source_sha] += 1
            split_counts[split] += 1
            label_counts[label] += 1
            error_type_counts[error_type] += 1
            verdict_counts[verdict] += 1
            action_counts[action] += 1

            item = dict(row)
            item["loop95_wave_id"] = str(wave_id)
            item["loop95_wave_row_number"] = str(row_number)
            item["loop95_intake_row_number"] = str(len(combined_rows) + 1)
            combined_rows.append(item)

        wave_reports.append(
            {
                "wave_id": wave_id,
                "evidence_csv": str(evidence_csv),
                "rows": len(rows),
                "loop72_expected_rows": len(expected_wave_indices),
                "multiwave_summary_rows": summary_wave_rows.get(wave_id),
                "missing_loop72_sample_index_count": len(missing_from_evidence),
                "unexpected_sample_index_count": len(unexpected_in_evidence),
                "missing_required_columns": missing_columns,
            }
        )

    duplicate_sample_indices = {
        key: count for key, count in sorted(sample_index_counts.items()) if count > 1
    }
    if duplicate_sample_indices:
        blockers.append("duplicate_sample_index_across_intake")

    duplicate_source_sha_groups = {
        key: count for key, count in sorted(source_sha_counts.items()) if count > 1
    }
    if duplicate_source_sha_groups:
        warnings.append("duplicate_source_sha_groups_require_group_review")

    if len(combined_rows) != expected_rows:
        blockers.append("combined_row_count_mismatch_expected")
    if queue_rows and expected_rows != queue_rows:
        warnings.append("expected_rows_differs_from_queue_rows")
    if summary_combined_rows and len(combined_rows) != summary_combined_rows:
        blockers.append("combined_row_count_mismatch_multiwave_summary")

    output_fieldnames = _merge_fieldnames(output_fieldnames, OUTPUT_EXTRAS)
    write_csv_rows(output_csv, combined_rows, output_fieldnames)

    summary = {
        "schema": "axon_loop95_full_queue_verdict_intake_v1",
        "protocol": PROTOCOL,
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "inputs": {
            "loop72_plan_csv": str(loop72_plan_csv),
            "multiwave_summary_json": str(multiwave_summary_json),
            "waves": [{"wave_id": wave_id, "evidence_csv": str(path)} for wave_id, path in sorted(waves)],
        },
        "rows": len(combined_rows),
        "expected_rows": expected_rows,
        "queue_rows": queue_rows,
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "missing_required_columns_by_wave": missing_columns_by_wave,
        "wave_reports": wave_reports,
        "counts": {
            "split_counts": dict(sorted(split_counts.items())),
            "label_counts": dict(sorted(label_counts.items())),
            "loop57_error_type_counts": dict(sorted(error_type_counts.items())),
            "manual_label_verdict_counts": dict(sorted(verdict_counts.items())),
            "recommended_action_counts": dict(sorted(action_counts.items())),
            "duplicate_sample_index_rows": int(sum(count - 1 for count in sample_index_counts.values() if count > 1)),
            "duplicate_sample_index_examples": [
                {"sample_index": key, "count": count}
                for key, count in list(duplicate_sample_indices.items())[:20]
            ],
            "duplicate_source_sha_group_count": len(duplicate_source_sha_groups),
            "duplicate_source_sha_row_count": int(sum(duplicate_source_sha_groups.values())),
        },
        "decisions": {
            "automatic_relabel_allowed": False,
            "automatic_replacement_allowed": False,
            "training_allowed": False,
            "test10k_allowed": False,
            "full_test_allowed": False,
            "ready_for_loop87_full_queue_import": not blockers,
            "next_allowed_step": (
                "run Loop87 verdict gate on the combined intake CSV"
                if not blockers
                else "fix intake blockers before Loop87 import"
            ),
        },
        "outputs": {
            "combined_evidence_csv": str(output_csv),
            "summary_json": str(output_json),
        },
        "notes": [
            "sample_index is used only to prove the evidence intake covers the same Loop72 rows without folding duplicates.",
            "source_sha256 duplicates are counted for group review but are not blocked by themselves and are not verdict evidence.",
            "This builder does not validate verdict quality; Loop87 remains the strict manual/external verdict gate.",
        ],
    }
    write_json(output_json, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build full-queue Loop87 verdict intake from wave evidence CSVs.")
    parser.add_argument("--loop72-plan-csv", type=Path, required=True)
    parser.add_argument("--multiwave-summary-json", type=Path, required=True)
    parser.add_argument(
        "--wave",
        action="append",
        default=[],
        help="Wave spec in the form WAVE_ID=evidence_package.csv",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_intake(
        loop72_plan_csv=args.loop72_plan_csv,
        multiwave_summary_json=args.multiwave_summary_json,
        waves=[_parse_wave_spec(spec) for spec in args.wave],
        output_csv=args.output_csv,
        output_json=args.output_json,
        expected_rows=args.expected_rows,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not summary["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
