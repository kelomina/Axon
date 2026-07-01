#!/usr/bin/env python3
"""Build a corrected split from a manual adjustment plan.

This script is deliberately non-destructive. It reads an existing split and a
review-derived plan, applies safe train/val label fixes, and replaces excluded
rows with fresh same-label candidates that were not already present in the
split. It refuses to emit a short split. Test replacements are rejected by
default and require an explicit data-hygiene override.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
OUTPUT_FIELDNAMES = ["source_path", "label", "sample_index", "split"]
ALLOWED_PLAN_ACTIONS = {"keep_label", "relabel", "exclude_and_replace"}
BLOCKED_PLAN_ACTIONS = {"needs_manual_target_label", "held_out_test_verdict_only"}

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from apply_manual_review_verdicts import source_keys  # noqa: E402
from build_random_20w_split import is_valid_pe_sample, iter_sorted_files  # noqa: E402


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: object) -> bool:
    return str(value or "").strip().casefold() == "true"


def write_split_csv(path: Path, rows: Sequence[dict]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def split_key(row: dict) -> str:
    keys = source_keys(row)
    if keys:
        return keys[0]
    return f"path:{str(row.get('source_path', '')).strip().casefold()}"


def exact_row_key(row: dict) -> str:
    return f"{str(row.get('sample_index', '')).strip()}|{str(row.get('source_path', '')).strip().casefold()}"


def key_set(row: dict) -> set[str]:
    keys = set(source_keys(row))
    keys.add(split_key(row))
    return keys


def plan_has_exact_row_identity(row: dict) -> bool:
    return bool(str(row.get("sample_index", "")).strip() and str(row.get("source_path", "")).strip())


def lookup_plan(original: dict, exact_plan_by_key: dict[str, dict], loose_plan_by_key: dict[str, dict]) -> Optional[dict]:
    plan = exact_plan_by_key.get(exact_row_key(original))
    if plan is not None:
        return plan
    for key in source_keys(original):
        plan = loose_plan_by_key.get(key)
        if plan is not None:
            return plan
    return None


def _row_sort_key(row: dict) -> tuple[int, int]:
    split = str(row.get("split", ""))
    try:
        sample_index = int(row.get("sample_index", 0))
    except (TypeError, ValueError):
        sample_index = 0
    return SPLIT_ORDER.get(split, 99), sample_index


def summarize_split(rows: Sequence[dict]) -> dict:
    return {
        "rows": len(rows),
        "split_counts": dict(sorted(Counter(row.get("split", "") for row in rows).items())),
        "label_counts": dict(sorted(Counter(str(row.get("label", "")) for row in rows).items())),
        "label_split_counts": {
            split: dict(sorted(Counter(str(row.get("label", "")) for row in rows if row.get("split") == split).items()))
            for split in ["train", "val", "test"]
        },
    }


def validate_plan_rows(plan_rows: Sequence[dict], *, allow_test_replacements: bool = False) -> list[str]:
    failures = []
    for index, row in enumerate(plan_rows, start=1):
        action = str(row.get("plan_action", "")).strip()
        split = str(row.get("split", "")).strip()
        replacement_required = truthy(row.get("replacement_required"))
        usable_for_training = truthy(row.get("usable_for_training_policy"))
        planned_label = str(row.get("planned_label", "")).strip()
        replacement_label = str(row.get("replacement_label", "")).strip()
        row_id = row.get("sample_index") or row.get("source_path") or f"row-{index}"

        if split == "test":
            if not allow_test_replacements:
                failures.append(f"{row_id}: test split plan rows are not accepted by corrected split builder")
            elif action != "exclude_and_replace":
                failures.append(f"{row_id}: test split plan rows may only be exclude_and_replace")
            elif usable_for_training:
                failures.append(f"{row_id}: test replacement plan must not be marked usable for training policy")
        if action in BLOCKED_PLAN_ACTIONS:
            failures.append(f"{row_id}: unresolved or held-out plan action {action}")
        elif action not in ALLOWED_PLAN_ACTIONS:
            failures.append(f"{row_id}: unsupported plan action {action or '<blank>'}")

        if action == "relabel":
            if not usable_for_training:
                failures.append(f"{row_id}: relabel plan is not marked usable for training policy")
            if planned_label not in {"0", "1"}:
                failures.append(f"{row_id}: relabel plan is missing an explicit planned_label")

        if action == "exclude_and_replace":
            if not replacement_required:
                failures.append(f"{row_id}: exclude_and_replace plan is not marked replacement_required")
            if replacement_label not in {"0", "1"}:
                failures.append(f"{row_id}: replacement plan is missing replacement_label")
        elif replacement_required:
            failures.append(f"{row_id}: replacement_required is true for non-replacement action {action or '<blank>'}")
    return failures


def _candidate_rows_from_manifest(manifest_json: Path) -> Iterable[dict]:
    manifest = json.loads(resolve_path(manifest_json).read_text(encoding="utf-8"))
    for sample in manifest.get("samples", []):
        yield {
            "source_path": sample.get("source_path", ""),
            "label": str(sample.get("label", "")),
            "source_sha256": sample.get("source_sha256", ""),
        }


def _candidate_rows_from_roots(data_dir: Path, max_file_size: int) -> Iterable[dict]:
    data_dir = resolve_path(data_dir)
    roots = [(data_dir / "待加入白名单", 0), (data_dir / "待拉黑", 1)]
    for root, label in roots:
        if not root.exists():
            continue
        for path in iter_sorted_files(root):
            if path.is_file() and is_valid_pe_sample(path, max_file_size):
                yield {"source_path": str(path), "label": str(label), "source_sha256": ""}


def load_candidate_rows(
    *,
    candidate_csv: Optional[Path],
    manifest_json: Optional[Path],
    data_dir: Optional[Path],
    max_file_size: int,
) -> list[dict]:
    if candidate_csv is not None:
        rows = read_csv_rows(candidate_csv)
    elif manifest_json is not None:
        rows = list(_candidate_rows_from_manifest(manifest_json))
    elif data_dir is not None:
        rows = list(_candidate_rows_from_roots(data_dir, max_file_size))
    else:
        rows = []
    cleaned = []
    for row in rows:
        label = str(row.get("label", "")).strip()
        source_path = str(row.get("source_path", "")).strip()
        if label not in {"0", "1"} or not source_path:
            continue
        cleaned.append({"source_path": source_path, "label": label, "source_sha256": row.get("source_sha256", "")})
    return cleaned


def choose_replacements(
    *,
    candidate_rows: Sequence[dict],
    used_keys: set[str],
    forbidden_replacement_keys: set[str],
    replacement_requests: Sequence[dict],
    seed: int,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {"0": [], "1": []}
    seen_candidate_keys: set[str] = set()
    self_replacement_candidates_skipped = 0
    for row in candidate_rows:
        keys = key_set(row)
        if keys & forbidden_replacement_keys:
            self_replacement_candidates_skipped += 1
            continue
        if keys & used_keys or keys & seen_candidate_keys:
            continue
        label = str(row.get("label", ""))
        if label in by_label:
            by_label[label].append(row)
            seen_candidate_keys.update(keys)
    for rows in by_label.values():
        rng.shuffle(rows)

    selected: list[dict] = []
    shortfall: dict[str, int] = {}
    available_before = {label: len(rows) for label, rows in by_label.items()}
    for request in replacement_requests:
        label = str(request["replacement_label"])
        if not by_label.get(label):
            shortfall[label] = shortfall.get(label, 0) + 1
            continue
        candidate = by_label[label].pop()
        selected.append(
            {
                "source_path": candidate["source_path"],
                "label": label,
                "split": request["split"],
                "replacement_for_sample_index": request.get("sample_index", ""),
                "replacement_for_source_path": request.get("source_path", ""),
            }
        )
        used_keys.update(key_set(candidate))
    summary = {
        "candidate_available_before": available_before,
        "forbidden_replacement_key_count": len(forbidden_replacement_keys),
        "self_replacement_candidates_skipped": self_replacement_candidates_skipped,
        "replacement_requests": len(replacement_requests),
        "selected_replacements": len(selected),
        "shortfall": dict(sorted(shortfall.items())),
    }
    return selected, summary


def build_corrected_split(
    *,
    split_csv: Path,
    plan_csv: Path,
    candidate_csv: Optional[Path] = None,
    manifest_json: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    seed: int = 42,
    max_file_size: int = 1 * 1024 * 1024 * 1024,
    allow_test_replacements: bool = False,
) -> tuple[list[dict], dict]:
    original_rows = read_csv_rows(split_csv)
    plan_rows = read_csv_rows(plan_csv)
    plan_failures = validate_plan_rows(plan_rows, allow_test_replacements=allow_test_replacements)
    if plan_failures:
        raise ValueError(
            "Unsafe manual adjustment plan; fix the plan before building a corrected split: "
            + json.dumps(plan_failures, ensure_ascii=False)
        )
    exact_plan_by_key: dict[str, dict] = {}
    loose_plan_by_key: dict[str, dict] = {}
    for row in plan_rows:
        if plan_has_exact_row_identity(row):
            exact_plan_by_key.setdefault(exact_row_key(row), row)
        else:
            for key in source_keys(row):
                loose_plan_by_key.setdefault(key, row)

    kept_rows: list[dict] = []
    excluded_rows: list[dict] = []
    relabeled_rows: list[dict] = []
    used_keys: set[str] = set()

    for original in original_rows:
        plan = lookup_plan(original, exact_plan_by_key, loose_plan_by_key)
        if plan is None:
            kept = dict(original)
            kept_rows.append(kept)
            used_keys.update(key_set(kept))
            continue

        action = str(plan.get("plan_action", ""))
        if action == "relabel" and str(plan.get("usable_for_training_policy", "")).casefold() == "true":
            kept = dict(original)
            kept["label"] = str(plan.get("planned_label", kept.get("label", "")))
            kept_rows.append(kept)
            relabeled_rows.append(kept)
            used_keys.update(key_set(kept))
        elif str(plan.get("replacement_required", "")).casefold() == "true":
            excluded = dict(original)
            excluded_rows.append(excluded)
        else:
            kept = dict(original)
            kept_rows.append(kept)
            used_keys.update(key_set(kept))

    replacement_requests = []
    forbidden_replacement_keys: set[str] = set()
    for row in plan_rows:
        if str(row.get("replacement_required", "")).casefold() == "true":
            replacement_requests.append(row)
            forbidden_replacement_keys.update(key_set(row))

    candidate_rows = load_candidate_rows(
        candidate_csv=candidate_csv,
        manifest_json=manifest_json,
        data_dir=data_dir,
        max_file_size=max_file_size,
    )
    replacements, replacement_summary = choose_replacements(
        candidate_rows=candidate_rows,
        used_keys=used_keys,
        forbidden_replacement_keys=forbidden_replacement_keys,
        replacement_requests=replacement_requests,
        seed=seed,
    )
    if replacement_summary["shortfall"]:
        raise ValueError(
            "Not enough unused same-label replacement candidates: "
            + json.dumps(replacement_summary["shortfall"], ensure_ascii=False)
        )

    corrected_rows = kept_rows + [
        {
            "source_path": row["source_path"],
            "label": row["label"],
            "split": row["split"],
            "sample_index": "",
        }
        for row in replacements
    ]
    corrected_rows.sort(key=_row_sort_key)
    for index, row in enumerate(corrected_rows):
        row["sample_index"] = index

    if len(corrected_rows) != len(original_rows):
        raise ValueError(f"Corrected split row count changed: {len(corrected_rows)} != {len(original_rows)}")

    summary = {
        "schema": "axon_corrected_split_from_manual_plan_v1",
        "split_csv": str(resolve_path(split_csv)),
        "plan_csv": str(resolve_path(plan_csv)),
        "seed": int(seed),
        "allow_test_replacements": bool(allow_test_replacements),
        "original_summary": summarize_split(original_rows),
        "corrected_summary": summarize_split(corrected_rows),
        "plan_rows": len(plan_rows),
        "excluded_rows": len(excluded_rows),
        "relabeled_rows": len(relabeled_rows),
        "replacement_summary": replacement_summary,
        "notes": [
            "The original split is not modified.",
            "Excluded rows are replaced with unused same-label candidates.",
            "Replacement candidates that match an excluded row are skipped, even if they appear in the candidate source.",
            "The script refuses to emit a split with fewer rows than the original.",
        ],
    }
    return corrected_rows, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a corrected split from a manual adjustment plan.")
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--plan-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--candidate-csv", type=Path)
    source_group.add_argument("--manifest-json", type=Path)
    source_group.add_argument("--data-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-file-size", type=int, default=1 * 1024 * 1024 * 1024)
    parser.add_argument(
        "--allow-test-replacements",
        action="store_true",
        help="Allow explicit exclude-and-replace rows in the test split for data hygiene only.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows, summary = build_corrected_split(
        split_csv=args.split_csv,
        plan_csv=args.plan_csv,
        candidate_csv=args.candidate_csv,
        manifest_json=args.manifest_json,
        data_dir=args.data_dir,
        seed=args.seed,
        max_file_size=args.max_file_size,
        allow_test_replacements=bool(args.allow_test_replacements),
    )
    write_split_csv(args.output_csv, rows)
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    summary["outputs"] = {"csv": str(resolve_path(args.output_csv)), "json": str(output_json)}
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
