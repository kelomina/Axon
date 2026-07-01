#!/usr/bin/env python3
"""Convert manual review verdicts into a non-destructive split adjustment plan.

The script does not edit the original split, cache, or raw files. It records
which reviewed rows have enough human evidence to relabel, keep, quarantine, or
replace in a future regenerated split.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SPLITS = {"train", "val"}

KEEP_VERDICTS = {"label_correct", "correct", "keep", "benign_correct", "malicious_correct"}
RELABEL_VERDICTS = {"label_wrong", "wrong", "mislabeled", "flip_label"}
EXCLUDE_VERDICTS = {"out_of_scope", "feature_broken", "corrupt", "invalid_pe", "bad_feature"}
UNCERTAIN_VERDICTS = {"", "uncertain", "needs_more_evidence", "unknown", "review_later"}

KEEP_ACTIONS = {"", "keep_label", "model_blindspot", "needs_more_evidence"}
RELABEL_ACTIONS = {"relabel", "relabel_train_only", "flip_label"}
EXCLUDE_ACTIONS = {"exclude", "quarantine", "quarantine_source_group", "drop", "replace_sample", "resample"}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_text(value: object) -> str:
    return str(value or "").strip().casefold()


def normalize_source_path(value: object) -> str:
    return str(value or "").strip().casefold()


def source_path_stem_sha(row: dict) -> str:
    source_path = str(row.get("source_path") or "").strip()
    if not source_path:
        return ""
    name = Path(source_path).name.casefold()
    stem = Path(name).stem if "." in name else name
    if len(stem) == 64 and all(char in "0123456789abcdef" for char in stem):
        return stem
    return ""


def read_csv_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_key(row: dict) -> str:
    sha = normalize_text(row.get("source_sha256"))
    if sha:
        return f"sha:{sha}"
    source_path = normalize_source_path(row.get("source_path"))
    return f"path:{source_path}"


def source_keys(row: dict) -> list[str]:
    keys = []
    sha = normalize_text(row.get("source_sha256"))
    if sha:
        keys.append(f"sha:{sha}")

    source_path = normalize_source_path(row.get("source_path"))
    if source_path:
        keys.append(f"path:{source_path}")
        stem = source_path_stem_sha(row)
        if stem:
            keys.append(f"sha:{stem}")

    deduped = []
    seen = set()
    for key in keys:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def load_split_index(split_csv: Path) -> tuple[dict[str, dict[str, dict]], dict]:
    rows = read_csv_rows(split_csv)
    split_index: dict[str, dict[str, dict]] = {
        "by_sha": {},
        "by_path": {},
        "by_path_stem_sha": {},
    }
    for row in rows:
        sha = normalize_text(row.get("source_sha256"))
        source_path = normalize_source_path(row.get("source_path"))
        stem_sha = source_path_stem_sha(row)
        if sha:
            split_index["by_sha"].setdefault(sha, row)
        if source_path:
            split_index["by_path"].setdefault(source_path, row)
        if stem_sha and not sha:
            split_index["by_path_stem_sha"].setdefault(stem_sha, row)
    summary = {
        "rows": len(rows),
        "split_counts": dict(sorted(Counter(row.get("split", "") for row in rows).items())),
        "label_split_counts": {
            split: dict(sorted(Counter(row.get("label", "") for row in rows if row.get("split") == split).items()))
            for split in ["train", "val", "test"]
        },
    }
    return split_index, summary


def find_split_row(review_row: dict, split_index: dict[str, dict[str, dict]]) -> Optional[dict]:
    sha = normalize_text(review_row.get("source_sha256"))
    source_path = normalize_source_path(review_row.get("source_path"))
    stem_sha = source_path_stem_sha(review_row)

    if sha:
        split_row = split_index["by_sha"].get(sha)
        if split_row is not None:
            return split_row
    if source_path:
        split_row = split_index["by_path"].get(source_path)
        if split_row is not None:
            return split_row
    for fallback_sha in [sha, stem_sha]:
        if fallback_sha:
            split_row = split_index["by_path_stem_sha"].get(fallback_sha)
            if split_row is not None:
                return split_row
    return None


def classify_manual_row(row: dict) -> tuple[str, str]:
    verdict = normalize_text(row.get("manual_label_verdict"))
    action = normalize_text(row.get("recommended_action"))
    if verdict in UNCERTAIN_VERDICTS and action in KEEP_ACTIONS:
        return "ignore", "no_manual_decision"
    if verdict in EXCLUDE_VERDICTS or action in EXCLUDE_ACTIONS:
        return "exclude_and_replace", "manual_exclude_or_replace"
    if verdict in KEEP_VERDICTS and action not in EXCLUDE_ACTIONS and action not in RELABEL_ACTIONS:
        return "keep_label", "manual_label_kept"
    if verdict in RELABEL_VERDICTS or action in RELABEL_ACTIONS:
        return "relabel", "manual_label_wrong"
    if verdict in UNCERTAIN_VERDICTS or action in {"needs_more_evidence", "model_blindspot"}:
        return "ignore", "manual_uncertain"
    return "ignored_unknown_verdict", f"unknown verdict/action: {verdict}/{action}"


def infer_relabel_target(row: dict) -> Optional[int]:
    for key in ["corrected_label", "new_label", "target_label", "manual_label"]:
        value = normalize_text(row.get(key))
        if value in {"0", "benign", "white", "clean"}:
            return 0
        if value in {"1", "malicious", "black", "malware"}:
            return 1
    return None


def _plan_row(review_row: dict, split_row: dict, action: str, reason: str, allow_test_actions: bool) -> dict:
    split = split_row.get("split", "")
    label = int(split_row.get("label", review_row.get("label", 0)))
    row = {
        "source_path": split_row.get("source_path", review_row.get("source_path", "")),
        "source_sha256": review_row.get("source_sha256", ""),
        "sample_index": split_row.get("sample_index", review_row.get("sample_index", "")),
        "split": split,
        "original_label": label,
        "planned_label": label,
        "plan_action": action,
        "reason": reason,
        "manual_label_verdict": review_row.get("manual_label_verdict", ""),
        "recommended_action": review_row.get("recommended_action", ""),
        "manual_verdict_note": review_row.get("manual_verdict_note", ""),
        "replacement_required": "false",
        "replacement_label": "",
        "usable_for_training_policy": "true" if split in TRAINING_SPLITS else "false",
    }
    if split == "test" and not allow_test_actions:
        row["plan_action"] = "held_out_test_verdict_only"
        row["reason"] = "test split verdict withheld from training plan"
        row["usable_for_training_policy"] = "false"
        return row
    if action == "relabel":
        target = infer_relabel_target(review_row)
        if target is None:
            row["plan_action"] = "needs_manual_target_label"
            row["reason"] = "relabel verdict did not provide an inferable target label"
            row["usable_for_training_policy"] = "false"
        else:
            row["planned_label"] = target
    elif action == "exclude_and_replace":
        row["replacement_required"] = "true"
        row["replacement_label"] = str(label)
        row["usable_for_training_policy"] = "false"
    return row


def build_plan(
    *,
    review_csv: Path,
    split_csv: Path,
    allow_test_actions: bool = False,
) -> tuple[list[dict], dict]:
    review_rows = read_csv_rows(review_csv)
    split_index, split_summary = load_split_index(split_csv)
    planned_rows: list[dict] = []
    ignored_rows = 0
    missing_split_rows = 0
    unknown_rows = 0
    duplicate_review_keys: Counter[str] = Counter()
    seen_review_keys: set[str] = set()
    review_split_counts: Counter[str] = Counter()
    review_label_split_counts: Counter[str] = Counter()

    for review_row in review_rows:
        key = source_key(review_row)
        duplicate_review_keys[key] += 1
        if key in seen_review_keys:
            continue
        seen_review_keys.add(key)

        split_row = find_split_row(review_row, split_index)
        if split_row is None:
            missing_split_rows += 1
            continue
        split_name = str(split_row.get("split", ""))
        review_split_counts[split_name] += 1
        review_label_split_counts[f"{split_name}:{split_row.get('label', '')}"] += 1

        action, reason = classify_manual_row(review_row)
        if action == "ignore":
            ignored_rows += 1
            continue
        if action == "ignored_unknown_verdict":
            unknown_rows += 1
            continue
        planned_rows.append(_plan_row(review_row, split_row, action, reason, allow_test_actions))

    action_counts = Counter(row["plan_action"] for row in planned_rows)
    replacement_counts = Counter(row["replacement_label"] for row in planned_rows if row["replacement_required"] == "true")
    split_action_counts = Counter(f"{row['split']}:{row['plan_action']}" for row in planned_rows)
    train_policy_rows = [row for row in planned_rows if row["split"] in TRAINING_SPLITS and row["usable_for_training_policy"] == "true"]
    summary = {
        "schema": "axon_manual_review_adjustment_plan_v1",
        "review_csv": str(resolve_path(review_csv)),
        "split_csv": str(resolve_path(split_csv)),
        "allow_test_actions": bool(allow_test_actions),
        "split_summary": split_summary,
        "review_rows": len(review_rows),
        "planned_rows": len(planned_rows),
        "ignored_rows": ignored_rows,
        "unknown_verdict_rows": unknown_rows,
        "missing_split_rows": missing_split_rows,
        "duplicate_review_rows": int(sum(count - 1 for count in duplicate_review_keys.values() if count > 1)),
        "review_split_counts": dict(sorted(review_split_counts.items())),
        "review_label_split_counts": dict(sorted(review_label_split_counts.items())),
        "review_rows_in_test_split": int(review_split_counts.get("test", 0)),
        "action_counts": dict(sorted(action_counts.items())),
        "split_action_counts": dict(sorted(split_action_counts.items())),
        "replacement_required": int(sum(row["replacement_required"] == "true" for row in planned_rows)),
        "replacement_counts_by_original_label": dict(sorted(replacement_counts.items())),
        "training_policy_rows": len(train_policy_rows),
        "notes": [
            "This plan is non-destructive; it does not edit the original split, cache, or raw files.",
            "Excluded or feature-broken rows require fresh replacement sampling; they are not used to fill their own slots.",
            "Test split verdicts are withheld from train/val policy unless --allow-test-actions is explicitly set.",
        ],
    }
    return planned_rows, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply manual review verdicts into a non-destructive plan.")
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--allow-test-actions",
        action="store_true",
        help="Allow test-split verdicts to appear as actions. Default withholds them from train/val policy.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows, summary = build_plan(
        review_csv=args.review_csv,
        split_csv=args.split_csv,
        allow_test_actions=bool(args.allow_test_actions),
    )
    fieldnames = [
        "source_path",
        "source_sha256",
        "sample_index",
        "split",
        "original_label",
        "planned_label",
        "plan_action",
        "reason",
        "manual_label_verdict",
        "recommended_action",
        "manual_verdict_note",
        "replacement_required",
        "replacement_label",
        "usable_for_training_policy",
    ]
    write_csv_rows(args.output_csv, rows, fieldnames)
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    summary["outputs"] = {"csv": str(resolve_path(args.output_csv)), "json": str(output_json)}
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
