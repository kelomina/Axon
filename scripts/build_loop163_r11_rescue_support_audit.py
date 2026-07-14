#!/usr/bin/env python3
"""Audit support size for R11 rescue selector work.

Loop159/160 showed that R11-style recall rescue is tempting but unstable. This
guard counts the actual disagreement rows available on Val/Test-10k and checks
whether there is enough Val support to justify another selector or rule search.
Full-test rows, when provided, are posthoc-only diagnostics.
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
    "source_path/cache_path/source_sha256/sample_index/split are private alignment fields only. "
    "Loop163 public rows use synthetic ids and bucketed non-identity scores. Full-test disagreement rows are "
    "posthoc diagnostics only and must not select thresholds, rules, feature masks, signer terms, replacements, "
    "or production inference behavior."
)
PUBLIC_FIELDS = [
    "loop163_focus_id",
    "evaluation_split",
    "direction",
    "label",
    "base_prediction",
    "candidate_prediction",
    "candidate_outcome",
    "baseline_prob_bucket",
    "selector_score_bucket",
]
PRIVATE_FIELDS = [
    "loop163_focus_id",
    "evaluation_split",
    "source_path",
    "cache_path",
    "source_sha256",
    "sample_index",
    "label",
    "baseline_prob_malicious",
    "selector_score",
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


def _key(row: dict[str, str]) -> tuple[str, str]:
    return (str(row.get("sample_index") or "").strip(), str(row.get("source_sha256") or "").strip().casefold())


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


def prediction(row: dict[str, str]) -> int:
    if str(row.get("trusted_signer_guard_prediction") or "").strip():
        return _int(row.get("trusted_signer_guard_prediction"))
    if str(row.get("loop160_prediction") or "").strip():
        return _int(row.get("loop160_prediction"))
    return _int(row.get("prediction"))


def score_bucket(value: float, *, kind: str) -> str:
    if kind == "selector":
        if value <= 0.25:
            return "<=0.25"
        if value <= 0.50:
            return "(0.25,0.50]"
        if value <= 0.75:
            return "(0.50,0.75]"
        return ">0.75"
    if value <= 0.10:
        return "<=0.10"
    if value <= 0.20:
        return "(0.10,0.20]"
    if value <= 0.25:
        return "(0.20,0.25]"
    if value <= 0.50:
        return "(0.25,0.50]"
    return ">0.50"


def align(base_csv: Path, candidate_csv: Path) -> list[tuple[dict[str, str], dict[str, str]]]:
    base_rows = read_rows(base_csv)
    candidate_rows = read_rows(candidate_csv)
    by_key = {_key(row): row for row in candidate_rows}
    if len(by_key) != len(candidate_rows):
        raise ValueError("candidate_csv contains duplicate sample_index/source_sha256 keys")
    aligned: list[tuple[dict[str, str], dict[str, str]]] = []
    missing = 0
    label_mismatch = 0
    for row in base_rows:
        candidate = by_key.get(_key(row))
        if candidate is None:
            missing += 1
            continue
        if str(row.get("label")) != str(candidate.get("label")):
            label_mismatch += 1
        aligned.append((row, candidate))
    if missing or label_mismatch or len(aligned) != len(base_rows):
        raise ValueError(
            f"alignment failed: missing={missing}, label_mismatch={label_mismatch}, aligned={len(aligned)}, base={len(base_rows)}"
        )
    return aligned


def summarize_split(
    *,
    split_name: str,
    aligned: Sequence[tuple[dict[str, str], dict[str, str]]],
    public_rows: list[dict[str, Any]],
    private_rows: list[dict[str, Any]],
    focus_prefix: str,
) -> dict[str, Any]:
    direction_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    baseline_prob_bucket_counts: Counter[str] = Counter()
    selector_score_bucket_counts: Counter[str] = Counter()
    disagreement_rows = 0

    for base, candidate in aligned:
        base_pred = prediction(base)
        candidate_pred = prediction(candidate)
        if base_pred == candidate_pred:
            continue
        disagreement_rows += 1
        label = str(base.get("label") or "").strip()
        label_int = _int(label)
        direction = f"{base_pred}_to_{candidate_pred}"
        if candidate_pred == label_int and base_pred != label_int:
            outcome = "candidate_fixes_base_error"
        elif candidate_pred != label_int and base_pred == label_int:
            outcome = "candidate_breaks_base_correct"
        else:
            outcome = "candidate_changes_without_net_correctness"
        baseline_prob = _float(candidate.get("baseline_prob_malicious"), _float(base.get("baseline_prob_malicious")))
        selector_score = _float(candidate.get("selector_score"), _float(base.get("selector_score")))
        baseline_bucket = score_bucket(baseline_prob, kind="prob")
        selector_bucket = score_bucket(selector_score, kind="selector")
        focus_id = f"{focus_prefix}_{split_name}_{disagreement_rows:06d}"

        direction_counts[direction] += 1
        outcome_counts[outcome] += 1
        label_counts[label] += 1
        baseline_prob_bucket_counts[baseline_bucket] += 1
        selector_score_bucket_counts[selector_bucket] += 1
        public_rows.append(
            {
                "loop163_focus_id": focus_id,
                "evaluation_split": split_name,
                "direction": direction,
                "label": label,
                "base_prediction": str(base_pred),
                "candidate_prediction": str(candidate_pred),
                "candidate_outcome": outcome,
                "baseline_prob_bucket": baseline_bucket,
                "selector_score_bucket": selector_bucket,
            }
        )
        private_rows.append(
            {
                "loop163_focus_id": focus_id,
                "evaluation_split": split_name,
                "source_path": base.get("source_path", ""),
                "cache_path": base.get("cache_path", ""),
                "source_sha256": base.get("source_sha256", ""),
                "sample_index": base.get("sample_index", ""),
                "label": label,
                "baseline_prob_malicious": candidate.get("baseline_prob_malicious", base.get("baseline_prob_malicious", "")),
                "selector_score": candidate.get("selector_score", base.get("selector_score", "")),
            }
        )
    return {
        "rows": len(aligned),
        "disagreement_rows": disagreement_rows,
        "direction_counts": dict(sorted(direction_counts.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "baseline_prob_bucket_counts": dict(sorted(baseline_prob_bucket_counts.items())),
        "selector_score_bucket_counts": dict(sorted(selector_score_bucket_counts.items())),
    }


def build_loop163_support_audit(
    *,
    val_base_csv: Path,
    val_candidate_csv: Path,
    test10k_base_csv: Path,
    test10k_candidate_csv: Path,
    output_json: Path,
    output_public_csv: Path,
    output_private_map_csv: Path,
    output_md: Optional[Path] = None,
    full_base_csv: Optional[Path] = None,
    full_candidate_csv: Optional[Path] = None,
    min_val_disagreements_for_selector: int = 30,
    min_val_fix_rows_for_selector: int = 10,
    max_val_break_rows_for_selector: int = 0,
    focus_prefix: str = "loop163_r11_support",
) -> dict[str, Any]:
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    split_summaries = {
        "val": summarize_split(
            split_name="val",
            aligned=align(val_base_csv, val_candidate_csv),
            public_rows=public_rows,
            private_rows=private_rows,
            focus_prefix=focus_prefix,
        ),
        "test10k": summarize_split(
            split_name="test10k",
            aligned=align(test10k_base_csv, test10k_candidate_csv),
            public_rows=public_rows,
            private_rows=private_rows,
            focus_prefix=focus_prefix,
        ),
    }
    if full_base_csv is not None and full_candidate_csv is not None:
        split_summaries["full_test_posthoc"] = summarize_split(
            split_name="full_test",
            aligned=align(full_base_csv, full_candidate_csv),
            public_rows=public_rows,
            private_rows=private_rows,
            focus_prefix=focus_prefix,
        )
    val = split_summaries["val"]
    val_fix = int(val["outcome_counts"].get("candidate_fixes_base_error", 0))
    val_break = int(val["outcome_counts"].get("candidate_breaks_base_correct", 0))
    support_failures: list[str] = []
    if int(val["disagreement_rows"]) < int(min_val_disagreements_for_selector):
        support_failures.append("val_disagreement_support_below_minimum")
    if val_fix < int(min_val_fix_rows_for_selector):
        support_failures.append("val_fix_support_below_minimum")
    if val_break > int(max_val_break_rows_for_selector):
        support_failures.append("val_break_rows_exceed_limit")
    decision = "reject_low_support_no_selector_training" if support_failures else "support_sufficient_for_val_only_probe"
    payload = {
        "schema": "axon_loop163_r11_rescue_support_audit_v1",
        "protocol": (
            "Read-only support audit for R11 rescue disagreements; decides whether another Val-only selector/rule "
            "search is statistically supported"
        ),
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "support_thresholds": {
            "min_val_disagreements_for_selector": int(min_val_disagreements_for_selector),
            "min_val_fix_rows_for_selector": int(min_val_fix_rows_for_selector),
            "max_val_break_rows_for_selector": int(max_val_break_rows_for_selector),
        },
        "split_summaries": split_summaries,
        "support_failures": support_failures,
        "selection_policy": {
            "train_selector_allowed": not support_failures,
            "test10k_allowed": False,
            "full_test_allowed": False,
            "full_test_posthoc_only": full_base_csv is not None and full_candidate_csv is not None,
        },
        "outputs": {
            "public_csv": str(resolve_path(output_public_csv)),
            "private_map_csv": str(resolve_path(output_private_map_csv)),
            "output_json": str(resolve_path(output_json)),
            "output_md": str(resolve_path(output_md)) if output_md else "",
        },
        "decision": decision,
        "next_action": (
            "Stop probability-only/R11-only selector searches until a new independent Val-side content or external "
            "evidence source provides broader support."
        )
        if support_failures
        else "A tightly scoped Val-only selector probe may be considered, still behind Loop161 promotion guard.",
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
    val = payload["split_summaries"]["val"]
    test10k = payload["split_summaries"]["test10k"]
    lines = [
        "# Loop163 R11 Rescue Support Audit",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Support failures: `{payload['support_failures']}`",
        f"- Val disagreements: `{val['disagreement_rows']}`",
        f"- Val outcomes: `{val['outcome_counts']}`",
        f"- Test-10k disagreements: `{test10k['disagreement_rows']}`",
        f"- Test-10k outcomes: `{test10k['outcome_counts']}`",
        "",
        "## Policy",
        "",
        IDENTITY_FEATURE_POLICY,
        "",
    ]
    resolved.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop163 R11 rescue support audit.")
    parser.add_argument("--val-base-csv", type=Path, required=True)
    parser.add_argument("--val-candidate-csv", type=Path, required=True)
    parser.add_argument("--test10k-base-csv", type=Path, required=True)
    parser.add_argument("--test10k-candidate-csv", type=Path, required=True)
    parser.add_argument("--full-base-csv", type=Path, default=None)
    parser.add_argument("--full-candidate-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-public-csv", type=Path, required=True)
    parser.add_argument("--output-private-map-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--min-val-disagreements-for-selector", type=int, default=30)
    parser.add_argument("--min-val-fix-rows-for-selector", type=int, default=10)
    parser.add_argument("--max-val-break-rows-for-selector", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_loop163_support_audit(
        val_base_csv=args.val_base_csv,
        val_candidate_csv=args.val_candidate_csv,
        test10k_base_csv=args.test10k_base_csv,
        test10k_candidate_csv=args.test10k_candidate_csv,
        full_base_csv=args.full_base_csv,
        full_candidate_csv=args.full_candidate_csv,
        output_json=args.output_json,
        output_public_csv=args.output_public_csv,
        output_private_map_csv=args.output_private_map_csv,
        output_md=args.output_md,
        min_val_disagreements_for_selector=args.min_val_disagreements_for_selector,
        min_val_fix_rows_for_selector=args.min_val_fix_rows_for_selector,
        max_val_break_rows_for_selector=args.max_val_break_rows_for_selector,
    )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "support_failures": payload["support_failures"],
                "val_disagreement_rows": payload["split_summaries"]["val"]["disagreement_rows"],
                "test10k_disagreement_rows": payload["split_summaries"]["test10k"]["disagreement_rows"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
