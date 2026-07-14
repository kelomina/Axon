#!/usr/bin/env python3
"""Build a non-destructive cache recovery plan from coverage audit artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_missing_csv(path: Path) -> dict:
    """Summarize a missing-cache CSV without touching cache files."""
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    label_counts = Counter(str(row.get("label", "")) for row in rows)
    split_counts = Counter(str(row.get("split", "")) for row in rows)
    suffix_counts = Counter(Path(row.get("source_path", "")).suffix.lower() or "<none>" for row in rows)
    parent_counts = Counter(str(Path(row.get("source_path", "")).parent) for row in rows)

    return {
        "path": str(path),
        "rows": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "top_parent_dirs": [
            {"parent_dir": parent, "count": count}
            for parent, count in parent_counts.most_common(10)
        ],
        "examples": rows[:10],
    }


def _check_action(check: dict) -> str:
    name = check["name"]
    if check.get("missing", 0) == 0:
        return "No cache recovery needed for this check."
    if name == "official_test_current_cache_subset":
        if check.get("missing_output"):
            return (
                "Use the official-test missing-cache CSV as the bounded recovery input. "
                "Only rebuild or restore those fixed-v2 rows, then rerun the strict "
                "calibrator evaluation without allow-missing-cache behavior."
            )
        return (
            "Recover the missing official-test fixed-v2 cache rows first. "
            "The current audit only stores example missing cache paths, so the "
            "next authorized step should generate a full missing-row manifest "
            "before rebuilding anything."
        )
    if "hard_fn" in name:
        return (
            "Use the hard-FN missing-cache CSV as the bounded recovery input. "
            "Only rebuild or restore those rows, then re-export strict hard-FN "
            "predictions without allow-missing-cache behavior."
        )
    if "hard_error" in name:
        return (
            "Use the hard-error missing-cache CSV as the bounded recovery input. "
            "Only rebuild or restore those rows, then re-export strict hard-error "
            "predictions without allow-missing-cache behavior."
        )
    return "Recover only the rows identified by this check, then rerun the strict evaluation."


def build_recovery_plan(audit_path: Path) -> dict:
    audit = load_json(audit_path)
    targets = []
    for check in audit.get("checks", []):
        missing_output = check.get("missing_output")
        missing_csv_summary = None
        if missing_output:
            missing_path = Path(missing_output)
            if missing_path.exists():
                missing_csv_summary = _read_missing_csv(missing_path)

        targets.append(
            {
                "name": check["name"],
                "total": int(check.get("total", 0)),
                "covered": int(check.get("covered", 0)),
                "missing": int(check.get("missing", 0)),
                "coverage_ratio": float(check.get("coverage_ratio", 0.0)),
                "blocked_recommendations": list(check.get("blocked_recommendations", [])),
                "source": check.get("source"),
                "missing_output": missing_output,
                "missing_examples": list(check.get("missing_examples", [])),
                "missing_csv_summary": missing_csv_summary,
                "recommended_recovery_action": _check_action(check),
            }
        )

    blocked = sorted({item for target in targets for item in target["blocked_recommendations"] if target["missing"] > 0})
    return {
        "schema": "axon_cache_recovery_plan_v1",
        "source_audit": str(audit_path),
        "all_full_coverage": all(target["missing"] == 0 for target in targets),
        "blocked_recommendations": blocked,
        "targets": targets,
        "guardrails": [
            "Do not delete or clear data/.cache.",
            "Do not use scripts/rebuild_cache_64.py or scripts/rebuild_cache_8192.py for the current 512-byte fixed-v2 path.",
            "Do not treat allow-missing-cache diagnostic subsets as formal completion evidence.",
            "Generate a full missing manifest before any bounded rebuild when the audit only has sample missing paths.",
            "After recovery, rerun strict evaluations and require missing=0 before removing a recommendation from pending.",
        ],
    }


def write_markdown(plan: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cache Recovery Plan for ML Recommendation Completion",
        "",
        "This plan is non-destructive. It does not rebuild cache, delete cache files, train models, or run evaluation.",
        "",
        f"- Source audit: `{plan['source_audit']}`",
        f"- Full coverage already available: `{plan['all_full_coverage']}`",
        f"- Blocked recommendation ids: `{plan['blocked_recommendations']}`",
        "",
        "## Recovery Targets",
        "",
        "| Target | Covered | Missing | Coverage | Blocks | Recovery action |",
        "|---|---:|---:|---:|---|---|",
    ]
    for target in plan["targets"]:
        lines.append(
            "| {name} | {covered}/{total} | {missing} | {coverage:.1%} | {blocks} | {action} |".format(
                name=target["name"],
                covered=target["covered"],
                total=target["total"],
                missing=target["missing"],
                coverage=target["coverage_ratio"],
                blocks=", ".join(target["blocked_recommendations"]),
                action=target["recommended_recovery_action"],
            )
        )

    lines.extend(["", "## Missing CSV Summaries", ""])
    for target in plan["targets"]:
        summary = target.get("missing_csv_summary")
        if not summary:
            if target.get("missing_examples"):
                lines.append(f"### {target['name']}")
                lines.append("")
                lines.append("The audit has only example missing cache paths, not a full missing-row CSV.")
                lines.append("")
            continue
        lines.append(f"### {target['name']}")
        lines.append("")
        lines.append(f"- Missing CSV: `{summary['path']}`")
        lines.append(f"- Rows: `{summary['rows']}`")
        lines.append(f"- Label counts: `{summary['label_counts']}`")
        lines.append(f"- Split counts: `{summary['split_counts']}`")
        lines.append(f"- File suffix counts: `{summary['suffix_counts']}`")
        lines.append("- Top parent directories:")
        for item in summary["top_parent_dirs"]:
            lines.append(f"  - `{item['parent_dir']}`: `{item['count']}`")
        lines.append("")

    lines.extend(["## Guardrails", ""])
    for guardrail in plan["guardrails"]:
        lines.append(f"- {guardrail}")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a non-destructive cache recovery plan.")
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_recovery_plan(args.audit_json.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(plan, args.output_md)
    print(f"JSON: {args.output_json}")
    print(f"Markdown: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
