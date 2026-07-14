#!/usr/bin/env python3
"""Run the strict Loop106 focus annotation verdict pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_loop96_blinded_review_package import unblind_verdicts  # noqa: E402
from import_loop87_review_evidence_verdicts import validate_loop86_verdicts  # noqa: E402
from merge_loop106_focus_annotations import merge_focus_annotations  # noqa: E402
from preflight_loop106_focus_annotations import preflight_focus_annotations  # noqa: E402


PROTOCOL = (
    "read-only Loop110 focus verdict pipeline; runs preflight -> merge -> unblind -> Loop87 import; "
    "no training, no threshold tuning, no automatic relabeling, no replacement sampling, no split/cache mutation"
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _stage_status(*, ran: bool, passed: bool, summary: Optional[dict[str, Any]], blocker_key: str) -> dict[str, Any]:
    return {
        "ran": ran,
        "passed": passed,
        "blockers": [] if summary is None else list(summary.get(blocker_key, [])),
        "decision": None if summary is None else summary.get("decision"),
    }


def run_focus_verdict_pipeline(
    *,
    full_blinded_csv: Path,
    focus_annotations_csv: Path,
    private_map_csv: Path,
    output_dir: Path,
    output_json: Path,
    expected_full_rows: int = 1868,
    expected_focus_rows: int = 240,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "preflight_csv": output_dir / "loop110_focus_preflight_validated.csv",
        "preflight_json": output_dir / "loop110_focus_preflight.json",
        "merged_blinded_csv": output_dir / "loop110_merged_full_blinded.csv",
        "merge_json": output_dir / "loop110_focus_merge.json",
        "unblinded_loop87_csv": output_dir / "loop110_unblinded_loop87_input.csv",
        "unblind_json": output_dir / "loop110_unblind.json",
        "loop87_validated_csv": output_dir / "loop110_loop87_validated.csv",
        "loop87_json": output_dir / "loop110_loop87_import.json",
    }
    blockers: list[str] = []

    preflight = preflight_focus_annotations(
        focus_annotations_csv=focus_annotations_csv,
        output_csv=paths["preflight_csv"],
        output_json=paths["preflight_json"],
        expected_rows=expected_focus_rows,
    )
    preflight_passed = bool(preflight.get("ready_for_focus_merge"))
    if not preflight_passed:
        blockers.append("focus_annotation_preflight_not_ready")

    merge_summary: Optional[dict[str, Any]] = None
    unblind_summary: Optional[dict[str, Any]] = None
    loop87_summary: Optional[dict[str, Any]] = None
    if preflight_passed:
        merge_summary = merge_focus_annotations(
            full_blinded_csv=full_blinded_csv,
            focus_annotations_csv=focus_annotations_csv,
            output_csv=paths["merged_blinded_csv"],
            output_json=paths["merge_json"],
            expected_full_rows=expected_full_rows,
            expected_focus_rows=expected_focus_rows,
        )
        if merge_summary.get("blockers"):
            blockers.append("focus_merge_not_ready")

    merge_passed = merge_summary is not None and not merge_summary.get("blockers")
    if merge_passed:
        unblind_summary = unblind_verdicts(
            annotated_blinded_csv=paths["merged_blinded_csv"],
            private_map_csv=private_map_csv,
            output_csv=paths["unblinded_loop87_csv"],
            output_json=paths["unblind_json"],
            expected_rows=expected_full_rows,
        )
        if unblind_summary.get("blockers"):
            blockers.append("loop96_unblind_not_ready")

    unblind_passed = unblind_summary is not None and not unblind_summary.get("blockers")
    if unblind_passed:
        loop87_summary = validate_loop86_verdicts(
            evidence_csv=paths["unblinded_loop87_csv"],
            output_csv=paths["loop87_validated_csv"],
            output_json=paths["loop87_json"],
            expected_rows=expected_full_rows,
        )
        if not loop87_summary.get("import_ready"):
            blockers.append("loop87_import_not_ready")

    loop87_passed = loop87_summary is not None and bool(loop87_summary.get("import_ready"))
    actionable_rows = int((loop87_summary or {}).get("actionable_rows", 0) or 0)
    replacement_required_rows = int((loop87_summary or {}).get("replacement_required_rows", 0) or 0)
    training_policy_rows = int((loop87_summary or {}).get("training_policy_rows", 0) or 0)
    if blockers:
        decision = "blocked_before_redraw_preflight"
    elif actionable_rows == 0:
        decision = "ready_noop_no_actionable_verdicts"
    else:
        decision = "ready_for_redraw_preflight_review_only"

    summary = {
        "schema": "axon_loop110_focus_verdict_pipeline_v1",
        "protocol": PROTOCOL,
        "inputs": {
            "full_blinded_csv": str(full_blinded_csv),
            "focus_annotations_csv": str(focus_annotations_csv),
            "private_map_csv": str(private_map_csv),
            "expected_full_rows": expected_full_rows,
            "expected_focus_rows": expected_focus_rows,
        },
        "output_dir": str(output_dir),
        "decision": decision,
        "blockers": sorted(set(blockers)),
        "stages": {
            "focus_annotation_preflight": _stage_status(
                ran=True,
                passed=preflight_passed,
                summary=preflight,
                blocker_key="blockers",
            ),
            "focus_merge": _stage_status(
                ran=merge_summary is not None,
                passed=merge_passed,
                summary=merge_summary,
                blocker_key="blockers",
            ),
            "loop96_unblind": _stage_status(
                ran=unblind_summary is not None,
                passed=unblind_passed,
                summary=unblind_summary,
                blocker_key="blockers",
            ),
            "loop87_import": _stage_status(
                ran=loop87_summary is not None,
                passed=loop87_passed,
                summary=loop87_summary,
                blocker_key="blocking_issues",
            ),
        },
        "counts": {
            "preflight_rows": preflight.get("rows"),
            "preflight_annotated_rows": preflight.get("annotated_rows"),
            "preflight_actionable_rows": preflight.get("actionable_rows"),
            "merged_annotated_rows": None if merge_summary is None else merge_summary.get("rows", {}).get("merged_annotated_rows"),
            "loop87_rows": None if loop87_summary is None else loop87_summary.get("rows"),
            "loop87_actionable_rows": actionable_rows,
            "loop87_replacement_required_rows": replacement_required_rows,
            "loop87_training_policy_rows": training_policy_rows,
        },
        "decisions": {
            "ready_for_redraw_preflight": bool(not blockers and actionable_rows > 0 and replacement_required_rows > 0 and training_policy_rows == 0),
            "automatic_relabel_allowed": False,
            "automatic_replacement_allowed": False,
            "training_allowed": False,
            "test10k_allowed": False,
            "full_test_allowed": False,
            "next_allowed_step": (
                "build non-destructive redraw preflight"
                if not blockers and actionable_rows > 0
                else "fill focus manual fields with independent content/external verdicts"
            ),
        },
        "outputs": {name: str(path) for name, path in paths.items()},
        "summary_json": str(output_json),
        "notes": [
            "The pipeline preserves the blinded review boundary until the preflight and merge gates pass.",
            "Loop87 remains the strict unblinded verdict quality gate.",
            "This pipeline never trains, tunes thresholds, mutates split/cache, or samples replacements.",
        ],
    }
    write_json(output_json, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Loop110 focus verdict pipeline.")
    parser.add_argument("--full-blinded-csv", type=Path, required=True)
    parser.add_argument("--focus-annotations-csv", type=Path, required=True)
    parser.add_argument("--private-map-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-full-rows", type=int, default=1868)
    parser.add_argument("--expected-focus-rows", type=int, default=240)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_focus_verdict_pipeline(
        full_blinded_csv=args.full_blinded_csv,
        focus_annotations_csv=args.focus_annotations_csv,
        private_map_csv=args.private_map_csv,
        output_dir=args.output_dir,
        output_json=args.output_json,
        expected_full_rows=args.expected_full_rows,
        expected_focus_rows=args.expected_focus_rows,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not summary["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
