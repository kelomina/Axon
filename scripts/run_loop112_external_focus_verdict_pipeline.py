#!/usr/bin/env python3
"""Run external focus annotation import followed by the strict verdict pipeline."""

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

from import_loop111_focus_external_annotations import import_focus_external_annotations  # noqa: E402
from run_loop110_focus_verdict_pipeline import run_focus_verdict_pipeline  # noqa: E402


PROTOCOL = (
    "Loop112 external focus verdict pipeline; runs Loop111 import -> Loop110 preflight/merge/unblind/Loop87; "
    "no training, no threshold tuning, no automatic relabeling, no replacement sampling, no split/cache mutation"
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _stage_status(*, ran: bool, passed: bool, summary: Optional[dict[str, Any]], blocker_key: str) -> dict[str, Any]:
    return {
        "ran": ran,
        "passed": passed,
        "decision": None if summary is None else summary.get("decision"),
        "blockers": [] if summary is None else list(summary.get(blocker_key, [])),
    }


def run_external_focus_verdict_pipeline(
    *,
    full_blinded_csv: Path,
    focus_csv: Path,
    external_annotations: Path,
    private_map_csv: Path,
    output_dir: Path,
    output_json: Path,
    expected_full_rows: int = 1868,
    expected_focus_rows: int = 240,
    input_format: str = "auto",
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "imported_focus_csv": output_dir / "loop112_imported_focus.csv",
        "import_json": output_dir / "loop112_import.json",
        "import_preflight_csv": output_dir / "loop112_import_preflight_validated.csv",
        "import_preflight_json": output_dir / "loop112_import_preflight.json",
        "loop110_output_dir": output_dir / "loop112_loop110_focus_pipeline",
        "loop110_json": output_dir / "loop112_loop110_summary.json",
    }

    blockers: list[str] = []
    import_summary = import_focus_external_annotations(
        focus_csv=focus_csv,
        external_annotations=external_annotations,
        output_csv=paths["imported_focus_csv"],
        output_json=paths["import_json"],
        preflight_output_csv=paths["import_preflight_csv"],
        preflight_output_json=paths["import_preflight_json"],
        expected_focus_rows=expected_focus_rows,
        input_format=input_format,
        allow_overwrite=allow_overwrite,
    )
    import_passed = bool(import_summary.get("ready_for_loop110_focus_pipeline"))
    if not import_passed:
        blockers.append("loop111_import_not_ready")

    loop110_summary: Optional[dict[str, Any]] = None
    if import_passed:
        loop110_summary = run_focus_verdict_pipeline(
            full_blinded_csv=full_blinded_csv,
            focus_annotations_csv=paths["imported_focus_csv"],
            private_map_csv=private_map_csv,
            output_dir=paths["loop110_output_dir"],
            output_json=paths["loop110_json"],
            expected_full_rows=expected_full_rows,
            expected_focus_rows=expected_focus_rows,
        )
        if loop110_summary.get("blockers"):
            blockers.append("loop110_focus_verdict_pipeline_not_ready")

    loop110_passed = loop110_summary is not None and not loop110_summary.get("blockers")
    loop110_counts = {} if loop110_summary is None else loop110_summary.get("counts", {})
    actionable_rows = int(loop110_counts.get("loop87_actionable_rows", 0) or 0)
    replacement_required_rows = int(loop110_counts.get("loop87_replacement_required_rows", 0) or 0)
    training_policy_rows = int(loop110_counts.get("loop87_training_policy_rows", 0) or 0)

    if blockers:
        decision = "blocked_before_redraw_preflight"
    elif actionable_rows == 0:
        decision = "ready_noop_no_actionable_verdicts"
    else:
        decision = "ready_for_redraw_preflight_review_only"

    summary = {
        "schema": "axon_loop112_external_focus_verdict_pipeline_v1",
        "protocol": PROTOCOL,
        "inputs": {
            "full_blinded_csv": str(full_blinded_csv),
            "focus_csv": str(focus_csv),
            "external_annotations": str(external_annotations),
            "private_map_csv": str(private_map_csv),
            "expected_full_rows": expected_full_rows,
            "expected_focus_rows": expected_focus_rows,
            "input_format": input_format,
            "allow_overwrite": allow_overwrite,
        },
        "output_dir": str(output_dir),
        "decision": decision,
        "blockers": sorted(set(blockers)),
        "stages": {
            "loop111_import": _stage_status(
                ran=True,
                passed=import_passed,
                summary=import_summary,
                blocker_key="blockers",
            ),
            "loop110_focus_verdict_pipeline": _stage_status(
                ran=loop110_summary is not None,
                passed=loop110_passed,
                summary=loop110_summary,
                blocker_key="blockers",
            ),
        },
        "counts": {
            "external_rows": import_summary.get("external", {}).get("rows"),
            "imported_rows": import_summary.get("counts", {}).get("imported_rows"),
            "post_import_actionable_rows": import_summary.get("counts", {}).get("post_preflight_actionable_rows"),
            "loop87_rows": loop110_counts.get("loop87_rows"),
            "loop87_actionable_rows": actionable_rows,
            "loop87_replacement_required_rows": replacement_required_rows,
            "loop87_training_policy_rows": training_policy_rows,
        },
        "decisions": {
            "ready_for_redraw_preflight": bool(not blockers and actionable_rows > 0 and replacement_required_rows > 0 and training_policy_rows == 0),
            "automatic_verdict_allowed": False,
            "automatic_relabel_allowed": False,
            "automatic_replacement_allowed": False,
            "training_allowed": False,
            "test10k_allowed": False,
            "full_test_allowed": False,
            "next_allowed_step": (
                "build non-destructive redraw preflight"
                if not blockers and actionable_rows > 0
                else "collect independent content/external focus annotations"
                if not blockers
                else "fix external focus annotation blockers"
            ),
        },
        "outputs": {name: str(path) for name, path in paths.items()},
        "summary_json": str(output_json),
        "notes": [
            "Loop111 is the only external ingress stage and only accepts blind_review_id plus the three manual fields.",
            "Loop110 does not run when Loop111 import or post-import focus preflight blocks.",
            "This command does not train, tune thresholds, mutate split/cache, or sample replacements.",
        ],
    }
    write_json(output_json, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Loop112 external focus verdict pipeline.")
    parser.add_argument("--full-blinded-csv", type=Path, required=True)
    parser.add_argument("--focus-csv", type=Path, required=True)
    parser.add_argument("--external-annotations", type=Path, required=True)
    parser.add_argument("--private-map-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-full-rows", type=int, default=1868)
    parser.add_argument("--expected-focus-rows", type=int, default=240)
    parser.add_argument("--input-format", choices=["auto", "csv", "jsonl"], default="auto")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_external_focus_verdict_pipeline(
        full_blinded_csv=args.full_blinded_csv,
        focus_csv=args.focus_csv,
        external_annotations=args.external_annotations,
        private_map_csv=args.private_map_csv,
        output_dir=args.output_dir,
        output_json=args.output_json,
        expected_full_rows=args.expected_full_rows,
        expected_focus_rows=args.expected_focus_rows,
        input_format=args.input_format,
        allow_overwrite=bool(args.allow_overwrite),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not summary["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
