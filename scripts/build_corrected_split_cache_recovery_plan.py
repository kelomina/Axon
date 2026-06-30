#!/usr/bin/env python3
"""Build a non-destructive recovery plan for corrected split missing cache rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_missing_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_missing_rows(rows: Sequence[dict]) -> dict:
    return {
        "rows": len(rows),
        "label_counts": dict(sorted(Counter(str(row.get("label", "")) for row in rows).items())),
        "split_counts": dict(sorted(Counter(str(row.get("split", "")) for row in rows).items())),
        "reason_counts": dict(sorted(Counter(str(row.get("reason", "")) for row in rows).items())),
        "suffix_counts": dict(sorted(Counter(Path(str(row.get("source_path", ""))).suffix.lower() or "<none>" for row in rows).items())),
        "examples": list(rows[:10]),
    }


def build_recovery_command(
    *,
    missing_csv: Path,
    checkpoint: Path,
    cache_dir: Path,
    output_json: Path,
    workers: int,
    backend: str,
    storage_format: str,
    dry_run: bool,
) -> str:
    parts = [
        r".\vnev\Scripts\python.exe",
        "scripts\\recover_missing_feature_cache.py",
        f"--missing-csv {missing_csv}",
        f"--checkpoint {checkpoint}",
        f"--cache-dir {cache_dir}",
        f"--workers {int(workers)}",
        f"--backend {backend}",
        f"--storage-format {storage_format}",
        f"--output-json {output_json}",
    ]
    if dry_run:
        parts.append("--dry-run")
    return " ".join(str(part) for part in parts)


def build_plan(
    *,
    missing_csv: Path,
    checkpoint: Path,
    cache_dir: Path,
    recovery_output_json: Path,
    audit_command: str,
    workers: int = 4,
    backend: str = "process",
    storage_format: str = "uncompressed",
) -> dict:
    rows = read_missing_rows(missing_csv)
    missing_summary = summarize_missing_rows(rows)
    dry_run_command = build_recovery_command(
        missing_csv=missing_csv,
        checkpoint=checkpoint,
        cache_dir=cache_dir,
        output_json=recovery_output_json,
        workers=workers,
        backend=backend,
        storage_format=storage_format,
        dry_run=True,
    )
    recovery_command = build_recovery_command(
        missing_csv=missing_csv,
        checkpoint=checkpoint,
        cache_dir=cache_dir,
        output_json=recovery_output_json,
        workers=workers,
        backend=backend,
        storage_format=storage_format,
        dry_run=False,
    )
    return {
        "schema": "axon_corrected_split_cache_recovery_plan_v1",
        "missing_csv": str(resolve_path(missing_csv)),
        "checkpoint": str(resolve_path(checkpoint)),
        "cache_dir": str(resolve_path(cache_dir)),
        "recovery_output_json": str(resolve_path(recovery_output_json)),
        "workers": int(workers),
        "backend": backend,
        "storage_format": storage_format,
        "missing_summary": missing_summary,
        "needs_recovery": missing_summary["rows"] > 0,
        "commands": {
            "dry_run": dry_run_command,
            "recover": recovery_command,
            "post_recovery_audit": audit_command,
        },
        "guardrails": [
            "Do not edit the corrected split while recovering cache.",
            "Run the dry-run recovery command first.",
            "Only run recovery for rows listed in the bounded missing CSV.",
            "After recovery, rerun the strict corrected split cache readiness audit and require cache_ready=true.",
        ],
    }


def write_markdown(plan: dict, output_md: Path) -> None:
    resolved = resolve_path(output_md)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Corrected Split Cache Recovery Plan",
        "",
        f"- Missing rows: `{plan['missing_summary']['rows']}`",
        f"- Needs recovery: `{plan['needs_recovery']}`",
        f"- Label counts: `{plan['missing_summary']['label_counts']}`",
        f"- Split counts: `{plan['missing_summary']['split_counts']}`",
        f"- Reason counts: `{plan['missing_summary']['reason_counts']}`",
        "",
        "## Commands",
        "",
        "Dry run first:",
        "",
        "```powershell",
        plan["commands"]["dry_run"],
        "```",
        "",
        "Recovery command:",
        "",
        "```powershell",
        plan["commands"]["recover"],
        "```",
        "",
        "Post-recovery strict audit:",
        "",
        "```powershell",
        plan["commands"]["post_recovery_audit"],
        "```",
        "",
        "## Guardrails",
        "",
    ]
    for guardrail in plan["guardrails"]:
        lines.append(f"- {guardrail}")
    lines.append("")
    resolved.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build corrected split cache recovery plan.")
    parser.add_argument("--missing-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/.cache"))
    parser.add_argument("--recovery-output-json", type=Path, required=True)
    parser.add_argument("--post-recovery-audit-command", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backend", choices=["process", "thread"], default="process")
    parser.add_argument("--storage-format", choices=["compressed", "uncompressed"], default="uncompressed")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_plan(
        missing_csv=args.missing_csv,
        checkpoint=args.checkpoint,
        cache_dir=args.cache_dir,
        recovery_output_json=args.recovery_output_json,
        audit_command=args.post_recovery_audit_command,
        workers=args.workers,
        backend=args.backend,
        storage_format=args.storage_format,
    )
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(plan, args.output_md)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
