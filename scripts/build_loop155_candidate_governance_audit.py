#!/usr/bin/env python3
"""Build a governance audit for Loop151-adjacent candidates.

The goal is to make full-test-looking improvements auditable without allowing
full-test results to select a model. A candidate remains deployable only if it
passes the Val -> Test-10k -> full-test funnel against the current reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


REFERENCE = {
    "id": "loop151_current_strict_best",
    "title": "Loop151 trusted signer guard",
    "val": ("reports/phase3_loop151/loop151_trusted_signer_guard_val_eval.json", "candidate"),
    "test10k": ("reports/phase3_loop151/loop151_trusted_signer_guard_test10k_eval.json", "candidate"),
    "full": ("reports/phase3_loop151/loop151_trusted_signer_guard_full_eval.json", "candidate"),
}


CANDIDATES = [
    {
        "id": "loop151_current_strict_best",
        "title": "Loop151 trusted signer guard",
        "val": ("reports/phase3_loop151/loop151_trusted_signer_guard_val_eval.json", "candidate"),
        "test10k": ("reports/phase3_loop151/loop151_trusted_signer_guard_test10k_eval.json", "candidate"),
        "full": ("reports/phase3_loop151/loop151_trusted_signer_guard_full_eval.json", "candidate"),
        "notes": "Canonical current best under the strict Val -> Test-10k -> full-test funnel.",
    },
    {
        "id": "loop151_on_loop144_union",
        "title": "Loop144 union plus Loop151 trusted signer guard",
        "val": ("reports/phase3_loop151/loop151_trusted_signer_guard_on_loop144_union_val_eval.json", "candidate"),
        "test10k": ("reports/phase3_loop151/loop151_trusted_signer_guard_on_loop144_union_test10k_eval.json", "candidate"),
        "notes": "Val improves strongly, but frozen Test-10k regresses versus Loop151.",
    },
    {
        "id": "loop151_on_oof_noise_r5",
        "title": "OOF-noise/R5 plus Loop151 trusted signer guard",
        "val": ("reports/phase3_loop151/loop151_trusted_signer_guard_on_oof_noise_val_eval_reuse_loop136_sigs.json", "candidate"),
        "full": ("reports/phase3_loop151/loop151_trusted_signer_guard_on_r5_full_eval.json", "candidate"),
        "notes": "Full-test errors are lower than Loop151, but the candidate fails the Val gate.",
    },
    {
        "id": "loop154_trusted_signer_t0995",
        "title": "Loop154 trusted signer score threshold 0.995",
        "val": ("reports/phase3_loop154/loop154_trusted_signer_guard_t0995_val_eval.json", "candidate"),
        "test10k": ("reports/phase3_loop154/loop154_trusted_signer_guard_t0995_test10k_eval.json", "candidate"),
        "full": ("reports/phase3_loop154/loop154_trusted_signer_guard_t0995_full_eval.json", "candidate"),
        "notes": "Equivalent to Loop151 on all evaluated splits.",
    },
]


METRIC_KEYS = ("f1", "errors", "false_positive", "false_negative")


def resolve_path(path: str | Path, root: Path) -> Path:
    item = Path(path)
    return item if item.is_absolute() else root / item


def read_json(path: str | Path, root: Path) -> dict:
    return json.loads(resolve_path(path, root).read_text(encoding="utf-8"))


def extract_metrics(path: str, metric_key: str, root: Path) -> dict[str, object]:
    payload = read_json(path, root)
    metrics = payload.get(metric_key)
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing metric key {metric_key!r} in {path}")
    return {key: metrics.get(key) for key in METRIC_KEYS}


def _split_entry(spec: tuple[str, str] | None, root: Path) -> dict[str, object] | None:
    if spec is None:
        return None
    path, metric_key = spec
    resolved = resolve_path(path, root)
    if not resolved.exists():
        return {"path": path, "present": False, "metrics": None}
    return {
        "path": path,
        "present": True,
        "metric_key": metric_key,
        "metrics": extract_metrics(path, metric_key, root),
    }


def _errors(entry: dict[str, object] | None) -> Optional[int]:
    if not entry or not entry.get("present"):
        return None
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("errors")
    return int(value) if value is not None else None


def _delta(candidate: dict[str, object] | None, reference: dict[str, object] | None) -> Optional[int]:
    cand_errors = _errors(candidate)
    ref_errors = _errors(reference)
    if cand_errors is None or ref_errors is None:
        return None
    return cand_errors - ref_errors


def decide(row: dict[str, object], reference: dict[str, object]) -> tuple[str, str]:
    if row["id"] == reference["id"]:
        return "adopted_current_strict_best", "Reference candidate passed the full funnel."

    val_delta = row["deltas_vs_reference"].get("val_errors")
    test_delta = row["deltas_vs_reference"].get("test10k_errors")
    full_delta = row["deltas_vs_reference"].get("full_errors")

    if val_delta is not None and val_delta > 0:
        if full_delta is not None and full_delta < 0:
            return (
                "reject_val_gate_full_test_mirage",
                "Full-test errors are lower, but Val is worse than the current reference; do not select from full-test.",
            )
        return "reject_val_gate", "Val is worse than the current reference."
    if test_delta is not None and test_delta > 0:
        return "reject_test10k_gate", "Val passed, but frozen Test-10k is worse than the current reference."
    if full_delta == 0 and val_delta == 0 and test_delta == 0:
        return "reject_equivalent_to_current_best", "No row-level or metric improvement over the current reference."
    if full_delta is not None and full_delta < 0 and (val_delta or 0) <= 0 and (test_delta or 0) <= 0:
        return "candidate_beats_reference_full_funnel", "Candidate appears to beat the current reference after passing funnel gates."
    return "retain_as_diagnostic", "Candidate is useful diagnostic evidence but not a replacement."


def build_audit(root: Path) -> dict[str, object]:
    reference_entries = {
        split: _split_entry(REFERENCE.get(split), root)
        for split in ("val", "test10k", "full")
    }
    reference_row = {"id": REFERENCE["id"], **reference_entries}

    rows = []
    for candidate in CANDIDATES:
        split_entries = {
            split: _split_entry(candidate.get(split), root)
            for split in ("val", "test10k", "full")
        }
        row = {
            "id": candidate["id"],
            "title": candidate["title"],
            "notes": candidate.get("notes", ""),
            **split_entries,
            "deltas_vs_reference": {
                "val_errors": _delta(split_entries["val"], reference_entries["val"]),
                "test10k_errors": _delta(split_entries["test10k"], reference_entries["test10k"]),
                "full_errors": _delta(split_entries["full"], reference_entries["full"]),
            },
        }
        decision, reason = decide(row, reference_row)
        row["governance_decision"] = decision
        row["decision_reason"] = reason
        rows.append(row)

    return {
        "schema": "axon_loop155_candidate_governance_audit_v1",
        "protocol": (
            "Val-first candidate governance audit. Full-test results are recorded as diagnostics, "
            "but a candidate that fails Val or Test-10k cannot replace the current reference."
        ),
        "identity_feature_policy": (
            "source_path/cache_path/source_sha256/sample_index/split/row order are logistics and audit fields only; "
            "they are not model, threshold, verdict, replacement-sampling, or production inference evidence."
        ),
        "reference": {
            "id": REFERENCE["id"],
            "title": REFERENCE["title"],
            **reference_entries,
        },
        "candidates": rows,
        "summary": {
            "candidate_count": len(rows),
            "rejected_val_gate": sum(1 for row in rows if str(row["governance_decision"]).startswith("reject_val_gate")),
            "rejected_test10k_gate": sum(1 for row in rows if row["governance_decision"] == "reject_test10k_gate"),
            "full_test_mirage_count": sum(1 for row in rows if row["governance_decision"] == "reject_val_gate_full_test_mirage"),
        },
    }


def format_metric(entry: dict[str, object] | None) -> str:
    if not entry:
        return "`not_run`"
    if not entry.get("present"):
        return "`missing`"
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        return "`invalid`"
    return (
        f"`F1={float(metrics['f1']):.10f}`, errors `{int(metrics['errors'])}`, "
        f"FP/FN `{int(metrics['false_positive'])}/{int(metrics['false_negative'])}`"
    )


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    rows = payload["candidates"]
    lines = [
        "# Loop155 Candidate Governance Audit",
        "",
        "更新时间：2026-07-08",
        "",
        "## Decision",
        "",
        "Loop151 remains the deployable strict best. A lower full-test error count is not enough if the candidate failed the Val or Test-10k gate.",
        "",
        "| Candidate | Val | Test-10k | Full-test | Decision |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {title} | {val} | {test10k} | {full} | `{decision}` |".format(
                title=row["title"],
                val=format_metric(row.get("val")),
                test10k=format_metric(row.get("test10k")),
                full=format_metric(row.get("full")),
                decision=row["governance_decision"],
            )
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            str(payload["protocol"]),
            "",
            str(payload["identity_feature_policy"]),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop155 candidate governance audit.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    payload = build_audit(root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
