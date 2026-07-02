#!/usr/bin/env python3
"""Audit whether checkpoints are safe candidates for current split stacking.

The script is intentionally read-only. It inspects checkpoint metadata and
reports whether a checkpoint is likely compatible with the current random 20w
8192/fixed-v2 split. It does not evaluate model quality and does not use path,
filename, extension, hash, sample id, split, or row order as model features.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


CURRENT_MODEL_SIGNATURE = {
    "max_byte_length": 8192,
    "pe_feature_dim": 256,
    "stat_feature_dim": 49,
    "pe_schema_version": "fixed_v2",
    "pe_fixed_section_slots": 32,
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _safe_load_checkpoint(path: Path) -> Optional[dict]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _read_split_summary(split_csv: Path) -> dict:
    counts = {"total": 0, "by_split": {}, "by_label": {}, "by_split_label": {}}
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row.get("split", "")
            label = str(row.get("label", ""))
            counts["total"] += 1
            counts["by_split"][split] = counts["by_split"].get(split, 0) + 1
            counts["by_label"][label] = counts["by_label"].get(label, 0) + 1
            key = f"{split}:{label}"
            counts["by_split_label"][key] = counts["by_split_label"].get(key, 0) + 1
    return counts


def _config_summary(config: dict) -> dict:
    return {name: config.get(name) for name in CURRENT_MODEL_SIGNATURE}


def _train_config_summary(train_config: dict) -> dict:
    keys = [
        "max_epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "label_smoothing",
        "focal_gamma",
        "focal_alpha",
        "decision_threshold",
    ]
    return {key: train_config.get(key) for key in keys if key in train_config}


def _status_for_checkpoint(path: Path, payload: Optional[dict], current_signature: dict) -> tuple[str, list[str]]:
    reasons = []
    if payload is None:
        return "unreadable", ["torch_load_failed_or_not_checkpoint_dict"]
    config = payload.get("config")
    if not isinstance(config, dict):
        return "unknown", ["missing_config_dict"]
    for key, expected in current_signature.items():
        value = config.get(key)
        if value != expected:
            reasons.append(f"{key}={value!r} != expected {expected!r}")
    train_config = payload.get("train_config")
    if not isinstance(train_config, dict):
        reasons.append("missing_train_config")

    lower_path = str(path).replace("/", "\\").casefold()
    if reasons and any("!= expected" in reason for reason in reasons):
        return "incompatible", reasons
    if "\\random_20w_8192\\" in lower_path:
        return ("compatible_current_random20w", reasons) if not reasons else ("incompatible", reasons)
    if "\\random_20w_baseline_seed42_e10\\" in lower_path:
        reasons.append("older_random20w_baseline_not_current_loop27_corrected_split")
        return "provenance_mismatch", reasons
    if "\\generalization_group_isolated" in lower_path:
        reasons.append("trained_on_group_isolated_subset_not_current_20w_split")
        return "provenance_mismatch", reasons
    if "\\comparison_experiments_from_cache" in lower_path:
        reasons.append("trained_on_comparison_cache_subset_not_current_20w_split")
        return "provenance_mismatch", reasons
    if "\\balanced_replay" in lower_path or "\\hard_family" in lower_path:
        reasons.append("fine_tune_or_hard_replay_provenance_not_current_clean_train_only")
        return "provenance_mismatch", reasons
    if "\\models\\best_model.pt" in lower_path or "\\models\\final_model.pt" in lower_path:
        reasons.append("legacy_root_model_unknown_split")
        return "unknown", reasons
    return "unknown", reasons or ["path_does_not_identify_current_training_split"]


def audit_checkpoints(
    *,
    models_dir: Path,
    split_csv: Path,
    output_json: Path,
    max_rows: Optional[int],
) -> dict:
    split_summary = _read_split_summary(split_csv)
    checkpoints = sorted(models_dir.rglob("*.pt"), key=lambda path: str(path).casefold())
    if max_rows is not None:
        checkpoints = checkpoints[:max_rows]

    rows = []
    status_counts: dict[str, int] = {}
    for checkpoint_path in checkpoints:
        payload = _safe_load_checkpoint(checkpoint_path)
        status, reasons = _status_for_checkpoint(checkpoint_path, payload, CURRENT_MODEL_SIGNATURE)
        status_counts[status] = status_counts.get(status, 0) + 1
        config = payload.get("config", {}) if isinstance(payload, dict) else {}
        train_config = payload.get("train_config", {}) if isinstance(payload, dict) else {}
        rows.append(
            {
                "path": str(checkpoint_path),
                "status": status,
                "reasons": reasons,
                "config": _config_summary(config if isinstance(config, dict) else {}),
                "train_config": _train_config_summary(train_config if isinstance(train_config, dict) else {}),
                "best_f1": payload.get("best_f1") if isinstance(payload, dict) else None,
                "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
                "last_epoch": payload.get("last_epoch") if isinstance(payload, dict) else None,
            }
        )

    compatible = [row for row in rows if row["status"] == "compatible_current_random20w"]
    report = {
        "schema": "axon_checkpoint_provenance_audit_v1",
        "protocol": (
            "Read-only checkpoint metadata audit. Identity fields are audit-only and never model features."
        ),
        "split_csv": str(split_csv),
        "split_summary": split_summary,
        "current_signature": CURRENT_MODEL_SIGNATURE,
        "models_dir": str(models_dir),
        "checkpoint_count": len(rows),
        "status_counts": status_counts,
        "compatible_current_random20w_count": len(compatible),
        "compatible_current_random20w": compatible,
        "checkpoints": rows,
        "decision": (
            "current repo has no safe diverse current-split checkpoint pool"
            if len(compatible) <= 1
            else "candidate pool exists but must still export train/val predictions before stacking"
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit checkpoint provenance for current split stacking.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args(argv)
    report = audit_checkpoints(
        models_dir=resolve_path(args.models_dir),
        split_csv=resolve_path(args.split_csv),
        output_json=resolve_path(args.output_json),
        max_rows=args.max_rows,
    )
    print(json.dumps({key: report[key] for key in ["checkpoint_count", "status_counts", "decision"]}, indent=2))
    print(f"JSON: {resolve_path(args.output_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
