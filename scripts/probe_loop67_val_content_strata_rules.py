#!/usr/bin/env python3
"""Probe a small fixed set of Loop66-derived Val-only content strata rules.

This script is deliberately narrow: it reads frozen Loop57 validation
predictions plus content-derived sidecar caches, evaluates predeclared rules,
and writes a report. It does not train, fit thresholds, touch Test-10k/full-test,
relabel samples, or mutate split/cache files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for item in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from identity_feature_guard import assert_no_identity_feature_names  # noqa: E402
from kvd_features.content_pe_v1 import CONTENT_PE_V1_FEATURE_NAMES  # noqa: E402
from train_loop55_overlay_boundary import OVERLAY_BOUNDARY_FEATURE_NAMES  # noqa: E402


@dataclass(frozen=True)
class RuleSpec:
    name: str
    action: str
    description: str


RULES = [
    RuleSpec(
        name="repair_signed_overlay_complex_c515_g80",
        action="repair",
        description=(
            "Flip 0->1 only for signed/overlay-complex PE rows with candidate>=0.515 and gate>=0.80."
        ),
    ),
    RuleSpec(
        name="repair_signed_overlay_complex_c250_g80",
        action="repair",
        description=(
            "Same signed/overlay-complex stratum with a looser candidate score but still high gate>=0.80."
        ),
    ),
    RuleSpec(
        name="repair_overlay_export_or_reloc_c515_g80",
        action="repair",
        description="Flip 0->1 for overlay rows with export/basereloc/exception evidence and high gate.",
    ),
    RuleSpec(
        name="rollback_lowconf_unsigned_importheavy",
        action="rollback",
        description="Flip 1->0 for low-confidence unsigned import-heavy rows.",
    ),
    RuleSpec(
        name="rollback_lowconf_unsigned_payload",
        action="rollback",
        description="Flip 1->0 for low-confidence unsigned payload-like rows.",
    ),
    RuleSpec(
        name="rollback_lowconf_no_security",
        action="rollback",
        description="Flip 1->0 for low-confidence rows with weak security-directory evidence.",
    ),
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _to_int(value: object) -> int:
    return int(float(str(value).strip()))


def _cache_key(row: dict) -> str:
    source_sha = str(row.get("source_sha256") or "").strip().casefold()
    if source_sha:
        return source_sha
    source_path = str(row.get("source_path") or "")
    return hashlib.sha256(str(resolve_path(Path(source_path))).encode("utf-8", errors="ignore")).hexdigest()


def _load_features(row: dict, cache_dir: Path, expected_dim: int, family: str) -> np.ndarray:
    cache_path = resolve_path(cache_dir) / f"{_cache_key(row)}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing {family} sidecar cache: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as data:
        if "features" not in data.files:
            raise ValueError(f"{family} cache missing features array: {cache_path}")
        features = data["features"].astype(np.float32, copy=False)
    if features.shape != (expected_dim,):
        raise ValueError(f"Bad {family} shape for {cache_path}: {features.shape} != {(expected_dim,)}")
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite {family} features: {cache_path}")
    return features


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    labels = labels.astype(np.int64, copy=False)
    predictions = predictions.astype(np.int64, copy=False)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float((tp + tn) / max(len(labels), 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "errors": int(fp + fn),
    }


def _feature_getters(content_matrix: np.ndarray, overlay_matrix: np.ndarray) -> dict[str, np.ndarray]:
    c = {name: index for index, name in enumerate(CONTENT_PE_V1_FEATURE_NAMES)}
    o = {name: index for index, name in enumerate(OVERLAY_BOUNDARY_FEATURE_NAMES)}
    return {
        "security": content_matrix[:, c["content_dir_security_log_size"]],
        "overlay": content_matrix[:, c["content_overlay_log_size"]],
        "basereloc": content_matrix[:, c["content_dir_basereloc_log_size"]],
        "export": content_matrix[:, c["content_dir_export_log_size"]],
        "exception": content_matrix[:, c["content_dir_exception_log_size"]],
        "iat": content_matrix[:, c["content_dir_iat_log_size"]],
        "avg_imports_per_dll": content_matrix[:, c["content_avg_imports_per_dll"]],
        "import_api": content_matrix[:, c["content_import_api_count_log"]],
        "is_dll": content_matrix[:, c["content_is_dll"]],
        "payload_log_size": overlay_matrix[:, o["overlay_boundary_payload_log_size"]],
    }


def _rule_mask(
    rule_name: str,
    *,
    features: dict[str, np.ndarray],
    candidate_scores: np.ndarray,
    gate_scores: np.ndarray,
    final_scores: np.ndarray,
) -> np.ndarray:
    security = features["security"]
    overlay = features["overlay"]
    basereloc = features["basereloc"]
    export = features["export"]
    exception = features["exception"]
    iat = features["iat"]
    avg_imports = features["avg_imports_per_dll"]
    import_api = features["import_api"]
    is_dll = features["is_dll"]
    payload = features["payload_log_size"]

    signed_overlay_complex = (
        (security >= 3.0)
        & (overlay >= 4.0)
        & ((basereloc >= 3.0) | (export >= 1.0) | (exception >= 1.0) | (iat >= 5.0))
    )
    overlay_export_or_reloc = (
        (security >= 2.0)
        & (overlay >= 3.0)
        & ((basereloc >= 3.0) | (export >= 1.0) | (exception >= 1.0))
    )

    if rule_name == "repair_signed_overlay_complex_c515_g80":
        return signed_overlay_complex & (candidate_scores >= 0.515) & (gate_scores >= 0.80)
    if rule_name == "repair_signed_overlay_complex_c250_g80":
        return signed_overlay_complex & (candidate_scores >= 0.25) & (gate_scores >= 0.80)
    if rule_name == "repair_overlay_export_or_reloc_c515_g80":
        return overlay_export_or_reloc & (candidate_scores >= 0.515) & (gate_scores >= 0.80)
    if rule_name == "rollback_lowconf_unsigned_importheavy":
        return (
            (final_scores < 0.90)
            & (is_dll < 0.5)
            & (security < 3.0)
            & ((avg_imports >= 15.0) | (import_api >= 3.5))
        )
    if rule_name == "rollback_lowconf_unsigned_payload":
        return (final_scores < 0.90) & (security < 3.0) & (payload > 0.0)
    if rule_name == "rollback_lowconf_no_security":
        return (final_scores < 0.85) & (security < 2.0)
    raise ValueError(f"Unknown Loop67 rule: {rule_name}")


def _apply_rule(predictions: np.ndarray, rule: RuleSpec, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    updated = predictions.copy()
    if rule.action == "repair":
        action_mask = (updated == 0) & mask
        updated[action_mask] = 1
    elif rule.action == "rollback":
        action_mask = (updated == 1) & mask
        updated[action_mask] = 0
    else:
        raise ValueError(f"Unsupported rule action: {rule.action}")
    return updated, action_mask


def _action_breakdown(labels: np.ndarray, predictions: np.ndarray, action: str, action_mask: np.ndarray) -> dict:
    labels = labels.astype(np.int64, copy=False)
    predictions = predictions.astype(np.int64, copy=False)
    if action == "repair":
        beneficial = int((action_mask & (labels == 1)).sum())
        harmful = int((action_mask & (labels == 0)).sum())
        return {
            "action_rows": int(action_mask.sum()),
            "beneficial_fn_repairs": beneficial,
            "harmful_new_fp": harmful,
            "neutral_rows": 0,
        }
    beneficial = int((action_mask & (labels == 0)).sum())
    harmful = int((action_mask & (labels == 1)).sum())
    return {
        "action_rows": int(action_mask.sum()),
        "beneficial_fp_repairs": beneficial,
        "harmful_new_fn": harmful,
        "neutral_rows": 0,
    }


def write_candidate_csv(path: Path, rows: Sequence[dict]) -> None:
    fieldnames = [
        "rank",
        "name",
        "action",
        "decision",
        "errors",
        "delta_errors_vs_loop57",
        "f1",
        "false_positive",
        "false_negative",
        "action_rows",
        "beneficial_rows",
        "harmful_rows",
        "description",
    ]
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def probe_val_content_strata_rules(
    *,
    loop57_val_predictions: Path,
    content_pe_cache_dir: Path,
    overlay_boundary_cache_dir: Path,
    output_json: Path,
    output_csv: Path,
    min_error_reduction_for_test10k: int = 10,
) -> dict:
    assert_no_identity_feature_names(CONTENT_PE_V1_FEATURE_NAMES, context="Loop67 content PE v1 features")
    assert_no_identity_feature_names(OVERLAY_BOUNDARY_FEATURE_NAMES, context="Loop67 overlay boundary features")

    rows = read_rows(loop57_val_predictions)
    if not rows:
        raise ValueError("No Loop57 validation rows found")

    split_counts: Counter[str] = Counter()
    labels = []
    predictions = []
    candidate_scores = []
    gate_scores = []
    final_scores = []
    content = []
    overlay = []
    for row in rows:
        split = str(row.get("split", "")).strip()
        split_counts[split] += 1
        if split != "val":
            raise ValueError(f"Loop67 is Val-only, got split={split!r}")
        labels.append(_to_int(row["label"]))
        predictions.append(_to_int(row["prediction"]))
        candidate_scores.append(_to_float(row.get("candidate_prob_malicious")))
        gate_scores.append(_to_float(row.get("gate_prob_override")))
        final_scores.append(_to_float(row.get("final_prob_malicious")))
        content.append(
            _load_features(
                row,
                resolve_path(content_pe_cache_dir),
                len(CONTENT_PE_V1_FEATURE_NAMES),
                "content_pe_v1",
            )
        )
        overlay.append(
            _load_features(
                row,
                resolve_path(overlay_boundary_cache_dir),
                len(OVERLAY_BOUNDARY_FEATURE_NAMES),
                "overlay_boundary",
            )
        )

    labels_array = np.asarray(labels, dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    candidate_array = np.asarray(candidate_scores, dtype=np.float32)
    gate_array = np.asarray(gate_scores, dtype=np.float32)
    final_array = np.asarray(final_scores, dtype=np.float32)
    content_matrix = np.vstack(content).astype(np.float32, copy=False)
    overlay_matrix = np.vstack(overlay).astype(np.float32, copy=False)
    features = _feature_getters(content_matrix, overlay_matrix)

    baseline_metrics = _metrics(labels_array, prediction_array)
    required_errors = int(baseline_metrics["errors"]) - max(1, int(min_error_reduction_for_test10k))
    candidate_rows = []
    detailed = []
    for rule in RULES:
        mask = _rule_mask(
            rule.name,
            features=features,
            candidate_scores=candidate_array,
            gate_scores=gate_array,
            final_scores=final_array,
        )
        updated, action_mask = _apply_rule(prediction_array, rule, mask)
        metrics = _metrics(labels_array, updated)
        breakdown = _action_breakdown(labels_array, prediction_array, rule.action, action_mask)
        if rule.action == "repair":
            beneficial = breakdown["beneficial_fn_repairs"]
            harmful = breakdown["harmful_new_fp"]
        else:
            beneficial = breakdown["beneficial_fp_repairs"]
            harmful = breakdown["harmful_new_fn"]
        delta_errors = int(metrics["errors"]) - int(baseline_metrics["errors"])
        decision = "eligible_for_test10k" if int(metrics["errors"]) <= required_errors else "reject_val_margin_too_thin"
        item = {
            "name": rule.name,
            "action": rule.action,
            "description": rule.description,
            "metrics": metrics,
            "delta_errors_vs_loop57": delta_errors,
            "action_breakdown": breakdown,
            "decision": decision,
        }
        detailed.append(item)
        candidate_rows.append(
            {
                "name": rule.name,
                "action": rule.action,
                "decision": decision,
                "errors": metrics["errors"],
                "delta_errors_vs_loop57": delta_errors,
                "f1": metrics["f1"],
                "false_positive": metrics["false_positive"],
                "false_negative": metrics["false_negative"],
                "action_rows": breakdown["action_rows"],
                "beneficial_rows": beneficial,
                "harmful_rows": harmful,
                "description": rule.description,
            }
        )

    candidate_rows.sort(key=lambda row: (int(row["errors"]), -float(row["f1"]), int(row["action_rows"])))
    for rank, row in enumerate(candidate_rows, start=1):
        row["rank"] = rank
    detailed_by_name = {item["name"]: item for item in detailed}
    selected = detailed_by_name[candidate_rows[0]["name"]]
    overall_decision = (
        "eligible_for_test10k"
        if selected["decision"] == "eligible_for_test10k"
        else "reject_no_candidate_with_sufficient_val_margin"
    )
    write_candidate_csv(output_csv, candidate_rows)

    report = {
        "schema": "axon_loop67_val_content_strata_rule_probe_v1",
        "protocol": (
            "read-only Val-only fixed-rule probe; no training, no free-form threshold search, "
            "no Test-10k/full-test use, no relabeling, no split/cache mutation"
        ),
        "identity_feature_policy": (
            "source_path/source_sha256/cache_path/sample_index/split are used only for row alignment "
            "and sidecar cache lookup; they are not model evidence"
        ),
        "loop57_val_predictions": str(resolve_path(loop57_val_predictions)),
        "content_pe_cache_dir": str(resolve_path(content_pe_cache_dir)),
        "overlay_boundary_cache_dir": str(resolve_path(overlay_boundary_cache_dir)),
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "feature_shapes": {
            "content_pe_v1": list(content_matrix.shape),
            "overlay_boundary": list(overlay_matrix.shape),
        },
        "baseline_loop57": baseline_metrics,
        "min_error_reduction_for_test10k": int(min_error_reduction_for_test10k),
        "required_errors_for_test10k": required_errors,
        "selected_by_val": selected,
        "overall_decision": overall_decision,
        "candidates": detailed,
        "candidate_ranking": candidate_rows,
        "outputs": {
            "summary_json": str(resolve_path(output_json)),
            "candidate_csv": str(resolve_path(output_csv)),
        },
    }
    resolved_json = resolve_path(output_json)
    resolved_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Probe fixed Loop67 content-strata rules on Loop57 Val only.")
    parser.add_argument("--loop57-val-predictions", type=Path, required=True)
    parser.add_argument("--content-pe-cache-dir", type=Path, required=True)
    parser.add_argument("--overlay-boundary-cache-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--min-error-reduction-for-test10k", type=int, default=10)
    args = parser.parse_args(argv)
    report = probe_val_content_strata_rules(
        loop57_val_predictions=args.loop57_val_predictions,
        content_pe_cache_dir=args.content_pe_cache_dir,
        overlay_boundary_cache_dir=args.overlay_boundary_cache_dir,
        output_json=args.output_json,
        output_csv=args.output_csv,
        min_error_reduction_for_test10k=args.min_error_reduction_for_test10k,
    )
    print(
        json.dumps(
            {
                "baseline_loop57": report["baseline_loop57"],
                "selected_by_val": {
                    "name": report["selected_by_val"]["name"],
                    "metrics": report["selected_by_val"]["metrics"],
                    "delta_errors_vs_loop57": report["selected_by_val"]["delta_errors_vs_loop57"],
                    "decision": report["selected_by_val"]["decision"],
                },
                "overall_decision": report["overall_decision"],
                "outputs": report["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
