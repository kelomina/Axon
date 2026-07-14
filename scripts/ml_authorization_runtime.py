#!/usr/bin/env python3
"""Runtime checks for ML train/eval authorization preflight reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_PREFLIGHT = Path("reports/random_20w_split/loop104_ml_authorization_preflight.json")


class AuthorizationError(RuntimeError):
    """Raised when an ML operation is not authorized by the preflight report."""


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def load_preflight(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AuthorizationError(f"ML authorization preflight is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def required_operations_for_train(*, fast: bool, skip_test_eval: bool) -> list[str]:
    if fast:
        return []
    operations = ["train_val"]
    if not skip_test_eval:
        operations.append("full_test")
    return operations


def required_operations_for_eval(
    *,
    split: str,
    max_eval_samples: int | None,
    sweep_thresholds: str | None,
    decision_threshold: float | None,
) -> list[str]:
    operations: list[str] = []
    split = split.lower()
    if split in {"test", "all"}:
        if max_eval_samples is None:
            operations.append("full_test")
        else:
            operations.append("test10k")
    if sweep_thresholds or decision_threshold is not None:
        operations.append("threshold_sweep")
    return _unique(operations)


def assert_operations_authorized(
    preflight: dict[str, Any],
    operations: Sequence[str],
) -> dict[str, Any]:
    if not operations:
        return {
            "authorized": True,
            "required_operations": [],
            "preflight_schema": preflight.get("schema"),
        }

    decisions = (
        preflight.get("operation_authorization", {})
        .get("decisions", {})
    )
    blockers = (
        preflight.get("operation_authorization", {})
        .get("operation_blockers", {})
    )
    failed: dict[str, list[str]] = {}
    for operation in operations:
        decision_key = f"{operation}_allowed"
        if decisions.get(decision_key) is True:
            continue
        failed[operation] = list(blockers.get(operation, []))

    if failed:
        details = "; ".join(
            f"{operation}: {', '.join(reasons) if reasons else 'not allowed'}"
            for operation, reasons in failed.items()
        )
        raise AuthorizationError(
            "ML operation is blocked by authorization preflight. "
            f"required={list(operations)}; {details}"
        )

    return {
        "authorized": True,
        "required_operations": list(operations),
        "preflight_schema": preflight.get("schema"),
    }


def authorize_from_file(path: Path, operations: Sequence[str]) -> dict[str, Any]:
    return assert_operations_authorized(load_preflight(path), operations)
