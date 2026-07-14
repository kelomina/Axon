#!/usr/bin/env python3
"""Authorized wrapper for official Axon train/eval operations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ml_authorization_runtime import (
    DEFAULT_PREFLIGHT,
    AuthorizationError,
    authorize_from_file,
    required_operations_for_eval,
    required_operations_for_train,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run scripts/main.py only after the ML authorization preflight allows "
            "the requested official train/eval operation."
        )
    )
    parser.add_argument(
        "--ml-preflight",
        type=Path,
        default=DEFAULT_PREFLIGHT,
        help="Path to the ML authorization preflight JSON.",
    )
    parser.add_argument(
        "main_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to scripts/main.py, e.g. train --config ...",
    )
    return parser


def _parse_main_args(main_args: list[str]) -> argparse.Namespace:
    if main_args and main_args[0] == "--":
        main_args = main_args[1:]
    if not main_args:
        raise AuthorizationError("No scripts/main.py command was provided.")
    command = main_args[0]
    parser = argparse.ArgumentParser(add_help=False)
    parser.set_defaults(command=command)
    if command == "train":
        parser.add_argument("--fast", action="store_true", default=False)
        parser.add_argument("--skip-test-eval", action="store_true", default=False)
    elif command == "eval":
        parser.add_argument("--split", type=str, default="test")
        parser.add_argument("--max-eval-samples", type=int, default=None)
        parser.add_argument("--sweep-thresholds", type=str, default=None)
        parser.add_argument("--decision-threshold", type=float, default=None)
    return parser.parse_known_args(main_args[1:])[0]


def _required_operations(args: argparse.Namespace) -> list[str]:
    if args.command == "train":
        return required_operations_for_train(
            fast=bool(getattr(args, "fast", False)),
            skip_test_eval=bool(getattr(args, "skip_test_eval", False)),
        )
    if args.command == "eval":
        return required_operations_for_eval(
            split=str(getattr(args, "split", "test")),
            max_eval_samples=getattr(args, "max_eval_samples", None),
            sweep_thresholds=getattr(args, "sweep_thresholds", None),
            decision_threshold=getattr(args, "decision_threshold", None),
        )
    return []


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    main_args = list(args.main_args)
    main_namespace = _parse_main_args(main_args)
    operations = _required_operations(main_namespace)
    if operations:
        receipt = authorize_from_file(args.ml_preflight, operations)
        print(
            "[ML Authorization] allowed "
            f"operations={receipt['required_operations']} "
            f"preflight={args.ml_preflight}"
        )
    else:
        print("[ML Authorization] no heavy official operation requested")

    if main_args and main_args[0] == "--":
        main_args = main_args[1:]
    import main as axon_main

    old_argv = sys.argv
    try:
        sys.argv = [str(Path(axon_main.__file__)), *main_args]
        axon_main.main()
    finally:
        sys.argv = old_argv
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuthorizationError as exc:
        print(f"[ML Authorization Blocked] {exc}", file=sys.stderr)
        raise SystemExit(2)
