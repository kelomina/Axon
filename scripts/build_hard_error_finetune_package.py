#!/usr/bin/env python3
"""Build a second-round fine-tune package from real FP/FN errors."""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_COLUMNS = [
    "source_path",
    "label",
    "sample_index",
    "group_id",
    "source_group_id",
    "group_size",
    "split",
    "sample_weight",
    "hard_family_role",
    "hard_error_type",
    "hard_error_prob",
    "hard_error_margin",
    "is_rare_group",
    "group_source",
]


@dataclass(frozen=True)
class SplitRatios:
    train: float
    val: float

    @property
    def holdout(self) -> float:
        return 1.0 - self.train - self.val


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def normalize_path_text(path_text: str) -> str:
    return str(Path(path_text)).casefold()


def stable_sample_key(row: dict) -> tuple[int, str]:
    try:
        sample_index = int(row.get("sample_index", ""))
    except ValueError:
        sample_index = 0
    return sample_index, row.get("source_path", "")


def source_group_key(row: dict) -> str:
    return str(row.get("source_group_id") or row.get("group_id") or "")


def error_type(row: dict) -> str:
    explicit = str(row.get("error_type", "")).upper()
    if explicit in {"FP", "FN"}:
        return explicit
    label = int(row.get("label", 0))
    pred = int(row.get("prediction", 0))
    if label == 0 and pred == 1:
        return "FP"
    if label == 1 and pred == 0:
        return "FN"
    return ""


def load_error_rows(fp_csv: Path, fn_csv: Path, focus: str = "both") -> list[dict]:
    allowed_types = {
        "both": {"FP", "FN"},
        "fp": {"FP"},
        "fn": {"FN"},
    }[focus]
    rows = []
    for path in [resolve_path(fp_csv), resolve_path(fn_csv)]:
        for row in read_csv_rows(path):
            kind = error_type(row)
            if not kind or kind not in allowed_types:
                continue
            row = dict(row)
            row["error_type"] = kind
            rows.append(row)
    if not rows:
        raise ValueError(f"No {focus} error rows were loaded")
    return rows


def allocate_groups(error_rows: Sequence[dict], ratios: SplitRatios, seed: int) -> dict[str, str]:
    """Allocate whole source groups to train/val/holdout, stratified by FP/FN."""
    allocation = {}
    by_error_type = defaultdict(lambda: defaultdict(list))
    for row in error_rows:
        by_error_type[row["error_type"]][source_group_key(row)].append(row)

    for kind, groups in sorted(by_error_type.items()):
        units = []
        for group_id, rows in groups.items():
            sorted_rows = sorted(rows, key=stable_sample_key)
            units.append((group_id, len(sorted_rows), sorted_rows[0].get("source_path", "")))
        random.Random(seed + (17 if kind == "FP" else 31)).shuffle(units)

        total = sum(size for _group_id, size, _path in units)
        target_train = round(total * ratios.train)
        target_val = round(total * ratios.val)
        assigned_counts = Counter()
        for group_id, size, _path in units:
            if assigned_counts["train"] < target_train:
                split = "train"
            elif assigned_counts["val"] < target_val:
                split = "val"
            else:
                split = "test"
            allocation[group_id] = split
            assigned_counts[split] += size
    return allocation


def build_hard_error_rows(
    source_rows: Sequence[dict],
    error_rows: Sequence[dict],
    ratios: SplitRatios,
    seed: int,
    fp_weight: float,
    fn_weight: float,
    eligible_split: str = "test",
    strict_source_group_isolation: bool = False,
) -> tuple[list[dict], dict]:
    errors_by_path = {
        normalize_path_text(row["source_path"]): row
        for row in error_rows
    }
    error_group_keys = {source_group_key(row) for row in error_rows}
    allocation_by_group = allocate_groups(error_rows, ratios, seed)

    output_rows = []
    for source_row in source_rows:
        source_row = dict(source_row)
        key = normalize_path_text(source_row.get("source_path", ""))
        group_key = source_group_key(source_row)
        error_row = errors_by_path.get(key)
        source_split = source_row.get("split", "")
        group_is_hard = group_key in error_group_keys
        split_is_eligible = source_split == eligible_split
        hard_split = None
        if group_is_hard and (split_is_eligible or strict_source_group_isolation):
            hard_split = allocation_by_group.get(group_key)

        if hard_split is None:
            split = source_split
            row = {
                "source_path": source_row.get("source_path", ""),
                "label": source_row.get("label", ""),
                "sample_index": source_row.get("sample_index", ""),
                "group_id": source_row.get("group_id", ""),
                "source_group_id": source_row.get("source_group_id") or source_row.get("group_id", ""),
                "group_size": source_row.get("group_size", ""),
                "split": split,
                "sample_weight": "",
                "hard_family_role": f"base_{split}",
                "hard_error_type": "",
                "hard_error_prob": "",
                "hard_error_margin": "",
                "is_rare_group": source_row.get("is_rare_group", ""),
                "group_source": source_row.get("group_source", ""),
            }
        else:
            hard_suffix = "train" if hard_split == "train" else ("val" if hard_split == "val" else "holdout")
            if error_row is None:
                role = f"hard_error_context_{hard_suffix}"
                sample_weight = ""
                kind = ""
                prob = ""
                margin = ""
            else:
                kind = error_row["error_type"]
                role = f"hard_error_{kind.lower()}_{hard_suffix}"
                sample_weight = f"{fp_weight:.6g}" if kind == "FP" and hard_split == "train" else (
                    f"{fn_weight:.6g}" if kind == "FN" and hard_split == "train" else ""
                )
                prob = error_row.get("prob_malicious", "")
                margin = error_row.get("margin_to_threshold", "")
            row = {
                "source_path": source_row.get("source_path", ""),
                "label": source_row.get("label", ""),
                "sample_index": source_row.get("sample_index", ""),
                "group_id": f"{source_row.get('group_id', group_key)}_hard_error_{hard_suffix}",
                "source_group_id": source_row.get("source_group_id") or source_row.get("group_id", ""),
                "group_size": source_row.get("group_size", ""),
                "split": hard_split,
                "sample_weight": sample_weight,
                "hard_family_role": role,
                "hard_error_type": kind,
                "hard_error_prob": prob,
                "hard_error_margin": margin,
                "is_rare_group": source_row.get("is_rare_group", ""),
                "group_source": source_row.get("group_source", ""),
            }
        output_rows.append(row)

    plan_counts = summarize_rows(output_rows)
    plan_counts["allocated_error_groups"] = dict(sorted(Counter(allocation_by_group.values()).items()))
    plan_counts["eligible_split"] = eligible_split
    plan_counts["strict_source_group_isolation"] = strict_source_group_isolation
    return output_rows, plan_counts


def summarize_rows(rows: Sequence[dict]) -> dict:
    split_counts = Counter(row.get("split", "") for row in rows)
    role_counts = Counter(row.get("hard_family_role", "") for row in rows)
    error_type_counts = Counter(row.get("hard_error_type", "") for row in rows if row.get("hard_error_type"))
    hard_rows = [row for row in rows if str(row.get("hard_family_role", "")).startswith("hard_error")]
    hard_by_split = Counter(row.get("split", "") for row in hard_rows)
    weighted_train = sum(1 for row in rows if row.get("split") == "train" and str(row.get("sample_weight", "")).strip())
    return {
        "total_rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "hard_role_counts": dict(sorted(role_counts.items())),
        "hard_error_type_counts": dict(sorted(error_type_counts.items())),
        "hard_rows_by_split": dict(sorted(hard_by_split.items())),
        "weighted_train_samples": weighted_train,
    }


def validate_no_group_cross_split(rows: Sequence[dict]) -> None:
    group_splits = {}
    for row in rows:
        group_id = row.get("group_id")
        split = row.get("split")
        if not group_id:
            continue
        existing = group_splits.get(group_id)
        if existing is not None and existing != split:
            raise ValueError(f"group_id {group_id} crosses splits: {existing}, {split}")
        group_splits[group_id] = split


def validate_no_hard_source_group_cross_split(rows: Sequence[dict]) -> None:
    """Ensure each source group touched by hard errors is fully isolated to one split."""
    hard_source_groups = {
        source_group_key(row)
        for row in rows
        if str(row.get("hard_family_role", "")).startswith("hard_error")
    }
    source_group_splits = defaultdict(set)
    for row in rows:
        group_key = source_group_key(row)
        if group_key in hard_source_groups:
            source_group_splits[group_key].add(row.get("split", ""))
    crossing = {
        group_key: sorted(splits)
        for group_key, splits in source_group_splits.items()
        if len(splits) > 1
    }
    if crossing:
        examples = ", ".join(
            f"{group_key}:{'/'.join(splits)}"
            for group_key, splits in list(crossing.items())[:5]
        )
        raise ValueError(
            "Hard source_group_id crosses splits. "
            "Use --strict-source-group-isolation for leakage-safe replay packages. "
            f"Examples: {examples}"
        )


def ps_command(*parts: str) -> str:
    return " ".join(parts)


def build_commands(args, paths: dict[str, Path]) -> dict[str, str]:
    python_exe = r'& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe"'
    authorized_main = (
        'scripts\\authorized_main.py --ml-preflight '
        '"reports\\random_20w_split\\loop104_ml_authorization_preflight.json" --'
    )
    checkpoint = str(resolve_path(args.checkpoint))
    model_output_dir = str(resolve_path(args.model_output_dir))
    finetuned_checkpoint = str(resolve_path(args.model_output_dir) / "best_model.pt")
    data_dir = str(resolve_path(args.data_dir))
    config = str(resolve_path(args.config))
    sweep_thresholds = sorted({0.45, 0.50, 0.55, 0.60, float(args.decision_threshold), 0.65, 0.70})
    sweep_text = ",".join(f"{value:.3g}" for value in sweep_thresholds)

    return {
        "fine_tune": ps_command(
            python_exe, authorized_main, "train",
            "--config", f'"{config}"',
            "--data-dir", f'"{data_dir}"',
            "--split-file", f'"{paths["split_csv"]}"',
            "--init-checkpoint", f'"{checkpoint}"',
            "--output-dir", f'"{model_output_dir}"',
            "--fast",
            "--samples-per-class", str(args.samples_per_class),
            "--epochs", str(args.epochs),
            "--lr", str(args.learning_rate),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--rare-group-weighting",
            "--singleton-group-weight", str(args.singleton_group_weight),
            "--rare-group-weight", str(args.rare_group_weight),
            "--medium-group-weight", str(args.medium_group_weight),
            "--extract-workers", str(args.extract_workers),
            "--extract-backend", args.extract_backend,
        ),
        "full_threshold_sweep": ps_command(
            python_exe, authorized_main, "eval",
            "--checkpoint", f'"{finetuned_checkpoint}"',
            "--data-dir", f'"{data_dir}"',
            "--split-file", f'"{paths["split_csv"]}"',
            "--split test",
            "--samples-per-class 0",
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--sweep-thresholds", f'"{sweep_text}"',
            "--output", f'"{paths["output_dir"] / "hard_error_finetuned_full_threshold_sweep.json"}"',
        ),
        "holdout_predictions": ps_command(
            python_exe, "scripts\\export_sample_predictions.py",
            "--checkpoint", f'"{finetuned_checkpoint}"',
            "--config", f'"{config}"',
            "--data-dir", f'"{data_dir}"',
            "--samples", f'"{paths["hard_holdout_csv"]}"',
            "--decision-threshold", str(args.decision_threshold),
            "--output", f'"{paths["output_dir"] / "hard_error_holdout_predictions.csv"}"',
            "--batch-size", str(args.batch_size),
            "--device", args.device,
        ),
    }


def write_readme(path: Path, plan: dict) -> None:
    lines = [
        "# Hard-Error Fine-Tune Package",
        "",
        f"This package is built from real false positives and false negatives exported at threshold {plan['decision_threshold']:.3g}.",
        "Group allocation is done at source_group_id level for test rows, and only real error rows receive explicit training weights.",
        "",
        "## Files",
        f"- Split file: `{plan['outputs']['split_csv']}`",
        f"- Hard train samples: `{plan['outputs']['hard_train_csv']}`",
        f"- Hard val samples: `{plan['outputs']['hard_val_csv']}`",
        f"- Hard holdout samples: `{plan['outputs']['hard_holdout_csv']}`",
        "",
        "## Suggested Command Order",
    ]
    for name, command in plan["commands"].items():
        lines.extend(["", f"### {name}", "```powershell", command, "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_package(args) -> dict:
    ratios = SplitRatios(train=args.hard_train_ratio, val=args.hard_val_ratio)
    if ratios.train <= 0 or ratios.val <= 0 or ratios.holdout <= 0:
        raise ValueError("hard split ratios must be positive and leave room for holdout")
    if args.fp_weight <= 0 or args.fn_weight <= 0:
        raise ValueError("hard sample weights must be positive")

    source_rows = read_csv_rows(resolve_path(args.source_split))
    error_rows = load_error_rows(args.false_positives, args.false_negatives, focus=args.error_focus)
    rows, counts = build_hard_error_rows(
        source_rows,
        error_rows,
        ratios=ratios,
        seed=args.seed,
        fp_weight=args.fp_weight,
        fn_weight=args.fn_weight,
        eligible_split=args.eligible_split,
        strict_source_group_isolation=args.strict_source_group_isolation,
    )
    validate_no_group_cross_split(rows)
    if args.strict_source_group_isolation:
        validate_no_hard_source_group_cross_split(rows)

    output_dir = resolve_path(args.output_dir)
    paths = {
        "output_dir": output_dir,
        "split_csv": output_dir / "hard_error_finetune_split.csv",
        "hard_train_csv": output_dir / "hard_error_train_samples.csv",
        "hard_val_csv": output_dir / "hard_error_val_samples.csv",
        "hard_holdout_csv": output_dir / "hard_error_holdout_samples.csv",
        "plan_json": output_dir / "hard_error_finetune_plan.json",
        "readme": output_dir / "README.md",
    }

    hard_train_rows = [
        row for row in rows
        if row["split"] == "train" and row["hard_family_role"].startswith("hard_error_")
    ]
    hard_val_rows = [
        row for row in rows
        if row["split"] == "val" and row["hard_family_role"].startswith("hard_error_")
    ]
    hard_holdout_rows = [
        row for row in rows
        if row["split"] == "test" and row["hard_family_role"].startswith("hard_error_")
    ]

    write_csv(paths["split_csv"], rows, OUTPUT_COLUMNS)
    write_csv(paths["hard_train_csv"], hard_train_rows, OUTPUT_COLUMNS)
    write_csv(paths["hard_val_csv"], hard_val_rows, OUTPUT_COLUMNS)
    write_csv(paths["hard_holdout_csv"], hard_holdout_rows, OUTPUT_COLUMNS)

    plan = {
        "purpose": f"Second-round fine-tuning package from real {args.error_focus.upper()} errors.",
        "source_split": str(resolve_path(args.source_split)),
        "false_positives": str(resolve_path(args.false_positives)),
        "false_negatives": str(resolve_path(args.false_negatives)),
        "error_focus": args.error_focus,
        "eligible_split": args.eligible_split,
        "strict_source_group_isolation": args.strict_source_group_isolation,
        "decision_threshold": args.decision_threshold,
        "hard_split_ratios": {"train": ratios.train, "val": ratios.val, "holdout": ratios.holdout},
        "weights": {"FP": args.fp_weight, "FN": args.fn_weight},
        "counts": counts,
        "outputs": {name: str(path) for name, path in paths.items() if name != "output_dir"},
    }
    plan["commands"] = build_commands(args, paths)

    output_dir.mkdir(parents=True, exist_ok=True)
    with paths["plan_json"].open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    write_readme(paths["readme"], plan)
    return plan


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Build hard-error fine-tuning split and command package.")
    parser.add_argument("--source-split", type=Path, default=Path("reports/hard_family_finetune/hard_family_finetune_split.csv"))
    parser.add_argument("--false-positives", type=Path, default=Path("reports/hard_family_finetune/error_analysis_threshold055/false_positives.csv"))
    parser.add_argument("--false-negatives", type=Path, default=Path("reports/hard_family_finetune/error_analysis_threshold055/false_negatives.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hard_family_finetune/hard_error_finetune_threshold055"))
    parser.add_argument("--hard-train-ratio", type=float, default=0.60)
    parser.add_argument("--hard-val-ratio", type=float, default=0.20)
    parser.add_argument("--fp-weight", type=float, default=4.0)
    parser.add_argument("--fn-weight", type=float, default=4.0)
    parser.add_argument("--error-focus", type=str, default="both", choices=["both", "fp", "fn"],
                        help="Use both FP/FN errors, only FP errors, or only FN errors.")
    parser.add_argument("--eligible-split", type=str, default="test", choices=["train", "val", "test"],
                        help="Only source rows from this split can be reallocated as hard examples.")
    parser.add_argument("--strict-source-group-isolation", action="store_true", default=False,
                        help=(
                            "When any source_group_id has hard errors, move all rows from that "
                            "source_group_id together with the allocated hard split."
                        ))
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--decision-threshold", type=float, default=0.55)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/group_isolated_rare_weighted_ft_rebuilt_cache/best_model.pt"))
    parser.add_argument("--model-output-dir", type=Path, default=Path("models/group_isolated_hard_error_ft_threshold055"))
    parser.add_argument("--config", type=Path, default=Path("config/default_config.toml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--samples-per-class", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--singleton-group-weight", type=float, default=1.8)
    parser.add_argument("--rare-group-weight", type=float, default=1.5)
    parser.add_argument("--medium-group-weight", type=float, default=1.2)
    parser.add_argument("--extract-workers", type=int, default=16)
    parser.add_argument("--extract-backend", type=str, default="process", choices=["thread", "process"])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    plan = build_package(args)
    counts = plan["counts"]
    print("=" * 60)
    print("Hard-Error Fine-Tune Package")
    print("=" * 60)
    print(f"Split file: {plan['outputs']['split_csv']}")
    print(f"Hard rows by split: {counts['hard_rows_by_split']}")
    print(f"Hard error types: {counts['hard_error_type_counts']}")
    print(f"Weighted train samples: {counts['weighted_train_samples']}")
    print(f"Plan: {plan['outputs']['plan_json']}")
    print("Training was not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
