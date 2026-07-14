#!/usr/bin/env python3
"""Build a hard-family fine-tuning package from known false-negative groups.

The generated split file is only used to organize training and evaluation.
It does not add path, hash, group id, or any rule feature to the model input.
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


BASE_COLUMNS = [
    "source_path",
    "label",
    "sample_index",
    "group_id",
    "group_size",
    "split",
    "is_rare_group",
    "group_source",
]

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
    "is_rare_group",
    "group_source",
]


@dataclass(frozen=True)
class HardSplitRatios:
    train: float
    val: float


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


def parse_group_ids(text: str) -> set[str]:
    group_ids = {item.strip() for item in text.split(",") if item.strip()}
    if not group_ids:
        raise ValueError("--hard-groups must contain at least one group id")
    return group_ids


def stable_sample_key(row: dict) -> tuple[int, str]:
    try:
        sample_index = int(row.get("sample_index", ""))
    except ValueError:
        sample_index = 0
    return sample_index, row.get("source_path", "")


def allocate_hard_group(
    rows: list[dict],
    ratios: HardSplitRatios,
    seed: int,
    group_id: str,
) -> dict[str, set[str]]:
    """Return source_path sets for train/val/test within one hard group."""
    if len(rows) < 3:
        raise ValueError(f"Hard group {group_id} needs at least 3 samples, got {len(rows)}")

    shuffled = sorted(rows, key=stable_sample_key)
    random.Random(seed + int(group_id)).shuffle(shuffled)

    n_total = len(shuffled)
    n_train = max(1, round(n_total * ratios.train))
    n_val = max(1, round(n_total * ratios.val))
    if n_train + n_val >= n_total:
        n_train = max(1, n_total - 2)
        n_val = 1
    n_test = n_total - n_train - n_val
    if n_test <= 0:
        raise ValueError(f"Hard group {group_id} produced empty holdout split")

    train_rows = shuffled[:n_train]
    val_rows = shuffled[n_train:n_train + n_val]
    test_rows = shuffled[n_train + n_val:]
    return {
        "train": {normalize_path_text(row["source_path"]) for row in train_rows},
        "val": {normalize_path_text(row["source_path"]) for row in val_rows},
        "test": {normalize_path_text(row["source_path"]) for row in test_rows},
    }


def load_predictions_by_path(predictions_path: Optional[Path]) -> dict[str, dict]:
    if predictions_path is None:
        return {}
    path = resolve_path(predictions_path)
    if not path.exists():
        return {}
    return {normalize_path_text(row["source_path"]): row for row in read_csv_rows(path)}


def build_finetune_rows(
    source_rows: Sequence[dict],
    hard_group_ids: set[str],
    ratios: HardSplitRatios,
    hard_weight: float,
    seed: int,
) -> tuple[list[dict], dict]:
    by_group = defaultdict(list)
    for row in source_rows:
        if row.get("group_id") in hard_group_ids:
            by_group[row["group_id"]].append(row)

    missing_groups = sorted(hard_group_ids - set(by_group), key=int)
    if missing_groups:
        raise ValueError(f"Hard groups not found in split file: {missing_groups}")

    allocation_by_path = {}
    allocation_summary = {}
    for group_id in sorted(by_group, key=int):
        allocation = allocate_hard_group(by_group[group_id], ratios, seed, group_id)
        allocation_summary[group_id] = {split: len(paths) for split, paths in allocation.items()}
        for split, keys in allocation.items():
            for key in keys:
                allocation_by_path[key] = split

    output_rows = []
    for row in source_rows:
        original_group_id = row.get("group_id", "")
        key = normalize_path_text(row.get("source_path", ""))
        hard_split = allocation_by_path.get(key)
        if hard_split is None:
            split = row.get("split", "")
            sample_weight = ""
            hard_role = "base_" + split
            group_id = original_group_id
        else:
            split = hard_split
            sample_weight = f"{hard_weight:.6g}" if hard_split == "train" else ""
            hard_role = "hard_train" if hard_split == "train" else ("hard_val" if hard_split == "val" else "hard_holdout")
            # create_split_from_file enforces one group_id per split. For this
            # deliberate hard-family experiment we keep source_group_id for
            # analysis and use a split-specific group_id only for validation.
            group_id = f"{original_group_id}_hard_{hard_split}"

        output_rows.append({
            "source_path": row.get("source_path", ""),
            "label": row.get("label", ""),
            "sample_index": row.get("sample_index", ""),
            "group_id": group_id,
            "source_group_id": original_group_id,
            "group_size": row.get("group_size", ""),
            "split": split,
            "sample_weight": sample_weight,
            "hard_family_role": hard_role,
            "is_rare_group": row.get("is_rare_group", ""),
            "group_source": row.get("group_source", ""),
        })

    return output_rows, allocation_summary


def summarize_predictions(rows: Sequence[dict], predictions_by_path: dict[str, dict]) -> dict:
    hard_rows = [row for row in rows if row["hard_family_role"].startswith("hard_")]
    matched = []
    for row in hard_rows:
        prediction = predictions_by_path.get(normalize_path_text(row["source_path"]))
        if prediction is not None:
            matched.append((row, prediction))

    by_role = defaultdict(Counter)
    for row, prediction in matched:
        role = row["hard_family_role"]
        correct = str(prediction.get("correct", "")).lower() in {"true", "1", "yes"}
        by_role[role]["matched"] += 1
        by_role[role]["correct" if correct else "error"] += 1
    return {role: dict(counter) for role, counter in sorted(by_role.items())}


def ps_command(*parts: str) -> str:
    return " ".join(parts)


def build_commands(args, paths: dict[str, Path]) -> dict[str, str]:
    python_exe = r'& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe"'
    authorized_main = (
        'scripts\\authorized_main.py --ml-preflight '
        '"reports\\random_20w_split\\loop104_ml_authorization_preflight.json" --'
    )
    checkpoint = str(resolve_path(args.checkpoint))
    finetuned_checkpoint = str(resolve_path(args.model_output_dir) / "best_model.pt")
    data_dir = str(resolve_path(args.data_dir))
    config = str(resolve_path(args.config))

    baseline_predictions = paths["output_dir"] / "baseline_eval_predictions.csv"
    finetuned_predictions = paths["output_dir"] / "finetuned_eval_predictions.csv"
    baseline_holdout_predictions = paths["output_dir"] / "baseline_hard_holdout_predictions.csv"
    finetuned_holdout_predictions = paths["output_dir"] / "finetuned_hard_holdout_predictions.csv"

    return {
        "baseline_overall_eval": ps_command(
            python_exe, authorized_main, "eval",
            "--checkpoint", f'"{checkpoint}"',
            "--data-dir", f'"{data_dir}"',
            "--split-file", f'"{paths["split_csv"]}"',
            "--split test",
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--output", f'"{paths["output_dir"] / "baseline_overall_eval.json"}"',
        ),
        "baseline_eval_predictions": ps_command(
            python_exe, "scripts\\export_sample_predictions.py",
            "--checkpoint", f'"{checkpoint}"',
            "--config", f'"{config}"',
            "--data-dir", f'"{data_dir}"',
            "--samples", f'"{paths["eval_samples_csv"]}"',
            "--output", f'"{baseline_predictions}"',
            "--batch-size", str(args.batch_size),
            "--device", args.device,
        ),
        "baseline_hard_holdout_predictions": ps_command(
            python_exe, "scripts\\export_sample_predictions.py",
            "--checkpoint", f'"{checkpoint}"',
            "--config", f'"{config}"',
            "--data-dir", f'"{data_dir}"',
            "--samples", f'"{paths["hard_holdout_csv"]}"',
            "--output", f'"{baseline_holdout_predictions}"',
            "--batch-size", str(args.batch_size),
            "--device", args.device,
        ),
        "fine_tune": ps_command(
            python_exe, authorized_main, "train",
            "--config", f'"{config}"',
            "--data-dir", f'"{data_dir}"',
            "--split-file", f'"{paths["split_csv"]}"',
            "--init-checkpoint", f'"{checkpoint}"',
            "--output-dir", f'"{resolve_path(args.model_output_dir)}"',
            "--epochs", str(args.epochs),
            "--lr", str(args.learning_rate),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
        ),
        "finetuned_overall_eval": ps_command(
            python_exe, authorized_main, "eval",
            "--checkpoint", f'"{finetuned_checkpoint}"',
            "--data-dir", f'"{data_dir}"',
            "--split-file", f'"{paths["split_csv"]}"',
            "--split test",
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--output", f'"{paths["output_dir"] / "finetuned_overall_eval.json"}"',
        ),
        "finetuned_eval_predictions": ps_command(
            python_exe, "scripts\\export_sample_predictions.py",
            "--checkpoint", f'"{finetuned_checkpoint}"',
            "--config", f'"{config}"',
            "--data-dir", f'"{data_dir}"',
            "--samples", f'"{paths["eval_samples_csv"]}"',
            "--output", f'"{finetuned_predictions}"',
            "--batch-size", str(args.batch_size),
            "--device", args.device,
        ),
        "finetuned_hard_holdout_predictions": ps_command(
            python_exe, "scripts\\export_sample_predictions.py",
            "--checkpoint", f'"{finetuned_checkpoint}"',
            "--config", f'"{config}"',
            "--data-dir", f'"{data_dir}"',
            "--samples", f'"{paths["hard_holdout_csv"]}"',
            "--output", f'"{finetuned_holdout_predictions}"',
            "--batch-size", str(args.batch_size),
            "--device", args.device,
        ),
    }


def write_readme(path: Path, plan: dict) -> None:
    lines = [
        "# Hard-Family Fine-Tune Package",
        "",
        "This package is for real model fine-tuning, not rule-based override.",
        "The split uses group ids only to choose train/val/holdout rows.",
        "The model still receives only byte sequence, PE features, and statistical features.",
        "",
        "## Files",
        f"- Split file: `{plan['outputs']['split_csv']}`",
        f"- Overall eval samples: `{plan['outputs']['eval_samples_csv']}`",
        f"- Hard holdout samples: `{plan['outputs']['hard_holdout_csv']}`",
        f"- Hard train samples: `{plan['outputs']['hard_train_csv']}`",
        "",
        "## Suggested Command Order",
    ]
    for name, command in plan["commands"].items():
        lines.extend(["", f"### {name}", "```powershell", command, "```"])
    lines.extend([
        "",
        "Compare baseline and fine-tuned outputs on the same hard holdout samples.",
        "A real improvement should raise hard-family recall without a large false-positive increase on the overall test split.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_package(args) -> dict:
    source_split = resolve_path(args.source_split)
    output_dir = resolve_path(args.output_dir)
    hard_group_ids = parse_group_ids(args.hard_groups)
    ratios = HardSplitRatios(train=args.hard_train_ratio, val=args.hard_val_ratio)
    if ratios.train <= 0 or ratios.val <= 0 or ratios.train + ratios.val >= 1:
        raise ValueError("Hard split ratios must be positive and leave room for holdout")
    if args.hard_weight <= 0:
        raise ValueError("--hard-weight must be positive")

    source_rows = read_csv_rows(source_split)
    rows, allocation_summary = build_finetune_rows(
        source_rows,
        hard_group_ids=hard_group_ids,
        ratios=ratios,
        hard_weight=args.hard_weight,
        seed=args.seed,
    )

    paths = {
        "output_dir": output_dir,
        "split_csv": output_dir / "hard_family_finetune_split.csv",
        "eval_samples_csv": output_dir / "hard_family_eval_samples.csv",
        "hard_holdout_csv": output_dir / "hard_family_holdout_samples.csv",
        "hard_train_csv": output_dir / "hard_family_train_samples.csv",
        "plan_json": output_dir / "hard_family_finetune_plan.json",
        "readme": output_dir / "README.md",
    }

    eval_rows = [row for row in rows if row["split"] == "test"]
    hard_holdout_rows = [row for row in rows if row["hard_family_role"] == "hard_holdout"]
    hard_train_rows = [row for row in rows if row["hard_family_role"] == "hard_train"]

    write_csv(paths["split_csv"], rows, OUTPUT_COLUMNS)
    write_csv(paths["eval_samples_csv"], eval_rows, OUTPUT_COLUMNS)
    write_csv(paths["hard_holdout_csv"], hard_holdout_rows, OUTPUT_COLUMNS)
    write_csv(paths["hard_train_csv"], hard_train_rows, OUTPUT_COLUMNS)

    predictions_by_path = load_predictions_by_path(args.predictions)
    split_counts = Counter(row["split"] for row in rows)
    role_counts = Counter(row["hard_family_role"] for row in rows)
    hard_counts_by_group = defaultdict(Counter)
    for row in rows:
        if row["hard_family_role"].startswith("hard_"):
            hard_counts_by_group[row["source_group_id"]][row["hard_family_role"]] += 1

    plan = {
        "purpose": "Hard-example fine-tuning package. No rule features are added to model input.",
        "source_split": str(source_split),
        "hard_groups": sorted(hard_group_ids, key=int),
        "hard_weight": args.hard_weight,
        "hard_split_ratios": {"train": ratios.train, "val": ratios.val, "holdout": 1 - ratios.train - ratios.val},
        "counts": {
            "total_rows": len(rows),
            "split_counts": dict(sorted(split_counts.items())),
            "hard_role_counts": dict(sorted(role_counts.items())),
            "hard_counts_by_group": {
                group_id: dict(counter)
                for group_id, counter in sorted(hard_counts_by_group.items(), key=lambda item: int(item[0]))
            },
            "allocation_by_group": allocation_summary,
        },
        "baseline_prediction_summary": summarize_predictions(rows, predictions_by_path),
        "outputs": {name: str(path) for name, path in paths.items() if name != "output_dir"},
        "model_input_guardrail": [
            "source_path is used only to locate cached sample arrays",
            "group_id/source_group_id are used only for split construction and reporting",
            "sample_weight affects only training loss, not validation/test metrics",
            "SHA or family-rule matching is not used",
        ],
    }
    plan["commands"] = build_commands(args, paths)

    output_dir.mkdir(parents=True, exist_ok=True)
    with paths["plan_json"].open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    write_readme(paths["readme"], plan)
    return plan


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Build hard-family fine-tuning split and command package.")
    parser.add_argument("--source-split", type=Path, default=Path("reports/raw_group_diagnostics/group_isolated_split.csv"))
    parser.add_argument("--predictions", type=Path, default=Path("reports/group_isolated_rare_weighted_diagnostics/sample_predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hard_family_finetune"))
    parser.add_argument("--hard-groups", type=str, default="10,19,26,39,52,59")
    parser.add_argument("--hard-train-ratio", type=float, default=0.60)
    parser.add_argument("--hard-val-ratio", type=float, default=0.20)
    parser.add_argument("--hard-weight", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/group_isolated_rare_weighted_ft/best_model.pt"))
    parser.add_argument("--model-output-dir", type=Path, default=Path("models/hard_family_finetune_v1"))
    parser.add_argument("--config", type=Path, default=Path("config/default_config.toml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    plan = build_package(args)
    print("=" * 60)
    print("Hard-Family Fine-Tune Package")
    print("=" * 60)
    print(f"Hard groups: {','.join(plan['hard_groups'])}")
    print(f"Split file: {plan['outputs']['split_csv']}")
    print(f"Hard train samples: {plan['counts']['hard_role_counts'].get('hard_train', 0)}")
    print(f"Hard val samples: {plan['counts']['hard_role_counts'].get('hard_val', 0)}")
    print(f"Hard holdout samples: {plan['counts']['hard_role_counts'].get('hard_holdout', 0)}")
    print(f"Plan: {plan['outputs']['plan_json']}")
    print("Training was not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
