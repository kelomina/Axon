#!/usr/bin/env python3
"""Build raw-file group distribution and group-isolated split reports."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from raw_group_tools import (
    build_group_membership,
    load_config,
    read_json,
    load_similarity_groups,
    make_group_isolated_split,
    resolve_path,
    scan_raw_sample_records,
    summarize_group_distribution,
    write_csv,
    write_json,
)


GROUP_COLUMNS = [
    "group_id",
    "source",
    "size",
    "labels",
    "splits",
    "label_count",
    "dominant_label",
    "dominant_label_count",
    "train_count",
    "val_count",
    "test_count",
    "has_label_conflict",
    "has_cross_split",
    "has_leakage",
    "is_rare_group",
    "is_singleton",
    "train_too_small",
]

MEMBER_COLUMNS = [
    "group_id",
    "group_source",
    "group_size",
    "is_rare_group",
    "is_singleton",
    "sample_index",
    "source_path",
    "label",
    "split",
]

SPLIT_COLUMNS = [
    "source_path",
    "label",
    "sample_index",
    "group_id",
    "group_size",
    "split",
    "is_rare_group",
    "group_source",
]


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Analyze raw-file similarity groups and create group-isolated split suggestions."
    )
    parser.add_argument("--config", type=Path, default=Path("config/default_config.toml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--raw-report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/raw_group_diagnostics"))
    parser.add_argument("--rare-threshold", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _raw_config, config = load_config(args.config)
    raw_report_dir = resolve_path(args.raw_report_dir)
    summary_path = raw_report_dir / "raw_similarity_summary.json"
    max_samples_per_class = None
    if summary_path.exists():
        max_samples_per_class = read_json(summary_path).get("max_samples_per_class")
    records = scan_raw_sample_records(config, args.data_dir, max_samples_per_class=max_samples_per_class)

    output_dir = resolve_path(args.output_dir)
    raw_group_rows = load_similarity_groups(raw_report_dir)
    group_rows, member_rows, _path_to_group_id = build_group_membership(
        raw_group_rows,
        records,
        rare_threshold=args.rare_threshold,
    )
    distribution_summary = summarize_group_distribution(group_rows, records, args.rare_threshold)

    split_rows, split_summary = make_group_isolated_split(group_rows, member_rows, config)
    summary = {
        "raw_report_dir": str(raw_report_dir),
        "output_dir": str(output_dir),
        "distribution": distribution_summary,
        "group_isolated_split": split_summary,
        "outputs": {
            "group_distribution_csv": str(output_dir / "group_distribution.csv"),
            "group_members_csv": str(output_dir / "group_members.csv"),
            "group_distribution_json": str(output_dir / "group_distribution_summary.json"),
            "group_isolated_split_csv": str(output_dir / "group_isolated_split.csv"),
            "group_isolated_split_json": str(output_dir / "group_isolated_split_summary.json"),
        },
    }

    write_csv(output_dir / "group_distribution.csv", group_rows, GROUP_COLUMNS)
    write_csv(output_dir / "group_members.csv", member_rows, MEMBER_COLUMNS)
    write_json(output_dir / "group_distribution_summary.json", summary)
    write_csv(output_dir / "group_isolated_split.csv", split_rows, SPLIT_COLUMNS)
    write_json(output_dir / "group_isolated_split_summary.json", split_summary)

    print("=" * 60)
    print("Raw Group Diagnostics")
    print("=" * 60)
    print(f"Samples: {distribution_summary['total_samples']}")
    print(f"Groups: {distribution_summary['total_groups']}")
    print(f"Rare groups: {distribution_summary['rare_groups']}")
    print(f"Singleton groups: {distribution_summary['singleton_groups']}")
    print(f"Leakage groups: {distribution_summary['leakage_groups']}")
    print(f"Largest group size: {distribution_summary['largest_group_size']}")
    print(f"Group distribution: {summary['outputs']['group_distribution_csv']}")
    print(f"Group-isolated split: {summary['outputs']['group_isolated_split_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
