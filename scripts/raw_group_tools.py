"""Shared helpers for raw sample group diagnostics."""

import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
for path in [SCRIPT_DIR, SRC_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_raw_similarity import iter_sorted_files  # noqa: E402
from analyze_similarity import assign_splits, build_experiment_config, read_toml_config, resolve_path  # noqa: E402

csv.field_size_limit(sys.maxsize)


@dataclass
class RawSample:
    index: int
    source_path: str
    label: int
    split: str = "unknown"


class RawSampleRecords:
    def __init__(self, records: Sequence[RawSample]):
        self.records = list(records)
        self.label_list = [record.label for record in self.records]

    def __len__(self):
        return len(self.records)


def normalize_path_text(path: str) -> str:
    return str(Path(path)).casefold()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def configured_roots(data_dir: Path, config) -> List[Tuple[Path, int]]:
    roots = []
    for dirname in getattr(config, "benign_dir_names_fs", ["benign", "待加入白名单"]):
        root = data_dir / dirname
        if root.exists():
            roots.append((root, 0))
    for dirname in getattr(config, "malicious_dir_names_fs", ["malicious", "待拉黑"]):
        root = data_dir / dirname
        if root.exists():
            roots.append((root, 1))
    return roots


def load_config(config_path: Optional[Path]):
    raw_config = read_toml_config(resolve_path(config_path) if config_path else None)
    return raw_config, build_experiment_config(raw_config)


def scan_raw_sample_records(
    config,
    data_dir: Optional[Path] = None,
    max_samples_per_class: Optional[int] = None,
) -> List[RawSample]:
    data_root = resolve_path(data_dir or Path(config.data_dir or "data"))
    records = []
    for root, label in configured_roots(data_root, config):
        count = 0
        for path in iter_sorted_files(root):
            if max_samples_per_class is not None and count >= max_samples_per_class:
                break
            if not path.is_file():
                continue
            records.append(
                RawSample(
                    index=len(records),
                    source_path=str(path),
                    label=label,
                )
            )
            count += 1
    container = RawSampleRecords(records)
    assign_splits(container.records, config)
    return container.records


def parse_count_map(text: str) -> Dict[str, int]:
    result = {}
    if not text:
        return result
    for item in str(text).split("|"):
        if not item:
            continue
        key, _, value = item.partition(":")
        if key:
            result[key] = int(value or 0)
    return result


def format_count_map(counter: Counter) -> str:
    return "|".join(f"{key}:{counter[key]}" for key in sorted(counter))


def parse_pipe_paths(text: str) -> List[str]:
    if not text:
        return []
    return [item for item in str(text).split("|") if item]


def load_similarity_groups(raw_report_dir: Path) -> List[dict]:
    groups_path = raw_report_dir / "raw_similarity_groups.csv"
    if not groups_path.exists():
        raise FileNotFoundError(f"Raw similarity groups report not found: {groups_path}")
    return read_csv_rows(groups_path)


def build_group_membership(
    raw_group_rows: Sequence[dict],
    records: Sequence[RawSample],
    rare_threshold: int,
) -> Tuple[List[dict], List[dict], Dict[str, int]]:
    path_to_sample = {normalize_path_text(record.source_path): record for record in records}
    assigned_paths = set()
    group_rows = []
    member_rows = []
    path_to_group_id: Dict[str, int] = {}

    for row in raw_group_rows:
        group_id = int(row["group_id"])
        paths = parse_pipe_paths(row.get("source_paths", ""))
        members = []
        for path in paths:
            sample = path_to_sample.get(normalize_path_text(path))
            if sample is not None:
                members.append(sample)
                assigned_paths.add(normalize_path_text(sample.source_path))
                path_to_group_id[normalize_path_text(sample.source_path)] = group_id

        if not members:
            continue
        group_rows.append(build_group_distribution_row(group_id, members, rare_threshold, source="similarity_group"))
        member_rows.extend(build_group_member_rows(group_rows[-1], members))

    next_group_id = max([int(row["group_id"]) for row in raw_group_rows], default=0) + 1
    for sample in records:
        key = normalize_path_text(sample.source_path)
        if key in assigned_paths:
            continue
        group_rows.append(
            build_group_distribution_row(
                next_group_id,
                [sample],
                rare_threshold,
                source="singleton",
            )
        )
        member_rows.extend(build_group_member_rows(group_rows[-1], [sample]))
        path_to_group_id[key] = next_group_id
        next_group_id += 1

    group_rows.sort(key=lambda item: (item["group_id"]))
    member_rows.sort(key=lambda item: (int(item["group_id"]), int(item["sample_index"])))
    return group_rows, member_rows, path_to_group_id


def build_group_distribution_row(group_id: int, members: Sequence[RawSample], rare_threshold: int, source: str) -> dict:
    labels = Counter(member.label for member in members)
    splits = Counter(member.split for member in members)
    size = len(members)
    train_count = splits.get("train", 0)
    val_count = splits.get("val", 0)
    test_count = splits.get("test", 0)
    return {
        "group_id": group_id,
        "source": source,
        "size": size,
        "labels": format_count_map(labels),
        "splits": format_count_map(splits),
        "label_count": len(labels),
        "dominant_label": labels.most_common(1)[0][0] if labels else "",
        "dominant_label_count": labels.most_common(1)[0][1] if labels else 0,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "has_label_conflict": len(labels) > 1,
        "has_cross_split": len(splits) > 1,
        "has_leakage": bool(train_count and (val_count or test_count)),
        "is_rare_group": size <= rare_threshold,
        "is_singleton": size == 1,
        "train_too_small": train_count <= 1,
    }


def build_group_member_rows(group_row: dict, members: Sequence[RawSample]) -> List[dict]:
    return [
        {
            "group_id": group_row["group_id"],
            "group_source": group_row["source"],
            "group_size": group_row["size"],
            "is_rare_group": group_row["is_rare_group"],
            "is_singleton": group_row["is_singleton"],
            "sample_index": member.index,
            "source_path": member.source_path,
            "label": member.label,
            "split": member.split,
        }
        for member in members
    ]


def group_size_bucket(size: int) -> str:
    if size == 1:
        return "1"
    if size <= 5:
        return "2-5"
    if size <= 20:
        return "6-20"
    if size <= 100:
        return "21-100"
    return "101+"


def summarize_group_distribution(rows: Sequence[dict], records: Sequence[RawSample], rare_threshold: int) -> dict:
    buckets = Counter(group_size_bucket(int(row["size"])) for row in rows)
    return {
        "total_samples": len(records),
        "total_groups": len(rows),
        "rare_threshold": rare_threshold,
        "rare_groups": sum(1 for row in rows if row["is_rare_group"]),
        "singleton_groups": sum(1 for row in rows if row["is_singleton"]),
        "leakage_groups": sum(1 for row in rows if row["has_leakage"]),
        "cross_split_groups": sum(1 for row in rows if row["has_cross_split"]),
        "label_conflict_groups": sum(1 for row in rows if row["has_label_conflict"]),
        "largest_group_size": max((int(row["size"]) for row in rows), default=0),
        "group_size_buckets": dict(sorted(buckets.items())),
    }


def make_group_isolated_split(
    group_rows: Sequence[dict],
    member_rows: Sequence[dict],
    config,
) -> Tuple[List[dict], dict]:
    """Greedy group-level split that keeps every group in one split."""
    total_samples = sum(int(row["size"]) for row in group_rows)
    targets = {
        "val": int(total_samples * config.val_ratio),
        "test": int(total_samples * config.test_ratio),
    }
    targets["train"] = max(0, total_samples - targets["val"] - targets["test"])
    assigned_counts = Counter()
    sorted_groups = sorted(group_rows, key=lambda row: (-int(row["size"]), int(row["group_id"])))
    split_rows = []
    members_by_group = defaultdict(list)
    for member in member_rows:
        members_by_group[int(member["group_id"])].append(member)

    for row in sorted_groups:
        size = int(row["size"])
        split = min(
            ["train", "val", "test"],
            key=lambda name: (
                (assigned_counts[name] + size) / max(targets[name], 1),
                assigned_counts[name],
            ),
        )
        assigned_counts[split] += size
        for member in members_by_group[int(row["group_id"])]:
            split_rows.append({
                "source_path": member["source_path"],
                "label": member["label"],
                "sample_index": member["sample_index"],
                "group_id": row["group_id"],
                "group_size": row["size"],
                "split": split,
                "is_rare_group": row["is_rare_group"],
                "group_source": row["source"],
            })

    summary = {
        "total_samples": total_samples,
        "target_counts": targets,
        "assigned_counts": {name: int(assigned_counts[name]) for name in ["train", "val", "test"]},
        "split_ratios": {
            name: (float(assigned_counts[name]) / total_samples if total_samples else 0.0)
            for name in ["train", "val", "test"]
        },
        "group_count": len(group_rows),
    }
    return split_rows, summary
