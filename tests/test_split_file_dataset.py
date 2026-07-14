import csv
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import create_split_from_file  # noqa: E402


class DummyDataset(Dataset):
    def __init__(self, file_list, labels=None, source_sha256=None):
        self.file_list = [Path(path) for path in file_list]
        self.label_list = list(labels) if labels is not None else [0 for _ in file_list]
        if source_sha256 is not None:
            self.samples = [
                {"source_path": str(path), "label": label, "source_sha256": sha}
                for path, label, sha in zip(self.file_list, self.label_list, source_sha256)
            ]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return torch.tensor(index)


def _write_split_file(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_path",
                "source_sha256",
                "label",
                "sample_index",
                "group_id",
                "group_size",
                "split",
                "sample_weight",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_create_split_from_file_matches_relative_dataset_paths(tmp_path):
    dataset = DummyDataset([
        "data/benign/a.exe",
        "data/benign/b.exe",
        "data/malicious/c.exe",
    ])
    split_file = tmp_path / "group_isolated_split.csv"
    root = Path.cwd()
    _write_split_file(split_file, [
        {"source_path": str(root / "data" / "benign" / "a.exe"), "label": 0, "sample_index": 0, "group_id": 1, "split": "train"},
        {"source_path": str(root / "data" / "benign" / "b.exe"), "label": 0, "sample_index": 1, "group_id": 2, "split": "val"},
        {"source_path": str(root / "data" / "malicious" / "c.exe"), "label": 1, "sample_index": 2, "group_id": 3, "split": "test"},
    ])

    train_dataset, val_dataset, test_dataset = create_split_from_file(dataset, split_file)

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1
    assert len(test_dataset) == 1
    assert train_dataset[0].item() == 0
    assert val_dataset[0].item() == 1
    assert test_dataset[0].item() == 2


def test_create_split_from_file_rejects_group_cross_split(tmp_path):
    dataset = DummyDataset(["data/a.exe", "data/b.exe", "data/c.exe"])
    split_file = tmp_path / "bad_split.csv"
    _write_split_file(split_file, [
        {"source_path": "data/a.exe", "label": 0, "sample_index": 0, "group_id": 1, "split": "train"},
        {"source_path": "data/b.exe", "label": 0, "sample_index": 1, "group_id": 1, "split": "test"},
        {"source_path": "data/c.exe", "label": 1, "sample_index": 2, "group_id": 2, "split": "val"},
    ])

    with pytest.raises(ValueError, match="Group 1"):
        create_split_from_file(dataset, split_file)


def test_create_split_from_file_adds_train_weights_only(tmp_path):
    dataset = DummyDataset(["data/a.exe", "data/b.exe", "data/c.exe"])
    split_file = tmp_path / "weighted_split.csv"
    _write_split_file(split_file, [
        {"source_path": "data/a.exe", "label": 0, "sample_index": 0, "group_id": 1, "split": "train", "group_size": 1},
        {"source_path": "data/b.exe", "label": 0, "sample_index": 1, "group_id": 2, "split": "val", "group_size": 1},
        {"source_path": "data/c.exe", "label": 1, "sample_index": 2, "group_id": 3, "split": "test", "group_size": 10},
    ])

    train_dataset, val_dataset, test_dataset = create_split_from_file(
        dataset,
        split_file,
        rare_group_weighting=True,
        singleton_group_weight=1.8,
        rare_group_weight=1.5,
        medium_group_weight=1.2,
    )

    train_sample = train_dataset[0]
    assert len(train_sample) == 2
    assert train_sample[0].item() == 0
    assert train_sample[1].item() == pytest.approx(1.8)
    assert not isinstance(val_dataset[0], tuple)
    assert not isinstance(test_dataset[0], tuple)


def test_create_split_from_file_uses_explicit_train_sample_weight(tmp_path):
    dataset = DummyDataset(["data/a.exe", "data/b.exe", "data/c.exe"])
    split_file = tmp_path / "explicit_weight_split.csv"
    _write_split_file(split_file, [
        {"source_path": "data/a.exe", "label": 1, "sample_index": 0, "group_id": 1, "split": "train", "sample_weight": 6.0},
        {"source_path": "data/b.exe", "label": 0, "sample_index": 1, "group_id": 2, "split": "val"},
        {"source_path": "data/c.exe", "label": 1, "sample_index": 2, "group_id": 3, "split": "test"},
    ])

    train_dataset, val_dataset, test_dataset = create_split_from_file(dataset, split_file)

    train_sample = train_dataset[0]
    assert len(train_sample) == 2
    assert train_sample[0].item() == 0
    assert train_sample[1].item() == pytest.approx(6.0)
    assert not isinstance(val_dataset[0], tuple)
    assert not isinstance(test_dataset[0], tuple)


def test_strict_split_requires_source_sha256_column(tmp_path):
    dataset = DummyDataset(["data/a.exe", "data/b.exe", "data/c.exe"], labels=[0, 0, 1])
    split_file = tmp_path / "missing_sha_split.csv"
    with split_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_path", "label", "sample_index", "split"])
        writer.writeheader()
        writer.writerows([
            {"source_path": "data/a.exe", "label": 0, "sample_index": 0, "split": "train"},
            {"source_path": "data/b.exe", "label": 0, "sample_index": 1, "split": "val"},
            {"source_path": "data/c.exe", "label": 1, "sample_index": 2, "split": "test"},
        ])

    with pytest.raises(ValueError, match="source_sha256"):
        create_split_from_file(dataset, split_file, require_explicit_metadata=True)


def test_strict_split_rejects_label_mismatch(tmp_path):
    dataset = DummyDataset(
        ["data/a.exe", "data/b.exe", "data/c.exe"],
        labels=[0, 0, 1],
        source_sha256=["a" * 64, "b" * 64, "c" * 64],
    )
    split_file = tmp_path / "label_mismatch_split.csv"
    _write_split_file(split_file, [
        {"source_path": "data/a.exe", "source_sha256": "a" * 64, "label": 1, "sample_index": 0, "split": "train"},
        {"source_path": "data/b.exe", "source_sha256": "b" * 64, "label": 0, "sample_index": 1, "split": "val"},
        {"source_path": "data/c.exe", "source_sha256": "c" * 64, "label": 1, "sample_index": 2, "split": "test"},
    ])

    with pytest.raises(ValueError, match="label_mismatch"):
        create_split_from_file(dataset, split_file, require_explicit_metadata=True)


def test_strict_split_rejects_source_sha256_mismatch(tmp_path):
    dataset = DummyDataset(
        ["data/a.exe", "data/b.exe", "data/c.exe"],
        labels=[0, 0, 1],
        source_sha256=["a" * 64, "b" * 64, "c" * 64],
    )
    split_file = tmp_path / "sha_mismatch_split.csv"
    _write_split_file(split_file, [
        {"source_path": "data/a.exe", "source_sha256": "0" * 64, "label": 0, "sample_index": 0, "split": "train"},
        {"source_path": "data/b.exe", "source_sha256": "b" * 64, "label": 0, "sample_index": 1, "split": "val"},
        {"source_path": "data/c.exe", "source_sha256": "c" * 64, "label": 1, "sample_index": 2, "split": "test"},
    ])

    with pytest.raises(ValueError, match="source_sha256_mismatch"):
        create_split_from_file(dataset, split_file, require_explicit_metadata=True)


def test_strict_split_accepts_explicit_label_and_content_hash_metadata(tmp_path):
    dataset = DummyDataset(
        ["data/a.exe", "data/b.exe", "data/c.exe"],
        labels=[0, 0, 1],
        source_sha256=["a" * 64, "b" * 64, "c" * 64],
    )
    split_file = tmp_path / "strict_ok_split.csv"
    _write_split_file(split_file, [
        {"source_path": "data/a.exe", "source_sha256": "a" * 64, "label": 0, "sample_index": 0, "split": "train"},
        {"source_path": "data/b.exe", "source_sha256": "b" * 64, "label": 0, "sample_index": 1, "split": "val"},
        {"source_path": "data/c.exe", "source_sha256": "c" * 64, "label": 1, "sample_index": 2, "split": "test"},
    ])

    train_dataset, val_dataset, test_dataset = create_split_from_file(
        dataset,
        split_file,
        require_explicit_metadata=True,
    )

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1
    assert len(test_dataset) == 1
