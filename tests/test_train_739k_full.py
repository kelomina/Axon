#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_train_739k_full.py

对 scripts/train_739k_full.py 的 DataLoader 构建与 7:1:2 划分比例进行单元测试。

覆盖范围：
- 数据划分比例常量必须为 7:1:2（训练 70% / 验证 10% / 测试 20%）；
- _build_dataloaders 生成的三个 DataLoader 必须带 num_workers=8、
  pin_memory=True、persistent_workers=True，训练集 shuffle=True + drop_last=True；
- 使用小规模 fake 数据集验证 create_stratified_split 实际划分比例接近 7:1:2。

说明：
- 测试不迭代 DataLoader（不触发子进程实际加载），只校验构造参数，
  避免 Windows spawn 进程在测试环境中残留。
"""

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_739k_full import (  # noqa: E402
    BATCH_SIZE,
    NUM_WORKERS,
    TEST_RATIO,
    TRAIN_RATIO,
    TRUNCATE_BYTE_LENGTH,
    VAL_RATIO,
    _build_dataloaders,
    _TruncatedByteDataset,
)

from dataset import SubDataset, create_stratified_split  # noqa: E402


class _FakeDataset(torch.utils.data.Dataset):
    """小规模 fake 数据集：仅提供长度与 getitem，用于 DataLoader 构造参数校验。"""

    def __init__(self, size: int, label: int = 0):
        self.size = size
        self.label = label

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        return torch.zeros(4, dtype=torch.float32), torch.tensor(self.label, dtype=torch.long)


class _LabeledDataset(torch.utils.data.Dataset):
    """带 label_list 属性的 fake 数据集，用于验证 create_stratified_split 分层比例。"""

    def __init__(self, labels):
        self.label_list = list(labels)
        self._labels = list(labels)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx: int):
        return torch.zeros(4, dtype=torch.float32), torch.tensor(self._labels[idx], dtype=torch.long)


def test_split_ratios_are_7_1_2():
    assert TRAIN_RATIO == 0.70
    assert VAL_RATIO == 0.10
    assert TEST_RATIO == 0.20
    assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-9


def test_default_batch_size_and_num_workers_constants():
    assert BATCH_SIZE == 64
    assert NUM_WORKERS == 8


def test_build_dataloaders_constructs_expected_parameters():
    train_dataset = _FakeDataset(size=10)
    val_dataset = _FakeDataset(size=4)
    test_dataset = _FakeDataset(size=4)

    train_loader, val_loader, test_loader = _build_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=2,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        seed=42,
    )

    assert train_loader.batch_size == 2
    assert train_loader.num_workers == 2
    assert train_loader.pin_memory is True
    assert train_loader.persistent_workers is True
    assert isinstance(train_loader.sampler, torch.utils.data.RandomSampler)
    assert train_loader.drop_last is True

    assert val_loader.batch_size == 2
    assert val_loader.num_workers == 2
    assert val_loader.pin_memory is True
    assert val_loader.persistent_workers is True
    assert isinstance(val_loader.sampler, torch.utils.data.SequentialSampler)
    assert val_loader.drop_last is False

    assert test_loader.batch_size == 2
    assert test_loader.num_workers == 2
    assert test_loader.pin_memory is True
    assert test_loader.persistent_workers is True
    assert isinstance(test_loader.sampler, torch.utils.data.SequentialSampler)
    assert test_loader.drop_last is False


def test_build_dataloaders_propagates_script_defaults():
    train_dataset = _FakeDataset(size=10)
    val_dataset = _FakeDataset(size=4)
    test_dataset = _FakeDataset(size=4)

    train_loader, val_loader, test_loader = _build_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
    )

    assert train_loader.batch_size == BATCH_SIZE
    assert train_loader.num_workers == NUM_WORKERS
    assert train_loader.pin_memory is True
    assert train_loader.persistent_workers is True
    for loader in (val_loader, test_loader):
        assert loader.batch_size == BATCH_SIZE
        assert loader.num_workers == NUM_WORKERS
        assert loader.pin_memory is True
        assert loader.persistent_workers is True


def test_create_stratified_split_produces_7_1_2_ratio_on_small_dataset():
    # 每类 1000 个样本，验证比例接近 70%/10%/20%
    labels = [0] * 1000 + [1] * 1000
    dataset = _LabeledDataset(labels)

    train_ds, val_ds, test_ds = create_stratified_split(
        dataset,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=42,
    )

    n = len(dataset)
    assert len(train_ds) == pytest.approx(n * TRAIN_RATIO, abs=4)
    assert len(val_ds) == pytest.approx(n * VAL_RATIO, abs=2)
    assert len(test_ds) == pytest.approx(n * TEST_RATIO, abs=2)

    # 分层划分要求每类在训练/验证/测试中都被保留
    def _labels_in(ds):
        return sorted(set(int(ds[i][1]) for i in range(len(ds))))

    assert _labels_in(train_ds) == [0, 1]
    assert _labels_in(val_ds) == [0, 1]
    assert _labels_in(test_ds) == [0, 1]


def test_subdataset_wrapping_preserves_len():
    base = _LabeledDataset([0] * 10 + [1] * 10)
    train_ds, val_ds, test_ds = create_stratified_split(
        base,
        val_ratio=0.1,
        test_ratio=0.2,
        seed=42,
    )
    assert isinstance(train_ds, SubDataset)
    assert isinstance(val_ds, SubDataset)
    assert isinstance(test_ds, SubDataset)
    assert len(train_ds) + len(val_ds) + len(test_ds) == len(base)


class _LongByteDataset(torch.utils.data.Dataset):
    """返回长字节序列样本的 fake 数据集，用于验证截断包装。"""

    def __init__(self, size: int, seq_len: int):
        self.size = size
        self.seq_len = seq_len

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        byte_seq = torch.zeros(self.seq_len, dtype=torch.long)
        pe = torch.zeros(1500, dtype=torch.float32)
        st = torch.zeros(49, dtype=torch.float32)
        label = torch.tensor(idx % 2, dtype=torch.long)
        return byte_seq, pe, st, label


def test_truncate_byte_dataset_cuts_long_sequences_to_constant_length():
    base = _LongByteDataset(size=8, seq_len=65536)
    wrapped = _TruncatedByteDataset(base, max_len=TRUNCATE_BYTE_LENGTH)

    assert len(wrapped) == len(base)
    assert wrapped.max_len == TRUNCATE_BYTE_LENGTH == 4096
    for i in range(len(wrapped)):
        byte_seq, pe, st, label = wrapped[i]
        assert byte_seq.shape[0] == TRUNCATE_BYTE_LENGTH
        assert pe.shape[0] == 1500
        assert st.shape[0] == 49


def test_truncate_byte_dataset_keeps_short_sequences_untouched():
    base = _LongByteDataset(size=4, seq_len=1024)
    wrapped = _TruncatedByteDataset(base, max_len=TRUNCATE_BYTE_LENGTH)

    for i in range(len(wrapped)):
        byte_seq = wrapped[i][0]
        assert byte_seq.shape[0] == 1024


def test_truncate_byte_dataset_is_picklable():
    import pickle

    base = _LongByteDataset(size=4, seq_len=8192)
    wrapped = _TruncatedByteDataset(base, max_len=TRUNCATE_BYTE_LENGTH)
    restored = pickle.loads(pickle.dumps(wrapped))
    assert len(restored) == 4
    assert restored.max_len == TRUNCATE_BYTE_LENGTH
    assert restored[0][0].shape[0] == TRUNCATE_BYTE_LENGTH
