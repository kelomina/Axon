"""Loop185 数据适配器（支持参数化 fold 划分，用于 4-fold OOF）。

复用 Loop175 region cache，支持任意 fit_folds / selection_fold 组合。
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from .contracts import (
    B0_FEATURE_DIM,
    BUCKET_COUNT,
    EXPECTED_REGION_BYTES,
    EXPECTED_REGIONS,
    OOF_FOLD_CONFIGS,
    OOF_FOLDS,
    OOF_FORBIDDEN_FOLD,
    PADDING_TOKEN,
    REGION_TYPE_COUNT,
    VOCABULARY_SIZE,
)

# ---------------------------------------------------------------------------
# Train fold 划分（与 Loop175 seed41 OOF 合同对齐）
# ---------------------------------------------------------------------------

ROWS_PER_FOLD: Final[int] = 4_000
FULL_TRAIN_ROWS: Final[int] = 20_000

CANONICAL_REGION_CACHE_RELATIVE_PATH: Final[str] = (
    "reports/roadmap_9997/loop175/phase_b_region_cache_v1.npz"
)
CANONICAL_REGION_CACHE_SHA256: Final[str] = (
    "6e4ffb2382b986b1c4bd8bd1ac8ca211e3ca01f28643d7303c2baec1d338249d"
)
CANONICAL_FOLD_MANIFEST_RELATIVE_PATH: Final[str] = (
    "reports/roadmap_9997/loop164/local_train_diagnostic_folds.jsonl"
)
CANONICAL_FOLD_MANIFEST_SHA256: Final[str] = (
    "00a31a1bd86d7b887447f3e86e5e753ebcaaee45be74311199332e073a3880a5"
)
CANONICAL_B0_CACHE_RELATIVE_PATH: Final[str] = (
    "reports/roadmap_9997/loop167/phase_b_v12_dual_identity_job_attestation_remediation/"
    "phase_b_feature_cache_v12.npz"
)

REGION_CACHE_SCHEMA_NAME: Final[str] = "axon_loop175_identity_free_ragged_region_cache_v1"
REGION_ARRAY_NAMES: Final[tuple[str, ...]] = (
    "row_region_offsets",
    "file_sizes",
    "region_token_offsets",
    "token_values",
    "region_types",
    "region_starts",
    "offset_buckets",
    "length_buckets",
)


@dataclass(frozen=True)
class RegionBatch:
    """单个训练 batch 的冻结 ABI，与 HGConvRegionNet.forward 对齐。"""

    region_tokens: object
    region_lengths: object
    region_types: object
    offset_buckets: object
    length_buckets: object
    b0_features: object
    labels: object
    row_ids: object


@dataclass(frozen=True)
class FoldSplit:
    """Phase A fold 划分契约（参数化版本，支持 4-fold OOF）。"""

    fit_folds: tuple[int, ...]
    selection_fold: int
    forbidden_folds: tuple[int, ...]
    fit_rows_expected: int
    selection_rows_expected: int

    def validate(self) -> None:
        if self.selection_fold in self.fit_folds:
            raise ValueError("selection fold must not overlap with fit folds")
        if self.selection_fold in self.forbidden_folds:
            raise ValueError("selection fold must not be forbidden")
        for fold in self.fit_folds:
            if fold in self.forbidden_folds:
                raise ValueError(f"fit fold {fold} is forbidden")
        if self.fit_rows_expected != len(self.fit_folds) * ROWS_PER_FOLD:
            raise ValueError("fit_rows_expected must match fit_folds * ROWS_PER_FOLD")
        if self.selection_rows_expected != ROWS_PER_FOLD:
            raise ValueError("selection_rows_expected must match ROWS_PER_FOLD")


def make_fold_split(selection_fold: int) -> FoldSplit:
    """根据 selection_fold 构造 FoldSplit（用于 4-fold OOF）。

    例如 selection_fold=1 → fit_folds=(2,3,4), forbidden=(0,)
         selection_fold=2 → fit_folds=(1,3,4), forbidden=(0,)
    """
    if selection_fold not in OOF_FOLDS:
        raise ValueError(f"selection_fold must be in {OOF_FOLDS}, got {selection_fold}")
    fit_folds = tuple(f for f in OOF_FOLDS if f != selection_fold)
    return FoldSplit(
        fit_folds=fit_folds,
        selection_fold=selection_fold,
        forbidden_folds=(OOF_FORBIDDEN_FOLD,),
        fit_rows_expected=len(fit_folds) * ROWS_PER_FOLD,
        selection_rows_expected=ROWS_PER_FOLD,
    )


# 默认 fold 划分（与 Loop184 一致，用于兼容性）
PHASE_A_FIT_FOLDS: Final[tuple[int, ...]] = (2, 3, 4)
PHASE_A_SELECTION_FOLD: Final[int] = 1
PHASE_A_FORBIDDEN_FOLDS: Final[tuple[int, ...]] = (0,)
PHASE_A_FOLD_SPLIT: Final[FoldSplit] = FoldSplit(
    fit_folds=PHASE_A_FIT_FOLDS,
    selection_fold=PHASE_A_SELECTION_FOLD,
    forbidden_folds=PHASE_A_FORBIDDEN_FOLDS,
    fit_rows_expected=12_000,
    selection_rows_expected=4_000,
)


def validate_region_tokens_shape(tokens: object) -> None:
    shape = tuple(getattr(tokens, "shape", ()))
    dtype = str(getattr(tokens, "dtype", ""))
    if len(shape) != 3:
        raise ValueError(f"region_tokens must be 3D, got shape {shape}")
    if shape[1] != EXPECTED_REGIONS:
        raise ValueError(f"region_tokens dim 1 must be {EXPECTED_REGIONS}, got {shape[1]}")
    if shape[2] != EXPECTED_REGION_BYTES:
        raise ValueError(f"region_tokens dim 2 must be {EXPECTED_REGION_BYTES}, got {shape[2]}")
    if "int" not in dtype:
        raise ValueError(f"region_tokens must be integer dtype, got {dtype}")


def validate_metadata_shape(metadata: object, name: str) -> None:
    shape = tuple(getattr(metadata, "shape", ()))
    dtype = str(getattr(metadata, "dtype", ""))
    if len(shape) != 2:
        raise ValueError(f"{name} must be 2D, got shape {shape}")
    if shape[1] != EXPECTED_REGIONS:
        raise ValueError(f"{name} dim 1 must be {EXPECTED_REGIONS}, got {shape[1]}")
    if "int" not in dtype:
        raise ValueError(f"{name} must be integer dtype, got {dtype}")


def validate_b0_features_shape(b0: object) -> None:
    shape = tuple(getattr(b0, "shape", ()))
    dtype = str(getattr(b0, "dtype", ""))
    if len(shape) != 2:
        raise ValueError(f"b0_features must be 2D, got shape {shape}")
    if shape[1] != B0_FEATURE_DIM:
        raise ValueError(f"b0_features dim 1 must be {B0_FEATURE_DIM}, got {shape[1]}")
    if "float" not in dtype:
        raise ValueError(f"b0_features must be float dtype, got {dtype}")


def validate_token_ranges(tokens: object) -> None:
    mn = getattr(tokens, "min", None)
    mx = getattr(tokens, "max", None)
    if mn is None or mx is None:
        raise ValueError("tokens must expose min()/max()")
    min_value = mn().item() if callable(mn) else mn
    max_value = mx().item() if callable(mx) else mx
    if min_value < 0:
        raise ValueError(f"tokens contain negative value {min_value}")
    if max_value >= VOCABULARY_SIZE:
        raise ValueError(f"tokens contain out-of-range value {max_value}")


def validate_metadata_ranges(
    region_types: object,
    offset_buckets: object,
    length_buckets: object,
) -> None:
    for tensor, name, upper in [
        (region_types, "region_types", REGION_TYPE_COUNT),
        (offset_buckets, "offset_buckets", BUCKET_COUNT),
        (length_buckets, "length_buckets", BUCKET_COUNT),
    ]:
        mn = getattr(tensor, "min", None)()
        mx = getattr(tensor, "max", None)()
        min_value = mn.item() if callable(mn) else mn
        max_value = mx.item() if callable(mx) else mx
        if min_value < 0:
            raise ValueError(f"{name} contain negative value {min_value}")
        if max_value >= upper:
            raise ValueError(f"{name} contain out-of-range value {max_value} >= {upper}")


def make_synthetic_batch(
    batch_size: int,
    *,
    seed: int = 41,
    include_b0: bool = True,
) -> dict[str, object]:
    """生成合成 batch 用于 Phase 0 测试。"""

    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)

    tokens = torch.randint(
        0,
        PADDING_TOKEN,
        (batch_size, EXPECTED_REGIONS, EXPECTED_REGION_BYTES),
        generator=generator,
        dtype=torch.int64,
    )
    lengths = torch.randint(
        0,
        EXPECTED_REGION_BYTES + 1,
        (batch_size, EXPECTED_REGIONS),
        generator=generator,
        dtype=torch.int64,
    )
    positions = torch.arange(EXPECTED_REGION_BYTES).view(1, 1, -1)
    mask = positions < lengths.unsqueeze(-1)
    tokens = torch.where(mask, tokens, torch.full_like(tokens, PADDING_TOKEN))

    # type 0 保留给缺失区域，有效区域从 type 1 开始
    types = torch.randint(
        1,
        REGION_TYPE_COUNT,
        (batch_size, EXPECTED_REGIONS),
        generator=generator,
        dtype=torch.int64,
    )
    types = torch.where(lengths == 0, torch.zeros_like(types), types)

    offsets = torch.randint(
        0,
        BUCKET_COUNT,
        (batch_size, EXPECTED_REGIONS),
        generator=generator,
        dtype=torch.int64,
    )
    length_buckets = torch.randint(
        0,
        BUCKET_COUNT,
        (batch_size, EXPECTED_REGIONS),
        generator=generator,
        dtype=torch.int64,
    )
    offsets = torch.where(lengths == 0, torch.zeros_like(offsets), offsets)
    length_buckets = torch.where(lengths == 0, torch.zeros_like(length_buckets), length_buckets)

    b0 = None
    if include_b0:
        b0 = torch.randn(
            (batch_size, B0_FEATURE_DIM),
            generator=generator,
            dtype=torch.float32,
        )

    labels = torch.randint(0, 2, (batch_size,), generator=generator, dtype=torch.int64)
    row_ids = torch.arange(batch_size, dtype=torch.int64)

    return {
        "region_tokens": tokens,
        "region_lengths": lengths,
        "region_types": types,
        "offset_buckets": offsets,
        "length_buckets": length_buckets,
        "b0_features": b0,
        "labels": labels,
        "row_ids": row_ids,
    }


class PhaseADataLoader:
    """Phase A 数据加载器（支持参数化 fold 划分，用于 4-fold OOF）。"""

    def __init__(self, fold_split: FoldSplit = PHASE_A_FOLD_SPLIT) -> None:
        fold_split.validate()
        self.fold_split = fold_split
        self._rows_loaded = False
        self._region_cache: dict[str, np.ndarray] | None = None
        self._fold_labels: np.ndarray | None = None
        self._fold_assignments: np.ndarray | None = None
        self._b0_values: np.ndarray | None = None

    def assert_not_forbidden_fold(self, fold: int) -> None:
        if fold in self.fold_split.forbidden_folds:
            raise ValueError(
                f"fold {fold} is forbidden (Phase A must not touch fold0)"
            )

    def validate_batch(self, batch: dict[str, object]) -> None:
        validate_region_tokens_shape(batch["region_tokens"])
        for key in ("region_lengths", "region_types", "offset_buckets", "length_buckets"):
            validate_metadata_shape(batch[key], key)
        validate_token_ranges(batch["region_tokens"])
        validate_metadata_ranges(
            batch["region_types"],
            batch["offset_buckets"],
            batch["length_buckets"],
        )
        if batch.get("b0_features") is not None:
            validate_b0_features_shape(batch["b0_features"])

    def load_real_data(
        self,
        *,
        project_root: str | Path | None = None,
        region_cache_path: str | Path | None = None,
        fold_manifest_path: str | Path | None = None,
        b0_cache_path: str | Path | None = None,
        region_cache_sha256: str = CANONICAL_REGION_CACHE_SHA256,
        fold_manifest_sha256: str = CANONICAL_FOLD_MANIFEST_SHA256,
    ) -> dict[str, object]:
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]
        root = Path(project_root).resolve(strict=True)

        cache_path = (
            Path(region_cache_path)
            if region_cache_path
            else root / CANONICAL_REGION_CACHE_RELATIVE_PATH
        )
        manifest_path = (
            Path(fold_manifest_path)
            if fold_manifest_path
            else root / CANONICAL_FOLD_MANIFEST_RELATIVE_PATH
        )
        b0_path = (
            Path(b0_cache_path)
            if b0_cache_path
            else root / CANONICAL_B0_CACHE_RELATIVE_PATH
        )

        for path, label in [
            (cache_path, "region cache"),
            (manifest_path, "fold manifest"),
            (b0_path, "B0 cache"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        labels, folds = self._load_fold_manifest(manifest_path, fold_manifest_sha256)
        region_arrays = self._load_region_cache(cache_path, region_cache_sha256)
        b0_values, b0_sha256 = self._load_b0_features(b0_path)

        row_count = int(region_arrays["file_sizes"].shape[0])
        if row_count != FULL_TRAIN_ROWS:
            raise ValueError(
                f"region cache has {row_count} rows, expected {FULL_TRAIN_ROWS}"
            )
        if labels.shape != (FULL_TRAIN_ROWS,):
            raise ValueError(f"labels shape {labels.shape} mismatch")
        if b0_values.shape != (FULL_TRAIN_ROWS, B0_FEATURE_DIM):
            raise ValueError(f"b0_values shape {b0_values.shape} mismatch")

        self._fold_labels = labels
        self._fold_assignments = folds
        self._region_cache = region_arrays
        self._b0_values = b0_values
        self._rows_loaded = True

        return {
            "schema": "axon_loop185_phase_a_data_receipt_v1",
            "region_cache_path": str(cache_path),
            "region_cache_sha256": region_cache_sha256,
            "fold_manifest_path": str(manifest_path),
            "fold_manifest_sha256": fold_manifest_sha256,
            "b0_cache_path": str(b0_path),
            "b0_cache_sha256": b0_sha256,
            "row_count": row_count,
            "region_count": int(region_arrays["region_types"].shape[0]),
            "token_count": int(region_arrays["token_values"].shape[0]),
            "fit_folds": list(self.fold_split.fit_folds),
            "selection_fold": self.fold_split.selection_fold,
            "forbidden_folds": list(self.fold_split.forbidden_folds),
        }

    def _load_fold_manifest(
        self,
        path: Path,
        expected_sha256: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        raw = path.read_bytes()
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha != expected_sha256:
            raise ValueError(
                f"fold manifest SHA256 drift: expected {expected_sha256[:16]}, "
                f"got {actual_sha[:16]}"
            )
        if not raw.endswith(b"\n"):
            raise ValueError("fold manifest must end with newline")

        lines = raw.splitlines()
        if len(lines) != FULL_TRAIN_ROWS:
            raise ValueError(
                f"fold manifest has {len(lines)} lines, expected {FULL_TRAIN_ROWS}"
            )

        labels = np.empty(FULL_TRAIN_ROWS, dtype=np.uint8)
        folds = np.empty(FULL_TRAIN_ROWS, dtype=np.int8)

        for ordinal, line in enumerate(lines):
            record = json.loads(line.decode("ascii"))
            if "label" not in record or "diagnostic_fold" not in record:
                raise ValueError(f"fold manifest line {ordinal} missing keys")
            labels[ordinal] = int(record["label"])
            folds[ordinal] = int(record["diagnostic_fold"])

        return labels, folds

    def _load_region_cache(
        self,
        path: Path,
        expected_sha256: str,
    ) -> dict[str, np.ndarray]:
        raw = path.read_bytes()
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha != expected_sha256:
            raise ValueError(
                f"region cache SHA256 drift: expected {expected_sha256[:16]}, "
                f"got {actual_sha[:16]}"
            )

        arrays: dict[str, np.ndarray] = {}
        with zipfile.ZipFile(path, mode="r") as archive:
            names = {info.filename for info in archive.infolist()}
            expected_names = {f"{name}.npy" for name in REGION_ARRAY_NAMES} | {"metadata.json"}
            if names != expected_names:
                raise ValueError(f"region cache members mismatch: {names}")

            for name in REGION_ARRAY_NAMES:
                with archive.open(f"{name}.npy", mode="r") as member:
                    arrays[name] = np.lib.format.read_array(member, allow_pickle=False)

            metadata_raw = archive.read("metadata.json")

        metadata = json.loads(metadata_raw.decode("ascii"))
        if metadata.get("schema") != REGION_CACHE_SCHEMA_NAME:
            raise ValueError(f"region cache schema mismatch: {metadata.get('schema')}")
        if metadata.get("row_count") != FULL_TRAIN_ROWS:
            raise ValueError(f"region cache row_count mismatch: {metadata.get('row_count')}")

        row_offsets = arrays["row_region_offsets"]
        if row_offsets.shape != (FULL_TRAIN_ROWS + 1,):
            raise ValueError(f"row_region_offsets shape {row_offsets.shape} mismatch")
        if row_offsets[0] != 0 or np.any(np.diff(row_offsets) < 0):
            raise ValueError("row_region_offsets not canonical monotonic")

        regions_per_row = np.diff(row_offsets)
        if not np.all(regions_per_row == EXPECTED_REGIONS):
            raise ValueError("not all rows have exactly 16 regions")

        token_offsets = arrays["region_token_offsets"]
        if token_offsets[0] != 0 or np.any(np.diff(token_offsets) < 0):
            raise ValueError("region_token_offsets not canonical monotonic")
        if int(token_offsets[-1]) != arrays["token_values"].size:
            raise ValueError("token_values size does not match token_offsets[-1]")

        return arrays

    def _load_b0_features(self, path: Path) -> tuple[np.ndarray, str]:
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            b0_key = "b0_values.npy"
            if b0_key not in names:
                raise ValueError(f"b0 cache missing array: {b0_key}")
            with archive.open(b0_key) as handle:
                b0 = np.lib.format.read_array(handle, allow_pickle=False)
        b0_values = np.asarray(b0, dtype=np.float32)
        if b0_values.shape != (FULL_TRAIN_ROWS, B0_FEATURE_DIM):
            raise ValueError(f"b0_values shape {b0_values.shape} mismatch")
        if not np.isfinite(b0_values).all():
            raise ValueError("b0_values contains non-finite values")
        return b0_values, sha

    def _fold_indices(self, fold: int) -> np.ndarray:
        if self._fold_assignments is None:
            raise RuntimeError("fold assignments not loaded")
        return np.where(self._fold_assignments == fold)[0]

    def fit_indices(self) -> np.ndarray:
        indices: list[np.ndarray] = []
        for fold in self.fold_split.fit_folds:
            indices.append(self._fold_indices(fold))
        return np.concatenate(indices, axis=0)

    def selection_indices(self) -> np.ndarray:
        return self._fold_indices(self.fold_split.selection_fold)

    def prematerialize_all(self) -> None:
        """一次性将所有行的区域数据物化为密集 numpy 数组，消除 materialize_batch 的 Python 循环。"""
        if not self._rows_loaded:
            raise RuntimeError("call load_real_data() before prematerialize_all()")
        if self._region_cache is None:
            raise RuntimeError("region cache not loaded")

        cache = self._region_cache
        row_offsets = cache["row_region_offsets"]
        token_offsets = cache["region_token_offsets"]
        token_values = cache["token_values"]
        region_types_arr = cache["region_types"]
        offset_buckets_arr = cache["offset_buckets"]
        length_buckets_arr = cache["length_buckets"]

        n = FULL_TRAIN_ROWS
        all_tokens = np.full(
            (n, EXPECTED_REGIONS, EXPECTED_REGION_BYTES),
            PADDING_TOKEN,
            dtype=np.int64,
        )
        all_lengths = np.zeros((n, EXPECTED_REGIONS), dtype=np.int64)
        all_types = np.zeros((n, EXPECTED_REGIONS), dtype=np.int64)
        all_offsets = np.zeros((n, EXPECTED_REGIONS), dtype=np.int64)
        all_len_buckets = np.zeros((n, EXPECTED_REGIONS), dtype=np.int64)

        for row_idx in range(n):
            start = int(row_offsets[row_idx])
            end = int(row_offsets[row_idx + 1]) if row_idx + 1 < n else int(region_types_arr.shape[0])
            for region_ordinal in range(end - start):
                abs_idx = start + region_ordinal
                ts = int(token_offsets[abs_idx])
                te = int(token_offsets[abs_idx + 1]) if abs_idx + 1 < len(token_offsets) else int(token_values.shape[0])
                length = min(te - ts, EXPECTED_REGION_BYTES)
                if length > 0:
                    all_tokens[row_idx, region_ordinal, :length] = token_values[ts:ts + length]
                    all_lengths[row_idx, region_ordinal] = length
                    all_types[row_idx, region_ordinal] = int(region_types_arr[abs_idx])
                    all_offsets[row_idx, region_ordinal] = int(offset_buckets_arr[abs_idx])
                    all_len_buckets[row_idx, region_ordinal] = int(length_buckets_arr[abs_idx])

        self._premat_tokens = all_tokens
        self._premat_lengths = all_lengths
        self._premat_types = all_types
        self._premat_offsets = all_offsets
        self._premat_len_buckets = all_len_buckets

        # 预分配 GPU 张量（将在首次 materialize 时创建）

    def materialize_batch(self, row_indices: np.ndarray) -> dict[str, object]:
        if not self._rows_loaded:
            raise RuntimeError("call load_real_data() before materialize_batch()")
        if self._region_cache is None or self._fold_labels is None or self._b0_values is None:
            raise RuntimeError("data not loaded")

        import torch

        # 使用预物化数据（快速路径）
        if hasattr(self, "_premat_tokens") and self._premat_tokens is not None:
            return {
                "region_tokens": torch.from_numpy(self._premat_tokens[row_indices]),
                "region_lengths": torch.from_numpy(self._premat_lengths[row_indices]),
                "region_types": torch.from_numpy(self._premat_types[row_indices]),
                "offset_buckets": torch.from_numpy(self._premat_offsets[row_indices]),
                "length_buckets": torch.from_numpy(self._premat_len_buckets[row_indices]),
                "b0_features": torch.from_numpy(self._b0_values[row_indices].astype(np.float32, copy=False)),
                "labels": torch.from_numpy(self._fold_labels[row_indices].astype(np.int64, copy=False)),
                "row_ids": torch.from_numpy(np.asarray(row_indices, dtype=np.int64)),
            }

        # 原始慢速路径（逐行循环）
        cache = self._region_cache
        labels = self._fold_labels
        b0_values = self._b0_values

        batch_size = len(row_indices)
        region_tokens = np.full(
            (batch_size, EXPECTED_REGIONS, EXPECTED_REGION_BYTES),
            PADDING_TOKEN,
            dtype=np.int64,
        )
        region_lengths = np.zeros((batch_size, EXPECTED_REGIONS), dtype=np.int64)
        region_types = np.zeros((batch_size, EXPECTED_REGIONS), dtype=np.int64)
        offset_buckets = np.zeros((batch_size, EXPECTED_REGIONS), dtype=np.int64)
        length_buckets = np.zeros((batch_size, EXPECTED_REGIONS), dtype=np.int64)

        row_offsets = cache["row_region_offsets"]
        token_offsets = cache["region_token_offsets"]
        token_values = cache["token_values"]
        region_types_arr = cache["region_types"]
        region_starts = cache["region_starts"]
        offset_buckets_arr = cache["offset_buckets"]
        length_buckets_arr = cache["length_buckets"]
        file_sizes = cache["file_sizes"]

        for i, row_idx in enumerate(row_indices):
            row_idx = int(row_idx)
            start = int(row_offsets[row_idx])
            end = int(row_offsets[row_idx + 1]) if row_idx + 1 < len(row_offsets) else int(region_types_arr.shape[0])
            for region_ordinal in range(end - start):
                abs_idx = start + region_ordinal
                token_start = int(token_offsets[abs_idx])
                token_end = int(token_offsets[abs_idx + 1]) if abs_idx + 1 < len(token_offsets) else int(token_values.shape[0])
                length = min(token_end - token_start, EXPECTED_REGION_BYTES)
                if length > 0:
                    region_tokens[i, region_ordinal, :length] = token_values[token_start:token_start + length]
                    region_lengths[i, region_ordinal] = length
                    region_types[i, region_ordinal] = int(region_types_arr[abs_idx])
                    offset_buckets[i, region_ordinal] = int(offset_buckets_arr[abs_idx])
                    length_buckets[i, region_ordinal] = int(length_buckets_arr[abs_idx])

        batch_b0 = b0_values[row_indices].astype(np.float32, copy=False)
        batch_labels = labels[row_indices].astype(np.int64, copy=False)
        batch_row_ids = np.asarray(row_indices, dtype=np.int64)

        return {
            "region_tokens": torch.from_numpy(region_tokens),
            "region_lengths": torch.from_numpy(region_lengths),
            "region_types": torch.from_numpy(region_types),
            "offset_buckets": torch.from_numpy(offset_buckets),
            "length_buckets": torch.from_numpy(length_buckets),
            "b0_features": torch.from_numpy(batch_b0),
            "labels": torch.from_numpy(batch_labels),
            "row_ids": torch.from_numpy(batch_row_ids),
        }
