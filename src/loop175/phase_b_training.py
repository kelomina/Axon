"""Fail-closed training primitives for the Loop175 Phase-B experiment."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch
from torch import nn

from .model import RegionNet

B0_FEATURE_DIMENSION = 571
OUTER_FOLD_COUNT = 5
STRICT_THRESHOLD = 0.5
FROZEN_B0_HGB_PARAMETERS: Mapping[str, object] = MappingProxyType(
    {
        "loss": "log_loss",
        "learning_rate": 0.06,
        "max_iter": 260,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_bins": 255,
        "early_stopping": False,
        "random_state": 41,
    }
)

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}
_LOWERCASE_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_integer_tensor(values: torch.Tensor, *, name: str) -> None:
    if not isinstance(values, torch.Tensor) or values.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"{name} must be an integer tensor")


def _require_floating_tensor(values: torch.Tensor, *, name: str) -> None:
    if not isinstance(values, torch.Tensor) or not values.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if not torch.isfinite(values).all().item():
        raise ValueError(f"{name} must contain only finite values")


def validate_phase_b_region_batch(
    region_tokens: torch.Tensor,
    region_lengths: torch.Tensor,
    region_types: torch.Tensor,
    offset_buckets: torch.Tensor,
    length_buckets: torch.Tensor,
    b0_features: torch.Tensor | None,
    *,
    expected_regions: int = 16,
    expected_region_bytes: int = 8192,
    padding_token: int = 256,
    region_type_count: int = 6,
    bucket_count: int = 64,
) -> None:
    """Reject malformed tensors before RegionNet can clamp or normalize them."""

    _require_integer_tensor(region_tokens, name="region_tokens")
    if region_tokens.ndim != 3:
        raise ValueError("region_tokens must have shape [batch, regions, bytes]")
    batch_size, region_count, region_bytes = region_tokens.shape
    if batch_size <= 0 or region_count != expected_regions:
        raise ValueError("region_tokens region count drifted from the frozen contract")
    if region_bytes != expected_region_bytes:
        raise ValueError("region_tokens byte dimension drifted")
    if torch.any(region_tokens < 0).item() or torch.any(region_tokens > padding_token).item():
        raise ValueError("region_tokens contain an out-of-range token")

    expected_shape = (batch_size, region_count)
    for name, tensor in {
        "region_lengths": region_lengths,
        "region_types": region_types,
        "offset_buckets": offset_buckets,
        "length_buckets": length_buckets,
    }.items():
        _require_integer_tensor(tensor, name=name)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")

    if torch.any(region_lengths < 0).item() or torch.any(region_lengths > region_bytes).item():
        raise ValueError("region_lengths are outside the frozen range")
    if torch.any(region_types < 0).item() or torch.any(region_types >= region_type_count).item():
        raise ValueError("region_types are outside the frozen range")
    if torch.any(offset_buckets < 0).item() or torch.any(offset_buckets >= bucket_count).item():
        raise ValueError("offset_buckets are outside the frozen range")
    if torch.any(length_buckets < 0).item() or torch.any(length_buckets >= bucket_count).item():
        raise ValueError("length_buckets are outside the frozen range")

    positions = torch.arange(region_bytes, device=region_tokens.device).view(1, 1, -1)
    valid = positions < region_lengths.unsqueeze(-1)
    if torch.any(region_tokens[valid] == padding_token).item():
        raise ValueError("padding_token appeared inside a valid region span")
    if torch.any(region_tokens[~valid] != padding_token).item():
        raise ValueError("region padding bytes must use the frozen padding_token")
    missing = region_lengths == 0
    if torch.any(missing != (region_types == 0)).item():
        raise ValueError("zero-length regions and missing region types disagree")
    if torch.any(offset_buckets[missing] != 0).item() or torch.any(length_buckets[missing] != 0).item():
        raise ValueError("missing regions must use zero metadata buckets")

    if b0_features is not None:
        _require_floating_tensor(b0_features, name="b0_features")
        if tuple(b0_features.shape) != (batch_size, B0_FEATURE_DIMENSION):
            raise ValueError(
                f"b0_features must have shape {(batch_size, B0_FEATURE_DIMENSION)}"
            )


class FailClosedRegionNet(nn.Module):
    """Validate the frozen Phase-B tensor contract before forwarding RegionNet."""

    def __init__(
        self,
        model: RegionNet | None = None,
        *,
        expected_regions: int = 16,
        expected_region_bytes: int = 8192,
    ) -> None:
        super().__init__()
        self.model = model or RegionNet()
        self.expected_regions = expected_regions
        self.expected_region_bytes = expected_region_bytes

    def forward(
        self,
        region_tokens: torch.Tensor,
        region_lengths: torch.Tensor,
        region_types: torch.Tensor,
        offset_buckets: torch.Tensor,
        length_buckets: torch.Tensor,
        b0_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        config = self.model.config
        validate_phase_b_region_batch(
            region_tokens,
            region_lengths,
            region_types,
            offset_buckets,
            length_buckets,
            b0_features,
            expected_regions=self.expected_regions,
            expected_region_bytes=self.expected_region_bytes,
            padding_token=config.padding_token,
            region_type_count=config.region_type_count,
            bucket_count=config.bucket_count,
        )
        return self.model(
            region_tokens,
            region_lengths,
            region_types,
            offset_buckets,
            length_buckets,
            b0_features,
        )


def _finite_b0_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != B0_FEATURE_DIMENSION or matrix.shape[0] == 0:
        raise ValueError(f"{name} must have shape [rows, {B0_FEATURE_DIMENSION}]")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix)


def _binary_labels(values: np.ndarray, *, rows: int, name: str = "labels") -> np.ndarray:
    labels = np.asarray(values)
    if labels.shape != (rows,) or not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{name} must be a binary vector with one value per row")
    return np.ascontiguousarray(labels, dtype=np.uint8)


def fit_frozen_b0_hgb(values: np.ndarray, labels: np.ndarray):
    """Fit Arm A using the exact frozen 571-value HGB specification."""

    matrix = _finite_b0_matrix(values, name="values")
    targets = _binary_labels(labels, rows=matrix.shape[0])
    if np.unique(targets).size != 2:
        raise ValueError("B0 fit requires both binary classes")
    from sklearn.ensemble import HistGradientBoostingClassifier

    estimator = HistGradientBoostingClassifier(**dict(FROZEN_B0_HGB_PARAMETERS))
    estimator.fit(matrix, targets)
    if not np.array_equal(estimator.classes_, np.array([0, 1])):
        raise RuntimeError("frozen B0 HGB did not retain both binary classes")
    return estimator


def predict_b0_scores(estimator, values: np.ndarray) -> np.ndarray:
    matrix = _finite_b0_matrix(values, name="values")
    classes = np.asarray(getattr(estimator, "classes_", ()))
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError("B0 estimator classes drifted")
    probabilities = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)
    if probabilities.shape != (matrix.shape[0], 2):
        raise RuntimeError("B0 estimator returned an invalid probability matrix")
    scores = probabilities[:, 1]
    if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise RuntimeError("B0 estimator returned invalid malicious probabilities")
    return np.ascontiguousarray(scores)


def strict_hard_decisions(scores: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(scores, dtype=np.float64)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
        raise ValueError("scores must be a finite one-dimensional vector")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("scores must be probabilities in [0, 1]")
    return np.ascontiguousarray(probabilities > STRICT_THRESHOLD, dtype=np.uint8)


@dataclass(frozen=True)
class InnerOOFResult:
    outer_holdout_fold: int
    row_indices: np.ndarray
    inner_folds: np.ndarray
    scores: np.ndarray
    hard_decisions: np.ndarray


def generate_b0_inner_oof_scores(
    values: np.ndarray,
    labels: np.ndarray,
    component_folds: np.ndarray,
    *,
    outer_holdout_fold: int,
) -> InnerOOFResult:
    """Generate B0 OOF scores over only the four folds inside one outer-fit set."""

    matrix = _finite_b0_matrix(values, name="values")
    targets = _binary_labels(labels, rows=matrix.shape[0])
    folds = np.asarray(component_folds)
    if folds.shape != (matrix.shape[0],) or not np.issubdtype(folds.dtype, np.integer):
        raise ValueError("component_folds must be an integer vector with one value per row")
    if set(np.unique(folds).tolist()) != set(range(OUTER_FOLD_COUNT)):
        raise ValueError("component_folds must contain exactly the five frozen folds")
    if isinstance(outer_holdout_fold, bool) or not 0 <= outer_holdout_fold < OUTER_FOLD_COUNT:
        raise ValueError("outer_holdout_fold is invalid")

    outer_fit_indices = np.flatnonzero(folds != outer_holdout_fold)
    inner_fold_values = tuple(fold for fold in range(OUTER_FOLD_COUNT) if fold != outer_holdout_fold)
    scores_by_row = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    for inner_holdout_fold in inner_fold_values:
        inner_holdout = folds == inner_holdout_fold
        inner_fit = (folds != outer_holdout_fold) & ~inner_holdout
        if np.any(inner_fit & (folds == outer_holdout_fold)) or np.any(
            inner_holdout & (folds == outer_holdout_fold)
        ):
            raise RuntimeError("outer holdout leaked into the B0 inner-OOF scope")
        estimator = fit_frozen_b0_hgb(matrix[inner_fit], targets[inner_fit])
        scores_by_row[inner_holdout] = predict_b0_scores(estimator, matrix[inner_holdout])

    if not np.isfinite(scores_by_row[outer_fit_indices]).all():
        raise RuntimeError("B0 inner-OOF did not score every outer-fit row exactly once")
    if np.isfinite(scores_by_row[folds == outer_holdout_fold]).any():
        raise RuntimeError("B0 inner-OOF scored an outer-holdout row")
    scores = np.ascontiguousarray(scores_by_row[outer_fit_indices])
    row_indices = np.ascontiguousarray(outer_fit_indices, dtype=np.int64)
    inner_folds = np.ascontiguousarray(folds[outer_fit_indices], dtype=np.int8)
    decisions = strict_hard_decisions(scores)
    for array in (row_indices, inner_folds, scores, decisions):
        array.setflags(write=False)
    return InnerOOFResult(
        outer_holdout_fold=outer_holdout_fold,
        row_indices=row_indices,
        inner_folds=inner_folds,
        scores=scores,
        hard_decisions=decisions,
    )


def build_e_residual_weights(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Build the frozen 8/3/1 E-arm weights from inner-OOF predictions only."""

    probabilities = np.asarray(scores, dtype=np.float64)
    targets = _binary_labels(labels, rows=probabilities.size)
    decisions = strict_hard_decisions(probabilities)
    errors = decisions != targets
    near_boundary = (~errors) & (probabilities >= 0.35) & (probabilities <= 0.65)
    raw = np.where(errors, 8.0, np.where(near_boundary, 3.0, 1.0)).astype(np.float64)

    # 先消除类别频率和类别难度差，再保持全局均值为 1；8x 是不可突破的硬上限。
    normalized = raw.copy()
    for label in (0, 1):
        selected = targets == label
        if not selected.any():
            raise ValueError("E residual weights require both binary classes")
        normalized[selected] /= normalized[selected].mean()
    normalized /= normalized.mean()
    normalized = np.minimum(normalized, 8.0)
    if not np.isfinite(normalized).all() or np.any(normalized <= 0.0):
        raise RuntimeError("E residual weight normalization produced invalid values")
    if not np.isclose(normalized.mean(), 1.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("E residual weights lost the frozen global mean")
    return np.ascontiguousarray(normalized, dtype=np.float32)


@dataclass(frozen=True)
class EWeightResult:
    row_indices: np.ndarray
    weights: np.ndarray
    inner_oof: InnerOOFResult


def generate_e_weights_from_b0_inner_oof(
    values: np.ndarray,
    labels: np.ndarray,
    component_folds: np.ndarray,
    *,
    outer_holdout_fold: int,
) -> EWeightResult:
    inner_oof = generate_b0_inner_oof_scores(
        values,
        labels,
        component_folds,
        outer_holdout_fold=outer_holdout_fold,
    )
    targets = _binary_labels(labels, rows=np.asarray(values).shape[0])
    weights = build_e_residual_weights(targets[inner_oof.row_indices], inner_oof.scores)
    weights.setflags(write=False)
    return EWeightResult(inner_oof.row_indices, weights, inner_oof)


def deterministic_region_record_permutation(
    size: int,
    *,
    protocol_sha256: str,
    seed: int,
    outer_fold: int,
    role: str,
) -> np.ndarray:
    """Return a deterministic label-free circular derangement for one partition."""

    if isinstance(size, bool) or size < 2:
        raise ValueError("a zero-fixed-point permutation requires at least two rows")
    if not isinstance(protocol_sha256, str) or not _LOWERCASE_SHA256.fullmatch(
        protocol_sha256
    ):
        raise ValueError("protocol_sha256 must be a lowercase SHA-256 digest")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(outer_fold, bool) or not 0 <= outer_fold < OUTER_FOLD_COUNT:
        raise ValueError("outer_fold is invalid")
    if role not in {"fit", "holdout"}:
        raise ValueError("role must be fit or holdout")
    material = (
        f"loop175-phase-b-d|{protocol_sha256}|{seed}|{outer_fold}|{role}|{size}"
    ).encode("ascii")
    shift = 1 + int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (size - 1)
    # 整条 region record 使用同一个 donor，循环位移天然是双射且没有 fixed point。
    permutation = (np.arange(size, dtype=np.int64) + shift) % size
    if np.any(permutation == np.arange(size)) or np.unique(permutation).size != size:
        raise RuntimeError("counterfactual ownership permutation is not a derangement")
    permutation.setflags(write=False)
    return permutation


@dataclass(frozen=True)
class ShuffledRegionRecords:
    region_tokens: torch.Tensor
    region_lengths: torch.Tensor
    region_types: torch.Tensor
    offset_buckets: torch.Tensor
    length_buckets: torch.Tensor
    permutation: np.ndarray


def shuffle_region_record_ownership(
    region_tokens: torch.Tensor,
    region_lengths: torch.Tensor,
    region_types: torch.Tensor,
    offset_buckets: torch.Tensor,
    length_buckets: torch.Tensor,
    *,
    protocol_sha256: str,
    seed: int,
    outer_fold: int,
    role: str,
    expected_regions: int = 16,
    expected_region_bytes: int = 8192,
    padding_token: int = 256,
    region_type_count: int = 6,
    bucket_count: int = 64,
) -> ShuffledRegionRecords:
    """Move bytes and all region metadata together without accepting labels or B0."""

    validate_phase_b_region_batch(
        region_tokens,
        region_lengths,
        region_types,
        offset_buckets,
        length_buckets,
        None,
        expected_regions=expected_regions,
        expected_region_bytes=expected_region_bytes,
        padding_token=padding_token,
        region_type_count=region_type_count,
        bucket_count=bucket_count,
    )
    permutation = deterministic_region_record_permutation(
        region_tokens.shape[0],
        protocol_sha256=protocol_sha256,
        seed=seed,
        outer_fold=outer_fold,
        role=role,
    )
    indices = torch.as_tensor(permutation.copy(), dtype=torch.long, device=region_tokens.device)
    return ShuffledRegionRecords(
        region_tokens=region_tokens.index_select(0, indices),
        region_lengths=region_lengths.index_select(0, indices),
        region_types=region_types.index_select(0, indices),
        offset_buckets=offset_buckets.index_select(0, indices),
        length_buckets=length_buckets.index_select(0, indices),
        permutation=permutation,
    )
