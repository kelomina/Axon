"""Identity-free neural execution engine for Loop175 Phase B."""

from __future__ import annotations

import copy
import hashlib
import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from .model import RegionNet, RegionNetConfig
from .phase_b_data import IdentityFreePhaseBFitPayload, RaggedRegionCache
from .phase_b_training import (
    B0_FEATURE_DIMENSION,
    FailClosedRegionNet,
    deterministic_region_record_permutation,
)

NEURAL_ARMS = frozenset({"B", "C", "D", "E"})


@dataclass(frozen=True)
class EngineConfig:
    seed: int = 41
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2
    microbatch: int = 2
    gradient_accumulation: int = 16
    warmup_epochs: int = 1
    gradient_clip: float = 1.0
    ema_decay: float = 0.999
    expected_regions: int = 16
    expected_region_bytes: int = 8192
    device: str = "auto"

    def __post_init__(self) -> None:
        integer_fields = {
            "seed": self.seed,
            "microbatch": self.microbatch,
            "gradient_accumulation": self.gradient_accumulation,
            "warmup_epochs": self.warmup_epochs,
            "expected_regions": self.expected_regions,
            "expected_region_bytes": self.expected_region_bytes,
        }
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields.values()):
            raise ValueError("engine integer settings must be integers")
        if self.seed < 0 or self.microbatch <= 0 or self.gradient_accumulation <= 0:
            raise ValueError("seed and batch settings are invalid")
        if self.warmup_epochs < 0 or self.expected_regions <= 0 or self.expected_region_bytes <= 0:
            raise ValueError("warmup or region settings are invalid")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("optimizer settings are invalid")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")


@dataclass(frozen=True)
class FixedRegionBatch:
    receiver_indices: np.ndarray
    donor_indices: np.ndarray
    region_tokens: torch.Tensor
    region_lengths: torch.Tensor
    region_types: torch.Tensor
    offset_buckets: torch.Tensor
    length_buckets: torch.Tensor
    b0_features: torch.Tensor
    labels: torch.Tensor
    sample_weights: torch.Tensor | None


@dataclass(frozen=True)
class PilotResult:
    arm: str
    selected_epoch: int
    selection_losses: tuple[float, ...]
    selection_is_unweighted: bool


@dataclass(frozen=True)
class NeuralArmResult:
    arm: str
    frozen_epoch: int
    holdout_scores: np.ndarray
    optimizer_steps: int
    used_sample_weights: bool
    sample_weight_sum: float | None
    final_training_loss: float
    device_type: str
    autocast_dtype: str
    checkpoint_path: str | None
    checkpoint_sha256: str | None


def _normalized_indices(values: np.ndarray, *, rows: int, name: str) -> np.ndarray:
    indices = np.asarray(values)
    if indices.ndim != 1 or indices.size == 0 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"{name} must be a nonempty integer vector")
    normalized = np.ascontiguousarray(indices, dtype=np.int64)
    if np.any(normalized < 0) or np.any(normalized >= rows):
        raise ValueError(f"{name} contains an out-of-range row")
    if np.unique(normalized).size != normalized.size:
        raise ValueError(f"{name} contains duplicate rows")
    return normalized


def validate_fit_holdout_indices(
    fit_indices: np.ndarray,
    holdout_indices: np.ndarray,
    *,
    rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    fit = _normalized_indices(fit_indices, rows=rows, name="fit_indices")
    holdout = _normalized_indices(holdout_indices, rows=rows, name="holdout_indices")
    if np.intersect1d(fit, holdout, assume_unique=True).size:
        raise ValueError("fit_indices and holdout_indices must be disjoint")
    return fit, holdout


def deterministic_epoch_batches(
    fit_indices: np.ndarray,
    *,
    rows: int,
    microbatch: int,
    seed: int,
    epoch: int,
) -> tuple[np.ndarray, ...]:
    fit = _normalized_indices(fit_indices, rows=rows, name="fit_indices")
    if microbatch <= 0 or epoch <= 0:
        raise ValueError("microbatch and epoch must be positive")
    material = f"loop175-engine-order|{seed}|{epoch}|{fit.size}".encode("ascii")
    order_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    order = fit.copy()
    np.random.default_rng(order_seed).shuffle(order)
    return tuple(order[start : start + microbatch] for start in range(0, order.size, microbatch))


def _validate_payload(payload: IdentityFreePhaseBFitPayload) -> int:
    if not isinstance(payload, IdentityFreePhaseBFitPayload):
        raise TypeError("payload must be an IdentityFreePhaseBFitPayload")
    rows = int(np.asarray(payload.labels).size)
    if rows <= 0:
        raise ValueError("payload is empty")
    if np.asarray(payload.labels).shape != (rows,) or not np.isin(payload.labels, (0, 1)).all():
        raise ValueError("payload labels are invalid")
    if np.asarray(payload.folds).shape != (rows,):
        raise ValueError("payload folds are invalid")
    b0 = np.asarray(payload.b0_values)
    if b0.shape != (rows, B0_FEATURE_DIMENSION) or not np.isfinite(b0).all():
        raise ValueError("payload B0 values are invalid")
    regions = payload.regions
    if not isinstance(regions, RaggedRegionCache):
        raise TypeError("payload regions must be a RaggedRegionCache")
    if np.asarray(regions.row_region_offsets).shape != (rows + 1,):
        raise ValueError("payload region row offsets are invalid")
    return rows


def _sample_weight_vector(values: np.ndarray | None, *, rows: int) -> np.ndarray | None:
    if values is None:
        return None
    weights = np.asarray(values, dtype=np.float32)
    if weights.shape != (rows,) or not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("sample_weights must be a positive finite vector with one value per row")
    return np.ascontiguousarray(weights)


def collate_ragged_region_rows(
    payload: IdentityFreePhaseBFitPayload,
    receiver_indices: np.ndarray,
    *,
    donor_indices: np.ndarray | None = None,
    sample_weights: np.ndarray | None = None,
    expected_regions: int = 16,
    expected_region_bytes: int = 8192,
) -> FixedRegionBatch:
    """Materialize receiver targets/B0 and donor regions as separate planes."""

    rows = _validate_payload(payload)
    receivers = _normalized_indices(receiver_indices, rows=rows, name="receiver_indices")
    donors = (
        receivers.copy()
        if donor_indices is None
        else _normalized_indices(donor_indices, rows=rows, name="donor_indices")
    )
    if donors.shape != receivers.shape:
        raise ValueError("donor_indices must contain one donor per receiver")
    weights = _sample_weight_vector(sample_weights, rows=rows)

    cache = payload.regions
    tokens = torch.full(
        (receivers.size, expected_regions, expected_region_bytes),
        256,
        dtype=torch.int64,
    )
    lengths = torch.zeros((receivers.size, expected_regions), dtype=torch.int64)
    region_types = torch.zeros_like(lengths)
    offset_buckets = torch.zeros_like(lengths)
    length_buckets = torch.zeros_like(lengths)
    for batch_index, donor in enumerate(donors.tolist()):
        region_start = int(cache.row_region_offsets[donor])
        region_end = int(cache.row_region_offsets[donor + 1])
        if region_end - region_start != expected_regions:
            raise ValueError("every engine row must have exactly the frozen region count")
        for slot, region_index in enumerate(range(region_start, region_end)):
            token_start = int(cache.region_token_offsets[region_index])
            token_end = int(cache.region_token_offsets[region_index + 1])
            length = token_end - token_start
            if not 0 <= length <= expected_region_bytes:
                raise ValueError("cached region length drifted")
            if length:
                values = np.asarray(cache.token_values[token_start:token_end], dtype=np.uint8)
                tokens[batch_index, slot, :length] = torch.from_numpy(values.copy()).long()
            lengths[batch_index, slot] = length
            region_types[batch_index, slot] = int(cache.region_types[region_index])
            offset_buckets[batch_index, slot] = int(cache.offset_buckets[region_index])
            length_buckets[batch_index, slot] = int(cache.length_buckets[region_index])

    return FixedRegionBatch(
        receiver_indices=receivers,
        donor_indices=donors,
        region_tokens=tokens,
        region_lengths=lengths,
        region_types=region_types,
        offset_buckets=offset_buckets,
        length_buckets=length_buckets,
        b0_features=torch.from_numpy(np.asarray(payload.b0_values[receivers], dtype=np.float32).copy()),
        labels=torch.from_numpy(np.asarray(payload.labels[receivers], dtype=np.int64).copy()),
        sample_weights=(
            None
            if weights is None
            else torch.from_numpy(np.asarray(weights[receivers], dtype=np.float32).copy())
        ),
    )


def build_d_donor_mapping(
    *,
    rows: int,
    fit_indices: np.ndarray,
    holdout_indices: np.ndarray,
    protocol_sha256: str,
    seed: int,
    outer_fold: int,
) -> np.ndarray:
    """Build two isolated donor bijections without accepting labels or identities."""

    fit, holdout = validate_fit_holdout_indices(fit_indices, holdout_indices, rows=rows)
    mapping = np.full(rows, -1, dtype=np.int64)
    for role, receivers in (("fit", fit), ("holdout", holdout)):
        permutation = deterministic_region_record_permutation(
            receivers.size,
            protocol_sha256=protocol_sha256,
            seed=seed,
            outer_fold=outer_fold,
            role=role,
        )
        mapping[receivers] = receivers[permutation]
        if np.any(mapping[receivers] == receivers):
            raise RuntimeError("D donor mapping retained a fixed point")
        if set(mapping[receivers].tolist()) != set(receivers.tolist()):
            raise RuntimeError("D donor mapping crossed its fit/holdout partition")
    mapping.setflags(write=False)
    return mapping


def _resolve_device(requested: str) -> tuple[torch.device, bool]:
    if requested == "cpu":
        return torch.device("cpu"), False
    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    if requested in {"auto", "cuda"} and cuda_available:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Loop175 CUDA execution requires BF16 support")
        return torch.device("cuda"), True
    return torch.device("cpu"), False


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True)


class _ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            target = self.state[name]
            if target.is_floating_point():
                target.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                target.copy_(value.detach())

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.state, strict=True)


def _make_model(model_config: RegionNetConfig, engine: EngineConfig, device: torch.device) -> FailClosedRegionNet:
    model = FailClosedRegionNet(
        RegionNet(model_config),
        expected_regions=engine.expected_regions,
        expected_region_bytes=engine.expected_region_bytes,
    )
    return model.to(device=device, dtype=torch.float32)


def _derived_model_seed(seed: int, arm: str) -> int:
    material = f"loop175-engine-model|{seed}|{arm}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _save_checkpoint_exclusive(
    path: Path,
    *,
    model: FailClosedRegionNet,
    arm: str,
    frozen_epoch: int,
    model_config: RegionNetConfig,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise RuntimeError("refusing to overwrite Loop175 neural checkpoint") from error
    checkpoint = {
        "schema": "axon_loop175_neural_arm_checkpoint_v1",
        "arm": arm,
        "frozen_epoch": frozen_epoch,
        "model_config": asdict(model_config),
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
    }
    with os.fdopen(descriptor, "wb") as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_arm_weights(
    arm: str,
    sample_weights: np.ndarray | None,
    *,
    rows: int,
) -> np.ndarray | None:
    if arm not in NEURAL_ARMS:
        raise ValueError(f"arm must be one of {sorted(NEURAL_ARMS)}")
    weights = _sample_weight_vector(sample_weights, rows=rows)
    if arm == "E" and weights is None:
        raise ValueError("Arm E requires frozen B0 inner-OOF sample weights")
    if arm != "E" and weights is not None:
        raise ValueError("sample weights are forbidden outside Arm E")
    return weights


def _scheduler_multiplier(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def _move_batch(batch: FixedRegionBatch, device: torch.device) -> FixedRegionBatch:
    return FixedRegionBatch(
        receiver_indices=batch.receiver_indices,
        donor_indices=batch.donor_indices,
        region_tokens=batch.region_tokens.to(device),
        region_lengths=batch.region_lengths.to(device),
        region_types=batch.region_types.to(device),
        offset_buckets=batch.offset_buckets.to(device),
        length_buckets=batch.length_buckets.to(device),
        b0_features=batch.b0_features.to(device=device, dtype=torch.float32),
        labels=batch.labels.to(device),
        sample_weights=(
            None
            if batch.sample_weights is None
            else batch.sample_weights.to(device=device, dtype=torch.float32)
        ),
    )


def loss_numerator_and_normalizer(
    per_sample_losses: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return additive loss terms so microbatch partitioning cannot change the objective."""

    if per_sample_losses.ndim != 1 or per_sample_losses.numel() == 0:
        raise ValueError("per_sample_losses must be a nonempty vector")
    if not torch.isfinite(per_sample_losses).all().item():
        raise ValueError("per_sample_losses must be finite")
    if sample_weights is None:
        return (
            per_sample_losses.sum(),
            per_sample_losses.new_tensor(float(per_sample_losses.numel())),
        )
    if sample_weights.shape != per_sample_losses.shape:
        raise ValueError("sample_weights must match per_sample_losses")
    if not torch.isfinite(sample_weights).all().item() or torch.any(sample_weights <= 0).item():
        raise ValueError("sample_weights must be positive and finite")
    return (per_sample_losses * sample_weights).sum(), sample_weights.sum()


def _batch_loss_terms(
    logits: torch.Tensor,
    batch: FixedRegionBatch,
    *,
    arm: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_sample = functional.cross_entropy(logits.float(), batch.labels, reduction="none")
    if arm == "E":
        if batch.sample_weights is None:
            raise RuntimeError("Arm E batch lost its sample weights")
        return loss_numerator_and_normalizer(per_sample, batch.sample_weights)
    if batch.sample_weights is not None:
        raise RuntimeError("a non-E arm received sample weights")
    return loss_numerator_and_normalizer(per_sample)


def _forward_logits(model: FailClosedRegionNet, batch: FixedRegionBatch, *, arm: str) -> torch.Tensor:
    outputs = model(
        batch.region_tokens,
        batch.region_lengths,
        batch.region_types,
        batch.offset_buckets,
        batch.length_buckets,
        None if arm == "B" else batch.b0_features,
    )
    return outputs["region_logits" if arm == "B" else "fusion_logits"]


def _train_one_epoch(
    *,
    model: FailClosedRegionNet,
    ema: _ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    payload: IdentityFreePhaseBFitPayload,
    fit_indices: np.ndarray,
    donor_mapping: np.ndarray | None,
    sample_weights: np.ndarray | None,
    arm: str,
    epoch: int,
    engine: EngineConfig,
    device: torch.device,
    use_bf16: bool,
) -> tuple[float, int]:
    model.train()
    batches = deterministic_epoch_batches(
        fit_indices,
        rows=np.asarray(payload.labels).size,
        microbatch=engine.microbatch,
        seed=engine.seed,
        epoch=epoch,
    )
    epoch_numerator = 0.0
    epoch_normalizer = 0.0
    total_rows = 0
    optimizer_steps = 0
    for window_start in range(0, len(batches), engine.gradient_accumulation):
        window = batches[window_start : window_start + engine.gradient_accumulation]
        window_indices = np.concatenate(window)
        window_normalizer = (
            float(window_indices.size)
            if sample_weights is None
            else float(sample_weights[window_indices].sum(dtype=np.float64))
        )
        if not math.isfinite(window_normalizer) or window_normalizer <= 0.0:
            raise FloatingPointError("Loop175 accumulation window has an invalid normalizer")
        optimizer.zero_grad(set_to_none=True)
        for receiver_indices in window:
            donors = None if donor_mapping is None else donor_mapping[receiver_indices]
            batch = _move_batch(
                collate_ragged_region_rows(
                    payload,
                    receiver_indices,
                    donor_indices=donors,
                    sample_weights=sample_weights,
                    expected_regions=engine.expected_regions,
                    expected_region_bytes=engine.expected_region_bytes,
                ),
                device,
            )
            with _autocast(device, use_bf16):
                logits = _forward_logits(model, batch, arm=arm)
            numerator, normalizer = _batch_loss_terms(logits, batch, arm=arm)
            if not torch.isfinite(numerator).item() or not torch.isfinite(normalizer).item():
                raise FloatingPointError("Loop175 neural training produced a non-finite loss")
            (numerator / window_normalizer).backward()
            epoch_numerator += float(numerator.detach())
            epoch_normalizer += float(normalizer.detach())
            total_rows += receiver_indices.size
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), engine.gradient_clip)
        if not torch.isfinite(gradient_norm).item():
            raise FloatingPointError("Loop175 neural training produced a non-finite gradient norm")
        optimizer.step()
        scheduler.step()
        ema.update(model)
        optimizer_steps += 1
    if total_rows != fit_indices.size:
        raise RuntimeError("one training epoch did not consume exactly the fit partition")
    if not math.isfinite(epoch_normalizer) or epoch_normalizer <= 0.0:
        raise RuntimeError("one training epoch accumulated an invalid loss normalizer")
    return epoch_numerator / epoch_normalizer, optimizer_steps


@torch.no_grad()
def _evaluate(
    *,
    model: FailClosedRegionNet,
    payload: IdentityFreePhaseBFitPayload,
    receiver_indices: np.ndarray,
    donor_mapping: np.ndarray | None,
    arm: str,
    engine: EngineConfig,
    device: torch.device,
    use_bf16: bool,
    return_scores: bool,
) -> tuple[float, np.ndarray | None]:
    model.eval()
    losses: list[float] = []
    scores: list[np.ndarray] = []
    for start in range(0, receiver_indices.size, engine.microbatch):
        receivers = receiver_indices[start : start + engine.microbatch]
        donors = None if donor_mapping is None else donor_mapping[receivers]
        batch = _move_batch(
            collate_ragged_region_rows(
                payload,
                receivers,
                donor_indices=donors,
                sample_weights=None,
                expected_regions=engine.expected_regions,
                expected_region_bytes=engine.expected_region_bytes,
            ),
            device,
        )
        with _autocast(device, use_bf16):
            logits = _forward_logits(model, batch, arm=arm)
        per_sample = functional.cross_entropy(logits.float(), batch.labels, reduction="none")
        if not torch.isfinite(per_sample).all().item():
            raise FloatingPointError("Loop175 neural evaluation produced a non-finite loss")
        losses.extend(per_sample.cpu().tolist())
        if return_scores:
            probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
            if not torch.isfinite(probabilities).all().item():
                raise FloatingPointError("Loop175 neural evaluation produced a non-finite score")
            scores.append(probabilities.cpu().numpy())
    if len(losses) != receiver_indices.size:
        raise RuntimeError("evaluation did not consume exactly its receiver partition")
    combined = None if not return_scores else np.ascontiguousarray(np.concatenate(scores), dtype=np.float64)
    return float(np.mean(losses)), combined


def select_earliest_minimum_epoch(losses: Iterable[float]) -> int:
    values = tuple(float(value) for value in losses)
    if not values or not np.isfinite(values).all():
        raise ValueError("pilot losses must be a nonempty finite sequence")
    minimum = min(values)
    return values.index(minimum) + 1


def run_inner_pilot(
    payload: IdentityFreePhaseBFitPayload,
    *,
    pilot_fit_indices: np.ndarray,
    selection_indices: np.ndarray,
    arm: str,
    max_epochs: int,
    model_config: RegionNetConfig,
    engine: EngineConfig,
    sample_weights: np.ndarray | None = None,
    protocol_sha256: str | None = None,
    outer_fold: int = 0,
) -> PilotResult:
    """Select one epoch using unweighted CE on an explicit disjoint inner selection."""

    rows = _validate_payload(payload)
    fit, selection = validate_fit_holdout_indices(pilot_fit_indices, selection_indices, rows=rows)
    weights = _validate_arm_weights(arm, sample_weights, rows=rows)
    if max_epochs <= 0 or engine.warmup_epochs > max_epochs:
        raise ValueError("max_epochs must be at least warmup_epochs")
    donor_mapping = None
    if arm == "D":
        if protocol_sha256 is None:
            raise ValueError("Arm D requires protocol_sha256")
        donor_mapping = build_d_donor_mapping(
            rows=rows,
            fit_indices=fit,
            holdout_indices=selection,
            protocol_sha256=protocol_sha256,
            seed=engine.seed,
            outer_fold=outer_fold,
        )

    model_seed = _derived_model_seed(engine.seed, arm)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    device, use_bf16 = _resolve_device(engine.device)
    model = _make_model(model_config, engine, device)
    ema = _ExponentialMovingAverage(model, engine.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=engine.learning_rate,
        weight_decay=engine.weight_decay,
    )
    batches_per_epoch = math.ceil(math.ceil(fit.size / engine.microbatch) / engine.gradient_accumulation)
    total_steps = max_epochs * batches_per_epoch
    warmup_steps = engine.warmup_epochs * batches_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _scheduler_multiplier(
            step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        ),
    )
    losses: list[float] = []
    for epoch in range(1, max_epochs + 1):
        _train_one_epoch(
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            payload=payload,
            fit_indices=fit,
            donor_mapping=donor_mapping,
            sample_weights=weights,
            arm=arm,
            epoch=epoch,
            engine=engine,
            device=device,
            use_bf16=use_bf16,
        )
        # Pilot 选择指标始终是未加权 CE；E 权重只作用于 pilot fit 的反向传播。
        ema_model = copy.deepcopy(model)
        ema.copy_to(ema_model)
        selection_loss, _scores = _evaluate(
            model=ema_model,
            payload=payload,
            receiver_indices=selection,
            donor_mapping=donor_mapping,
            arm=arm,
            engine=engine,
            device=device,
            use_bf16=use_bf16,
            return_scores=False,
        )
        losses.append(selection_loss)
        del ema_model
    return PilotResult(
        arm=arm,
        selected_epoch=select_earliest_minimum_epoch(losses),
        selection_losses=tuple(losses),
        selection_is_unweighted=True,
    )


def train_neural_arm(
    payload: IdentityFreePhaseBFitPayload,
    *,
    fit_indices: np.ndarray,
    holdout_indices: np.ndarray,
    arm: str,
    frozen_epoch: int,
    model_config: RegionNetConfig,
    engine: EngineConfig,
    sample_weights: np.ndarray | None = None,
    protocol_sha256: str | None = None,
    outer_fold: int = 0,
    checkpoint_path: Path | None = None,
) -> NeuralArmResult:
    """Train one independent arm from scratch and score only its explicit holdout."""

    rows = _validate_payload(payload)
    fit, holdout = validate_fit_holdout_indices(fit_indices, holdout_indices, rows=rows)
    if fit.size + holdout.size != rows:
        raise ValueError("final outer fit and holdout must cover every payload row exactly once")
    weights = _validate_arm_weights(arm, sample_weights, rows=rows)
    if frozen_epoch <= 0 or engine.warmup_epochs > frozen_epoch:
        raise ValueError("frozen_epoch must be at least warmup_epochs")
    donor_mapping = None
    if arm == "D":
        if protocol_sha256 is None:
            raise ValueError("Arm D requires protocol_sha256")
        donor_mapping = build_d_donor_mapping(
            rows=rows,
            fit_indices=fit,
            holdout_indices=holdout,
            protocol_sha256=protocol_sha256,
            seed=engine.seed,
            outer_fold=outer_fold,
        )

    model_seed = _derived_model_seed(engine.seed, arm)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    device, use_bf16 = _resolve_device(engine.device)
    model = _make_model(model_config, engine, device)
    ema = _ExponentialMovingAverage(model, engine.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=engine.learning_rate,
        weight_decay=engine.weight_decay,
    )
    batches_per_epoch = math.ceil(math.ceil(fit.size / engine.microbatch) / engine.gradient_accumulation)
    total_steps = frozen_epoch * batches_per_epoch
    warmup_steps = engine.warmup_epochs * batches_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _scheduler_multiplier(
            step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        ),
    )
    optimizer_steps = 0
    final_loss = math.nan
    for epoch in range(1, frozen_epoch + 1):
        final_loss, epoch_steps = _train_one_epoch(
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            payload=payload,
            fit_indices=fit,
            donor_mapping=donor_mapping,
            sample_weights=weights,
            arm=arm,
            epoch=epoch,
            engine=engine,
            device=device,
            use_bf16=use_bf16,
        )
        optimizer_steps += epoch_steps
    ema.copy_to(model)
    _loss, scores = _evaluate(
        model=model,
        payload=payload,
        receiver_indices=holdout,
        donor_mapping=donor_mapping,
        arm=arm,
        engine=engine,
        device=device,
        use_bf16=use_bf16,
        return_scores=True,
    )
    if scores is None or scores.shape != (holdout.size,) or not np.isfinite(scores).all():
        raise RuntimeError("final EMA did not produce one finite score per holdout row")
    scores.setflags(write=False)
    checkpoint_sha256 = None
    if checkpoint_path is not None:
        checkpoint_sha256 = _save_checkpoint_exclusive(
            checkpoint_path,
            model=model,
            arm=arm,
            frozen_epoch=frozen_epoch,
            model_config=model_config,
        )
    return NeuralArmResult(
        arm=arm,
        frozen_epoch=frozen_epoch,
        holdout_scores=scores,
        optimizer_steps=optimizer_steps,
        used_sample_weights=weights is not None,
        sample_weight_sum=(None if weights is None else float(weights[fit].sum(dtype=np.float64))),
        final_training_loss=float(final_loss),
        device_type=device.type,
        autocast_dtype="bf16" if use_bf16 else "fp32",
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha256,
    )
