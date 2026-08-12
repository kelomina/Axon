from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop175.model import RegionNetConfig  # noqa: E402
from src.loop175.phase_b_data import (  # noqa: E402
    IdentityFreePhaseBFitPayload,
    RaggedRegionCache,
)
from src.loop175.phase_b_engine import (  # noqa: E402
    EngineConfig,
    build_d_donor_mapping,
    collate_ragged_region_rows,
    deterministic_epoch_batches,
    loss_numerator_and_normalizer,
    run_inner_pilot,
    select_earliest_minimum_epoch,
    train_neural_arm,
)


def _payload(rows: int = 8) -> IdentityFreePhaseBFitPayload:
    row_region_offsets = np.arange(0, 2 * rows + 1, 2, dtype="<i8")
    region_token_offsets = [0]
    token_values: list[int] = []
    region_types: list[int] = []
    region_starts: list[int] = []
    offset_buckets: list[int] = []
    length_buckets: list[int] = []
    for row in range(rows):
        values = [row + 1] * 32
        token_values.extend(values)
        region_token_offsets.append(len(token_values))
        region_types.append(1)
        region_starts.append(0)
        offset_buckets.append(0)
        length_buckets.append(1)
        region_token_offsets.append(len(token_values))
        region_types.append(0)
        region_starts.append(0)
        offset_buckets.append(0)
        length_buckets.append(0)
    regions = RaggedRegionCache(
        row_region_offsets=row_region_offsets,
        file_sizes=np.full(rows, 32, dtype="<i8"),
        region_token_offsets=np.asarray(region_token_offsets, dtype="<i8"),
        token_values=np.asarray(token_values, dtype="u1"),
        region_types=np.asarray(region_types, dtype="u1"),
        region_starts=np.asarray(region_starts, dtype="<i8"),
        offset_buckets=np.asarray(offset_buckets, dtype="u1"),
        length_buckets=np.asarray(length_buckets, dtype="u1"),
    )
    b0 = np.zeros((rows, 571), dtype="<f4")
    b0[:, 0] = np.arange(rows, dtype=np.float32) + 100.0
    labels = np.asarray([row % 2 for row in range(rows)], dtype="u1")
    folds = np.asarray([row % 5 for row in range(rows)], dtype="i1")
    return IdentityFreePhaseBFitPayload(b0_values=b0, labels=labels, folds=folds, regions=regions)


def _model_config() -> RegionNetConfig:
    return RegionNetConfig(
        byte_embedding_dim=8,
        model_dim=12,
        block_count=1,
        block_expansion=2,
        dilations=(1,),
        transformer_layers=1,
        transformer_heads=3,
        transformer_ffn_dim=24,
        dropout=0.0,
    )


def _engine() -> EngineConfig:
    return EngineConfig(
        seed=41,
        microbatch=2,
        gradient_accumulation=1,
        warmup_epochs=0,
        expected_regions=2,
        expected_region_bytes=32,
        device="cpu",
    )


def test_collate_keeps_receiver_plane_separate_from_d_region_donors() -> None:
    payload = _payload()
    fit = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    holdout = np.array([6, 7], dtype=np.int64)
    donor_map = build_d_donor_mapping(
        rows=8,
        fit_indices=fit,
        holdout_indices=holdout,
        protocol_sha256="a" * 64,
        seed=41,
        outer_fold=0,
    )
    receivers = np.array([0, 1], dtype=np.int64)
    batch = collate_ragged_region_rows(
        payload,
        receivers,
        donor_indices=donor_map[receivers],
        expected_regions=2,
        expected_region_bytes=32,
    )
    np.testing.assert_array_equal(batch.receiver_indices, receivers)
    torch.testing.assert_close(batch.b0_features[:, 0], torch.tensor([100.0, 101.0]))
    torch.testing.assert_close(batch.labels, torch.tensor([0, 1]))
    for position, donor in enumerate(batch.donor_indices.tolist()):
        assert torch.all(batch.region_tokens[position, 0] == donor + 1)
    assert not np.any(donor_map[fit] == fit)
    assert set(donor_map[fit].tolist()) == set(fit.tolist())
    assert set(donor_map[holdout].tolist()) == set(holdout.tolist())


def test_epoch_batches_never_include_outer_holdout() -> None:
    fit = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    holdout = {6, 7}
    batches = deterministic_epoch_batches(
        fit,
        rows=8,
        microbatch=2,
        seed=41,
        epoch=1,
    )
    observed = np.concatenate(batches)
    assert set(observed.tolist()) == set(fit.tolist())
    assert set(observed.tolist()).isdisjoint(holdout)


def test_loss_objective_is_invariant_to_microbatch_partitioning() -> None:
    losses = torch.tensor([0.2, 0.7, 1.1, 0.4, 1.8], dtype=torch.float32)
    weights = torch.tensor([1.0, 8.0, 3.0, 0.5, 2.0], dtype=torch.float32)
    full_numerator, full_normalizer = loss_numerator_and_normalizer(losses, weights)
    split_terms = [
        loss_numerator_and_normalizer(losses[:2], weights[:2]),
        loss_numerator_and_normalizer(losses[2:4], weights[2:4]),
        loss_numerator_and_normalizer(losses[4:], weights[4:]),
    ]
    split_numerator = sum(term[0] for term in split_terms)
    split_normalizer = sum(term[1] for term in split_terms)
    torch.testing.assert_close(
        full_numerator / full_normalizer,
        split_numerator / split_normalizer,
        rtol=0.0,
        atol=1.0e-7,
    )

    unweighted_full = loss_numerator_and_normalizer(losses)
    unweighted_split = [loss_numerator_and_normalizer(losses[:4]), loss_numerator_and_normalizer(losses[4:])]
    torch.testing.assert_close(
        unweighted_full[0] / unweighted_full[1],
        sum(term[0] for term in unweighted_split) / sum(term[1] for term in unweighted_split),
        rtol=0.0,
        atol=1.0e-7,
    )


def test_c_and_e_train_independently_and_emit_finite_final_ema_scores() -> None:
    payload = _payload()
    fit = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    holdout = np.array([6, 7], dtype=np.int64)
    c_result = train_neural_arm(
        payload,
        fit_indices=fit,
        holdout_indices=holdout,
        arm="C",
        frozen_epoch=1,
        model_config=_model_config(),
        engine=_engine(),
    )
    weights = np.asarray([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32)
    e_result = train_neural_arm(
        payload,
        fit_indices=fit,
        holdout_indices=holdout,
        arm="E",
        frozen_epoch=1,
        model_config=_model_config(),
        engine=_engine(),
        sample_weights=weights,
    )
    assert c_result.arm == "C" and not c_result.used_sample_weights
    assert e_result.arm == "E" and e_result.used_sample_weights
    assert e_result.sample_weight_sum == pytest.approx(float(weights[fit].sum()))
    for result in (c_result, e_result):
        assert result.device_type == "cpu"
        assert result.autocast_dtype == "fp32"
        assert result.holdout_scores.shape == (2,)
        assert np.isfinite(result.holdout_scores).all()
        assert np.all((0.0 <= result.holdout_scores) & (result.holdout_scores <= 1.0))
    with pytest.raises(ValueError, match="forbidden outside Arm E"):
        train_neural_arm(
            payload,
            fit_indices=fit,
            holdout_indices=holdout,
            arm="C",
            frozen_epoch=1,
            model_config=_model_config(),
            engine=_engine(),
            sample_weights=weights,
        )


def test_inner_pilot_uses_unweighted_ce_and_earliest_tie_rule() -> None:
    payload = _payload()
    weights = np.asarray([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32)
    pilot = run_inner_pilot(
        payload,
        pilot_fit_indices=np.array([0, 1, 2, 3], dtype=np.int64),
        selection_indices=np.array([4, 5], dtype=np.int64),
        arm="E",
        max_epochs=2,
        model_config=_model_config(),
        engine=_engine(),
        sample_weights=weights,
    )
    assert pilot.selection_is_unweighted
    assert pilot.selected_epoch in {1, 2}
    assert len(pilot.selection_losses) == 2
    assert np.isfinite(pilot.selection_losses).all()
    assert select_earliest_minimum_epoch([0.7, 0.4, 0.4, 0.5]) == 2


def test_fit_and_holdout_overlap_fails_before_training() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        train_neural_arm(
            _payload(),
            fit_indices=np.array([0, 1, 2], dtype=np.int64),
            holdout_indices=np.array([2, 3], dtype=np.int64),
            arm="B",
            frozen_epoch=1,
            model_config=_model_config(),
            engine=_engine(),
        )


def test_final_outer_partition_cannot_omit_rows() -> None:
    payload = _payload()
    with pytest.raises(ValueError, match="cover every payload row"):
        train_neural_arm(
            payload,
            fit_indices=np.array([0, 1, 2, 3]),
            holdout_indices=np.array([4, 5]),
            arm="B",
            frozen_epoch=1,
            model_config=_model_config(),
            engine=_engine(),
        )


def test_one_frozen_epoch_may_consist_entirely_of_warmup() -> None:
    engine = EngineConfig(
        seed=41,
        microbatch=2,
        gradient_accumulation=1,
        warmup_epochs=1,
        expected_regions=2,
        expected_region_bytes=32,
        device="cpu",
    )
    result = train_neural_arm(
        _payload(),
        fit_indices=np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        holdout_indices=np.array([6, 7], dtype=np.int64),
        arm="B",
        frozen_epoch=1,
        model_config=_model_config(),
        engine=engine,
    )
    assert result.frozen_epoch == 1


def test_final_ema_checkpoint_is_exclusive_and_loadable(tmp_path: Path) -> None:
    payload = _payload()
    checkpoint = tmp_path / "arm_c_fold_0.pt"
    result = train_neural_arm(
        payload,
        fit_indices=np.arange(6),
        holdout_indices=np.arange(6, 8),
        arm="C",
        frozen_epoch=1,
        model_config=_model_config(),
        engine=_engine(),
        checkpoint_path=checkpoint,
    )
    assert result.checkpoint_path == str(checkpoint)
    assert result.checkpoint_sha256 is not None
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert loaded["schema"] == "axon_loop175_neural_arm_checkpoint_v1"
    assert loaded["arm"] == "C"
    with pytest.raises(RuntimeError, match="overwrite"):
        train_neural_arm(
            payload,
            fit_indices=np.arange(6),
            holdout_indices=np.arange(6, 8),
            arm="C",
            frozen_epoch=1,
            model_config=_model_config(),
            engine=_engine(),
            checkpoint_path=checkpoint,
        )
    assert np.isfinite(result.holdout_scores).all()
