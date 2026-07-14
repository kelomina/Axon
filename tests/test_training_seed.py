import random
import sys
import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AxonExperimentConfig, DSRAArchitectureConfig  # noqa: E402
from main import _filter_partial_init_state_dict, _make_train_generator, _set_training_seed, train_command  # noqa: E402
from model import AxonMalwareModel  # noqa: E402


def _draw_random_state():
    generator = _make_train_generator(123)
    return {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "torch": torch.rand(3).tolist(),
        "loader_order": torch.randperm(8, generator=generator).tolist(),
    }


def test_set_training_seed_reproduces_python_numpy_torch_and_loader_generator():
    _set_training_seed(123)
    first = _draw_random_state()

    _set_training_seed(123)
    second = _draw_random_state()

    assert second == first


def test_set_training_seed_changes_random_sequence():
    _set_training_seed(123)
    first = _draw_random_state()

    _set_training_seed(456)
    second = _draw_random_state()

    assert second["python"] != first["python"]
    assert second["numpy"] != first["numpy"]
    assert second["torch"] != first["torch"]


def test_train_command_requires_explicit_split_file_outside_fast_mode(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    args = argparse.Namespace(
        config=None,
        data_dir=str(data_dir),
        samples_per_class=None,
        epochs=1,
        batch_size=1,
        lr=None,
        device="cpu",
        output_dir=str(tmp_path / "models"),
        resume=None,
        init_checkpoint=None,
        partial_init=False,
        fast=False,
        fp16=False,
        enable_swanlab=False,
        extract_workers=1,
        extract_backend="thread",
        split_file=None,
        rare_group_weighting=False,
        singleton_group_weight=None,
        rare_group_weight=None,
        medium_group_weight=None,
        skip_test_eval=True,
    )

    with pytest.raises(ValueError, match="Strict training requires --split-file"):
        train_command(args)


def _tiny_model_config(classifier_hidden_dim: int, max_byte_length: int = 16) -> AxonExperimentConfig:
    return AxonExperimentConfig(
        max_byte_length=max_byte_length,
        pe_feature_dim=16,
        byte_embedding_dim=8,
        dsra_dim=8,
        dsra_heads=2,
        dsra_slots=8,
        dsra_chunk_size=16,
        pe_projection_dim=8,
        pe_projector_hidden_dim=16,
        classifier_hidden_dim=classifier_hidden_dim,
        dropout=0.0,
        dsra_arch_config=DSRAArchitectureConfig(
            dim=8,
            heads=2,
            slots=8,
            local_window=8,
            use_local=False,
            use_retrieval=False,
        ),
    )


def test_filter_partial_init_state_dict_skips_shape_mismatches():
    checkpoint_model = AxonMalwareModel(_tiny_model_config(classifier_hidden_dim=8, max_byte_length=16))
    target_model = AxonMalwareModel(_tiny_model_config(classifier_hidden_dim=8, max_byte_length=20))

    filtered, mismatches, adapted = _filter_partial_init_state_dict(
        target_model,
        checkpoint_model.state_dict(),
    )

    mismatch_keys = {key for key, _checkpoint_shape, _model_shape in mismatches}
    assert mismatch_keys == {"dsra_encoder.pos_encoding.pe"}
    assert all(key not in filtered for key in mismatch_keys)
    assert adapted == []

    load_result = target_model.load_state_dict(filtered, strict=False)
    assert set(load_result.missing_keys) == mismatch_keys
    assert load_result.unexpected_keys == []


def test_filter_partial_init_state_dict_adapts_widened_classifier_head():
    checkpoint_model = AxonMalwareModel(_tiny_model_config(classifier_hidden_dim=8))
    target_model = AxonMalwareModel(_tiny_model_config(classifier_hidden_dim=12))

    checkpoint_state = checkpoint_model.state_dict()
    filtered, mismatches, adapted = _filter_partial_init_state_dict(
        target_model,
        checkpoint_state,
    )

    adapted_keys = {key for key, _checkpoint_shape, _model_shape in adapted}
    assert adapted_keys == {
        "classifier.2.weight",
        "classifier.2.bias",
        "classifier.5.weight",
    }
    assert mismatches == []

    assert torch.equal(filtered["classifier.2.weight"][:8, :], checkpoint_state["classifier.2.weight"])
    assert torch.equal(filtered["classifier.2.bias"][:8], checkpoint_state["classifier.2.bias"])
    assert torch.equal(filtered["classifier.5.weight"][:, :8], checkpoint_state["classifier.5.weight"])
    assert torch.count_nonzero(filtered["classifier.5.weight"][:, 8:]).item() == 0

    load_result = target_model.load_state_dict(filtered, strict=False)
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []

    checkpoint_model.eval()
    target_model.eval()
    byte_seq = torch.randint(0, 256, (3, 16), dtype=torch.long)
    pe_features = torch.randn(3, 16)
    stat_features = torch.randn(3, 49)
    with torch.no_grad():
        checkpoint_logits = checkpoint_model(byte_seq, pe_features, stat_features=stat_features)["logits"]
        target_logits = target_model(byte_seq, pe_features, stat_features=stat_features)["logits"]

    torch.testing.assert_close(target_logits, checkpoint_logits, rtol=1e-6, atol=1e-6)
