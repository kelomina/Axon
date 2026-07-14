import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AxonExperimentConfig, DSRAArchitectureConfig  # noqa: E402
from model import AxonMalwareModel, PositionalEncoding  # noqa: E402


def _chunked_config() -> AxonExperimentConfig:
    return AxonExperimentConfig(
        max_byte_length=32,
        pe_feature_dim=16,
        byte_embedding_dim=8,
        dsra_dim=8,
        dsra_heads=2,
        dsra_slots=8,
        dsra_read_topk=2,
        dsra_write_topk=2,
        dsra_local_window=8,
        dsra_chunk_size=8,
        pe_projection_dim=8,
        pe_projector_hidden_dim=16,
        classifier_hidden_dim=8,
        dropout=0.0,
        dsra_arch_config=DSRAArchitectureConfig(
            dim=8,
            heads=2,
            slots=8,
            read_topk=2,
            write_topk=2,
            local_window=8,
            use_local=False,
            use_retrieval=False,
        ),
    )


def _inputs(config: AxonExperimentConfig):
    return (
        torch.randint(0, 256, (2, config.max_byte_length), dtype=torch.long),
        torch.randn(2, config.pe_feature_dim),
        torch.randn(2, config.stat_feature_dim),
    )


def test_regular_forward_skips_diversity_loss(monkeypatch):
    config = _chunked_config()
    model = AxonMalwareModel(config)
    model.eval()
    byte_seq, pe_features, stat_features = _inputs(config)

    def fail_if_called(_state):
        raise AssertionError("diversity_loss should not run during regular eval forward")

    monkeypatch.setattr(model.dsra, "diversity_loss", fail_if_called)

    with torch.no_grad():
        output = model(byte_seq, pe_features, stat_features=stat_features)

    assert output["logits"].shape == (2, config.num_classes)
    assert "diversity_loss" not in output


def test_forward_accepts_uint8_byte_sequence():
    config = _chunked_config()
    model = AxonMalwareModel(config)
    model.eval()
    byte_seq, pe_features, stat_features = _inputs(config)
    byte_seq = byte_seq.to(torch.uint8)

    with torch.no_grad():
        output = model(byte_seq, pe_features, stat_features=stat_features)

    assert output["logits"].shape == (2, config.num_classes)


def test_return_state_forward_skips_diversity_loss_by_default(monkeypatch):
    config = _chunked_config()
    model = AxonMalwareModel(config)
    model.train()
    byte_seq, pe_features, stat_features = _inputs(config)

    def fail_if_called(_state):
        raise AssertionError("return_state alone should not compute diversity_loss")

    monkeypatch.setattr(model.dsra, "diversity_loss", fail_if_called)

    output = model(byte_seq, pe_features, stat_features=stat_features, return_state=True)

    assert output["logits"].shape == (2, config.num_classes)
    assert "dsra_state" in output
    assert "diversity_loss" not in output
    assert model.dsra_encoder._last_diversity_loss is None


def test_explicit_diversity_loss_forward_returns_loss(monkeypatch):
    config = _chunked_config()
    model = AxonMalwareModel(config)
    model.train()
    byte_seq, pe_features, stat_features = _inputs(config)
    calls = {"count": 0}

    def counted_diversity_loss(state):
        calls["count"] += 1
        return torch.zeros((), device=state.slot_k.device, dtype=state.slot_k.dtype)

    monkeypatch.setattr(model.dsra, "diversity_loss", counted_diversity_loss)

    output = model(
        byte_seq,
        pe_features,
        stat_features=stat_features,
        return_state=True,
        compute_diversity_loss=True,
    )

    assert output["logits"].shape == (2, config.num_classes)
    assert "dsra_state" in output
    assert "diversity_loss" in output
    assert calls["count"] == 1


def test_return_state_forward_does_not_attach_diversity_loss_to_state_or_module():
    config = _chunked_config()
    model = AxonMalwareModel(config)
    model.train()
    byte_seq, pe_features, stat_features = _inputs(config)

    output = model(
        byte_seq,
        pe_features,
        stat_features=stat_features,
        return_state=True,
        compute_diversity_loss=True,
    )
    dsra_state = output["dsra_state"]
    primary_state = dsra_state[0] if isinstance(dsra_state, (list, tuple)) else dsra_state

    assert "diversity_loss" in output
    assert not hasattr(primary_state, "_diversity_loss")
    assert not hasattr(model.dsra, "_slot_k_before_detach")
    assert getattr(model.dsra, "_capture_slot_k_before_detach", False) is False
    assert model.dsra_encoder._last_diversity_loss is None


def test_positional_encoding_offset_matches_full_sequence_slice():
    encoding = PositionalEncoding(d_model=8, max_len=32, mode="sinusoidal")
    full = torch.zeros(2, 32, 8)
    chunk_a = torch.zeros(2, 8, 8)
    chunk_b = torch.zeros(2, 8, 8)

    full_encoded = encoding(full)
    chunked_slice = torch.cat(
        [
            encoding(chunk_a, offset=8),
            encoding(chunk_b, offset=16),
        ],
        dim=1,
    )

    torch.testing.assert_close(chunked_slice, full_encoded[:, 8:24, :])
