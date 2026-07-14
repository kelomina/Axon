import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AxonExperimentConfig, DSRAArchitectureConfig  # noqa: E402
from model import AxonMalwareModel  # noqa: E402


def test_gated_fusion_forward_returns_normalized_gate_weights():
    config = AxonExperimentConfig(
        max_byte_length=16,
        pe_feature_dim=32,
        byte_embedding_dim=8,
        dsra_dim=8,
        dsra_heads=2,
        dsra_slots=8,
        dsra_chunk_size=16,
        pe_projection_dim=8,
        pe_projector_hidden_dim=16,
        classifier_hidden_dim=8,
        fusion_type="gated",
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
    model = AxonMalwareModel(config)
    model.eval()

    batch_size = 3
    byte_seq = torch.randint(0, 256, (batch_size, config.max_byte_length), dtype=torch.long)
    pe_features = torch.randn(batch_size, config.pe_feature_dim)
    stat_features = torch.randn(batch_size, config.stat_feature_dim)

    with torch.no_grad():
        output = model(byte_seq, pe_features, stat_features, return_features=True)

    assert output["logits"].shape == (batch_size, config.num_classes)
    assert output["features"].shape == (batch_size, config.pe_projection_dim)
    assert output["fusion_gate_weights"].shape == (batch_size, 3)
    assert torch.allclose(
        output["fusion_gate_weights"].sum(dim=-1),
        torch.ones(batch_size),
        atol=1e-6,
    )


def test_residual_stat_gate_forward_keeps_concat_feature_width():
    config = AxonExperimentConfig(
        max_byte_length=16,
        pe_feature_dim=32,
        byte_embedding_dim=8,
        dsra_dim=8,
        dsra_heads=2,
        dsra_slots=8,
        dsra_chunk_size=16,
        pe_projection_dim=8,
        pe_projector_hidden_dim=16,
        classifier_hidden_dim=8,
        fusion_type="residual_stat_gate",
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
    model = AxonMalwareModel(config)
    model.eval()

    batch_size = 3
    byte_seq = torch.randint(0, 256, (batch_size, config.max_byte_length), dtype=torch.long)
    pe_features = torch.randn(batch_size, config.pe_feature_dim)
    stat_features = torch.randn(batch_size, config.stat_feature_dim)

    with torch.no_grad():
        output = model(byte_seq, pe_features, stat_features, return_features=True)

    expected_feature_dim = config.dsra_arch_config.dim + config.pe_projection_dim * 2
    assert output["logits"].shape == (batch_size, config.num_classes)
    assert output["features"].shape == (batch_size, expected_feature_dim)
    assert output["stat_gate_weights"].shape == (batch_size, config.pe_projection_dim)
    assert torch.all(output["stat_gate_weights"] >= 0)
    assert torch.all(output["stat_gate_weights"] <= 1)


def test_residual_channel_gate_forward_keeps_concat_feature_width_and_gate_range():
    config = AxonExperimentConfig(
        max_byte_length=16,
        pe_feature_dim=32,
        byte_embedding_dim=8,
        dsra_dim=8,
        dsra_heads=2,
        dsra_slots=8,
        dsra_chunk_size=16,
        pe_projection_dim=8,
        pe_projector_hidden_dim=16,
        classifier_hidden_dim=8,
        fusion_type="residual_channel_gate",
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
    model = AxonMalwareModel(config)
    model.eval()

    batch_size = 3
    byte_seq = torch.randint(0, 256, (batch_size, config.max_byte_length), dtype=torch.long)
    pe_features = torch.randn(batch_size, config.pe_feature_dim)
    stat_features = torch.randn(batch_size, config.stat_feature_dim)

    with torch.no_grad():
        output = model(byte_seq, pe_features, stat_features, return_features=True)

    expected_feature_dim = config.dsra_arch_config.dim + config.pe_projection_dim * 2
    assert output["logits"].shape == (batch_size, config.num_classes)
    assert output["features"].shape == (batch_size, expected_feature_dim)
    assert output["channel_gate_weights"].shape == (batch_size, expected_feature_dim)
    assert torch.all(output["channel_gate_weights"] >= 0.5)
    assert torch.all(output["channel_gate_weights"] <= 1.5)
