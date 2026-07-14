import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import DSRAArchitectureConfig  # noqa: E402
from model import MalwareDSRAEncoder  # noqa: E402


class FakeByteEmbedding(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, byte_seq):
        return byte_seq.float().unsqueeze(-1).repeat(1, 1, self.dim)


class FakeDSRA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dsra = self

    def forward(self, x, state=None, return_aux=False):
        return x, state, None


def _encoder(pooling: str) -> MalwareDSRAEncoder:
    encoder = MalwareDSRAEncoder(
        byte_embedding_dim=4,
        max_byte_length=4,
        dsra_config=DSRAArchitectureConfig(dim=4, heads=1, slots=2, read_topk=1, write_topk=1),
        pe_feature_dim=1,
        pe_projection_dim=1,
        pe_projector_hidden_dim=4,
        use_pos_encoding=False,
        dropout=0.0,
        chunk_size=2,
        byte_chunk_pooling=pooling,
    )
    encoder.byte_embedding = FakeByteEmbedding(dim=4)
    encoder.input_proj = torch.nn.Identity()
    encoder.dsra_encoder = FakeDSRA()
    encoder.pe_projector = torch.nn.Identity()
    return encoder


def test_active_mean_chunk_pooling_ignores_zero_padding_chunks():
    byte_seq = torch.tensor([[1, 2, 0, 0]], dtype=torch.long)
    pe_features = torch.zeros(1, 1)

    active_repr, _pe_repr, _state = _encoder("active_mean")(byte_seq, pe_features)
    detached_repr, _pe_repr, _state = _encoder("active_mean_detached")(byte_seq, pe_features)
    mean_repr, _pe_repr, _state = _encoder("mean")(byte_seq, pe_features)
    last_repr, _pe_repr, _state = _encoder("last")(byte_seq, pe_features)

    assert torch.allclose(active_repr, torch.full((1, 4), 1.5))
    assert torch.allclose(detached_repr, torch.full((1, 4), 1.5))
    assert torch.allclose(mean_repr, torch.full((1, 4), 0.75))
    assert torch.allclose(last_repr, torch.zeros(1, 4))
