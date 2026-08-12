from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop175.model import RegionNet, parameter_count  # noqa: E402


def _inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(175)
    tokens = torch.randint(0, 256, (2, 4, 256), generator=generator)
    lengths = torch.tensor([[256, 129, 0, 64], [200, 256, 32, 0]])
    types = torch.tensor([[1, 2, 0, 5], [1, 3, 4, 0]])
    offsets = torch.tensor([[0, 8, 0, 48], [0, 12, 28, 0]])
    length_buckets = torch.tensor([[63, 32, 0, 16], [50, 63, 8, 0]])
    b0 = torch.randn(2, 571, generator=generator)
    return tokens, lengths, types, offsets, length_buckets, b0


def test_region_net_has_frozen_scale_and_finite_backward() -> None:
    torch.manual_seed(175)
    model = RegionNet()
    count = parameter_count(model)
    assert 5_000_000 <= count <= 8_000_000
    output = model(*_inputs())
    assert output["region_logits"].shape == (2, 2)
    assert output["fusion_logits"].shape == (2, 2)
    loss = output["fusion_logits"].float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_padding_bytes_do_not_change_eval_output() -> None:
    torch.manual_seed(175)
    model = RegionNet().eval()
    values = list(_inputs())
    changed = values[0].clone()
    lengths = values[1]
    for batch in range(changed.shape[0]):
        for region in range(changed.shape[1]):
            changed[batch, region, lengths[batch, region] :] = 17
    with torch.no_grad():
        baseline = model(*values)["fusion_logits"]
        candidate = model(changed, *values[1:])["fusion_logits"]
    torch.testing.assert_close(baseline, candidate, rtol=0.0, atol=0.0)
