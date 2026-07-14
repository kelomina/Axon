import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dsra.mhdsra2.improved_dsra_mha import MHDSRA2Config, MultiHeadDSRA2  # noqa: E402


def test_forward_step_detaches_external_kv_cache_when_state_detach_enabled():
    cfg = MHDSRA2Config(
        dim=8,
        heads=2,
        slots=4,
        read_topk=2,
        write_topk=2,
        local_window=4,
        use_retrieval=False,
        detach_state=True,
    )
    module = MultiHeadDSRA2(cfg)
    x_t = torch.randn(1, 1, cfg.dim, requires_grad=True)
    base_k = torch.randn(1, cfg.heads, 2, cfg.dim // cfg.heads, requires_grad=True)
    base_v = torch.randn(1, cfg.heads, 2, cfg.dim // cfg.heads, requires_grad=True)
    cached_k = base_k * 2.0
    cached_v = base_v * 3.0

    _out_t, next_state, next_kv_cache = module.forward_step(
        x_t,
        kv_cache=(cached_k, cached_v),
    )

    assert next_state.local_k is not None
    assert next_state.local_v is not None
    assert next_state.local_k.grad_fn is None
    assert next_state.local_v.grad_fn is None
    assert next_state.local_k.requires_grad is False
    assert next_state.local_v.requires_grad is False
    assert next_kv_cache[0] is next_state.local_k
    assert next_kv_cache[1] is next_state.local_v
