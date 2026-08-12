"""Loop219: System Integrity & Contract Test Suite.

Automated pytest suite for Axon expert networks, guards, and calibrators.
"""

from __future__ import annotations

from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loop198_trusted_signer_guard import Loop198TrustedSignerGuard
from src.loop202_whole_file_streamer import Loop202WholeFileStreamer
from src.loop206_signer_fingerprint_guard import Loop206SignerFingerprintGuard
from src.loop216_graph_expert import Loop216GraphExpert
from src.loop217_spline_annealer import Loop217SplineAnnealer


def test_loop198_signer_guard():
    guard = Loop198TrustedSignerGuard()
    pred, is_down = guard.evaluate_sample(1, "Valid", "Microsoft Corporation")
    assert pred == 0
    assert is_down is True


def test_loop202_whole_file_streamer():
    model = Loop202WholeFileStreamer(chunk_dim=192)
    x = torch.randn(2, 4, 192)
    out = model(x)
    assert out.shape == (2, 2)


def test_loop206_fingerprint_guard():
    guard = Loop206SignerFingerprintGuard()
    pred, is_down = guard.evaluate_sample(1, "Valid", "33000002ed1b794f728c312d8a0000000002ed")
    assert pred == 0
    assert is_down is True


def test_loop216_graph_expert():
    model = Loop216GraphExpert(node_dim=128)
    nodes = torch.randn(2, 5, 128)
    out = model(nodes)
    assert out.shape == (2, 2)


def test_loop217_spline_annealer():
    annealer = Loop217SplineAnnealer(low=0.29, high=0.33)
    p_annealed = annealer.anneal(0.31)
    assert 0.29 <= p_annealed <= 0.33
