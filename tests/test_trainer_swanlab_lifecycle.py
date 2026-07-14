from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trainer import AxonTrainer  # noqa: E402


def test_train_finishes_swanlab_when_training_impl_raises(monkeypatch):
    finish_calls = []
    monkeypatch.setitem(
        sys.modules,
        "swanlab",
        SimpleNamespace(finish=lambda: finish_calls.append("finish")),
    )
    trainer = AxonTrainer.__new__(AxonTrainer)
    trainer.swanlab_run = object()

    def raise_from_impl(_train_loader, _val_loader=None, _test_loader=None, _fast_mode=False):
        raise RuntimeError("training interrupted")

    trainer._train_impl = raise_from_impl

    with pytest.raises(RuntimeError, match="training interrupted"):
        AxonTrainer.train(trainer, object())

    assert finish_calls == ["finish"]
    assert trainer.swanlab_run is None


def test_finish_swanlab_is_idempotent(monkeypatch):
    finish_calls = []
    monkeypatch.setitem(
        sys.modules,
        "swanlab",
        SimpleNamespace(finish=lambda: finish_calls.append("finish")),
    )
    trainer = AxonTrainer.__new__(AxonTrainer)
    trainer.swanlab_run = object()

    trainer._finish_swanlab()
    trainer._finish_swanlab()

    assert finish_calls == ["finish"]
    assert trainer.swanlab_run is None
