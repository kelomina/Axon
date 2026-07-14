"""Security helpers for loading Axon model artifacts."""

from pathlib import Path
from typing import Iterable, Mapping, Optional

import torch


CHECKPOINT_SUFFIXES = {".pt", ".pth"}
BASE_CHECKPOINT_KEYS = {"model_state_dict", "config"}
BASE_CONFIG_KEYS = {
    "max_byte_length",
    "pe_feature_dim",
    "stat_feature_dim",
    "dsra_dim",
    "dsra_heads",
    "num_classes",
}


def _ensure_mapping(value, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_safe_checkpoint(
    checkpoint_path: Path,
    map_location="cpu",
    required_keys: Optional[Iterable[str]] = None,
) -> Mapping:
    """Load a trusted Axon checkpoint with PyTorch's restricted loader.

    The project only stores tensors and plain dictionaries in checkpoints. If a
    checkpoint needs Python object unpickling, it is not safe for this loader.
    """
    path = Path(checkpoint_path)
    if path.suffix.lower() not in CHECKPOINT_SUFFIXES:
        raise ValueError(f"Unsupported checkpoint suffix: {path.suffix}")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This PyTorch version does not support weights_only checkpoint loading; "
            "upgrade torch before loading Axon checkpoints."
        ) from exc

    checkpoint = _ensure_mapping(checkpoint, "checkpoint")
    required = set(BASE_CHECKPOINT_KEYS)
    if required_keys is not None:
        required.update(required_keys)
    missing = sorted(key for key in required if key not in checkpoint)
    if missing:
        raise ValueError(f"Checkpoint missing required keys: {missing}")

    config = _ensure_mapping(checkpoint["config"], "checkpoint['config']")
    missing_config = sorted(key for key in BASE_CONFIG_KEYS if key not in config)
    if missing_config:
        raise ValueError(f"Checkpoint config missing required keys: {missing_config}")

    return checkpoint
