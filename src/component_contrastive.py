"""Component-aware representation losses for isolated research experiments."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset
from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentBatch:
    """Explicit metadata contract for the isolated contrastive trainer."""

    byte_seq: Tensor
    pe_features: Tensor
    stat_features: Tensor
    labels: Tensor
    component_ids: Tensor
    sample_weights: Tensor | None = None


class ComponentSubDataset(Dataset):
    """Attach immutable component IDs without changing the base dataset."""

    def __init__(self, base_dataset: Dataset, indices: Tensor, component_ids: Tensor):
        if len(indices) != len(component_ids):
            raise ValueError("indices and component_ids must have equal length")
        self.base_dataset = base_dataset
        self.indices = indices.detach().cpu().long()
        self.component_ids = component_ids.detach().cpu().long()

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, index: int):
        sample = self.base_dataset[int(self.indices[index])]
        if len(sample) == 4:
            byte_seq, pe_features, stat_features, labels = sample
            return ComponentBatch(
                byte_seq, pe_features, stat_features, labels, self.component_ids[index]
            )
        if len(sample) == 5:
            byte_seq, pe_features, stat_features, labels, sample_weight = sample
            return ComponentBatch(
                byte_seq,
                pe_features,
                stat_features,
                labels,
                self.component_ids[index],
                sample_weight,
            )
        raise ValueError("base dataset samples must contain four or five fields")


def component_supervised_contrastive_loss(
    embeddings: Tensor,
    labels: Tensor,
    component_ids: Tensor,
    *,
    temperature: float = 0.1,
) -> Tensor:
    """Pull same-label samples together while excluding same-component pairs.

    Component identity is used only to prevent near-duplicate leakage: samples
    from the same component are neither positives nor negatives.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [batch, features]")
    if labels.ndim != 1 or component_ids.ndim != 1:
        raise ValueError("labels and component_ids must be one-dimensional")
    if embeddings.shape[0] != labels.shape[0] or labels.shape != component_ids.shape:
        raise ValueError("batch dimensions must match")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    count = embeddings.shape[0]
    if count < 2:
        return embeddings.sum() * 0.0
    normalized = F.normalize(embeddings, dim=1)
    logits = normalized @ normalized.T / temperature
    same_component = component_ids[:, None] == component_ids[None, :]
    valid = ~same_component
    valid.fill_diagonal_(False)
    positive = (labels[:, None] == labels[None, :]) & valid
    if not bool(positive.any()):
        return embeddings.sum() * 0.0
    logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1)
    per_anchor = -(log_prob * positive).sum(dim=1) / positive_count.clamp_min(1)
    return per_anchor[positive_count > 0].mean()
