import pytest
import torch

from src.component_contrastive import ComponentBatch, ComponentSubDataset, component_supervised_contrastive_loss


def test_component_loss_excludes_same_component_pairs() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1])
    components = torch.tensor([10, 10, 20])
    loss = component_supervised_contrastive_loss(embeddings, labels, components)
    assert loss.item() == pytest.approx(0.0)


def test_component_loss_is_finite_with_no_valid_pairs() -> None:
    embeddings = torch.eye(2)
    labels = torch.tensor([0, 1])
    components = torch.tensor([1, 2])
    loss = component_supervised_contrastive_loss(embeddings, labels, components)
    assert loss.item() == pytest.approx(0.0)


def test_component_loss_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError):
        component_supervised_contrastive_loss(
            torch.eye(2), torch.tensor([0, 1]), torch.tensor([1, 2]), temperature=0
        )


def test_component_subdataset_preserves_sample_and_attaches_id() -> None:
    base = [
        (torch.tensor([1]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])),
        (torch.tensor([2]), torch.tensor([1]), torch.tensor([1]), torch.tensor([1])),
    ]
    dataset = ComponentSubDataset(base, torch.tensor([1, 0]), torch.tensor([42, 7]))
    sample = dataset[0]
    assert isinstance(sample, ComponentBatch)
    assert sample.byte_seq.item() == 2
    assert sample.labels.item() == 1
    assert sample.component_ids.item() == 42
