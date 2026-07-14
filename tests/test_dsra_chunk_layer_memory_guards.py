import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DSRAArchitectureConfig  # noqa: E402
from dsra.dsra_layer import DSRA_Chunk_Layer  # noqa: E402
from dsra.mhdsra2.paged_exact_memory import PagedExactMemory  # noqa: E402


def test_dsra_chunk_reset_memory_preserves_paged_memory_type_and_restarts_positions():
    layer = DSRA_Chunk_Layer(
        dim=8,
        K=4,
        kr=2,
        dsra_arch_config=DSRAArchitectureConfig(
            dim=8,
            heads=2,
            paged_memory_page_size=2,
            paged_memory_max_pages=2,
        ),
    )
    key = torch.randn(1, 2, 4, 4)
    value = torch.randn(1, 2, 4, 4)
    layer.memory_repository.append(key, value)
    original_memory = layer.memory_repository.memory

    layer.reset_memory()
    layer.memory_repository.append(key[:, :, :2, :], value[:, :, :2, :])

    assert layer.memory_repository.memory is original_memory
    assert isinstance(layer.memory_repository.memory, PagedExactMemory)
    assert len(layer.memory_repository.memory) == 1
    assert layer.memory_repository.memory.next_position == 2
    assert layer.memory_repository.memory.pages[0].start == 0


def test_dsra_chunk_paged_memory_honors_configured_max_pages():
    layer = DSRA_Chunk_Layer(
        dim=8,
        K=4,
        kr=2,
        dsra_arch_config=DSRAArchitectureConfig(
            dim=8,
            heads=2,
            paged_memory_page_size=2,
            paged_memory_max_pages=2,
        ),
    )
    key = torch.randn(1, 2, 8, 4)
    value = torch.randn(1, 2, 8, 4)

    layer.memory_repository.append(key, value)

    assert len(layer.memory_repository.memory) == 2
    assert [page.start for page in layer.memory_repository.memory.pages] == [4, 6]
    assert [page.end for page in layer.memory_repository.memory.pages] == [6, 8]
