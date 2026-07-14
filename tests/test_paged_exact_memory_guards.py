import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dsra.mhdsra2.paged_exact_memory import PagedExactMemory  # noqa: E402


def test_invalidate_before_physically_removes_obsolete_pages():
    memory = PagedExactMemory(page_size=2)
    key = torch.randn(1, 2, 8, 4)
    value = torch.randn(1, 2, 8, 4)

    memory.append(key, value)
    assert len(memory) == 4

    memory.invalidate_before(5)

    assert len(memory) == 2
    assert [page.start for page in memory.pages] == [4, 6]
    assert [page.end for page in memory.pages] == [6, 8]


def test_paged_exact_memory_enforces_max_pages_capacity():
    memory = PagedExactMemory(page_size=2, max_pages=3)
    key = torch.randn(1, 2, 10, 4)
    value = torch.randn(1, 2, 10, 4)

    memory.append(key, value)

    assert len(memory) == 3
    assert [page.start for page in memory.pages] == [4, 6, 8]
    assert [page.end for page in memory.pages] == [6, 8, 10]


def test_paged_exact_memory_clear_can_reset_position_counter():
    memory = PagedExactMemory(page_size=2)
    key = torch.randn(1, 2, 4, 4)
    value = torch.randn(1, 2, 4, 4)

    memory.append(key, value)
    assert memory.next_position == 4

    memory.clear(reset_position=True)
    memory.append(key[:, :, :2, :], value[:, :, :2, :])

    assert len(memory) == 1
    assert memory.next_position == 2
    assert memory.pages[0].start == 0
    assert memory.pages[0].end == 2
