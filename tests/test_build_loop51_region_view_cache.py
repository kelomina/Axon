import numpy as np

from scripts.build_loop51_region_view_cache import build_region_view_sequence, region_slot_sizes
from train_loop44_region_byte_ngram import REGION_NAMES


def test_region_slot_sizes_cover_full_length():
    slots = region_slot_sizes(8192, REGION_NAMES)

    assert sum(slots.values()) == 8192
    assert set(slots) == set(REGION_NAMES)


def test_region_view_sequence_uses_content_not_region_names():
    first, copied = build_region_view_sequence(
        [("head", b"abc"), ("tail", b"xyz")],
        byte_length=22,
        region_names=["head", "tail"],
    )
    second, second_copied = build_region_view_sequence(
        [("head", b"abc"), ("tail", b"xyz")],
        byte_length=22,
        region_names=["head", "tail"],
    )

    assert copied == {"head": 3, "tail": 3}
    assert second_copied == copied
    assert np.array_equal(first, second)
    assert bytes(first[:3]) == b"abc"
    assert bytes(first[11:14]) == b"xyz"
