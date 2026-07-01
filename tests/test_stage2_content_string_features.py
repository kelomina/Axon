from pathlib import Path

import numpy as np

from scripts.train_stage2_cache_matrix import CONTENT_STRING_FEATURE_NAMES, _content_string_features_from_path


def test_content_string_features_do_not_depend_on_filename(tmp_path: Path):
    payload = (
        b"MZ"
        + b"http://example.invalid\x00"
        + b"VirtualAlloc\x00WriteProcessMemory\x00"
        + b"CompanyName\x00Microsoft Corporation\x00"
        + bytes(range(128))
    )
    first = tmp_path / "installer.exe"
    second = tmp_path / "sha256_like_name_without_extension"
    first.write_bytes(payload)
    second.write_bytes(payload)

    first_features = _content_string_features_from_path(first)
    second_features = _content_string_features_from_path(second)

    assert first_features.shape == (len(CONTENT_STRING_FEATURE_NAMES),)
    assert second_features.shape == (len(CONTENT_STRING_FEATURE_NAMES),)
    assert np.count_nonzero(first_features) > 0
    np.testing.assert_array_equal(first_features, second_features)
