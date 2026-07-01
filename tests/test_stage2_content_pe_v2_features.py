from pathlib import Path

import numpy as np

from scripts.train_stage2_cache_matrix import CONTENT_PE_V2_FEATURE_NAMES, _content_pe_v2_features_from_path


def test_content_pe_v2_features_do_not_depend_on_filename(tmp_path: Path):
    payload = b"MZ" + bytes(range(64)) + b"same-invalid-pe-content"
    first = tmp_path / "benign-looking-name.exe"
    second = tmp_path / "random_hash_without_extension"
    first.write_bytes(payload)
    second.write_bytes(payload)

    first_features = _content_pe_v2_features_from_path(first)
    second_features = _content_pe_v2_features_from_path(second)

    assert first_features.shape == (len(CONTENT_PE_V2_FEATURE_NAMES),)
    assert second_features.shape == (len(CONTENT_PE_V2_FEATURE_NAMES),)
    np.testing.assert_array_equal(first_features, second_features)
