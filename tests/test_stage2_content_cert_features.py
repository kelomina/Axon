from pathlib import Path

import numpy as np

from scripts.train_stage2_cache_matrix import CONTENT_CERT_FEATURE_NAMES, _content_cert_features_from_path


def test_content_cert_features_do_not_depend_on_filename(tmp_path: Path):
    payload = b"MZ" + bytes(range(128)) + b"same unsigned content"
    first = tmp_path / "signed-looking-name.exe"
    second = tmp_path / "sha256_like_name_without_extension"
    first.write_bytes(payload)
    second.write_bytes(payload)

    first_features = _content_cert_features_from_path(first)
    second_features = _content_cert_features_from_path(second)

    assert first_features.shape == (len(CONTENT_CERT_FEATURE_NAMES),)
    assert second_features.shape == (len(CONTENT_CERT_FEATURE_NAMES),)
    np.testing.assert_array_equal(first_features, second_features)
