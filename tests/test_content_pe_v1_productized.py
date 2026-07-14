from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for item in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from kvd_features.content_pe_v1 import CONTENT_PE_FEATURE_NAMES, _content_pe_features_from_path
from scripts.identity_feature_guard import identity_feature_violations
from scripts.train_stage2_cache_matrix import (
    CONTENT_PE_FEATURE_NAMES as STAGE2_CONTENT_PE_FEATURE_NAMES,
    _content_pe_features_from_path as stage2_content_pe_features_from_path,
)


def test_content_pe_v1_schema_matches_stage2_alias():
    assert CONTENT_PE_FEATURE_NAMES == STAGE2_CONTENT_PE_FEATURE_NAMES
    assert len(CONTENT_PE_FEATURE_NAMES) == 100


def test_content_pe_v1_schema_has_no_identity_fields():
    assert identity_feature_violations(CONTENT_PE_FEATURE_NAMES) == []


def test_content_pe_v1_stage2_features_match(tmp_path: Path):
    payload = b"MZ" + bytes(range(64)) + b"same-content"
    sample = tmp_path / "renamed_sample.bin"
    sample.write_bytes(payload)

    productized = _content_pe_features_from_path(sample)
    stage2 = stage2_content_pe_features_from_path(sample)

    assert productized.shape == (len(CONTENT_PE_FEATURE_NAMES),)
    np.testing.assert_array_equal(productized, stage2)
