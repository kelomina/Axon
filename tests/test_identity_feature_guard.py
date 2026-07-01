import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from identity_feature_guard import assert_no_identity_feature_names, identity_feature_violations  # noqa: E402
from train_stage2_cache_matrix import FeatureConfig, assert_stage2_feature_names_safe  # noqa: E402


def test_identity_guard_rejects_external_identity_features():
    violations = identity_feature_violations(
        [
            "source_path",
            "file_extension_exe",
            "sample_index",
            "split_train",
            "content_entropy",
        ]
    )

    assert violations == ["source_path", "file_extension_exe", "sample_index", "split_train"]


def test_identity_guard_allows_content_derived_path_and_sha_terms():
    assert_no_identity_feature_names(
        [
            "string_windows_path_count_log",
            "cert_oid_sha256_present",
            "content_dir_import_present",
            "content_resource_entry_count_log",
        ],
        context="unit test",
    )


def test_stage2_feature_groups_remain_identity_safe():
    feature_config = FeatureConfig(
        prefix_len=4,
        chunk_count=2,
        include_pe=True,
        include_stat=True,
        include_lightweight=True,
        include_byte_summary=True,
        include_content_pe=True,
        include_content_pe_v2=True,
        content_pe_v2_groups=("imports",),
        include_content_string=True,
        include_content_cert=True,
    )

    groups = assert_stage2_feature_names_safe(feature_config)

    assert groups["content_pe_feature_names"]
    assert groups["content_pe_v2_feature_names"]
    assert groups["content_string_feature_names"]
    assert groups["content_cert_feature_names"]
