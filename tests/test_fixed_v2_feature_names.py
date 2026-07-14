from kvd_features.schema_names import fixed_v2_feature_names


def test_fixed_v2_feature_names_match_used_dim():
    names = fixed_v2_feature_names(section_slots=32)

    assert len(names) == 143
    assert names[:3] == [
        "fixed_v2_file_size",
        "fixed_v2_log_size",
        "fixed_v2_size_of_optional_header",
    ]
    assert names[18:21] == [
        "fixed_v2_section_00_is_executable",
        "fixed_v2_section_00_is_writable",
        "fixed_v2_section_00_is_readable",
    ]
    assert names[-1] == "fixed_v2_packer_keyword_hits_ratio"


def test_fixed_v2_feature_names_include_reserved_padding():
    names = fixed_v2_feature_names(section_slots=32, pe_feature_dim=256)

    assert len(names) == 256
    assert names[143] == "fixed_v2_reserved_143"
    assert names[-1] == "fixed_v2_reserved_255"


def test_fixed_v2_feature_names_reject_too_small_dim():
    try:
        fixed_v2_feature_names(section_slots=32, pe_feature_dim=142)
    except ValueError as exc:
        assert "must be at least 143" in str(exc)
    else:
        raise AssertionError("expected ValueError")
