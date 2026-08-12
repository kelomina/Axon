from kvd_features.schema_names import fixed_v2_feature_names, fixed_v3_feature_names


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


def test_fixed_v3_drops_only_the_dead_signature_column():
    v2 = fixed_v2_feature_names(section_slots=32)
    v3 = fixed_v3_feature_names(section_slots=32)

    assert len(v2) == 143
    assert len(v3) == 142
    assert v2[16] == "fixed_v2_has_signature"
    assert not any("has_signature" in name for name in v3)

    # every other column keeps its order, shifted down by one past index 16
    expected = [
        name.replace("fixed_v2_", "fixed_v3_", 1)
        for index, name in enumerate(v2)
        if index != 16
    ]
    assert v3 == expected


def test_fixed_v3_feature_names_include_reserved_padding():
    names = fixed_v3_feature_names(section_slots=32, pe_feature_dim=256)

    assert len(names) == 256
    assert names[142] == "fixed_v3_reserved_142"
    assert names[-1] == "fixed_v3_reserved_255"


def test_fixed_v3_feature_names_reject_too_small_dim():
    try:
        fixed_v3_feature_names(section_slots=32, pe_feature_dim=141)
    except ValueError as exc:
        assert "must be at least 142" in str(exc)
    else:
        raise AssertionError("expected ValueError")
