import numpy as np

from scripts.train_loop43_content_cross import (
    CONTENT_CROSS_FEATURE_NAMES,
    CONTENT_PE_FEATURE_NAMES,
    CONTENT_PE_V2_FEATURE_NAMES,
    content_cross_features_from_arrays,
)


def _vector(names, values):
    mapping = {name: index for index, name in enumerate(names)}
    result = np.zeros(len(names), dtype=np.float32)
    for name, value in values.items():
        result[mapping[name]] = value
    return result


def test_content_cross_features_have_stable_width_and_no_nan():
    pe1 = _vector(
        CONTENT_PE_FEATURE_NAMES,
        {
            "content_is_dll": 1.0,
            "content_dir_security_present": 1.0,
            "content_dir_security_log_size": 3.0,
            "content_overlay_present": 1.0,
            "content_overlay_log_size": 2.0,
            "content_overlay_entropy": 0.8,
            "content_export_count_log": 1.5,
            "content_section_high_entropy_ratio": 0.5,
            "content_section_combo_rwx_ratio": 0.25,
            "content_section_zero_raw_ratio": 0.2,
            "content_section_name_packer_hit_ratio": 0.1,
            "content_system_dll_ratio": 0.9,
            "content_import_api_count_log": 4.0,
        },
    )
    pe2 = _vector(
        CONTENT_PE_V2_FEATURE_NAMES,
        {
            "v2_api_driver_present": 1.0,
            "v2_api_driver_count_log": 2.0,
            "v2_export_pattern_service_present": 1.0,
            "v2_section_exec_write_count_log": 1.0,
            "v2_section_exec_high_entropy_ratio": 0.7,
            "v2_last_section_entropy": 0.9,
        },
    )

    features = content_cross_features_from_arrays(pe1, pe2)

    assert features.shape == (len(CONTENT_CROSS_FEATURE_NAMES),)
    assert np.isfinite(features).all()
    assert features[CONTENT_CROSS_FEATURE_NAMES.index("cross_dll_security_log_size")] == 3.0
    assert features[CONTENT_CROSS_FEATURE_NAMES.index("cross_system_dll_high_import")] == 3.6


def test_content_cross_features_do_not_fire_for_unsigned_overlay_security_cross():
    pe1 = _vector(
        CONTENT_PE_FEATURE_NAMES,
        {
            "content_dir_security_present": 0.0,
            "content_overlay_present": 1.0,
            "content_overlay_log_size": 5.0,
            "content_overlay_entropy": 0.75,
        },
    )
    pe2 = np.zeros(len(CONTENT_PE_V2_FEATURE_NAMES), dtype=np.float32)

    features = content_cross_features_from_arrays(pe1, pe2)

    assert features[CONTENT_CROSS_FEATURE_NAMES.index("cross_security_overlay_log_size")] == 0.0
    assert features[CONTENT_CROSS_FEATURE_NAMES.index("cross_unsigned_overlay_log_size")] == 5.0
