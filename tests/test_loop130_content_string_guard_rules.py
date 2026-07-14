from __future__ import annotations

import numpy as np

from scripts import evaluate_loop130_content_string_guard_rules as loop130


def test_r5_adds_vendor_string_guard_only_on_remaining_possible_rows():
    rows = [
        {"label": "0", "stage2_prob_malicious": "0.50", "prediction": "1"},
        {"label": "0", "stage2_prob_malicious": "0.95", "prediction": "1"},
        {"label": "1", "stage2_prob_malicious": "0.95", "prediction": "1"},
        {"label": "0", "stage2_prob_malicious": "0.95", "prediction": "1"},
    ]
    conservative_rows = [
        {"prediction": "0"},
        {"prediction": "0"},
        {"prediction": "0"},
        {"prediction": "1"},
    ]
    old_rows = [{"prediction": "1"}] * len(rows)
    feature_table = {
        "v2_resource_data_entry_count_log": np.array([2.5, 0.0, 0.0, 0.0], dtype=np.float32),
        "v2_resource_type_icon_count_log": np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "v2_resource_type_dialog_count_log": np.zeros(4, dtype=np.float32),
        "content_dir_resource_size_ratio": np.array([0.01, 0.0, 0.0, 0.0], dtype=np.float32),
        "content_resource_entry_count_log": np.zeros(4, dtype=np.float32),
        "string_benign_vendor_count_log": np.array([0.0, 3.1, 3.1, 3.1], dtype=np.float32),
        "string_version_resource_count_log": np.zeros(4, dtype=np.float32),
    }

    results, masks, predictions_by_rule, *_ = loop130.evaluate_rules(
        rows,
        conservative_rows,
        old_rows,
        feature_table,
        ["R4_resource_icon_lowconf_resource_ratio_floor", "R5_r4_plus_vendor_strings"],
    )

    assert masks["R4_resource_icon_lowconf_resource_ratio_floor"].tolist() == [True, False, False, False]
    assert masks["R5_r4_plus_vendor_strings"].tolist() == [True, True, True, False]
    assert predictions_by_rule["R5_r4_plus_vendor_strings"].tolist() == [0, 0, 0, 1]
    assert results["R5_r4_plus_vendor_strings"]["extra_flips_over_r4"] == 2
    assert results["R5_r4_plus_vendor_strings"]["flipped_label0"] == 2
    assert results["R5_r4_plus_vendor_strings"]["flipped_label1"] == 1


def test_r8_restores_dialog_rich_r5_flips_to_primary_positive():
    rows = [
        {"label": "0", "stage2_prob_malicious": "0.50", "prediction": "1"},
        {"label": "1", "stage2_prob_malicious": "0.95", "prediction": "1"},
        {"label": "0", "stage2_prob_malicious": "0.95", "prediction": "1"},
    ]
    conservative_rows = [{"prediction": "0"}] * len(rows)
    old_rows = [{"prediction": "1"}] * len(rows)
    feature_table = {
        "v2_resource_data_entry_count_log": np.array([2.5, 0.0, 0.0], dtype=np.float32),
        "v2_resource_type_icon_count_log": np.array([2.0, 0.0, 0.0], dtype=np.float32),
        "v2_resource_type_dialog_count_log": np.array([0.0, 3.1, 0.0], dtype=np.float32),
        "content_dir_resource_size_ratio": np.array([0.01, 0.0, 0.0], dtype=np.float32),
        "content_resource_entry_count_log": np.zeros(3, dtype=np.float32),
        "string_benign_vendor_count_log": np.array([0.0, 3.1, 3.1], dtype=np.float32),
        "string_version_resource_count_log": np.zeros(3, dtype=np.float32),
    }

    _results, masks, predictions_by_rule, *_ = loop130.evaluate_rules(
        rows,
        conservative_rows,
        old_rows,
        feature_table,
        ["R5_r4_plus_vendor_strings", "R8_r5_dialog_protector"],
    )

    assert masks["R5_r4_plus_vendor_strings"].tolist() == [True, True, True]
    assert masks["R8_r5_dialog_protector"].tolist() == [True, False, True]
    assert predictions_by_rule["R8_r5_dialog_protector"].tolist() == [0, 1, 0]


def test_unknown_rule_is_rejected():
    rows = [{"label": "0", "stage2_prob_malicious": "0.5", "prediction": "1"}]
    feature_table = {
        "v2_resource_data_entry_count_log": np.array([0.0], dtype=np.float32),
        "v2_resource_type_icon_count_log": np.array([0.0], dtype=np.float32),
        "v2_resource_type_dialog_count_log": np.array([0.0], dtype=np.float32),
        "content_dir_resource_size_ratio": np.array([0.0], dtype=np.float32),
        "content_resource_entry_count_log": np.array([0.0], dtype=np.float32),
        "string_benign_vendor_count_log": np.array([0.0], dtype=np.float32),
        "string_version_resource_count_log": np.array([0.0], dtype=np.float32),
    }

    try:
        loop130.evaluate_rules(rows, [{"prediction": "0"}], [None], feature_table, ["bad_rule"])
    except ValueError as exc:
        assert "Unknown rule" in str(exc)
    else:
        raise AssertionError("unknown rule should fail")
