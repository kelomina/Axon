from __future__ import annotations

import numpy as np

from scripts import evaluate_loop132_fn_recovery_rules as loop132


def test_r11_unions_file_version_and_virtual_raw_recovery_masks():
    base_pred = np.array([0, 0, 0, 1], dtype=np.int64)
    features = {
        "primary_prob_malicious": np.array([0.25, 0.25, 0.19, 0.9], dtype=np.float32),
        "v2_api_file_mutation_ratio": np.array([0.05, 0.0, 0.05, 0.05], dtype=np.float32),
        "v2_import_dll_version_api_ratio": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "v2_section_max_virtual_raw_ratio_log": np.array([0.0, 3.6, 3.6, 3.6], dtype=np.float32),
    }

    assert loop132.rule_mask("R9_file_or_version_api_recovery", base_pred, features).tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert loop132.rule_mask("R10_virtual_raw_ratio_recovery", base_pred, features).tolist() == [
        False,
        True,
        False,
        False,
    ]
    assert loop132.rule_mask("R11_union_file_version_or_virtual_raw", base_pred, features).tolist() == [
        True,
        True,
        False,
        False,
    ]


def test_loop132_evaluate_rules_only_flips_base_zero_to_one():
    rows = [
        {"label": "1", "prediction": "0"},
        {"label": "0", "prediction": "0"},
        {"label": "1", "prediction": "1"},
    ]
    features = {
        "primary_prob_malicious": np.array([0.25, 0.25, 0.25], dtype=np.float32),
        "v2_api_file_mutation_ratio": np.array([0.05, 0.05, 0.05], dtype=np.float32),
        "v2_import_dll_version_api_ratio": np.zeros(3, dtype=np.float32),
        "v2_section_max_virtual_raw_ratio_log": np.zeros(3, dtype=np.float32),
    }

    results, masks, predictions_by_rule, *_ = loop132.evaluate_rules(
        rows,
        features,
        ["R9_file_or_version_api_recovery"],
    )

    assert masks["R9_file_or_version_api_recovery"].tolist() == [True, True, False]
    assert predictions_by_rule["R9_file_or_version_api_recovery"].tolist() == [1, 1, 1]
    assert results["R9_file_or_version_api_recovery"]["flipped_label1"] == 1
    assert results["R9_file_or_version_api_recovery"]["flipped_label0"] == 1


def test_loop132_unknown_rule_is_rejected():
    features = {
        "primary_prob_malicious": np.array([0.0], dtype=np.float32),
        "v2_api_file_mutation_ratio": np.array([0.0], dtype=np.float32),
        "v2_import_dll_version_api_ratio": np.array([0.0], dtype=np.float32),
        "v2_section_max_virtual_raw_ratio_log": np.array([0.0], dtype=np.float32),
    }

    try:
        loop132.rule_mask("bad_rule", np.array([0], dtype=np.int64), features)
    except ValueError as exc:
        assert "Unknown rule" in str(exc)
    else:
        raise AssertionError("unknown rule should fail")
