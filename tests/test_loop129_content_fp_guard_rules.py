import numpy as np

from scripts.evaluate_loop129_content_fp_guard_rules import (
    RULE_FEATURE_NAMES,
    evaluate_rules,
    metrics,
)
from scripts.identity_feature_guard import identity_feature_violations


def test_rule_feature_names_are_identity_safe():
    assert identity_feature_violations(RULE_FEATURE_NAMES) == []


def test_r2_and_r3_only_flip_primary_positive_rows():
    rows = [
        {"label": "0", "prediction": "1", "stage2_prob_malicious": "0.60"},
        {"label": "1", "prediction": "1", "stage2_prob_malicious": "0.60"},
        {"label": "0", "prediction": "0", "stage2_prob_malicious": "0.10"},
    ]
    conservative_rows = [
        {"prediction": "0"},
        {"prediction": "1"},
        {"prediction": "0"},
    ]
    old_rows = [None, None, None]
    feature_table = {
        "v2_resource_data_entry_count_log": np.asarray([2.1, 2.1, 2.1], dtype=np.float32),
        "v2_resource_type_icon_count_log": np.asarray([1.6, 1.6, 1.6], dtype=np.float32),
        "content_is_dll": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    }

    results, masks, predictions_by_rule, labels, primary_pred, _ = evaluate_rules(
        rows=rows,
        conservative_rows=conservative_rows,
        old_rows=old_rows,
        feature_table=feature_table,
        rule_names=["R2_resource_icon_lowconf", "R3_resource_icon_lowconf_not_dll"],
    )

    assert primary_pred.tolist() == [1, 1, 0]
    assert masks["R2_resource_icon_lowconf"].tolist() == [True, False, False]
    assert predictions_by_rule["R2_resource_icon_lowconf"].tolist() == [0, 1, 0]
    assert results["R2_resource_icon_lowconf"]["flipped_label0"] == 1
    assert results["R2_resource_icon_lowconf"]["flipped_label1"] == 0


def test_metrics_counts_errors():
    labels = np.asarray([0, 1, 1], dtype=np.int64)
    predictions = np.asarray([1, 1, 0], dtype=np.int64)

    result = metrics(labels, predictions)

    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["errors"] == 2
