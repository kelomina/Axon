from scripts.audit_content_pe_productization import build_report


def test_content_pe_productization_audit_identifies_gaps():
    report = build_report(section_slots=32, pe_feature_dim=256)
    content = report["loop28_content_pe"]

    assert report["schema"] == "axon_loop49_content_pe_productization_audit_v1"
    assert content["feature_count"] == 100
    assert content["covered_or_partial_count"] > 0
    assert content["productization_gap_count"] > 0
    assert "import_shape" in content["high_value_gap_groups"]
    assert "overlay" in content["high_value_gap_groups"]
    assert report["decision"]["test10k_allowed"] is False
