from scripts.train_native_student import SELECTED_FEATURE_INDICES


def test_native_student_feature_selection_matches_runtime_layout() -> None:
    assert SELECTED_FEATURE_INDICES[:3] == [6, 7, 8]
    assert SELECTED_FEATURE_INDICES[48] == 54
    assert SELECTED_FEATURE_INDICES[49] == 55
    assert SELECTED_FEATURE_INDICES[304] == 310
    assert SELECTED_FEATURE_INDICES[305] == 311
    assert SELECTED_FEATURE_INDICES[560] == 566
    assert SELECTED_FEATURE_INDICES[-1] == 1419
    assert len(SELECTED_FEATURE_INDICES) == 666
