from scripts.train_loop51_region_view_neural import balanced_limit


def test_balanced_limit_keeps_both_labels():
    labels = [0, 0, 0, 1, 1, 1]
    selected = balanced_limit(list(range(6)), labels, 4)

    assert len(selected) == 4
    assert {labels[index] for index in selected} == {0, 1}
