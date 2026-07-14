import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_family_classifier import build_family_classifier, source_path_keys  # noqa: E402


def _write_group_members(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_id",
                "group_source",
                "group_size",
                "is_rare_group",
                "is_singleton",
                "sample_index",
                "source_path",
                "label",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "group_id": 7,
                "group_source": "similarity_group",
                "group_size": 2,
                "is_rare_group": "True",
                "is_singleton": "False",
                "sample_index": 0,
                "source_path": "mal_a.exe",
                "label": 1,
                "split": "train",
            }
        )
        writer.writerow(
            {
                "group_id": 7,
                "group_source": "similarity_group",
                "group_size": 2,
                "is_rare_group": "True",
                "is_singleton": "False",
                "sample_index": 1,
                "source_path": "mal_b.exe",
                "label": 1,
                "split": "train",
            }
        )
        writer.writerow(
            {
                "group_id": 8,
                "group_source": "singleton",
                "group_size": 1,
                "is_rare_group": "True",
                "is_singleton": "True",
                "sample_index": 2,
                "source_path": "benign.exe",
                "label": 0,
                "split": "train",
            }
        )


def _map_feature(path, feature):
    return {
        key: {
            "source_path": path,
            "cache_path": f"{path}.npz",
            "label": 1,
            "feature": np.asarray(feature, dtype=np.float32),
        }
        for key in source_path_keys(path)
    }


def test_family_classifier_export_builds_scaled_centroid_and_ignores_benign(tmp_path):
    members_path = tmp_path / "group_members.csv"
    _write_group_members(members_path)
    cache_by_source = {}
    cache_by_source.update(_map_feature("mal_a.exe", [1.0, 2.0, 3.0, 4.0]))
    cache_by_source.update(_map_feature("mal_b.exe", [1.5, 2.5, 3.5, 4.5]))
    cache_by_source.update(_map_feature("benign.exe", [9.0, 9.0, 9.0, 9.0]))

    payload, summary = build_family_classifier(
        members_path,
        cache_by_source,
        pe_feature_dim=2,
        stat_feature_dim=2,
        min_family_size=2,
        threshold_scale=1.25,
        threshold_margin=0.05,
        min_threshold=0.25,
        max_threshold=None,
        family_name_prefix="axon_group_",
        include_singletons=False,
    )

    assert payload["schema"] == "axon_family_classifier_v1"
    assert payload["feature_dim"] == 4
    assert payload["cluster_ids"] == [7]
    assert payload["family_names"] == ["axon_group_7"]
    assert len(payload["centroids"]) == 1
    assert len(payload["centroids"][0]) == 4
    assert payload["thresholds"][0] >= 0.25
    assert summary["eligible_family_count"] == 1
    assert summary["skipped_non_malicious_rows"] == 1
