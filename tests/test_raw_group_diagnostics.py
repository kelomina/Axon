import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_raw_groups import main as analyze_groups_main  # noqa: E402
from evaluate_groups import evaluate_groups  # noqa: E402


def _write_config(path, data_dir):
    path.write_text(
        f"""
[experiment]
name = "group_diag_test"
seed = 42

[model]
pe_feature_dim = 256
pe_schema_version = "fixed_v2"

[data]
data_dir = "{data_dir.as_posix()}"
val_ratio = 0.2
test_ratio = 0.2
benign_dir_names_fs = ["benign"]
malicious_dir_names_fs = ["malicious"]
""",
        encoding="utf-8",
    )


def _write_file(path, payload=b"MZtest"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_group_distribution_adds_singletons_and_split_keeps_groups_together(tmp_path):
    data_dir = tmp_path / "data"
    config_path = tmp_path / "config.toml"
    raw_report_dir = tmp_path / "raw_report"
    output_dir = tmp_path / "out"
    raw_report_dir.mkdir()
    _write_config(config_path, data_dir)

    paths = [
        data_dir / "benign" / "a.exe",
        data_dir / "benign" / "b.exe",
        data_dir / "benign" / "c.exe",
        data_dir / "malicious" / "d.exe",
    ]
    for path in paths:
        _write_file(path)

    with (raw_report_dir / "raw_similarity_groups.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_id",
                "size",
                "labels",
                "splits",
                "has_label_conflict",
                "has_cross_split",
                "has_leakage",
                "sample_indices",
                "source_paths",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "group_id": 1,
            "size": 2,
            "labels": "0:2",
            "splits": "train:1|test:1",
            "has_label_conflict": "False",
            "has_cross_split": "True",
            "has_leakage": "True",
            "sample_indices": "0|1",
            "source_paths": f"{paths[0]}|{paths[1]}",
        })

    rc = analyze_groups_main([
        "--config",
        str(config_path),
        "--data-dir",
        str(data_dir),
        "--raw-report-dir",
        str(raw_report_dir),
        "--output-dir",
        str(output_dir),
    ])
    assert rc == 0

    groups = _read_csv(output_dir / "group_distribution.csv")
    assert len(groups) == 3
    assert sum(1 for row in groups if row["source"] == "singleton") == 2
    members = _read_csv(output_dir / "group_members.csv")
    assert len(members) == 4

    split_rows = _read_csv(output_dir / "group_isolated_split.csv")
    by_group = {}
    for row in split_rows:
        by_group.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_group.values())

    summary = json.loads((output_dir / "group_distribution_summary.json").read_text(encoding="utf-8"))
    assert summary["distribution"]["total_samples"] == 4
    assert summary["distribution"]["rare_groups"] == 3


def test_group_evaluation_reports_rare_and_worst_groups(tmp_path):
    groups_path = tmp_path / "groups.csv"
    predictions_path = tmp_path / "predictions.csv"
    output_dir = tmp_path / "eval"

    with groups_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_id",
                "source",
                "size",
                "labels",
                "splits",
                "is_rare_group",
                "is_singleton",
                "has_leakage",
                "train_count",
                "source_paths",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "group_id": 1,
            "source": "similarity_group",
            "size": 2,
            "labels": "1:2",
            "splits": "train:1|test:1",
            "is_rare_group": "True",
            "is_singleton": "False",
            "has_leakage": "True",
            "train_count": 1,
            "source_paths": "a.exe|b.exe",
        })
        writer.writerow({
            "group_id": 2,
            "source": "singleton",
            "size": 1,
            "labels": "0:1",
            "splits": "test:1",
            "is_rare_group": "True",
            "is_singleton": "True",
            "has_leakage": "False",
            "train_count": 0,
            "source_paths": "c.exe",
        })

    with predictions_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_path",
                "cache_path",
                "sample_index",
                "label",
                "split",
                "prob_malicious",
                "prediction",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerow({"source_path": "a.exe", "label": 1, "prediction": 1, "prob_malicious": 0.9, "correct": "True"})
        writer.writerow({"source_path": "b.exe", "label": 1, "prediction": 0, "prob_malicious": 0.2, "correct": "False"})
        writer.writerow({"source_path": "c.exe", "label": 0, "prediction": 1, "prob_malicious": 0.7, "correct": "False"})

    summary = evaluate_groups(groups_path, predictions_path, output_dir)
    assert summary["overall"]["predicted_samples"] == 3
    assert summary["overall"]["error_count"] == 2
    assert summary["rare_groups"]["groups"] == 2
    assert summary["singleton_groups"]["error_count"] == 1
    assert summary["worst_groups"][0]["error_count"] >= 1

    rows = _read_csv(output_dir / "group_evaluation.csv")
    assert len(rows) == 2
    assert rows[0]["accuracy"] != ""
