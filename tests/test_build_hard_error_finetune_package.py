import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_hard_error_finetune_package import (  # noqa: E402
    SplitRatios,
    build_commands,
    build_hard_error_rows,
    load_error_rows,
    write_readme,
)


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = ["source_path", "label", "prediction", "prob_malicious", "sample_index", "source_group_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_load_error_rows_can_focus_only_false_negatives():
    with _case_dir("hard_error_package_focus") as tmp_path:
        fp_csv = tmp_path / "false_positives.csv"
        fn_csv = tmp_path / "false_negatives.csv"
        _write_rows(fp_csv, [{
            "source_path": "data/benign-a",
            "label": 0,
            "prediction": 1,
            "prob_malicious": 0.88,
            "sample_index": 1,
            "source_group_id": "fp-group",
        }])
        _write_rows(fn_csv, [{
            "source_path": "data/malicious-a.exe",
            "label": 1,
            "prediction": 0,
            "prob_malicious": 0.42,
            "sample_index": 2,
            "source_group_id": "fn-group",
        }])

        rows = load_error_rows(fp_csv, fn_csv, focus="fn")

    assert len(rows) == 1
    assert rows[0]["error_type"] == "FN"
    assert rows[0]["source_path"] == "data/malicious-a.exe"


def test_build_hard_error_rows_can_limit_hard_examples_to_val_split():
    source_rows = [
        {
            "source_path": "data/benign-val",
            "label": "0",
            "sample_index": "1",
            "group_id": "g-val-fp",
            "source_group_id": "g-val-fp",
            "group_size": "1",
            "split": "val",
        },
        {
            "source_path": "data/malicious-test",
            "label": "1",
            "sample_index": "2",
            "group_id": "g-test-fn",
            "source_group_id": "g-test-fn",
            "group_size": "1",
            "split": "test",
        },
    ]
    error_rows = [
        {
            "source_path": "data/benign-val",
            "label": "0",
            "prediction": "1",
            "prob_malicious": "0.80",
            "sample_index": "1",
            "source_group_id": "g-val-fp",
            "error_type": "FP",
        },
        {
            "source_path": "data/malicious-test",
            "label": "1",
            "prediction": "0",
            "prob_malicious": "0.20",
            "sample_index": "2",
            "source_group_id": "g-test-fn",
            "error_type": "FN",
        },
    ]

    rows, counts = build_hard_error_rows(
        source_rows,
        error_rows,
        ratios=SplitRatios(train=0.6, val=0.2),
        seed=123,
        fp_weight=4.0,
        fn_weight=5.0,
        eligible_split="val",
    )

    by_path = {row["source_path"]: row for row in rows}
    assert by_path["data/benign-val"]["split"] == "train"
    assert by_path["data/benign-val"]["hard_family_role"] == "hard_error_fp_train"
    assert by_path["data/malicious-test"]["split"] == "test"
    assert by_path["data/malicious-test"]["hard_family_role"] == "base_test"
    assert counts["eligible_split"] == "val"


def test_strict_source_group_isolation_moves_context_rows_with_hard_errors():
    source_rows = [
        {
            "source_path": "data/group-a-train-context",
            "label": "1",
            "sample_index": "1",
            "group_id": "group-a_train",
            "source_group_id": "group-a",
            "group_size": "2",
            "split": "train",
        },
        {
            "source_path": "data/group-a-test-hard",
            "label": "1",
            "sample_index": "2",
            "group_id": "group-a_test",
            "source_group_id": "group-a",
            "group_size": "2",
            "split": "test",
        },
    ]
    error_rows = [
        {
            "source_path": "data/group-a-test-hard",
            "label": "1",
            "prediction": "0",
            "prob_malicious": "0.20",
            "sample_index": "2",
            "source_group_id": "group-a",
            "error_type": "FN",
        },
    ]

    rows, counts = build_hard_error_rows(
        source_rows,
        error_rows,
        ratios=SplitRatios(train=0.6, val=0.2),
        seed=123,
        fp_weight=4.0,
        fn_weight=5.0,
        eligible_split="test",
        strict_source_group_isolation=True,
    )

    assert counts["strict_source_group_isolation"] is True
    assert {row["split"] for row in rows} == {"train"}
    by_path = {row["source_path"]: row for row in rows}
    assert by_path["data/group-a-train-context"]["hard_family_role"] == "hard_error_context_train"
    assert by_path["data/group-a-test-hard"]["hard_family_role"] == "hard_error_fn_train"
    assert by_path["data/group-a-test-hard"]["sample_weight"] == "5"


def test_build_commands_and_readme_use_requested_decision_threshold():
    with _case_dir("hard_error_package_threshold") as tmp_path:
        class Args:
            checkpoint = Path("models/base.pt")
            model_output_dir = Path("models/replay063")
            data_dir = Path("data")
            config = Path("config/default_config.toml")
            samples_per_class = 20000
            epochs = 4
            learning_rate = 1e-5
            batch_size = 32
            device = "cpu"
            singleton_group_weight = 1.8
            rare_group_weight = 1.5
            medium_group_weight = 1.2
            extract_workers = 1
            extract_backend = "thread"
            decision_threshold = 0.63

        paths = {
            "output_dir": tmp_path,
            "split_csv": tmp_path / "split.csv",
            "hard_holdout_csv": tmp_path / "holdout.csv",
        }
        commands = build_commands(Args, paths)

        assert "scripts\\authorized_main.py" in commands["fine_tune"]
        assert "scripts\\main.py train" not in commands["fine_tune"]
        assert "scripts\\authorized_main.py" in commands["full_threshold_sweep"]
        assert "scripts\\main.py eval" not in commands["full_threshold_sweep"]
        assert '--sweep-thresholds "0.45,0.5,0.55,0.6,0.63,0.65,0.7"' in commands["full_threshold_sweep"]
        assert "--decision-threshold 0.63" in commands["holdout_predictions"]

        readme = tmp_path / "README.md"
        write_readme(readme, {
            "decision_threshold": 0.63,
            "outputs": {
                "split_csv": str(paths["split_csv"]),
                "hard_train_csv": str(tmp_path / "train.csv"),
                "hard_val_csv": str(tmp_path / "val.csv"),
                "hard_holdout_csv": str(paths["hard_holdout_csv"]),
            },
            "commands": commands,
        })

        text = readme.read_text(encoding="utf-8")
        assert "threshold 0.63" in text
        assert "threshold 0.55" not in text
