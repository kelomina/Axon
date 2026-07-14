import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_training_trick_summary import build_summary  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_training_trick_summary_classifies_decisions():
    with _case_dir("training_trick_summary") as tmp_path:
        cache_results = _write_json(
            tmp_path / "cache.json",
            [
                {"experiment": "exp0_baseline", "f1": 0.92, "status": "success"},
                {"experiment": "exp2_swa", "f1": 0.88, "status": "success"},
                {"experiment": "exp3_ema", "f1": 0.91, "status": "success"},
                {"experiment": "exp5_all_combined", "f1": 0.85, "status": "success"},
            ],
        )
        group_results = _write_json(
            tmp_path / "group.json",
            {
                "results": [
                    {
                        "experiment": "exp0_baseline",
                        "val_f1": 0.928,
                        "test_f1": 0.887,
                        "test": {"false_positive": 10, "false_negative": 20},
                        "status": "success",
                    },
                    {
                        "experiment": "exp1_byte_noise",
                        "val_f1": 0.927,
                        "test_f1": 0.889,
                        "test": {"false_positive": 8, "false_negative": 21},
                        "status": "success",
                    },
                    {
                        "experiment": "exp4_near_threshold",
                        "val_f1": 0.926,
                        "test_f1": 0.890,
                        "test": {"false_positive": 9, "false_negative": 18},
                        "status": "success",
                    },
                ]
            },
        )

        summary = build_summary(cache_results, group_results)

    assert summary["decisions"]["negative_do_not_prioritize"] == [
        "exp2_swa",
        "exp3_ema",
        "exp5_all_combined",
    ]
    assert summary["decisions"]["small_gain_needs_multiseed"] == [
        "exp1_byte_noise",
        "exp4_near_threshold",
    ]


def test_build_training_trick_summary_uses_multiseed_negative_confirmation():
    with _case_dir("training_trick_summary_multiseed") as tmp_path:
        cache_results = _write_json(
            tmp_path / "cache.json",
            [
                {"experiment": "exp0_baseline", "f1": 0.92, "status": "success"},
                {"experiment": "exp2_swa", "f1": 0.88, "status": "success"},
            ],
        )
        group_results = _write_json(
            tmp_path / "group.json",
            {
                "results": [
                    {
                        "experiment": "exp0_baseline",
                        "val_f1": 0.928,
                        "test_f1": 0.887,
                        "test": {"false_positive": 10, "false_negative": 20},
                        "status": "success",
                    },
                    {
                        "experiment": "exp1_byte_noise",
                        "val_f1": 0.927,
                        "test_f1": 0.889,
                        "test": {"false_positive": 8, "false_negative": 21},
                        "status": "success",
                    },
                ]
            },
        )
        multiseed_results = _write_json(
            tmp_path / "multiseed.json",
            {
                "multiseed_summary": {
                    "aggregate_by_base_experiment": {
                        "exp0_baseline": {
                            "runs": 3,
                            "seeds": [42, 43, 44],
                            "val_f1_mean": 0.95,
                            "val_f1_stdev": 0.01,
                            "test_f1_mean": 0.94,
                            "test_f1_stdev": 0.01,
                            "test_fp_mean": 10,
                            "test_fn_mean": 20,
                            "delta_test_f1_mean_vs_baseline": 0.0,
                            "delta_test_fp_mean_vs_baseline": 0.0,
                            "delta_test_fn_mean_vs_baseline": 0.0,
                        },
                        "exp1_byte_noise": {
                            "runs": 3,
                            "seeds": [42, 43, 44],
                            "val_f1_mean": 0.94,
                            "val_f1_stdev": 0.02,
                            "test_f1_mean": 0.92,
                            "test_f1_stdev": 0.05,
                            "test_fp_mean": 12,
                            "test_fn_mean": 40,
                            "delta_test_f1_mean_vs_baseline": -0.02,
                            "delta_test_fp_mean_vs_baseline": 2.0,
                            "delta_test_fn_mean_vs_baseline": 20.0,
                        },
                        "exp4_near_threshold": {
                            "runs": 3,
                            "seeds": [42, 43, 44],
                            "val_f1_mean": 0.93,
                            "val_f1_stdev": 0.03,
                            "test_f1_mean": 0.91,
                            "test_f1_stdev": 0.06,
                            "test_fp_mean": 11,
                            "test_fn_mean": 50,
                            "delta_test_f1_mean_vs_baseline": -0.03,
                            "delta_test_fp_mean_vs_baseline": 1.0,
                            "delta_test_fn_mean_vs_baseline": 30.0,
                        },
                    }
                }
            },
        )

        summary = build_summary(cache_results, group_results, multiseed_results)

    assert summary["decisions"]["small_gain_needs_multiseed"] == []
    assert "exp1_byte_noise" in summary["decisions"]["negative_do_not_prioritize"]
    assert "exp4_near_threshold" in summary["decisions"]["negative_do_not_prioritize"]
