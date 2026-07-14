from __future__ import annotations

import csv
from pathlib import Path

import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_loop160_lowprob_r11_gate import build_loop160_gate  # noqa: E402


FIELDS = [
    "source_path",
    "source_sha256",
    "sample_index",
    "split",
    "label",
    "baseline_prob_malicious",
    "prediction",
    "trusted_signer_guard_prediction",
]


def _sha(seed: int) -> str:
    return f"{seed:064x}"[-64:]


def _write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(seed: int, label: int, pred: int, *, score: float, split: str = "val") -> dict:
    return {
        "source_path": f"data/{seed}.exe",
        "source_sha256": _sha(seed),
        "sample_index": str(seed),
        "split": split,
        "label": str(label),
        "baseline_prob_malicious": f"{score:.6f}",
        "prediction": str(pred),
        "trusted_signer_guard_prediction": str(pred),
    }


def _case(tmp_path: Path):
    # Base has three FN and one correct benign. Candidate rescues all four.
    val_base = [
        _row(1, 1, 0, score=0.10),
        _row(2, 1, 0, score=0.20),
        _row(3, 1, 0, score=0.30),
        _row(4, 0, 0, score=0.40),
    ]
    val_candidate = [
        _row(1, 1, 1, score=0.10),
        _row(2, 1, 1, score=0.20),
        _row(3, 1, 1, score=0.30),
        _row(4, 0, 1, score=0.40),
    ]
    test_base = [
        _row(11, 1, 0, score=0.10, split="test"),
        _row(12, 0, 0, score=0.30, split="test"),
        _row(13, 1, 1, score=0.80, split="test"),
    ]
    test_candidate = [
        _row(11, 1, 1, score=0.10, split="test"),
        _row(12, 0, 1, score=0.30, split="test"),
        _row(13, 1, 1, score=0.80, split="test"),
    ]
    full_base = [
        _row(21, 1, 0, score=0.10, split="test"),
        _row(22, 0, 0, score=0.20, split="test"),
        _row(23, 0, 0, score=0.15, split="test"),
    ]
    full_candidate = [
        _row(21, 1, 1, score=0.10, split="test"),
        _row(22, 0, 1, score=0.20, split="test"),
        _row(23, 0, 1, score=0.15, split="test"),
    ]
    return {
        "val_base": _write_csv(tmp_path / "val_base.csv", val_base),
        "val_candidate": _write_csv(tmp_path / "val_candidate.csv", val_candidate),
        "test_base": _write_csv(tmp_path / "test_base.csv", test_base),
        "test_candidate": _write_csv(tmp_path / "test_candidate.csv", test_candidate),
        "full_base": _write_csv(tmp_path / "full_base.csv", full_base),
        "full_candidate": _write_csv(tmp_path / "full_candidate.csv", full_candidate),
    }


def test_loop160_selects_smallest_val_threshold_and_rejects_full_regression(tmp_path: Path):
    paths = _case(tmp_path)

    payload = build_loop160_gate(
        val_base_csv=paths["val_base"],
        val_candidate_csv=paths["val_candidate"],
        test10k_base_csv=paths["test_base"],
        test10k_candidate_csv=paths["test_candidate"],
        full_base_csv=paths["full_base"],
        full_candidate_csv=paths["full_candidate"],
        output_dir=tmp_path / "out",
        output_json=tmp_path / "summary.json",
        output_md=tmp_path / "summary.md",
        min_val_error_improvement=2,
        max_val_fp_delta=0,
    )

    assert payload["selection_policy"]["selected_threshold"] == 0.2
    assert payload["evaluations"]["val"]["delta_vs_baseline"]["errors"] == -2
    assert payload["evaluations"]["val"]["delta_vs_baseline"]["false_positive"] == 0
    assert payload["evaluations"]["test10k"]["delta_vs_baseline"]["errors"] == -1
    assert payload["evaluations"]["full_test"]["delta_vs_baseline"]["errors"] == 1
    assert payload["decision"] == "reject_full_test_confirmation"
    assert Path(payload["outputs"]["val_predictions_csv"]).exists()
