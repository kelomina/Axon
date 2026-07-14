from __future__ import annotations

import csv
from pathlib import Path

import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_loop162_loop160_failure_posthoc import build_loop162_posthoc  # noqa: E402


FIELDS = [
    "source_path",
    "cache_path",
    "source_sha256",
    "sample_index",
    "label",
    "trusted_signer_guard_prediction",
    "loop160_candidate_prediction",
    "loop160_prediction",
    "loop160_correct",
    "loop160_accept_candidate",
    "loop160_gate_score",
    "loop160_selected_threshold",
    "auth_status",
    "trusted_signer_guard_downgrade",
]


def _sha(seed: int) -> str:
    return f"{seed:064x}"[-64:]


def _row(seed: int, label: int, correct: bool, *, accepted: bool = True, score: float = 0.1) -> dict:
    return {
        "source_path": f"data/{seed}.exe",
        "cache_path": f"cache/{seed}.npz",
        "source_sha256": _sha(seed),
        "sample_index": str(seed),
        "label": str(label),
        "trusted_signer_guard_prediction": "0",
        "loop160_candidate_prediction": "1",
        "loop160_prediction": "1" if accepted else "0",
        "loop160_correct": str(correct),
        "loop160_accept_candidate": str(accepted),
        "loop160_gate_score": f"{score:.6f}",
        "loop160_selected_threshold": "0.25",
        "auth_status": "NotSigned",
        "trusted_signer_guard_downgrade": "False",
    }


def _write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_loop162_builds_public_and_private_posthoc_outputs(tmp_path: Path):
    val = _write_csv(tmp_path / "val.csv", [_row(1, 1, True), _row(2, 1, True, accepted=False)])
    test10k = _write_csv(tmp_path / "test10k.csv", [_row(3, 1, True)])
    full = _write_csv(tmp_path / "full.csv", [_row(4, 1, True), _row(5, 0, False), _row(6, 0, False)])

    payload = build_loop162_posthoc(
        val_predictions_csv=val,
        test10k_predictions_csv=test10k,
        full_predictions_csv=full,
        output_json=tmp_path / "summary.json",
        output_public_csv=tmp_path / "public.csv",
        output_private_map_csv=tmp_path / "private.csv",
        output_md=tmp_path / "summary.md",
    )

    assert payload["decision"] == "posthoc_failure_record_only"
    assert payload["split_summaries"]["val"]["accepted_correct"] == 1
    assert payload["split_summaries"]["test10k"]["accepted_wrong"] == 0
    assert payload["split_summaries"]["full_test"]["accepted_correct"] == 1
    assert payload["split_summaries"]["full_test"]["accepted_wrong"] == 2
    assert payload["failure_review"]["full_test_wrong_minus_correct"] == 1
    assert payload["selection_policy"]["may_select_model_or_threshold_from_this_report"] is False

    public_rows = list(csv.DictReader((tmp_path / "public.csv").open(encoding="utf-8-sig")))
    private_rows = list(csv.DictReader((tmp_path / "private.csv").open(encoding="utf-8-sig")))
    assert public_rows[0]["loop162_focus_id"].startswith("loop162_loop160_posthoc_val_")
    assert "source_sha256" not in public_rows[0]
    assert "sample_index" not in public_rows[0]
    assert private_rows[0]["source_sha256"] == _sha(1)
