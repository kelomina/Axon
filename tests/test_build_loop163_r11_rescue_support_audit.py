from __future__ import annotations

import csv
from pathlib import Path

import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_loop163_r11_rescue_support_audit import build_loop163_support_audit  # noqa: E402


FIELDS = [
    "source_path",
    "cache_path",
    "source_sha256",
    "sample_index",
    "label",
    "baseline_prob_malicious",
    "selector_score",
    "prediction",
    "trusted_signer_guard_prediction",
]


def _sha(seed: int) -> str:
    return f"{seed:064x}"[-64:]


def _row(seed: int, label: int, pred: int, *, score: float = 0.2, selector_score: float = 0.3) -> dict:
    return {
        "source_path": f"data/{seed}.exe",
        "cache_path": f"cache/{seed}.npz",
        "source_sha256": _sha(seed),
        "sample_index": str(seed),
        "label": str(label),
        "baseline_prob_malicious": f"{score:.6f}",
        "selector_score": f"{selector_score:.6f}",
        "prediction": str(pred),
        "trusted_signer_guard_prediction": str(pred),
    }


def _write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_loop163_blocks_low_support_and_public_rows_hide_identity(tmp_path: Path):
    val_base = [_row(1, 1, 0), _row(2, 0, 0), _row(3, 1, 1)]
    val_candidate = [_row(1, 1, 1), _row(2, 0, 1), _row(3, 1, 1)]
    test_base = [_row(11, 1, 0), _row(12, 1, 1)]
    test_candidate = [_row(11, 1, 1), _row(12, 1, 1)]

    payload = build_loop163_support_audit(
        val_base_csv=_write_csv(tmp_path / "val_base.csv", val_base),
        val_candidate_csv=_write_csv(tmp_path / "val_candidate.csv", val_candidate),
        test10k_base_csv=_write_csv(tmp_path / "test_base.csv", test_base),
        test10k_candidate_csv=_write_csv(tmp_path / "test_candidate.csv", test_candidate),
        output_json=tmp_path / "summary.json",
        output_public_csv=tmp_path / "public.csv",
        output_private_map_csv=tmp_path / "private.csv",
        output_md=tmp_path / "summary.md",
        min_val_disagreements_for_selector=5,
        min_val_fix_rows_for_selector=2,
        max_val_break_rows_for_selector=0,
    )

    assert payload["decision"] == "reject_low_support_no_selector_training"
    assert "val_disagreement_support_below_minimum" in payload["support_failures"]
    assert "val_fix_support_below_minimum" in payload["support_failures"]
    assert "val_break_rows_exceed_limit" in payload["support_failures"]
    assert payload["selection_policy"]["train_selector_allowed"] is False

    public_rows = list(csv.DictReader((tmp_path / "public.csv").open(encoding="utf-8-sig")))
    private_rows = list(csv.DictReader((tmp_path / "private.csv").open(encoding="utf-8-sig")))
    assert public_rows[0]["loop163_focus_id"].startswith("loop163_r11_support_val_")
    assert "source_sha256" not in public_rows[0]
    assert "sample_index" not in public_rows[0]
    assert private_rows[0]["source_sha256"] == _sha(1)
