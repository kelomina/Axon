from __future__ import annotations

import json
from pathlib import Path

import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_loop159_r11_only_candidate_audit import build_loop159_audit  # noqa: E402


def _eval_payload(*, f1: float, errors: int, fp: int, fn: int, decision: str = "allow_next_funnel_step") -> dict:
    return {
        "candidate": {
            "f1": f1,
            "errors": errors,
            "false_positive": fp,
            "false_negative": fn,
        },
        "fixed_fp": 3,
        "introduced_fn": 0,
        "decision": decision,
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_loop159_rejects_full_test_regression_after_val_and_test10k(tmp_path: Path):
    val = _write_json(tmp_path / "val.json", _eval_payload(f1=0.9922, errors=155, fp=106, fn=49))
    test10k = _write_json(
        tmp_path / "test10k.json",
        _eval_payload(f1=0.9921968788, errors=78, fp=52, fn=26, decision="reject_val_margin_too_small"),
    )
    full = _write_json(tmp_path / "full.json", _eval_payload(f1=0.9907064008, errors=1491, fp=962, fn=529))

    payload = build_loop159_audit(
        val_eval_json=val,
        test10k_eval_json=test10k,
        full_eval_json=full,
        output_json=tmp_path / "summary.json",
        output_md=tmp_path / "summary.md",
    )

    assert payload["decision"] == "reject_full_test_confirmation_not_strict_best"
    assert payload["metrics"]["val"]["delta_vs_loop151"]["errors"] == -7
    assert payload["metrics"]["test10k"]["delta_vs_loop151"]["errors"] == 0
    assert payload["metrics"]["full_test"]["delta_vs_loop151"]["errors"] == 25
    assert payload["metrics"]["full_test"]["delta_vs_loop151"]["fp"] == 83
    assert payload["metrics"]["full_test"]["delta_vs_loop151"]["fn"] == -58
    assert payload["gate_review"]["test10k_f1_non_regression"] is True
    assert "high-recall" in payload["next_action"]
