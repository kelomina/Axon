from __future__ import annotations

import json
from pathlib import Path

import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_loop161_test10k_promotion_margin_guard import build_loop161_guard  # noqa: E402


def _metric(errors: int, fp: int, fn: int, f1: float) -> dict:
    return {"errors": errors, "false_positive": fp, "false_negative": fn, "f1": f1}


def _auth_eval(path: Path, *, base_errors: int, cand_errors: int, base_f1: float, cand_f1: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "baseline": _metric(base_errors, base_errors // 2, base_errors - base_errors // 2, base_f1),
                "candidate": _metric(cand_errors, cand_errors // 2, cand_errors - cand_errors // 2, cand_f1),
            }
        ),
        encoding="utf-8",
    )
    return path


def _loop159(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "current_best_reference": {
                    "val": {"errors": 162, "fp": 105, "fn": 57, "f1": 0.9919},
                    "test10k": {"errors": 78, "fp": 49, "fn": 29, "f1": 0.9921},
                },
                "metrics": {
                    "val": {"errors": 155, "fp": 106, "fn": 49, "f1": 0.9922},
                    "test10k": {"errors": 78, "fp": 52, "fn": 26, "f1": 0.9922},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _loop160(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "evaluations": {
                    "val": {
                        "baseline": _metric(162, 105, 57, 0.9919),
                        "candidate": _metric(159, 105, 54, 0.9920),
                    },
                    "test10k": {
                        "baseline": _metric(78, 49, 29, 0.9921),
                        "candidate": _metric(77, 49, 28, 0.9922),
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_loop161_allows_only_material_test10k_margins(tmp_path: Path):
    payload = build_loop161_guard(
        loop151_val_eval=_auth_eval(tmp_path / "loop151_val.json", base_errors=179, cand_errors=162, base_f1=0.9910, cand_f1=0.9919),
        loop151_test10k_eval=_auth_eval(
            tmp_path / "loop151_test.json",
            base_errors=83,
            cand_errors=78,
            base_f1=0.9916,
            cand_f1=0.9921,
        ),
        loop144_val_eval=_auth_eval(tmp_path / "loop144_val.json", base_errors=167, cand_errors=150, base_f1=0.9916, cand_f1=0.9925),
        loop144_test10k_eval=_auth_eval(
            tmp_path / "loop144_test.json",
            base_errors=86,
            cand_errors=81,
            base_f1=0.9914,
            cand_f1=0.9919,
        ),
        loop159_audit=_loop159(tmp_path / "loop159.json"),
        loop160_audit=_loop160(tmp_path / "loop160.json"),
        output_json=tmp_path / "summary.json",
        output_md=tmp_path / "summary.md",
        min_val_error_improvement=3,
        min_test10k_error_improvement=3,
    )

    by_id = {row["candidate_id"]: row for row in payload["candidates"]}
    assert by_id["loop151_trusted_signer_guard"]["decision"] == "allow_full_test_confirmation"
    assert by_id["loop144_union_trusted_signer"]["decision"] == "reject_test10k_margin_too_small"
    assert by_id["loop159_r11_only_trusted_signer"]["decision"] == "reject_test10k_margin_too_small"
    assert by_id["loop160_lowprob_r11_gate"]["decision"] == "reject_test10k_margin_too_small"
    assert payload["summary"]["allow_full_test_confirmation"] == 1
    assert payload["summary"]["rejected_test10k_margin"] == 3
