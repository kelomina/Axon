from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_loop155_candidate_governance_audit import build_audit  # noqa: E402


def _write_eval(root: Path, relative: str, errors: int, fp: int, fn: int, *, key: str = "candidate") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    tp = 1000 - fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    path.write_text(
        json.dumps(
            {
                key: {
                    "f1": f1,
                    "errors": errors,
                    "false_positive": fp,
                    "false_negative": fn,
                }
            }
        ),
        encoding="utf-8",
    )


def test_loop155_audit_blocks_full_test_mirage(tmp_path: Path):
    _write_eval(tmp_path, "reports/phase3_loop151/loop151_trusted_signer_guard_val_eval.json", 162, 105, 57)
    _write_eval(tmp_path, "reports/phase3_loop151/loop151_trusted_signer_guard_test10k_eval.json", 78, 49, 29)
    _write_eval(tmp_path, "reports/phase3_loop151/loop151_trusted_signer_guard_full_eval.json", 1466, 879, 587)

    _write_eval(tmp_path, "reports/phase3_loop151/loop151_trusted_signer_guard_on_loop144_union_val_eval.json", 150, 104, 46)
    _write_eval(tmp_path, "reports/phase3_loop151/loop151_trusted_signer_guard_on_loop144_union_test10k_eval.json", 81, 55, 26)
    _write_eval(tmp_path, "reports/phase3_loop151/loop151_trusted_signer_guard_on_oof_noise_val_eval_reuse_loop136_sigs.json", 173, 110, 63)
    _write_eval(tmp_path, "reports/phase3_loop151/loop151_trusted_signer_guard_on_r5_full_eval.json", 1460, 906, 554)

    _write_eval(tmp_path, "reports/phase3_loop154/loop154_trusted_signer_guard_t0995_val_eval.json", 162, 105, 57)
    _write_eval(tmp_path, "reports/phase3_loop154/loop154_trusted_signer_guard_t0995_test10k_eval.json", 78, 49, 29)
    _write_eval(tmp_path, "reports/phase3_loop154/loop154_trusted_signer_guard_t0995_full_eval.json", 1466, 879, 587)

    payload = build_audit(tmp_path)
    by_id = {row["id"]: row for row in payload["candidates"]}

    assert by_id["loop151_current_strict_best"]["governance_decision"] == "adopted_current_strict_best"
    assert by_id["loop151_on_loop144_union"]["governance_decision"] == "reject_test10k_gate"
    assert by_id["loop151_on_loop144_union"]["deltas_vs_reference"]["val_errors"] == -12
    assert by_id["loop151_on_loop144_union"]["deltas_vs_reference"]["test10k_errors"] == 3
    assert by_id["loop151_on_oof_noise_r5"]["governance_decision"] == "reject_val_gate_full_test_mirage"
    assert by_id["loop151_on_oof_noise_r5"]["deltas_vs_reference"]["full_errors"] == -6
    assert by_id["loop154_trusted_signer_t0995"]["governance_decision"] == "reject_equivalent_to_current_best"
    assert payload["summary"]["full_test_mirage_count"] == 1
