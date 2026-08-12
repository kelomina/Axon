from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from audit_loop172_provenance_admission import audit_records


def _record(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "evidence_id": "evidence-a",
        "subject_commitment": "a" * 64,
        "source_name": "independent-archive",
        "source_independence_id": "archive-a",
        "published_at_utc": "2024-01-01T00:00:00Z",
        "available_at_utc": "2024-01-01T00:00:00Z",
        "snapshot_captured_at_utc": "2024-01-02T00:00:00Z",
        "verdict": "unknown",
    }
    payload.update(overrides)
    return payload


def _write(path: Path, rows: list[object]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_unknown_records_remain_unknown_after_admission_gate(tmp_path: Path) -> None:
    payload = audit_records(
        _write(tmp_path / "records.jsonl", [_record()]),
        score_time_utc=__import__("datetime").datetime(2024, 2, 1, tzinfo=__import__("datetime").timezone.utc),
    )

    assert payload["counts"]["unknown"] == 1
    assert payload["counts"]["accepted"] == 0
    assert payload["gates"]["unknown_never_promoted"] is True
    assert payload["gates"]["training_allowed"] is False


def test_future_record_and_schema_drift_are_explicitly_rejected(tmp_path: Path) -> None:
    payload = audit_records(
        _write(
            tmp_path / "records.jsonl",
            [
                _record(available_at_utc="2024-03-01T00:00:00Z", snapshot_captured_at_utc="2024-03-02T00:00:00Z"),
                {"raw_path": "forbidden"},
            ],
        ),
        score_time_utc=__import__("datetime").datetime(2024, 2, 1, tzinfo=__import__("datetime").timezone.utc),
    )

    assert payload["counts"]["rejected"] == 1
    assert payload["counts"]["parse_failures"] == 1
    assert payload["rejection_reasons"] == ["not_available_as_of_score_time"]
    assert payload["parse_failure_reasons"] == {"invalid_record": 1}
