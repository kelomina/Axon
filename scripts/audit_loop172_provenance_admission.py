#!/usr/bin/env python3
"""Audit metadata-only as-of provenance records before any data-governance action."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.loop172.provenance_ledger import (  # noqa: E402
    ProvenanceLedgerError,
    audit_as_of_records,
    parse_record,
)


def _parse_score_time(value: str) -> datetime:
    if not value.endswith("Z"):
        raise ValueError("score time must be a UTC Z timestamp")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def audit_records(path: Path, *, score_time_utc: datetime) -> dict[str, Any]:
    records = []
    parse_failures: Counter[str] = Counter()
    verdicts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                parse_failures["blank_record"] += 1
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ProvenanceLedgerError("record is not an object")
                record = parse_record(raw)
            except (json.JSONDecodeError, ProvenanceLedgerError):
                parse_failures["invalid_record"] += 1
                continue
            records.append(record)
            verdicts[record.verdict] += 1
    audit = audit_as_of_records(records, score_time_utc=score_time_utc)
    total = len(records) + sum(parse_failures.values())
    return {
        "schema": "axon_loop172_provenance_admission_audit_v1",
        "claim_scope": "metadata_governance_only_not_training_or_model_quality_evidence",
        "input": {"record_count": total, "score_time_utc": score_time_utc.isoformat().replace("+00:00", "Z")},
        "counts": {
            "parsed": len(records),
            "accepted": audit.accepted_records,
            "unknown": audit.unknown_records,
            "rejected": audit.rejected_records,
            "parse_failures": sum(parse_failures.values()),
            "verdicts_before_asof_gate": dict(sorted(verdicts.items())),
        },
        "rejection_reasons": list(audit.rejection_reasons),
        "parse_failure_reasons": dict(sorted(parse_failures.items())),
        "gates": {
            "all_records_parse": not parse_failures,
            "all_records_asof_admissible": audit.rejected_records == 0,
            "unknown_never_promoted": audit.unknown_records == verdicts.get("unknown", 0),
            "training_allowed": False,
            "heldout_allowed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--score-time-utc", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_records(args.records_jsonl, score_time_utc=_parse_score_time(args.score_time_utc))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
