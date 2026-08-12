#!/usr/bin/env python3
"""Build aggregate-only Loop151 review strata without exposing review identities."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
YEAR_MONTH = re.compile(r"(?:^|[\\/])(20\d{2})-\d{2}(?:[\\/]|$)")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _counts(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def build_strata(public_rows: list[dict[str, str]], private_rows: list[dict[str, str]]) -> dict[str, Any]:
    public_by_id = {row.get("review_focus_id", ""): row for row in public_rows}
    private_by_id = {row.get("review_focus_id", ""): row for row in private_rows}
    if not public_by_id or len(public_by_id) != len(public_rows) or len(private_by_id) != len(private_rows):
        raise ValueError("review focus identifiers must be present and unique")
    if set(public_by_id) != set(private_by_id):
        raise ValueError("public and private review identities do not align")
    lanes: Counter[str] = Counter()
    years: Counter[str] = Counter()
    source_groups: Counter[str] = Counter()
    for review_id, public_row in public_by_id.items():
        lane = public_row.get("review_lane", "").strip()
        if not lane:
            raise ValueError("review lane is missing")
        lanes[lane] += 1
        private_row = private_by_id[review_id]
        source_path = private_row.get("source_path", "")
        match = YEAR_MONTH.search(source_path.replace("\\", "/"))
        years[match.group(1) if match else "unknown"] += 1
        # 私有表没有独立来源字段；不可从文件路径、标签或模型输出猜造来源分层。
        source_groups["unknown"] += 1
    source_axis_ready = set(source_groups) != {"unknown"} and len(source_groups) >= 3
    return {
        "schema": "axon_loop173_loop151_review_strata_v1",
        "claim_scope": "aggregate_review_assignment_readiness_not_label_or_model_evidence",
        "input": {"review_rows": len(public_rows), "private_alignment_rows": len(private_rows)},
        "axes": {
            "component": {"field": "review_lane", "counts": _counts(lanes), "ready": len(lanes) >= 3},
            "time": {"field": "source_path_year_aggregate_only", "counts": _counts(years), "ready": "unknown" not in years and len(years) >= 3},
            "source": {"field": "independent_source_provenance", "counts": _counts(source_groups), "ready": source_axis_ready},
        },
        "identity_policy": "private paths are used only for aggregate year extraction and are not persisted",
        "decision": "ready_for_stratified_independent_review" if source_axis_ready else "blocked_missing_independent_source_provenance",
        "authorizations": {"training_allowed": False, "heldout_allowed": False, "automatic_verdict_allowed": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-context", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    payload = build_strata(_read_csv(args.public_context), _read_csv(args.private_map))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
