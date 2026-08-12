from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_loop175_region_coverage_probe import (  # noqa: E402
    select_balanced_rows,
    validate_phase0_receipt,
)


def test_balanced_selection_is_deterministic_and_does_not_mutate_rows() -> None:
    rows = [
        {
            "source_path": f"sample-{index}",
            "source_sha256": f"{index:064x}",
            "label": str(index % 2),
            "split": "train",
        }
        for index in range(40)
    ]
    first = select_balanced_rows(rows, count=20, seed=175)
    second = select_balanced_rows(rows, count=20, seed=175)
    assert first == second
    assert sum(int(row["label"]) == 0 for row in first) == 10
    assert sum(int(row["label"]) == 1 for row in first) == 10


def test_phase0_receipt_rejects_source_drift(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("frozen", encoding="ascii")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "axon_loop175_phase0_receipt_v1",
                "decision": "phase0_pass_phase_a_256_train_coverage_probe_ready",
                "sources": [
                    {
                        "path": str(source.relative_to(PROJECT_ROOT))
                        if source.is_relative_to(PROJECT_ROOT)
                        else "missing-source.txt",
                        "sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="ascii",
    )
    try:
        validate_phase0_receipt(receipt)
    except RuntimeError as error:
        assert "source drift" in str(error)
    else:
        raise AssertionError("source drift must fail closed")


def test_probe_output_schema_has_no_row_identity_fields() -> None:
    forbidden = {"source_path", "source_sha256", "sample_index", "cache_path"}
    output_fields = {
        "schema",
        "loop_id",
        "claim_scope",
        "inputs",
        "sampling",
        "coverage",
        "aggregate_region_kind_counts",
        "aggregate_missing_reason_counts",
        "timing",
        "resources",
        "blockers",
        "decision",
        "training_runs",
        "val_test_or_full_rows_opened",
    }
    assert forbidden.isdisjoint(output_fields)
