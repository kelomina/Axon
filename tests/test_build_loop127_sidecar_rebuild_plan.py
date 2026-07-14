from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.build_loop127_sidecar_rebuild_plan import build_loop127_sidecar_rebuild_plan


def _write_predictions(path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_path", "source_sha256", "cache_path", "label", "split", "sample_index"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_path": str(path.parent / "sample.exe"),
                "source_sha256": "a" * 64,
                "cache_path": str(path.parent / "cache.npz"),
                "label": "1",
                "split": "train",
                "sample_index": "0",
            }
        )


def test_sidecar_rebuild_plan_joins_readiness_examples_to_predictions(tmp_path: Path):
    readiness = {
        "train": {
            "missing_examples": {
                "content_pe_v1": [
                    {"split": "train", "sample_index": "0", "source_sha256": "a" * 64, "label": "1"}
                ],
                "content_pe_v2": [
                    {"split": "train", "sample_index": "0", "source_sha256": "a" * 64, "label": "1"}
                ],
                "cache_path": [],
            }
        },
        "val": {"missing_examples": {"content_pe_v1": [], "content_pe_v2": [], "cache_path": []}},
    }
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    train_predictions = tmp_path / "train.csv"
    val_predictions = tmp_path / "val.csv"
    _write_predictions(train_predictions)
    val_predictions.write_text("source_path,source_sha256,cache_path,label,split,sample_index\n", encoding="utf-8-sig")

    payload = build_loop127_sidecar_rebuild_plan(
        readiness_json=readiness_path,
        train_predictions=train_predictions,
        val_predictions=val_predictions,
        output_csv=tmp_path / "plan.csv",
        output_json=tmp_path / "plan.json",
    )
    rows = list(csv.DictReader((tmp_path / "plan.csv").open("r", encoding="utf-8-sig", newline="")))

    assert payload["rows"] == 2
    assert payload["sidecar_counts"] == {"content_pe_v1": 1, "content_pe_v2": 1}
    assert payload["ready_to_rebuild"] is True
    assert rows[0]["source_path"].endswith("sample.exe")
    assert rows[0]["action"] == "rebuild_sidecar_from_source_content"
