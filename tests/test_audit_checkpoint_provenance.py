from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_checkpoint_provenance import _safe_load_checkpoint, audit_checkpoints  # noqa: E402


def _write_split(path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "label", "sample_index", "split"])
        writer.writeheader()
        writer.writerow({"source_path": "a", "label": 0, "sample_index": 0, "split": "train"})
        writer.writerow({"source_path": "b", "label": 1, "sample_index": 1, "split": "val"})


def _checkpoint_payload() -> dict:
    return {
        "model_state_dict": {},
        "config": {
            "max_byte_length": 8192,
            "pe_feature_dim": 256,
            "stat_feature_dim": 49,
            "pe_schema_version": "fixed_v2",
            "pe_fixed_section_slots": 32,
        },
        "train_config": {"max_epochs": 5, "batch_size": 32},
        "best_f1": 0.9,
        "epoch": 3,
    }


def test_audit_checkpoint_provenance_flags_current_and_mismatch(tmp_path: Path):
    models = tmp_path / "models"
    current = models / "random_20w_8192"
    current_seed = models / "random_20w_8192_seed43"
    group = models / "generalization_group_isolated" / "exp0_baseline"
    unknown = models / "other"
    current.mkdir(parents=True)
    current_seed.mkdir(parents=True)
    group.mkdir(parents=True)
    unknown.mkdir(parents=True)
    torch.save(_checkpoint_payload(), current / "best_model.pt")
    torch.save(_checkpoint_payload(), current_seed / "best_model.pt")
    torch.save(_checkpoint_payload(), group / "best_model.pt")
    bad_payload = _checkpoint_payload()
    bad_payload["config"]["max_byte_length"] = 512
    torch.save(bad_payload, unknown / "best_model.pt")
    split = tmp_path / "split.csv"
    _write_split(split)

    report = audit_checkpoints(
        models_dir=models,
        split_csv=split,
        output_json=tmp_path / "audit.json",
        max_rows=None,
    )

    assert report["checkpoint_count"] == 4
    assert report["compatible_current_random20w_count"] == 2
    statuses = {Path(row["path"]).parent.name: row["status"] for row in report["checkpoints"]}
    assert statuses["random_20w_8192"] == "compatible_current_random20w"
    assert statuses["random_20w_8192_seed43"] == "compatible_current_random20w"
    assert statuses["exp0_baseline"] == "provenance_mismatch"
    assert statuses["other"] == "incompatible"


def test_audit_checkpoint_provenance_uses_restricted_torch_loader(monkeypatch, tmp_path: Path):
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"placeholder")
    captured = {}

    def fake_load(path, map_location=None, weights_only=None):
        captured["path"] = path
        captured["map_location"] = map_location
        captured["weights_only"] = weights_only
        return _checkpoint_payload()

    monkeypatch.setattr("audit_checkpoint_provenance.torch.load", fake_load)

    payload = _safe_load_checkpoint(checkpoint_path)

    assert payload is not None
    assert payload["config"]["pe_schema_version"] == "fixed_v2"
    assert captured == {"path": checkpoint_path, "map_location": "cpu", "weights_only": True}
