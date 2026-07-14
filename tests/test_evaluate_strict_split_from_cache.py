import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_strict_split_from_cache import (  # noqa: E402
    cache_eval_num_workers,
    collect_strict_records,
    compute_metrics,
    select_manifest_sample,
    evaluate_strict_split_from_cache,
)


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_split(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_path", "source_sha256", "label", "sample_index", "split"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path: Path, samples: list[dict]) -> None:
    path.write_text(json.dumps({"samples": samples}), encoding="utf-8")


def test_cache_eval_num_workers_rejects_windows_worker_copies():
    with pytest.raises(ValueError, match="num_workers > 0"):
        cache_eval_num_workers(1)


def test_collect_strict_records_matches_by_source_sha256_only():
    with _case_dir("strict_eval_sha_only") as tmp_path:
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        cache_path = cache_dir / "sample.npz"
        cache_path.write_bytes(b"placeholder")
        split_csv = tmp_path / "split.csv"
        manifest_json = cache_dir / "manifest.json"
        _write_split(
            split_csv,
            [
                {
                    "source_path": "renamed-anything.exe",
                    "source_sha256": "a" * 64,
                    "label": "1",
                    "sample_index": "1",
                    "split": "val",
                }
            ],
        )
        _write_manifest(
            manifest_json,
            [
                {
                    "source_path": "different-name.exe",
                    "cache_path": str(cache_path),
                    "label": 1,
                    "source_sha256": "a" * 64,
                }
            ],
        )

        records, summary = collect_strict_records(
            split_csv=split_csv,
            manifest_json=manifest_json,
            split="val",
        )

    assert len(records) == 1
    assert records[0]["source_sha256"] == "a" * 64
    assert summary["manifest_match_counts"] == {"source_sha256": 1}
    assert summary["issue_rows"] == 0


def test_collect_strict_records_rejects_path_match_when_hash_differs():
    with _case_dir("strict_eval_no_path_fallback") as tmp_path:
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        cache_path = cache_dir / "sample.npz"
        cache_path.write_bytes(b"placeholder")
        source_path = str(tmp_path / "same-name.exe")
        split_csv = tmp_path / "split.csv"
        manifest_json = cache_dir / "manifest.json"
        _write_split(
            split_csv,
            [
                {
                    "source_path": source_path,
                    "source_sha256": "b" * 64,
                    "label": "0",
                    "sample_index": "2",
                    "split": "val",
                }
            ],
        )
        _write_manifest(
            manifest_json,
            [
                {
                    "source_path": source_path,
                    "cache_path": str(cache_path),
                    "label": 0,
                    "source_sha256": "a" * 64,
                }
            ],
        )

        records, summary = collect_strict_records(
            split_csv=split_csv,
            manifest_json=manifest_json,
            split="val",
        )

    assert records == []
    assert summary["manifest_match_counts"] == {}
    assert summary["issue_counts"]["manifest_missing_source_sha256"] == 1


def test_select_manifest_sample_rejects_conflicting_manifest_labels():
    sample_sha = "c" * 64

    sample, issues = select_manifest_sample(
        {"source_sha256": sample_sha, "label": "1"},
        {
            sample_sha: [
                {"source_sha256": sample_sha, "label": "1"},
                {"source_sha256": sample_sha, "label": "0"},
            ]
        },
    )

    assert sample is None
    assert issues == ["manifest_conflicting_labels_for_source_sha256"]


def test_compute_metrics_reports_confusion_counts():
    metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.2], threshold=0.5)

    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["errors"] == 2


def test_collect_strict_records_accepts_locked_test10k_split():
    with _case_dir("strict_eval_test10k") as tmp_path:
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        cache_path = cache_dir / "sample.npz"
        cache_path.write_bytes(b"placeholder")
        split_csv = tmp_path / "split.csv"
        manifest_json = cache_dir / "manifest.json"
        _write_split(
            split_csv,
            [
                {
                    "source_path": "renamed-anything.exe",
                    "source_sha256": "d" * 64,
                    "label": "1",
                    "sample_index": "10",
                    "split": "test10k",
                }
            ],
        )
        _write_manifest(
            manifest_json,
            [
                {
                    "source_path": "different-name.exe",
                    "cache_path": str(cache_path),
                    "label": 1,
                    "source_sha256": "d" * 64,
                }
            ],
        )

        records, summary = collect_strict_records(
            split_csv=split_csv,
            manifest_json=manifest_json,
            split="test10k",
        )

    assert len(records) == 1
    assert records[0]["split"] == "test10k"
    assert summary["manifest_match_counts"] == {"source_sha256": 1}


def test_collect_strict_records_falls_back_to_locked_test_prefix_for_test10k():
    with _case_dir("strict_eval_test10k_fallback") as tmp_path:
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        samples = []
        split_rows = []
        for index, suffix in enumerate(["a", "b", "c"], start=1):
            cache_path = cache_dir / f"sample_{suffix}.npz"
            cache_path.write_bytes(b"placeholder")
            sha = suffix * 64
            split_rows.append(
                {
                    "source_path": f"sample_{suffix}.exe",
                    "source_sha256": sha,
                    "label": str(index % 2),
                    "sample_index": str(index),
                    "split": "test",
                }
            )
            samples.append(
                {
                    "source_path": f"sample_{suffix}.exe",
                    "cache_path": str(cache_path),
                    "label": index % 2,
                    "source_sha256": sha,
                }
            )

        split_csv = tmp_path / "split.csv"
        manifest_json = cache_dir / "manifest.json"
        _write_split(split_csv, split_rows)
        _write_manifest(manifest_json, samples)

        records, summary = collect_strict_records(
            split_csv=split_csv,
            manifest_json=manifest_json,
            split="test10k",
            max_rows=2,
        )

    assert [record["source_sha256"] for record in records] == ["a" * 64, "b" * 64]
    assert [record["split"] for record in records] == ["test", "test"]
    assert summary["raw_rows"] == 2
    assert summary["records"] == 2


def test_evaluate_strict_split_records_feature_mask_metadata(monkeypatch):
    with _case_dir("strict_eval_feature_mask") as tmp_path:
        split_csv = tmp_path / "split.csv"
        manifest_json = tmp_path / "manifest.json"
        output_json = tmp_path / "eval.json"
        mask_json = tmp_path / "mask.json"
        _write_split(
            split_csv,
            [
                {
                    "source_path": "anything.exe",
                    "source_sha256": "e" * 64,
                    "label": "0",
                    "sample_index": "1",
                    "split": "val",
                }
            ],
        )
        _write_manifest(manifest_json, [])
        mask_json.write_text("{}", encoding="utf-8")

        def fake_collect_strict_records(**_kwargs):
            return (
                [
                    {
                        "cache_path": str(tmp_path / "sample.npz"),
                        "source_path": "anything.exe",
                        "source_sha256": "e" * 64,
                        "label": 0,
                        "split": "val",
                        "sample_index": "1",
                    }
                ],
                {
                    "raw_rows": 1,
                    "records": 1,
                    "label_counts": {"0": 1},
                    "manifest_match_counts": {"source_sha256": 1},
                    "issue_counts": {},
                    "issue_rows": 0,
                    "issue_examples": [],
                },
            )

        class FakeDataset:
            def __init__(self, records, _config):
                self.records = records

            def __len__(self):
                return 1

            def __getitem__(self, index):
                import torch

                return (
                    torch.zeros(4, dtype=torch.long),
                    torch.ones(2, dtype=torch.float32),
                    torch.ones(2, dtype=torch.float32),
                    0,
                    index,
                )

        class FakeModel:
            def __init__(self, _config):
                pass

            def load_state_dict(self, _state):
                pass

            def to(self, _device):
                return self

            def eval(self):
                return self

            def __call__(self, byte_seq, pe_features, stat_features=None):
                import torch

                assert float(pe_features.sum()) == 0.0
                assert float(stat_features.sum()) == 0.0
                return {"logits": torch.tensor([[2.0, 0.0]], dtype=torch.float32)}

        class FakeConfig:
            max_byte_length = 4
            pe_feature_dim = 2
            stat_feature_dim = 2
            lightweight_feature_dim = 0
            pe_schema_version = "fixed_v2"
            pe_fixed_section_slots = 32

            @classmethod
            def from_dict(cls, _payload):
                return cls()

        import evaluate_strict_split_from_cache as module
        import torch

        monkeypatch.setattr(module, "collect_strict_records", fake_collect_strict_records)
        monkeypatch.setattr(module, "StrictCachedSplitDataset", FakeDataset)
        monkeypatch.setattr(module, "AxonMalwareModel", FakeModel)
        monkeypatch.setattr(module, "AxonExperimentConfig", FakeConfig)
        monkeypatch.setattr(module, "load_safe_checkpoint", lambda *_args, **_kwargs: {"config": {}, "model_state_dict": {}})
        monkeypatch.setattr(
            module,
            "load_feature_mask_tensors",
            lambda *_args, **_kwargs: (
                torch.zeros(2, dtype=torch.float32),
                torch.zeros(2, dtype=torch.float32),
                {"kept_total": 0, "kept_pe": 0, "kept_stat": 0},
            ),
        )

        payload = evaluate_strict_split_from_cache(
            checkpoint=tmp_path / "model.pt",
            split_csv=split_csv,
            manifest_json=manifest_json,
            output_json=output_json,
            split="val",
            threshold=0.5,
            batch_size=1,
            device_name="cpu",
            feature_mask_path=mask_json,
        )

    assert payload["feature_mask"] == str(mask_json)
    assert payload["feature_mask_summary"] == "kept_total=0, kept_pe=0, kept_stat=0"
