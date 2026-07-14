import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in [SRC_DIR, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import AxonExperimentConfig  # noqa: E402
from dataset import (  # noqa: E402
    FeatureCacheDataset,
    MalwareDataset,
    NPZDataset,
    _feature_cache_hash,
    _load_cached_feature_npz,
    _resolve_manifest_cache_path,
)
from kvd_features.extractor import extract_byte_sequence  # noqa: E402
from security import load_safe_checkpoint  # noqa: E402


def _base_checkpoint():
    return {
        "model_state_dict": {},
        "config": AxonExperimentConfig().to_dict(),
    }


def _write_cache_npz(path: Path, *, label: int, source_sha256: str | None = None):
    payload = {
        "byte_sequence": np.array([77, 90, 1, 2], dtype=np.uint8),
        "pe_features": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "stat_features": np.array([0.1, 0.2], dtype=np.float32),
        "lightweight_features": np.zeros(256, dtype=np.float32),
        "label": int(label),
    }
    if source_sha256 is not None:
        payload["source_sha256"] = source_sha256
    np.savez_compressed(path, **payload)


def _manifest_path(cache_dir: Path) -> Path:
    cache_hash = _feature_cache_hash(4, 2, 4, 256, True, False, "legacy_dynamic", 32)
    return cache_dir / f"manifest_{cache_hash}.json"


def _write_manifest(cache_dir: Path, samples: list[dict]):
    manifest = {
        "version": 1,
        "data_dir": str(cache_dir.parent),
        "cache_config_hash": _manifest_path(cache_dir).stem.replace("manifest_", ""),
        "max_byte_length": 4,
        "pe_feature_dim": 4,
        "stat_feature_dim": 2,
        "lightweight_feature_dim": 256,
        "strict_pe_parsing": True,
        "allow_pe_fallback": False,
        "pe_schema_version": "legacy_dynamic",
        "pe_fixed_section_slots": 32,
        "samples": samples,
    }
    _manifest_path(cache_dir).write_text(json.dumps(manifest), encoding="utf-8")


def _manifest_ready_malware_dataset(cache_dir: Path):
    dataset = MalwareDataset.__new__(MalwareDataset)
    dataset.cache_dir = cache_dir
    dataset.max_byte_length = 4
    dataset.pe_feature_dim = 4
    dataset.stat_feature_dim = 2
    dataset.lightweight_feature_dim = 256
    dataset.strict_pe_parsing = True
    dataset.allow_pe_fallback = False
    dataset.pe_schema_version = "legacy_dynamic"
    dataset.pe_fixed_section_slots = 32
    return dataset


def test_safe_checkpoint_uses_weights_only(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"placeholder")
    captured = {}

    def fake_load(path, map_location=None, weights_only=None):
        captured["weights_only"] = weights_only
        captured["map_location"] = map_location
        return _base_checkpoint()

    monkeypatch.setattr("security.torch.load", fake_load)

    checkpoint = load_safe_checkpoint(checkpoint_path, map_location="cpu")

    assert checkpoint["config"]["num_classes"] == 2
    assert captured == {"weights_only": True, "map_location": "cpu"}


def test_safe_checkpoint_defaults_to_cpu_map_location(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"placeholder")
    captured = {}

    def fake_load(path, map_location=None, weights_only=None):
        captured["weights_only"] = weights_only
        captured["map_location"] = map_location
        return _base_checkpoint()

    monkeypatch.setattr("security.torch.load", fake_load)

    checkpoint = load_safe_checkpoint(checkpoint_path)

    assert checkpoint["config"]["num_classes"] == 2
    assert captured == {"weights_only": True, "map_location": "cpu"}


def test_safe_checkpoint_rejects_bad_suffix_and_missing_config(tmp_path):
    bad_suffix = tmp_path / "model.pkl"
    bad_suffix.write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="Unsupported checkpoint suffix"):
        load_safe_checkpoint(bad_suffix)

    missing_config = tmp_path / "missing.pt"
    torch.save({"model_state_dict": {}, "config": {"max_byte_length": 4}}, missing_config)
    with pytest.raises(ValueError, match="Checkpoint config missing required keys"):
        load_safe_checkpoint(missing_config, map_location="cpu")


def test_manifest_cache_path_must_stay_inside_cache_dir(tmp_path):
    cache_dir = tmp_path / "data" / ".cache"
    cache_dir.mkdir(parents=True)
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"not-a-cache")

    with pytest.raises(ValueError, match="outside cache directory"):
        _resolve_manifest_cache_path(str(outside), cache_dir)


def test_feature_cache_manifest_rejects_label_conflict(tmp_path):
    cache_dir = tmp_path / "data" / ".cache"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / "sample_legacy.npz"
    _write_cache_npz(cache_path, label=0, source_sha256="a" * 64)
    _write_manifest(
        cache_dir,
        [{"source_path": "sample.exe", "cache_path": str(cache_path), "label": 1, "source_sha256": "a" * 64}],
    )

    with pytest.raises(ValueError, match="Cache label mismatch"):
        FeatureCacheDataset(
            data_dir=str(cache_dir.parent),
            max_byte_length=4,
            pe_feature_dim=4,
            stat_feature_dim=2,
            require_manifest=True,
        )


def test_strict_feature_cache_manifest_requires_source_sha256(tmp_path):
    cache_dir = tmp_path / "data" / ".cache"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / "sample_legacy.npz"
    _write_cache_npz(cache_path, label=0)
    _write_manifest(
        cache_dir,
        [{"source_path": "sample.exe", "cache_path": str(cache_path), "label": 0}],
    )

    with pytest.raises(ValueError, match="valid source_sha256"):
        FeatureCacheDataset(
            data_dir=str(cache_dir.parent),
            max_byte_length=4,
            pe_feature_dim=4,
            stat_feature_dim=2,
            require_manifest=True,
        )


def test_strict_feature_cache_manifest_rejects_source_sha256_mismatch(tmp_path):
    cache_dir = tmp_path / "data" / ".cache"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / "sample_legacy.npz"
    _write_cache_npz(cache_path, label=0, source_sha256="a" * 64)
    _write_manifest(
        cache_dir,
        [{"source_path": "sample.exe", "cache_path": str(cache_path), "label": 0, "source_sha256": "b" * 64}],
    )

    with pytest.raises(ValueError, match="Cache source SHA mismatch"):
        FeatureCacheDataset(
            data_dir=str(cache_dir.parent),
            max_byte_length=4,
            pe_feature_dim=4,
            stat_feature_dim=2,
            require_manifest=True,
        )


def test_cached_feature_npz_rejects_source_sha_mismatch(tmp_path):
    cache_path = tmp_path / "sample.npz"
    _write_cache_npz(cache_path, label=1, source_sha256="trusted-source")

    with pytest.raises(ValueError, match="Cache source SHA mismatch"):
        _load_cached_feature_npz(
            cache_path,
            max_byte_length=4,
            pe_feature_dim=4,
            stat_feature_dim=2,
            lightweight_feature_dim=256,
            expected_label=1,
            expected_source_sha256="tampered-source",
        )


def test_cached_feature_npz_closes_file_handle(tmp_path):
    cache_path = tmp_path / "sample.npz"
    _write_cache_npz(cache_path, label=1, source_sha256="a" * 64)

    _load_cached_feature_npz(
        cache_path,
        max_byte_length=4,
        pe_feature_dim=4,
        stat_feature_dim=2,
        lightweight_feature_dim=256,
        expected_label=1,
        expected_source_sha256="a" * 64,
    )

    renamed = tmp_path / "sample-renamed.npz"
    cache_path.rename(renamed)
    assert renamed.exists()
    renamed.unlink()
    assert not renamed.exists()


def test_malware_dataset_manifest_cache_samples_use_metadata_only(monkeypatch, tmp_path):
    cache_dir = tmp_path / "data" / ".cache"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / "sample_legacy.npz"
    source_sha = "a" * 64
    _write_cache_npz(cache_path, label=0, source_sha256=source_sha)
    _write_manifest(
        cache_dir,
        [{"source_path": "sample.exe", "cache_path": str(cache_path), "label": 0, "source_sha256": source_sha}],
    )

    def fail_if_full_npz_load_is_used(*_args, **_kwargs):
        raise AssertionError("manifest cache audit should only read metadata")

    monkeypatch.setattr("dataset._load_cached_feature_npz", fail_if_full_npz_load_is_used)
    dataset = _manifest_ready_malware_dataset(cache_dir)

    samples = dataset._load_manifest_cache_samples()

    assert samples == [
        {
            "source_path": "sample.exe",
            "cache_path": str(cache_path),
            "label": 0,
            "source_sha256": source_sha,
        }
    ]


def test_npz_dataset_bad_sample_fails_closed(tmp_path):
    split_dir = tmp_path / "npz" / "train"
    split_dir.mkdir(parents=True)
    np.savez_compressed(
        split_dir / "bad.npz",
        byte_sequence=np.array([77, 90], dtype=np.uint8),
        pe_features=np.zeros(4, dtype=np.float32),
    )
    dataset = NPZDataset(str(tmp_path / "npz"), split="train", max_byte_length=4, pe_feature_dim=4, stat_feature_dim=2)

    with pytest.raises(ValueError, match="missing required fields"):
        dataset[0]


def test_npz_dataset_returns_uint8_byte_tensor(tmp_path):
    split_dir = tmp_path / "npz" / "train"
    split_dir.mkdir(parents=True)
    np.savez_compressed(
        split_dir / "good.npz",
        byte_sequence=np.array([77, 90], dtype=np.uint8),
        pe_features=np.zeros(4, dtype=np.float32),
        stat_features=np.zeros(2, dtype=np.float32),
        label=1,
    )
    dataset = NPZDataset(str(tmp_path / "npz"), split="train", max_byte_length=4, pe_feature_dim=4, stat_feature_dim=2)

    byte_seq, _pe_features, _stat_features, label = dataset[0]

    assert byte_seq.dtype == torch.uint8
    assert byte_seq.tolist() == [77, 90, 0, 0]
    assert label.item() == 1


def test_npz_dataset_getitem_closes_file_handle(tmp_path):
    split_dir = tmp_path / "npz" / "train"
    split_dir.mkdir(parents=True)
    npz_path = split_dir / "good.npz"
    np.savez_compressed(
        npz_path,
        byte_sequence=np.array([77, 90], dtype=np.uint8),
        pe_features=np.zeros(4, dtype=np.float32),
        stat_features=np.zeros(2, dtype=np.float32),
        label=1,
    )
    dataset = NPZDataset(str(tmp_path / "npz"), split="train", max_byte_length=4, pe_feature_dim=4, stat_feature_dim=2)

    _sample = dataset[0]

    renamed = split_dir / "good-renamed.npz"
    npz_path.rename(renamed)
    assert renamed.exists()
    renamed.unlink()
    assert not renamed.exists()


def test_feature_cache_dataset_getitem_closes_file_handle(tmp_path):
    cache_dir = tmp_path / "data" / ".cache"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / "sample_legacy.npz"
    source_sha = "a" * 64
    _write_cache_npz(cache_path, label=0, source_sha256=source_sha)
    _write_manifest(
        cache_dir,
        [{"source_path": "sample.exe", "cache_path": str(cache_path), "label": 0, "source_sha256": source_sha}],
    )
    dataset = FeatureCacheDataset(
        data_dir=str(cache_dir.parent),
        max_byte_length=4,
        pe_feature_dim=4,
        stat_feature_dim=2,
        require_manifest=True,
    )

    _sample = dataset[0]

    renamed = cache_dir / "sample-renamed.npz"
    cache_path.rename(renamed)
    assert renamed.exists()
    renamed.unlink()
    assert not renamed.exists()


def test_malware_dataset_cached_getitem_closes_file_handle(tmp_path):
    cache_dir = tmp_path / "data" / ".cache"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / "sample_legacy.npz"
    _write_cache_npz(cache_path, label=1, source_sha256="a" * 64)
    source_path = tmp_path / "data" / "sample.exe"
    source_path.parent.mkdir(exist_ok=True)
    source_path.write_bytes(b"MZsample")
    dataset = MalwareDataset.__new__(MalwareDataset)
    dataset.file_list = [source_path]
    dataset.label_list = [1]
    dataset.cache_path_list = [cache_path]
    dataset.use_cache = True
    dataset.cache_dir = cache_dir
    dataset.max_byte_length = 4
    dataset.pe_feature_dim = 4
    dataset.stat_feature_dim = 2
    dataset.lightweight_feature_dim = 256
    dataset.strict_pe_parsing = True
    dataset.allow_pe_fallback = False
    dataset.pe_schema_version = "legacy_dynamic"
    dataset.pe_fixed_section_slots = 32
    dataset.transform = None
    dataset.target_transform = None

    _sample = dataset[0]

    renamed = cache_dir / "sample-renamed.npz"
    cache_path.rename(renamed)
    assert renamed.exists()
    renamed.unlink()
    assert not renamed.exists()


def _wait_for_dead_processes(pids: list[int], timeout_seconds: float = 10.0) -> list[int]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        alive = []
        for pid in pids:
            if psutil.pid_exists(pid):
                try:
                    process = psutil.Process(pid)
                    if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                        alive.append(pid)
                except psutil.NoSuchProcess:
                    pass
        if not alive:
            return []
        time.sleep(0.1)
    return [pid for pid in pids if psutil.pid_exists(pid)]


def test_npz_dataloader_workers_exit_and_release_directory(tmp_path):
    split_dir = tmp_path / "npz" / "train"
    split_dir.mkdir(parents=True)
    for index in range(4):
        np.savez_compressed(
            split_dir / f"sample-{index}.npz",
            byte_sequence=np.array([77, 90, index], dtype=np.uint8),
            pe_features=np.zeros(4, dtype=np.float32),
            stat_features=np.zeros(2, dtype=np.float32),
            label=index % 2,
        )
    dataset = NPZDataset(str(tmp_path / "npz"), split="train", max_byte_length=4, pe_feature_dim=4, stat_feature_dim=2)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        persistent_workers=False,
        pin_memory=False,
    )
    iterator = iter(loader)
    worker_pids = [worker.pid for worker in getattr(iterator, "_workers", []) if worker.pid is not None]

    batches = list(iterator)
    assert len(batches) == 4
    assert worker_pids

    del batches
    del iterator
    del loader
    del dataset
    gc.collect()

    alive = _wait_for_dead_processes(worker_pids)
    assert alive == []

    renamed_dir = tmp_path / "npz-renamed"
    (tmp_path / "npz").rename(renamed_dir)
    assert renamed_dir.exists()
    for npz_path in renamed_dir.rglob("*.npz"):
        unlinked_path = npz_path
        unlinked_path.unlink()
        assert not unlinked_path.exists()


def test_parallel_cache_preparation_backpressure_counts_completed_queue():
    source = Path(__file__).resolve().parents[1] / "src" / "dataset.py"
    body = source.read_text(encoding="utf-8")

    assert "(len(pending) + len(completed)) < max_pending" in body


def test_extract_byte_sequence_truncates_oversized_file(tmp_path):
    sample = tmp_path / "large.bin"
    sample.write_bytes(b"MZ123")

    byte_seq, orig_len = extract_byte_sequence(str(sample), max_file_size=4)

    assert byte_seq.tolist() == list(b"MZ12")
    assert orig_len == 5


def test_unknown_root_label_is_not_default_benign(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "unknown.exe").write_bytes(b"MZsample")

    with pytest.raises(ValueError, match="Cannot infer label"):
        MalwareDataset(
            data_dir=str(data_dir),
            max_byte_length=4,
            pe_feature_dim=4,
            stat_feature_dim=2,
            use_cache=False,
        )


def test_extract_command_output_names_include_relative_path_hash(monkeypatch, tmp_path):
    import main as main_script  # noqa: E402

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    (input_dir / "left").mkdir(parents=True)
    (input_dir / "right").mkdir(parents=True)
    (input_dir / "left" / "same.exe").write_bytes(b"MZleft")
    (input_dir / "right" / "same.exe").write_bytes(b"MZright")

    def fake_extract_all_features(file_path, extraction_config, axon_config=None):
        return (
            np.array([77, 90], dtype=np.uint8),
            np.zeros(4, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            2,
        )

    monkeypatch.setattr(main_script, "extract_all_features", fake_extract_all_features)

    main_script.extract_command(
        argparse.Namespace(data_dir=str(input_dir), output_dir=str(output_dir), max_workers=1)
    )

    outputs = sorted(output_dir.glob("*_same.npz"))
    assert len(outputs) == 2
    assert outputs[0].name != outputs[1].name
