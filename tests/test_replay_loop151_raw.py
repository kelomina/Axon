from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import replay_loop151_raw as replay  # noqa: E402


def _write_split(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_path", "source_sha256", "label", "sample_index", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _split_row(raw_path: Path, *, sample_index: object = 0, split: str = "train") -> dict:
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    return {
        "source_path": str(raw_path),
        "source_sha256": digest,
        "label": 0,
        "sample_index": sample_index,
        "split": split,
    }


def _fake_truth() -> tuple[dict, dict]:
    return (
        {
            "schema": replay.TRUTH_MANIFEST_SCHEMA,
            "decision": "artifact_freeze_complete_raw_replay_pending",
            "contract": {"champion_scope": "research_champion"},
        },
        {
            "path": "truth.json",
            "sha256": "a" * 64,
            "required_artifacts_verified": 1,
        },
    )


def _patch_lightweight_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        replay,
        "verify_a1_authorization",
        lambda *_args, **_kwargs: {
            "status": "authorized",
            "allowed_resolved_raw_roots": [],
        },
    )
    monkeypatch.setattr(replay, "verify_truth_manifest", lambda *_args: _fake_truth())
    monkeypatch.setattr(replay, "build_stage_contract", lambda *_args: ([], ["full_dag_open"]))


def test_read_split_samples_rejects_non_train_before_open(tmp_path: Path):
    with pytest.raises(replay.ReplayContractError, match="train-only"):
        replay.read_split_samples(
            tmp_path / "missing.csv",
            requested_split="val",
            max_samples=1,
        )


def test_read_split_samples_rejects_duplicate_identity_and_empty_index(tmp_path: Path):
    raw = tmp_path / "sample.exe"
    raw.write_bytes(b"MZ-duplicate")
    row = _split_row(raw)
    split_csv = tmp_path / "split.csv"
    _write_split(split_csv, [row, row])

    with pytest.raises(replay.ReplayContractError, match="Duplicate"):
        replay.read_split_samples(split_csv, requested_split="train", max_samples=1)

    row["sample_index"] = ""
    _write_split(split_csv, [row])
    with pytest.raises(replay.ReplayContractError, match="Invalid sample_index"):
        replay.read_split_samples(split_csv, requested_split="train", max_samples=1)


def test_verify_sample_file_rejects_sha_mismatch_and_root_escape(tmp_path: Path):
    raw_root = tmp_path / "data"
    raw_root.mkdir()
    raw = raw_root / "sample.exe"
    raw.write_bytes(b"MZ-safe")
    sample = replay.SampleIdentity(str(raw), "0" * 64, 7, 0, "train")

    with pytest.raises(replay.ReplayContractError, match="SHA-256 mismatch"):
        replay.verify_sample_file(sample, raw_root)

    outside = tmp_path / "outside.exe"
    outside.write_bytes(b"MZ-outside")
    outside_sample = replay.SampleIdentity(
        str(outside),
        hashlib.sha256(outside.read_bytes()).hexdigest(),
        8,
        0,
        "train",
    )
    with pytest.raises(replay.ReplayContractError, match="outside allowed root"):
        replay.verify_sample_file(outside_sample, raw_root)


def test_verify_sample_file_requires_explicit_resolved_root_for_directory_link(tmp_path: Path):
    raw_root = tmp_path / "data"
    raw_root.mkdir()
    physical_root = tmp_path / "physical"
    physical_root.mkdir()
    physical = physical_root / "sample.exe"
    physical.write_bytes(b"MZ-linked")
    logical_dir = raw_root / "linked"
    try:
        logical_dir.symlink_to(physical_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory links are unavailable: {exc}")
    logical = logical_dir / physical.name
    sample = replay.SampleIdentity(
        str(logical),
        hashlib.sha256(physical.read_bytes()).hexdigest(),
        9,
        0,
        "train",
    )

    with pytest.raises(replay.ReplayContractError, match="outside authorized roots"):
        replay.verify_sample_file(sample, raw_root)

    runtime_path, record = replay.verify_sample_file(
        sample,
        raw_root,
        allowed_resolved_roots=[physical_root],
    )

    assert runtime_path == physical.resolve()
    assert record["raw_sha256_verified"] is True
    assert record["logical_path"] != record["resolved_path"]


def test_truth_manifest_rejects_artifact_path_traversal(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    manifest = {
        "schema": replay.TRUTH_MANIFEST_SCHEMA,
        "integrity": {"artifact_freeze_complete": True, "blockers": []},
        "artifacts": [
            {
                "name": "escape",
                "required": True,
                "path": "../outside.bin",
                "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest_path = project_root / "truth.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(replay.ReplayContractError, match="escapes project root"):
        replay.verify_truth_manifest(project_root, manifest_path)


def test_a1_frozen_artifact_rejects_project_local_override(tmp_path: Path):
    project_root = tmp_path / "project"
    frozen = project_root / "models" / "frozen.bin"
    alternate = project_root / "models" / "alternate.bin"
    alternate.parent.mkdir(parents=True)
    frozen.write_bytes(b"frozen")
    alternate.write_bytes(b"alternate")

    with pytest.raises(replay.ReplayContractError, match="override is not authorized"):
        replay.resolve_frozen_a1_artifact(
            project_root,
            Path("models/alternate.bin"),
            Path("models/frozen.bin"),
            purpose="test artifact",
        )

    assert (
        replay.resolve_frozen_a1_artifact(
            project_root,
            Path("models/frozen.bin"),
            Path("models/frozen.bin"),
            purpose="test artifact",
        )
        == frozen.resolve()
    )


def test_cli_rejects_receipt_path_outside_project(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    outside = tmp_path / "outside.json"

    with pytest.raises(replay.ReplayContractError, match="Output receipt escapes"):
        replay.main(
            [
                "smoke",
                "--project-root",
                str(project_root),
                "--output-json",
                str(outside),
            ]
        )
    assert not outside.exists()


def test_runtime_uses_verified_private_snapshot_not_reopened_source(tmp_path: Path):
    raw_root = tmp_path / "data"
    raw_root.mkdir()
    raw = raw_root / "sample.exe"
    original = b"MZ-immutable-snapshot"
    raw.write_bytes(original)
    sample = replay.SampleIdentity(
        str(raw),
        hashlib.sha256(original).hexdigest(),
        10,
        0,
        "train",
    )
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()

    runtime_path, record = replay.snapshot_verified_sample(
        sample,
        allowed_raw_root=raw_root,
        allowed_resolved_roots=[],
        snapshot_root=snapshot_root,
    )
    raw.write_bytes(b"MZ-mutated-after-snapshot")

    assert runtime_path.read_bytes() == original
    assert record["snapshot_sha256"] == sample.source_sha256


def test_pickle_sha_guard_runs_before_predict_api_import(monkeypatch, tmp_path: Path):
    model = tmp_path / "model.pkl"
    model.write_bytes(b"not-loaded")
    metadata = tmp_path / "model.metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema": "axon_stage2_model_metadata_v1",
                "model_sha256": replay.file_sha256(model),
                "knn": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "schema": replay.PICKLE_ALLOWLIST_SCHEMA,
                "entries": [
                    {
                        "path": "model.pkl",
                        "sha256": "0" * 64,
                        "metadata_path": "model.metadata.json",
                        "metadata_sha256": replay.file_sha256(metadata),
                        "load_authorized": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sentinel = ModuleType("predict_api")
    sentinel.import_reached = False
    monkeypatch.setitem(sys.modules, "predict_api", sentinel)

    with pytest.raises(replay.ReplayContractError, match="Pickle SHA-256 mismatch"):
        replay.run_python_loop28(
            project_root=tmp_path,
            sample_path=tmp_path / "sample.exe",
            checkpoint_path=tmp_path / "checkpoint.pt",
            stage2_path=model,
            pickle_allowlist_path=allowlist,
        )
    assert sentinel.import_reached is False


def test_pickle_guard_requires_allowlisted_metadata_digest(tmp_path: Path):
    model = tmp_path / "model.pkl"
    model.write_bytes(b"trusted-by-test-only")
    metadata = tmp_path / "model.metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema": "axon_stage2_model_metadata_v1",
                "model_sha256": replay.file_sha256(model),
                "knn": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "schema": replay.PICKLE_ALLOWLIST_SCHEMA,
                "contract": "test",
                "entries": [
                    {
                        "path": "model.pkl",
                        "sha256": replay.file_sha256(model),
                        "metadata_path": "model.metadata.json",
                        "metadata_sha256": "f" * 64,
                        "load_authorized": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(replay.ReplayContractError, match="metadata SHA-256 mismatch"):
        replay.guard_pickle_before_load(tmp_path, model, allowlist)

    payload = json.loads(allowlist.read_text(encoding="utf-8"))
    payload["entries"][0]["metadata_sha256"] = replay.file_sha256(metadata)
    allowlist.write_text(json.dumps(payload), encoding="utf-8")
    record = replay.guard_pickle_before_load(tmp_path, model, allowlist)
    assert record["status"] == "verified_before_unpickle"


def _prediction(probability: float, *, prediction: int = 0) -> dict:
    return {
        "prediction": prediction,
        "prob_malicious": probability,
        "base_model": {"prediction": 0, "prob_malicious": probability},
        "stage2": {"prob_malicious": probability},
    }


def test_native_parity_uses_closed_probability_tolerance_boundary():
    python = _prediction(0.25)
    native = _prediction(0.2500005)
    exact_delta = abs(0.25 - 0.2500005)

    at_boundary = replay.compare_loop28_predictions(
        python,
        native,
        tolerance=exact_delta,
    )
    below_boundary = replay.compare_loop28_predictions(
        python,
        native,
        tolerance=exact_delta / 2,
    )

    assert at_boundary["passed"] is True
    assert below_boundary["passed"] is False
    with pytest.raises(replay.ReplayContractError, match="outside"):
        replay.normalize_loop28_prediction(_prediction(float("nan")))

    with pytest.raises(replay.ReplayContractError, match="frozen"):
        replay.compare_loop28_predictions(python, native, tolerance=float("inf"))


def test_native_parity_receipt_preserves_failed_row(monkeypatch, tmp_path: Path):
    _patch_lightweight_contract(monkeypatch)
    raw_root = tmp_path / "data"
    raw_root.mkdir()
    raw = raw_root / "sample.exe"
    raw.write_bytes(b"MZ-parity")
    split_csv = tmp_path / "split.csv"
    _write_split(split_csv, [_split_row(raw)])

    receipt = replay.build_native_parity_receipt(
        project_root=tmp_path,
        truth_manifest_path=tmp_path / "truth.json",
        split_csv=split_csv,
        allowed_raw_root=raw_root,
        max_samples=1,
        tolerance=1.0e-6,
        python_runner=lambda _path: {"prediction": _prediction(0.25)},
        native_runner=lambda _path, _root: {"prediction": _prediction(0.26)},
    )

    assert receipt["decision"] == "native_loop28_parity_blocked"
    assert receipt["parity"]["reported_count"] == 1
    assert receipt["parity"]["dropped_row_count"] == 0
    assert receipt["samples"][0]["status"] == "parity_failed"


def test_native_parity_rejects_tolerance_override_before_runtimes(tmp_path: Path):
    called = False

    def fail_if_called(*_args):
        nonlocal called
        called = True
        raise AssertionError("runtime called before tolerance rejection")

    with pytest.raises(replay.ReplayContractError, match="frozen tolerance"):
        replay.build_native_parity_receipt(
            project_root=tmp_path,
            truth_manifest_path=tmp_path / "truth.json",
            split_csv=tmp_path / "split.csv",
            allowed_raw_root=tmp_path / "data",
            max_samples=1,
            tolerance=1.0,
            python_runner=fail_if_called,
            native_runner=fail_if_called,
        )
    assert called is False


def test_verify_rejects_authorization_before_stage_artifact_inspection(
    monkeypatch,
    tmp_path: Path,
):
    truth = tmp_path / "truth.json"
    truth.write_text(
        json.dumps(
            {
                "schema": replay.TRUTH_MANIFEST_SCHEMA,
                "decision": "artifact_freeze_complete_raw_replay_pending",
                "contract": {"champion_scope": "research_champion"},
            }
        ),
        encoding="utf-8",
    )
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema": replay.VERIFY_AUTHORIZATION_SCHEMA,
                "authorization_level": "A1_scoped_change",
                "allow_complete_loop151_raw_replay": False,
                "truth_manifest_sha256": replay.file_sha256(truth),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        replay,
        "build_stage_contract",
        lambda *_args: (_ for _ in ()).throw(AssertionError("stage artifacts opened")),
    )
    receipt = replay.build_verify_receipt(
        project_root=tmp_path,
        truth_manifest_path=truth,
        authorization_path=authorization,
    )

    assert receipt["decision"] == "complete_loop151_raw_replay_blocked"
    assert "authorization_level_is_not_A2_or_A3" in receipt["verification"]["blockers"]
    assert all(stage["artifacts"] == [] for stage in receipt["dag"]["stages"])


def test_smoke_rejects_a1_authorization_before_truth_artifact_inspection(
    monkeypatch,
    tmp_path: Path,
):
    raw_root = tmp_path / "data"
    raw_root.mkdir()
    authorization = tmp_path / replay.DEFAULT_RAW_REPLAY_AUTHORIZATION
    authorization.parent.mkdir(parents=True)
    authorization.write_text(
        json.dumps(
            {
                "schema": "axon_roadmap_9997_authorization_v1",
                "loop_id": "p0_raw_replay_001",
                "authorization_level": "A1_scoped_change",
                "allowed_splits": ["train"],
                "allowed_logical_raw_root": "data",
                "max_raw_files": 8,
                "max_native_predictions": 2,
                "allow_train_identity_smoke": True,
                "allow_native_loop28_smoke": False,
                "allow_python_native_loop28_parity": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        replay,
        "verify_truth_manifest",
        lambda *_args: (_ for _ in ()).throw(AssertionError("truth artifacts opened")),
    )

    with pytest.raises(replay.ReplayContractError, match="not authorized"):
        replay.build_smoke_receipt(
            project_root=tmp_path,
            truth_manifest_path=tmp_path / "truth.json",
            split_csv=tmp_path / "split.csv",
            allowed_raw_root=raw_root,
            max_samples=1,
            runtime="native",
        )


def test_native_parity_cannot_exceed_authorized_native_count(tmp_path: Path):
    raw_root = tmp_path / "data"
    raw_root.mkdir()
    authorization = tmp_path / replay.DEFAULT_RAW_REPLAY_AUTHORIZATION
    authorization.parent.mkdir(parents=True)
    authorization.write_text(
        json.dumps(
            {
                "schema": "axon_roadmap_9997_authorization_v1",
                "loop_id": "p0_raw_replay_001",
                "authorization_level": "A1_scoped_change",
                "allowed_splits": ["train"],
                "allowed_logical_raw_root": "data",
                "max_raw_files": 8,
                "max_native_predictions": 2,
                "allow_train_identity_smoke": True,
                "allow_native_loop28_smoke": True,
                "allow_python_native_loop28_parity": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(replay.ReplayContractError, match="native predictions"):
        replay.verify_a1_authorization(
            tmp_path,
            mode="native-parity",
            max_samples=3,
            allowed_raw_root=raw_root,
        )


def test_a1_authorization_binds_exact_logical_raw_root(tmp_path: Path):
    raw_root = tmp_path / "data"
    raw_root.mkdir()
    alternate_root = tmp_path / "alternate-data"
    alternate_root.mkdir()
    authorization = tmp_path / replay.DEFAULT_RAW_REPLAY_AUTHORIZATION
    authorization.parent.mkdir(parents=True)
    authorization.write_text(
        json.dumps(
            {
                "schema": "axon_roadmap_9997_authorization_v1",
                "loop_id": "p0_raw_replay_001",
                "authorization_level": "A1_scoped_change",
                "allowed_splits": ["train"],
                "allowed_logical_raw_root": "data",
                "allowed_resolved_raw_roots": [],
                "max_raw_files": 8,
                "max_native_predictions": 2,
                "allow_train_identity_smoke": True,
                "allow_native_loop28_smoke": True,
                "allow_python_native_loop28_parity": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(replay.ReplayContractError, match="logical raw root"):
        replay.verify_a1_authorization(
            tmp_path,
            mode="identity-smoke",
            max_samples=1,
            allowed_raw_root=alternate_root,
        )

    record = replay.verify_a1_authorization(
        tmp_path,
        mode="identity-smoke",
        max_samples=1,
        allowed_raw_root=raw_root,
    )
    assert Path(record["allowed_logical_raw_root"]) == raw_root.resolve()


def test_a1_authorization_requires_logical_raw_root_field(tmp_path: Path):
    raw_root = tmp_path / "data"
    raw_root.mkdir()
    authorization = tmp_path / replay.DEFAULT_RAW_REPLAY_AUTHORIZATION
    authorization.parent.mkdir(parents=True)
    authorization.write_text(
        json.dumps(
            {
                "schema": "axon_roadmap_9997_authorization_v1",
                "loop_id": "p0_raw_replay_001",
                "authorization_level": "A1_scoped_change",
                "allowed_splits": ["train"],
                "max_raw_files": 8,
                "max_native_predictions": 2,
                "allow_train_identity_smoke": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(replay.ReplayContractError, match="no logical raw root"):
        replay.verify_a1_authorization(
            tmp_path,
            mode="identity-smoke",
            max_samples=1,
            allowed_raw_root=raw_root,
        )


def test_smoke_receipt_preserves_sha_failure_without_dropping_row(monkeypatch, tmp_path: Path):
    _patch_lightweight_contract(monkeypatch)
    raw_root = tmp_path / "data"
    raw_root.mkdir()
    raw = raw_root / "sample.exe"
    raw.write_bytes(b"MZ-before-mutation")
    row = _split_row(raw)
    split_csv = tmp_path / "split.csv"
    _write_split(split_csv, [row])
    raw.write_bytes(b"MZ-after-mutation")

    receipt = replay.build_smoke_receipt(
        project_root=tmp_path,
        truth_manifest_path=tmp_path / "truth.json",
        split_csv=split_csv,
        allowed_raw_root=raw_root,
        max_samples=1,
        runtime="identity",
    )

    assert receipt["decision"] == "train_smoke_blocked"
    assert receipt["execution"]["reported_count"] == 1
    assert receipt["execution"]["dropped_row_count"] == 0
    assert receipt["samples"][0]["status"] == "blocked"
