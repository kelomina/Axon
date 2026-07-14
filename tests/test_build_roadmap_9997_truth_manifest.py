from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_roadmap_9997_truth_manifest as truth_manifest  # noqa: E402


def test_default_artifacts_leave_downstream_diagnostic_out_of_parent_truth():
    artifacts_by_name = {artifact.name: artifact for artifact in truth_manifest.DEFAULT_ARTIFACTS}
    artifact_paths = [artifact.path.as_posix() for artifact in truth_manifest.DEFAULT_ARTIFACTS]

    expected = {
        "native_loop28_runtime_source": (
            "native_product_source",
            "tools/axon_onnx_dll/src/axon_onnx_predict.cpp",
        ),
        "native_loop28_public_header": (
            "native_product_abi",
            "tools/axon_onnx_dll/include/axon_onnx_predict.h",
        ),
        "native_loop28_selftest_source": (
            "native_product_test_source",
            "tools/axon_onnx_dll/examples/axon_onnx_selftest.cpp",
        ),
        "native_loop28_cmake": (
            "native_product_build_definition",
            "tools/axon_onnx_dll/CMakeLists.txt",
        ),
    }
    for name, (role, path) in expected.items():
        artifact = artifacts_by_name[name]
        assert artifact.role == role
        assert artifact.path.as_posix() == path
        assert artifact.required is True

    downstream_paths = {
        "manifests/roadmap_9997/p0_loop28_parity_diagnostic/proposal.json",
        "manifests/roadmap_9997/p0_loop28_parity_diagnostic/authorization.json",
        "manifests/roadmap_9997/p0_loop28_parity_diagnostic/preflight.json",
        "scripts/diagnose_loop28_parity.py",
        "tests/test_diagnose_loop28_parity.py",
        "scripts/build_loop28_parity_diagnostic_manifest.py",
        "tests/test_build_loop28_parity_diagnostic_manifest.py",
    }
    assert downstream_paths.isdisjoint(artifact_paths)
    forbidden_run_outputs = {
        "manifests/roadmap_9997/p0_loop28_parity_diagnostic/run_authorization.json",
        "manifests/roadmap_9997/p0_loop28_parity_diagnostic/run_attempt.final.json",
        "manifests/roadmap_9997/p0_loop28_parity_diagnostic/diagnostic_receipt.json",
        "manifests/roadmap_9997/p0_loop28_parity_diagnostic/post_diagnostic_manifest.json",
        "manifests/roadmap_9997/p0_loop28_parity_diagnostic/post_manifest.json",
        "reports/roadmap_9997/p0_loop28_parity_diagnostic/diagnostic_receipt.json",
    }
    assert forbidden_run_outputs.isdisjoint(artifact_paths)
    assert len(artifact_paths) == len(set(artifact_paths))


@pytest.mark.parametrize(
    "path",
    [
        "manifests/roadmap_9997/p0_loop28_parity_diagnostic/run_authorization.json",
        "reports/roadmap_9997/p0_loop28_parity_diagnostic/diagnostic_receipt.provisional.json",
        "reports/roadmap_9997/p0_loop28_parity_diagnostic/diagnostic_receipt.final.json",
        "reports/roadmap_9997/p0_loop28_parity_diagnostic/diagnostic_receipt.json",
    ],
)
def test_supplemental_artifacts_cannot_reinject_diagnostic_run_outputs(
    tmp_path: Path,
    path: str,
):
    with pytest.raises(ValueError, match="generated authorization/output path"):
        truth_manifest._reject_cyclic_supplemental_artifacts(
            tmp_path,
            Path("truth.json"),
            {"cycle": Path(path)},
        )


def test_supplemental_artifacts_cannot_include_manifest_output(tmp_path: Path):
    with pytest.raises(ValueError, match="generated authorization/output path"):
        truth_manifest._reject_cyclic_supplemental_artifacts(
            tmp_path,
            Path("truth.json"),
            {"self": Path("truth.json")},
        )


def _metrics(*, rows: int, false_positive: int, false_negative: int) -> dict:
    positive = rows // 2
    negative = rows - positive
    true_positive = positive - false_negative
    true_negative = negative - false_positive
    return {
        "accuracy": (true_positive + true_negative) / rows,
        "precision": true_positive / (true_positive + false_positive),
        "recall": true_positive / (true_positive + false_negative),
        "f1": 2 * true_positive / (2 * true_positive + false_positive + false_negative),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "errors": false_positive + false_negative,
    }


def _write_metric_report(
    root: Path,
    split_name: str,
    *,
    rows: int,
    false_positive: int,
    false_negative: int,
    threshold: float = 1.0,
) -> Path:
    lineage_paths = {
        "predictions_csv": Path("inputs") / f"{split_name}.csv",
        "signature_csv": Path("signatures") / f"{split_name}.csv",
        "output_predictions_csv": Path("outputs") / f"{split_name}.csv",
    }
    for path in (lineage_paths["predictions_csv"], lineage_paths["signature_csv"]):
        resolved = root / path
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(f"{split_name}\n", encoding="utf-8")

    candidate = _metrics(
        rows=rows,
        false_positive=false_positive,
        false_negative=false_negative,
    )
    output_predictions = root / lineage_paths["output_predictions_csv"]
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    with output_predictions.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_sha256",
                "sample_index",
                "label",
                "trusted_signer_guard_prediction",
            ],
        )
        writer.writeheader()
        confusion_rows = (
            (candidate["true_positive"], 1, 1),
            (candidate["false_negative"], 1, 0),
            (candidate["true_negative"], 0, 0),
            (candidate["false_positive"], 0, 1),
        )
        sample_index = 0
        for count, label, prediction in confusion_rows:
            for _ in range(count):
                writer.writerow(
                    {
                        "source_sha256": f"{sample_index + 1:064x}",
                        "sample_index": sample_index,
                        "label": label,
                        "trusted_signer_guard_prediction": prediction,
                    }
                )
                sample_index += 1
    payload = {
        "schema": "axon_authenticode_trusted_signer_guard_v1",
        "protocol": "frozen policy",
        "identity_feature_policy": "identity fields are audit-only",
        "predictions_csv": lineage_paths["predictions_csv"].as_posix(),
        "signature_csv": lineage_paths["signature_csv"].as_posix(),
        "trusted_terms": ["Example Publisher"],
        "score_column": "score",
        "score_threshold": threshold,
        "baseline": candidate,
        "candidate": candidate,
        "decision": "allow_next_funnel_step",
        "artifacts": {"output_predictions_csv": lineage_paths["output_predictions_csv"].as_posix()},
    }
    path = root / "metrics" / f"{split_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path.relative_to(root)


def _prepare_project(tmp_path: Path, *, missing_input: bool = False, mismatch_policy: bool = False):
    metric_paths = {
        "val": _write_metric_report(tmp_path, "val", rows=20, false_positive=1, false_negative=1),
        "test10k": _write_metric_report(
            tmp_path,
            "test10k",
            rows=10,
            false_positive=1,
            false_negative=0,
            threshold=0.9 if mismatch_policy else 1.0,
        ),
        "legacy_full_test": _write_metric_report(
            tmp_path,
            "legacy_full_test",
            rows=160,
            false_positive=8,
            false_negative=6,
        ),
    }
    if missing_input:
        (tmp_path / "inputs" / "legacy_full_test.csv").unlink()

    for path, value in {
        "config.toml": "seed = 42\n",
        "model.bin": "model",
        "split.csv": "split,label\nval,0\n",
        "evidence.md": "evidence",
    }.items():
        (tmp_path / path).write_text(value, encoding="utf-8")
    status = {
        "recommendations": [
            {
                "id": truth_manifest.CHAMPION_ID,
                "status": "current_strict_best",
                "evidence": ["evidence.md"],
            }
        ]
    }
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")
    artifacts = [
        truth_manifest.ArtifactSpec("status_ledger", "registry", Path("status.json")),
        truth_manifest.ArtifactSpec("config", "config", Path("config.toml")),
        truth_manifest.ArtifactSpec("model", "model", Path("model.bin")),
        truth_manifest.ArtifactSpec("split", "split", Path("split.csv")),
    ]
    return metric_paths, artifacts


def _fake_git_snapshot(_root: Path, output_path: Path) -> dict:
    return {
        "available": True,
        "head": "a" * 40,
        "branch": "test",
        "is_dirty": True,
        "status_entry_count": 1,
        "status_entries": [" M tracked.txt"],
        "status_sha256": "b" * 64,
        "tracked_patch_sha256": "c" * 64,
        "untracked_content_hashed": False,
        "excluded_output_path": output_path.name,
        "errors": [],
    }


def test_truth_manifest_freezes_metrics_hashes_and_target_gap(monkeypatch, tmp_path: Path):
    metric_paths, artifacts = _prepare_project(tmp_path)
    monkeypatch.setattr(truth_manifest, "_git_snapshot", _fake_git_snapshot)
    output = tmp_path / "truth.json"

    payload = truth_manifest.build_truth_manifest(
        project_root=tmp_path,
        output_json=output,
        metric_report_paths=metric_paths,
        artifact_specs=artifacts,
        invocation=["python", "build_truth_manifest.py"],
        expected_split_rows={"val": 20, "test10k": 10, "legacy_full_test": 160},
    )

    assert payload["decision"] == "artifact_freeze_complete_raw_replay_pending"
    assert payload["integrity"]["artifact_freeze_complete"] is True
    assert payload["metrics"]["legacy_full_test"]["candidate"]["errors"] == 14
    assert payload["metrics"]["legacy_full_test"]["recomputed_from_predictions"]["row_count"] == 160
    production_gap = truth_manifest._target_gap(
        {"positive_count": 80_000, "errors": 1466, "f1": 0.9908541911012403},
        0.9997,
    )
    assert production_gap["max_total_errors_any_fp_fn_mix"] == 48
    assert production_gap["minimum_errors_to_remove"] == 1418
    assert payload["capability_boundary"]["raw_file_to_report_replay_ready"] is False
    model_record = next(row for row in payload["artifacts"] if row["name"] == "model")
    assert model_record["sha256"] == hashlib.sha256(b"model").hexdigest()
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == payload["schema"]


def test_truth_manifest_blocks_missing_lineage_artifact(monkeypatch, tmp_path: Path):
    metric_paths, artifacts = _prepare_project(tmp_path, missing_input=True)
    monkeypatch.setattr(truth_manifest, "_git_snapshot", _fake_git_snapshot)

    payload = truth_manifest.build_truth_manifest(
        project_root=tmp_path,
        output_json=tmp_path / "truth.json",
        metric_report_paths=metric_paths,
        artifact_specs=artifacts,
        expected_split_rows={"val": 20, "test10k": 10, "legacy_full_test": 160},
    )

    assert payload["decision"] == "artifact_freeze_blocked"
    assert any(
        "legacy_full_test_input_predictions" in row for row in payload["integrity"]["blockers"]
    )


def test_truth_manifest_rejects_policy_drift_between_splits(monkeypatch, tmp_path: Path):
    metric_paths, artifacts = _prepare_project(tmp_path, mismatch_policy=True)
    monkeypatch.setattr(truth_manifest, "_git_snapshot", _fake_git_snapshot)

    payload = truth_manifest.build_truth_manifest(
        project_root=tmp_path,
        output_json=tmp_path / "truth.json",
        metric_report_paths=metric_paths,
        artifact_specs=artifacts,
        expected_split_rows={"val": 20, "test10k": 10, "legacy_full_test": 160},
    )

    assert payload["decision"] == "artifact_freeze_blocked"
    assert "metric_reports_do_not_share_one_frozen_policy" in payload["integrity"]["blockers"]


def test_truth_manifest_rejects_duplicate_prediction_identity(monkeypatch, tmp_path: Path):
    metric_paths, artifacts = _prepare_project(tmp_path)
    output_predictions = tmp_path / "outputs" / "legacy_full_test.csv"
    with output_predictions.open("a", encoding="utf-8", newline="") as handle:
        handle.write(f"{'1':0>64},0,1,1\n")
    monkeypatch.setattr(truth_manifest, "_git_snapshot", _fake_git_snapshot)

    payload = truth_manifest.build_truth_manifest(
        project_root=tmp_path,
        output_json=tmp_path / "truth.json",
        metric_report_paths=metric_paths,
        artifact_specs=artifacts,
        expected_split_rows={"val": 20, "test10k": 10, "legacy_full_test": 160},
    )

    blockers = payload["integrity"]["blockers"]
    assert payload["decision"] == "artifact_freeze_blocked"
    assert "duplicate_prediction_identity_keys:1" in blockers
    assert "legacy_full_test_prediction_row_count:161!=160" in blockers
