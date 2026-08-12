from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.loop151_runtime.raw_runtime import (
    FieldAblationPrediction,
    FieldAblationResult,
    FieldAblationScores,
    build_field_ablation_prediction,
    load_loop28_stage2,
)
from run_loop151_field_ablation import main, read_manifest, summarize_records


def test_build_field_ablation_prediction_preserves_frozen_stage_order() -> None:
    result = build_field_ablation_prediction(
        loop28_prediction=1,
        loop28_probability=0.8,
        primary_prediction=1,
        primary_probability=0.7,
        loop130_prediction=0,
        loop136_prediction=1,
        final_prediction=0,
        conservative_probability=0.2,
        content_cross_probability=0.3,
        noise_probability=0.9,
        selector_score=0.95,
        r5_flip=True,
        signer_downgraded=True,
    )

    assert isinstance(result, FieldAblationPrediction)
    assert result.arm_predictions == {"A": 1, "B": 1, "C": 0, "D": 1, "E": 0}
    assert result.r5_flip is True
    assert result.signer_downgraded is True


def test_build_field_ablation_prediction_rejects_non_binary_predictions() -> None:
    with pytest.raises(ValueError, match="binary"):
        build_field_ablation_prediction(
            loop28_prediction=2,
            loop28_probability=0.8,
            primary_prediction=1,
            primary_probability=0.7,
            loop130_prediction=0,
            loop136_prediction=1,
            final_prediction=0,
            conservative_probability=0.2,
            content_cross_probability=0.3,
            noise_probability=0.9,
            selector_score=None,
            r5_flip=True,
            signer_downgraded=True,
        )


def test_summarize_records_reports_each_arm_and_adjacent_repairs_breaks() -> None:
    records = [
        {"ok": True, "label": 1, "latency_ms": 10.0, "arms": {"A": 1, "B": 0, "C": 1, "D": 1, "E": 0}},
        {"ok": True, "label": 0, "latency_ms": 20.0, "arms": {"A": 1, "B": 0, "C": 0, "D": 1, "E": 0}},
        {"ok": False, "label": 1, "error": "failed"},
    ]

    summary = summarize_records(records)

    assert summary["rows"] == 3
    assert summary["successful_rows"] == 2
    assert summary["failed_rows"] == 1
    assert summary["arms"]["A"]["tp"] == 1
    assert summary["arms"]["A"]["fp"] == 1
    assert summary["arms"]["B"]["fn"] == 1
    assert summary["arms"]["B"]["tn"] == 1
    assert summary["transitions"]["A->B"] == {"repairs": 1, "breaks": 1, "unchanged": 0}
    assert summary["transitions"]["D->E"] == {"repairs": 1, "breaks": 1, "unchanged": 0}


def test_summarize_records_rejects_missing_arm() -> None:
    with pytest.raises(ValueError, match="A/B/C/D/E"):
        summarize_records([{"ok": True, "label": 0, "latency_ms": 1.0, "arms": {"A": 0}}])


def test_load_loop28_stage2_uses_frozen_metadata_and_threshold() -> None:
    bundle = load_loop28_stage2()

    assert bundle.threshold == 0.5
    assert bundle.feature_config.include_content_pe is True
    assert bundle.feature_config.include_content_pe_v2 is False


def test_load_loop28_stage2_rejects_metadata_threshold_override(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = project_root / "manifests/roadmap_9997/p0_raw_replay/loop28_stage2.metadata.json"
    payload = __import__("json").loads(source.read_text(encoding="utf-8"))
    payload["threshold"] = 0.4
    metadata = tmp_path / "loop28.metadata.json"
    metadata.write_text(__import__("json").dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="threshold"):
        load_loop28_stage2(metadata)


def test_read_manifest_verifies_sha_and_rejects_duplicate_identity(tmp_path: Path) -> None:
    sample = tmp_path / "sample.exe"
    sample.write_bytes(b"MZ-ablation")
    digest = __import__("hashlib").sha256(sample.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        f"sample_id,path,sha256,label\nrow-1,{sample},{digest},1\n",
        encoding="utf-8",
    )

    rows = read_manifest(manifest)

    assert rows == [{"sample_id": "row-1", "path": str(sample.resolve()), "sha256": digest, "label": 1}]
    manifest.write_text(
        f"sample_id,path,sha256,label\nrow-1,{sample},{digest},1\nrow-1,{sample},{digest},1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(manifest)


def test_read_manifest_rejects_sha_mismatch(tmp_path: Path) -> None:
    sample = tmp_path / "sample.exe"
    sample.write_bytes(b"MZ-ablation")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        f"sample_id,path,sha256,label\nrow-1,{sample},{'0' * 64},0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_manifest(manifest)


def test_field_ablation_prediction_from_result_round_trips_correctly() -> None:
    result = FieldAblationResult(
        loop28_score=0.85, arm_a=1, arm_b=1, arm_c=0, arm_d=1, arm_e=0,
        scores=FieldAblationScores(
            loop28_probability=0.85, primary_probability=0.7,
            conservative_probability=0.2, content_cross_probability=0.3,
            noise_probability=0.9, selector_score=0.95,
        ),
    )
    pred = FieldAblationPrediction.from_result(result)
    assert pred.arm_predictions == {"A": 1, "B": 1, "C": 0, "D": 1, "E": 0}
    assert pred.r5_flip is True
    assert pred.signer_downgraded is True
    assert pred.loop28_probability == 0.85
    assert pred.primary_probability == 0.7
    assert pred.selector_score == 0.95


def test_cli_rejects_output_outside_project(tmp_path: Path) -> None:
    exit_code = main(["--manifest", str(tmp_path / "missing.csv"), "--output", str(tmp_path / "out.json")])
    assert exit_code != 0


def test_cli_rejects_invalid_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.csv"
    manifest.write_text("sample_id,path,sha256,label\nid1,not_exist.exe,0000000000000000000000000000000000000000000000000000000000000000,0\n", encoding="utf-8")
    exit_code = main(["--manifest", str(manifest), "--output", str(tmp_path / "out.json")])
    assert exit_code != 0
