import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_strict_error_content_evidence_package import build_evidence_package  # noqa: E402
from kvd_features.schema_names import fixed_v2_feature_names  # noqa: E402


SHA_A = "a" * 64
SHA_B = "b" * 64


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_cache(path: Path, *, label: int, sha: str, pe: np.ndarray | None = None, stat: np.ndarray | None = None) -> None:
    pe = np.zeros(256, dtype=np.float32) if pe is None else pe
    stat = np.zeros(49, dtype=np.float32) if stat is None else stat
    np.savez(path, pe_features=pe, stat_features=stat, label=np.asarray(label), source_sha256=np.asarray(sha))


def _legacy_pe(*, overlay_high: float = 0.0) -> np.ndarray:
    pe = np.zeros(256, dtype=np.float32)
    pe[2] = 0.7
    pe[8] = 5
    pe[77] = 10.0
    pe[79] = 1.0
    pe[80] = 0.9
    pe[81] = overlay_high
    return pe


def _fixed_v2_pe() -> np.ndarray:
    names = fixed_v2_feature_names(section_slots=32, pe_feature_dim=256)
    columns = {name: index for index, name in enumerate(names)}
    pe = np.zeros(256, dtype=np.float32)
    pe[columns["fixed_v2_file_size"]] = 123456.0
    pe[columns["fixed_v2_log_size"]] = 11.72
    pe[columns["fixed_v2_has_signature"]] = 1.0
    pe[columns["fixed_v2_sections_count"]] = 2.0
    pe[columns["fixed_v2_section_00_is_executable"]] = 1.0
    pe[columns["fixed_v2_section_00_is_writable"]] = 1.0
    pe[columns["fixed_v2_section_00_is_readable"]] = 1.0
    pe[columns["fixed_v2_section_01_is_readable"]] = 1.0
    pe[columns["fixed_v2_section_entropy_max"]] = 0.91
    pe[columns["fixed_v2_section_entropy_avg"]] = 0.62
    pe[columns["fixed_v2_section_high_entropy_ratio"]] = 0.5
    pe[columns["fixed_v2_api_network_ratio"]] = 0.25
    pe[columns["fixed_v2_packer_keyword_hits_ratio"]] = 0.5
    return pe


def _stat_features() -> np.ndarray:
    stat = np.zeros(49, dtype=np.float32)
    stat[0] = 120.0
    stat[1] = 42.0
    stat[7] = 3.0
    stat[10] = 77.0
    stat[11] = 0.6
    return stat


def _write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_path",
                "source_sha256",
                "cache_path",
                "label",
                "split",
                "sample_index",
                "calibrated_prob_malicious",
                "calibrated_prediction",
                "calibrated_correct",
                "error_transition",
            ],
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_build_evidence_package_uses_legacy_content_features_and_keeps_lookup_fields():
    with _case_dir("strict_error_evidence") as tmp_path:
        cache_a = tmp_path / "a.npz"
        cache_b = tmp_path / "b.npz"
        predictions = tmp_path / "predictions.csv"
        output_csv = tmp_path / "evidence.csv"
        output_json = tmp_path / "evidence.json"
        _write_cache(cache_a, label=0, sha=SHA_A, pe=_legacy_pe(overlay_high=1.0), stat=_stat_features())
        _write_cache(cache_b, label=1, sha=SHA_B, pe=_legacy_pe(overlay_high=0.0), stat=_stat_features())
        _write_predictions(
            predictions,
            [
                {
                    "source_path": "looks-benign.exe",
                    "source_sha256": SHA_A,
                    "cache_path": str(cache_a),
                    "label": "0",
                    "split": "val",
                    "sample_index": "1",
                    "calibrated_prob_malicious": "0.95",
                    "calibrated_prediction": "1",
                    "calibrated_correct": "False",
                    "error_transition": "persistent_error",
                },
                {
                    "source_path": "looks-malicious.exe",
                    "source_sha256": SHA_B,
                    "cache_path": str(cache_b),
                    "label": "1",
                    "split": "val",
                    "sample_index": "2",
                    "calibrated_prob_malicious": "0.90",
                    "calibrated_prediction": "1",
                    "calibrated_correct": "True",
                    "error_transition": "fixed_by_calibrator",
                },
            ],
        )

        payload = build_evidence_package(
            predictions_csv=predictions,
            output_csv=output_csv,
            output_json=output_json,
            score_prefix="calibrated",
            pe_schema_version="legacy_dynamic",
        )
        rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8-sig", newline="")))

    assert payload["evidence_rows"] == 1
    assert payload["error_counts"] == {"FP": 1}
    assert rows[0]["source_path"] == "looks-benign.exe"
    assert float(rows[0]["overlay_high_entropy_flag"]) == 1.0
    assert abs(float(rows[0]["stat_byte_entropy"]) - 0.6) < 1e-6
    assert "directory" not in rows[0]
    assert "extension" not in rows[0]


def test_build_evidence_package_resolves_fixed_v2_schema_by_feature_names():
    with _case_dir("strict_error_evidence_fixed_v2") as tmp_path:
        cache_a = tmp_path / "a.npz"
        predictions = tmp_path / "predictions.csv"
        output_csv = tmp_path / "evidence.csv"
        output_json = tmp_path / "evidence.json"
        _write_cache(cache_a, label=0, sha=SHA_A, pe=_fixed_v2_pe(), stat=_stat_features())
        _write_predictions(
            predictions,
            [
                {
                    "source_path": "misleading-name.exe",
                    "source_sha256": SHA_A,
                    "cache_path": str(cache_a),
                    "label": "0",
                    "split": "val",
                    "sample_index": "1",
                    "calibrated_prob_malicious": "0.95",
                    "calibrated_prediction": "1",
                    "calibrated_correct": "False",
                    "error_transition": "persistent_error",
                }
            ],
        )

        payload = build_evidence_package(
            predictions_csv=predictions,
            output_csv=output_csv,
            output_json=output_json,
            score_prefix="calibrated",
            pe_schema_version="fixed_v2",
        )
        rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8-sig", newline="")))

    assert payload["pe_schema_counts"] == {"fixed_v2": 1}
    assert rows[0]["pe_schema_version"] == "fixed_v2"
    assert float(rows[0]["pe_file_size"]) == 123456.0
    assert abs(float(rows[0]["section_entropy_max"]) - 0.91) < 1e-6
    assert abs(float(rows[0]["section_high_entropy_ratio"]) - 0.5) < 1e-6
    assert abs(float(rows[0]["api_network_ratio"]) - 0.25) < 1e-6
    assert abs(float(rows[0]["packer_keyword_hits_ratio"]) - 0.5) < 1e-6
    assert float(rows[0]["rwx_sections_ratio"]) == 0.5
    assert rows[0]["overlay_high_entropy_flag"] == ""


def test_build_evidence_package_rejects_cache_sha_mismatch():
    with _case_dir("strict_error_evidence_sha_mismatch") as tmp_path:
        cache_a = tmp_path / "a.npz"
        predictions = tmp_path / "predictions.csv"
        _write_cache(cache_a, label=0, sha=SHA_B)
        _write_predictions(
            predictions,
            [
                {
                    "source_path": "a.exe",
                    "source_sha256": SHA_A,
                    "cache_path": str(cache_a),
                    "label": "0",
                    "split": "val",
                    "sample_index": "1",
                    "calibrated_prob_malicious": "0.95",
                    "calibrated_prediction": "1",
                    "calibrated_correct": "False",
                    "error_transition": "persistent_error",
                }
            ],
        )

        try:
            build_evidence_package(
                predictions_csv=predictions,
                output_csv=tmp_path / "evidence.csv",
                output_json=tmp_path / "evidence.json",
                score_prefix="calibrated",
            )
        except ValueError as exc:
            message = str(exc)
        else:
            message = ""

    assert "source_sha256 mismatch" in message
