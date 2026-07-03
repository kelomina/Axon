import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_strict_noise_audit_focus import build_noise_audit_focus  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_evidence(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "source_path",
        "source_sha256",
        "cache_path",
        "label",
        "split",
        "sample_index",
        "score",
        "prediction",
        "error_type",
        "error_transition",
        "severity_score",
        "pe_schema_version",
        "pe_file_size",
        "pe_log_size",
        "sections_count",
        "section_entropy_max",
        "section_entropy_avg",
        "section_high_entropy_ratio",
        "section_raw_size_cv",
        "long_sections_ratio",
        "short_sections_ratio",
        "executable_sections_ratio",
        "writable_sections_ratio",
        "readable_sections_ratio",
        "rwx_sections_ratio",
        "has_signature",
        "api_network_ratio",
        "api_process_ratio",
        "api_filesystem_ratio",
        "api_registry_ratio",
        "api_crypto_ratio",
        "api_injection_ratio",
        "packer_keyword_hits_count",
        "packer_keyword_hits_ratio",
        "stat_mean_byte",
        "stat_std_byte",
        "stat_count_0x00",
        "stat_count_0xff",
        "stat_count_ascii_printable",
        "stat_byte_entropy",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _base_row(**overrides):
    row = {
        "source_path": "C:/misleading/benign_name.exe",
        "source_sha256": "a" * 64,
        "cache_path": "C:/cache/a.npz",
        "label": "1",
        "split": "val",
        "sample_index": "10",
        "score": "0.01",
        "prediction": "0",
        "error_type": "FN",
        "error_transition": "persistent_error",
        "severity_score": "1.49",
        "pe_schema_version": "fixed_v2",
        "pe_file_size": "100000",
        "pe_log_size": "11.5",
        "sections_count": "4",
        "section_entropy_max": "0.91",
        "section_entropy_avg": "0.60",
        "section_high_entropy_ratio": "0.50",
        "section_raw_size_cv": "1.7",
        "long_sections_ratio": "0.25",
        "short_sections_ratio": "0.25",
        "executable_sections_ratio": "0.5",
        "writable_sections_ratio": "0.5",
        "readable_sections_ratio": "1.0",
        "rwx_sections_ratio": "0.25",
        "has_signature": "0",
        "api_network_ratio": "0.1",
        "api_process_ratio": "0.1",
        "api_filesystem_ratio": "0.1",
        "api_registry_ratio": "0",
        "api_crypto_ratio": "0",
        "api_injection_ratio": "0.1",
        "packer_keyword_hits_count": "1",
        "packer_keyword_hits_ratio": "0.25",
        "stat_mean_byte": "90",
        "stat_std_byte": "80",
        "stat_count_0x00": "3",
        "stat_count_0xff": "2",
        "stat_count_ascii_printable": "200",
        "stat_byte_entropy": "0.82",
    }
    row.update(overrides)
    return row


def test_focus_public_csv_is_blinded_and_private_map_keeps_lookup_fields():
    with _case_dir("strict_noise_focus") as tmp_path:
        evidence = tmp_path / "evidence.csv"
        focus = tmp_path / "focus.csv"
        private_map = tmp_path / "private_map.csv"
        summary = tmp_path / "focus.json"
        _write_evidence(evidence, [_base_row()])

        payload = build_noise_audit_focus(
            evidence_csv=evidence,
            output_focus_csv=focus,
            output_private_map_csv=private_map,
            output_json=summary,
            review_prefix="case",
        )
        public_rows = list(csv.DictReader(focus.open("r", encoding="utf-8-sig", newline="")))
        private_rows = list(csv.DictReader(private_map.open("r", encoding="utf-8-sig", newline="")))

    assert payload["focus_rows"] == 1
    public_header = set(public_rows[0])
    forbidden = {"source_path", "source_sha256", "cache_path", "sample_index", "score", "severity_score"}
    assert public_header.isdisjoint(forbidden)
    assert public_rows[0]["review_focus_id"] == "case_000001"
    assert "high_section_entropy" in public_rows[0]["content_tags"]
    assert "rwx_section_present" in public_rows[0]["content_tags"]
    assert private_rows[0]["source_sha256"] == "a" * 64
    assert private_rows[0]["sample_index"] == "10"


def test_focus_ranking_uses_content_and_error_state_not_misleading_paths():
    with _case_dir("strict_noise_focus_ranking") as tmp_path:
        evidence = tmp_path / "evidence.csv"
        focus = tmp_path / "focus.csv"
        private_map = tmp_path / "private_map.csv"
        summary = tmp_path / "focus.json"
        low_priority_bad_name = _base_row(
            source_path="C:/looks_scary/malware_family_name.exe",
            source_sha256="b" * 64,
            sample_index="2",
            score="0.45",
            severity_score="0.55",
            error_transition="broken_by_calibrator",
            section_entropy_max="0.1",
            section_high_entropy_ratio="0.0",
            section_raw_size_cv="0.1",
            executable_sections_ratio="0.0",
            writable_sections_ratio="0.0",
            rwx_sections_ratio="0.0",
            packer_keyword_hits_count="0",
            packer_keyword_hits_ratio="0",
            stat_byte_entropy="0.5",
        )
        high_priority_boring_name = _base_row(
            source_path="C:/plain/readme.tmp",
            source_sha256="c" * 64,
            sample_index="1",
        )
        _write_evidence(evidence, [low_priority_bad_name, high_priority_boring_name])

        build_noise_audit_focus(
            evidence_csv=evidence,
            output_focus_csv=focus,
            output_private_map_csv=private_map,
            output_json=summary,
            review_prefix="rank",
        )
        public_rows = list(csv.DictReader(focus.open("r", encoding="utf-8-sig", newline="")))
        private_rows = list(csv.DictReader(private_map.open("r", encoding="utf-8-sig", newline="")))

    assert public_rows[0]["review_focus_id"] == "rank_000001"
    assert private_rows[0]["source_sha256"] == "c" * 64
    assert private_rows[0]["source_path"] == "C:/plain/readme.tmp"
