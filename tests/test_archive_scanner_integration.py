import json
import subprocess
import sys

import pytest

from archive_scanner import (
    ArchiveScanOptions,
    HARD_MAX_ARCHIVE_FILES,
    MAX_SCANNER_ERROR_CHARS,
    MAX_SCANNER_OUTPUT_CHARS,
    ScannerProcessResult,
    _run_scanner_process,
    cleanup_scan_temp,
    iter_pe_prediction_targets,
    run_archive_scan,
    validate_scan_report,
)


def _base_report(entries):
    return {
        "version": 1,
        "input": "outer.zip",
        "temp_dir": None,
        "limits": {
            "max_depth": 4,
            "max_files": 4096,
            "max_total_bytes": 1,
            "max_file_bytes": 1,
        },
        "summary": {
            "total_entries": len(entries),
            "candidate_entries": len(entries),
            "blocked_entries": 0,
            "error_entries": 0,
            "total_observed_bytes": 0,
            "root_verdict_policy": "runtime any malicious; training unknown",
        },
        "entries": entries,
    }


def test_validate_scan_report_requires_entries_list():
    with pytest.raises(ValueError, match="entries list"):
        validate_scan_report({"version": 1, "entries": {}})


def test_iter_pe_prediction_targets_filters_to_extracted_pe_files():
    report = _base_report([
        {
            "id": 0,
            "parent_id": None,
            "depth": 0,
            "logical_path": "outer.zip",
            "extracted_path": "outer.zip",
            "kind": "zip",
            "candidate_for_axon": True,
            "archive": True,
            "status": "candidate",
            "training_label_policy": "unknown_training_label",
        },
        {
            "id": 1,
            "parent_id": 0,
            "depth": 1,
            "logical_path": "outer.zip/inner.exe",
            "extracted_path": "tmp/inner.exe",
            "kind": "pe",
            "candidate_for_axon": True,
            "archive": False,
            "status": "candidate",
            "training_label_policy": "unknown_training_label",
        },
        {
            "id": 2,
            "parent_id": 0,
            "depth": 1,
            "logical_path": "outer.zip/readme.txt",
            "extracted_path": "tmp/readme.txt",
            "kind": "other",
            "candidate_for_axon": False,
            "archive": False,
            "status": "scanned",
            "training_label_policy": "not_training_candidate",
        },
    ])

    validate_scan_report(report)
    targets = list(iter_pe_prediction_targets(report))

    assert len(targets) == 1
    assert targets[0]["logical_path"] == "outer.zip/inner.exe"
    assert "unknown_training_label" in targets[0]["training_label_policy"]


def test_cleanup_scan_temp_deletes_trusted_scanner_temp(tmp_path):
    temp_root = tmp_path / "trusted"
    temp_dir = temp_root / "axon-archive-scanner-test"
    temp_dir.mkdir(parents=True)
    (temp_dir / "inner.exe").write_bytes(b"MZ")

    status = cleanup_scan_temp({"temp_dir": str(temp_dir)}, trusted_roots=[temp_root])

    assert status["attempted"] is True
    assert status["deleted"] is True
    assert not temp_dir.exists()


def test_cleanup_scan_temp_refuses_untrusted_path(tmp_path):
    trusted_root = tmp_path / "trusted"
    untrusted_dir = tmp_path / "outside" / "axon-archive-scanner-test"
    trusted_root.mkdir()
    untrusted_dir.mkdir(parents=True)

    status = cleanup_scan_temp({"temp_dir": str(untrusted_dir)}, trusted_roots=[trusted_root])

    assert status["attempted"] is False
    assert status["reason"] == "temp_dir_outside_trusted_roots"
    assert untrusted_dir.exists()


def test_cleanup_scan_temp_does_not_trust_report_supplied_temp_root(tmp_path):
    trusted_root = tmp_path / "trusted"
    untrusted_root = tmp_path / "outside" / "axon-archive-scanner-root-evil"
    untrusted_dir = untrusted_root / "axon-archive-scanner-test"
    trusted_root.mkdir()
    untrusted_dir.mkdir(parents=True)

    status = cleanup_scan_temp(
        {"temp_dir": str(untrusted_dir), "_scanner_temp_root": str(untrusted_root)},
        trusted_roots=[trusted_root],
    )

    assert status["attempted"] is False
    assert status["reason"] == "temp_dir_outside_trusted_roots"
    assert untrusted_dir.exists()


def test_cleanup_scan_temp_reports_rmtree_failure(monkeypatch, tmp_path):
    temp_root = tmp_path / "trusted"
    temp_dir = temp_root / "axon-archive-scanner-test"
    temp_dir.mkdir(parents=True)

    def fail_rmtree(_path):
        raise PermissionError("locked")

    monkeypatch.setattr("archive_scanner.shutil.rmtree", fail_rmtree)

    status = cleanup_scan_temp({"temp_dir": str(temp_dir)}, trusted_roots=[temp_root])

    assert status["attempted"] is True
    assert status["deleted"] is False
    assert "PermissionError" in status["cleanup_error"]


def test_cleanup_scan_temp_deletes_owned_root_when_temp_dir_missing(tmp_path):
    trusted_root = tmp_path / "trusted"
    temp_root = trusted_root / "axon-archive-scanner-root-orphan"
    temp_root.mkdir(parents=True)
    (temp_root / "leftover.bin").write_bytes(b"x")

    status = cleanup_scan_temp({"_scanner_temp_root": str(temp_root)}, trusted_roots=[trusted_root])

    assert status["attempted"] is True
    assert status["temp_root_deleted"] is True
    assert not temp_root.exists()


def test_run_archive_scan_passes_owned_temp_root_and_cleans_success(monkeypatch, tmp_path):
    scanner = tmp_path / "scanner.exe"
    scanner.write_text("fake", encoding="utf-8")
    temp_root = tmp_path / "axon-archive-scanner-root-kept"
    captured = {}

    def fake_mkdtemp(prefix):
        assert prefix.startswith("axon-archive-scanner-root-")
        temp_root.mkdir()
        return str(temp_root)

    def fake_run_scanner_process(command):
        captured["command"] = command
        return ScannerProcessResult(returncode=0, stdout=json.dumps(_base_report([])), stderr="")

    monkeypatch.setattr("archive_scanner.resolve_scanner_binary", lambda _explicit=None: scanner)
    monkeypatch.setattr("archive_scanner.tempfile.mkdtemp", fake_mkdtemp)
    monkeypatch.setattr("archive_scanner._run_scanner_process", fake_run_scanner_process)

    report = run_archive_scan(tmp_path / "sample.zip")

    assert report["version"] == 1
    assert "--temp-root" in captured["command"]
    assert str(temp_root) in captured["command"]
    assert not temp_root.exists()


def test_run_archive_scan_rejects_extracted_pe_outside_owned_temp_root(monkeypatch, tmp_path):
    scanner = tmp_path / "scanner.exe"
    scanner.write_text("fake", encoding="utf-8")
    temp_root = tmp_path / "axon-archive-scanner-root-kept"
    outside_pe = tmp_path / "outside.exe"
    outside_pe.write_bytes(b"MZ")

    def fake_mkdtemp(prefix):
        assert prefix.startswith("axon-archive-scanner-root-")
        temp_root.mkdir()
        return str(temp_root)

    report = _base_report([
        {
            "id": 1,
            "parent_id": 0,
            "depth": 1,
            "logical_path": "outer.zip/inner.exe",
            "extracted_path": str(outside_pe),
            "kind": "pe",
            "candidate_for_axon": True,
            "archive": False,
            "status": "candidate",
            "training_label_policy": "unknown_training_label",
        }
    ])

    monkeypatch.setattr("archive_scanner.resolve_scanner_binary", lambda _explicit=None: scanner)
    monkeypatch.setattr("archive_scanner.tempfile.mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(
        "archive_scanner._run_scanner_process",
        lambda _command: ScannerProcessResult(returncode=0, stdout=json.dumps(report), stderr=""),
    )

    with pytest.raises(ValueError, match="extracted_path"):
        run_archive_scan(tmp_path / "sample.zip")

    assert not temp_root.exists()


def test_run_archive_scan_cleans_owned_temp_root_on_timeout(monkeypatch, tmp_path):
    scanner = tmp_path / "scanner.exe"
    scanner.write_text("fake", encoding="utf-8")
    temp_root = tmp_path / "axon-archive-scanner-root-kept"

    def fake_mkdtemp(prefix):
        assert prefix.startswith("axon-archive-scanner-root-")
        temp_root.mkdir()
        (temp_root / "partial.tmp").write_text("left behind", encoding="utf-8")
        return str(temp_root)

    def fake_run_scanner_process(command):
        raise subprocess.TimeoutExpired(command, timeout=1)

    monkeypatch.setattr("archive_scanner.resolve_scanner_binary", lambda _explicit=None: scanner)
    monkeypatch.setattr("archive_scanner.tempfile.mkdtemp", fake_mkdtemp)
    monkeypatch.setattr("archive_scanner._run_scanner_process", fake_run_scanner_process)

    with pytest.raises(subprocess.TimeoutExpired):
        run_archive_scan(tmp_path / "sample.zip")

    assert not temp_root.exists()


def test_run_archive_scan_keep_temp_defers_owned_root_cleanup(monkeypatch, tmp_path):
    scanner = tmp_path / "scanner.exe"
    scanner.write_text("fake", encoding="utf-8")
    temp_root = tmp_path / "axon-archive-scanner-root-keep"
    temp_dir = temp_root / "axon-archive-scanner-kept"

    def fake_mkdtemp(prefix):
        assert prefix.startswith("axon-archive-scanner-root-")
        temp_dir.mkdir(parents=True)
        (temp_dir / "inner.exe").write_bytes(b"MZ")
        return str(temp_root)

    monkeypatch.setattr("archive_scanner.resolve_scanner_binary", lambda _explicit=None: scanner)
    monkeypatch.setattr("archive_scanner.tempfile.mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(
        "archive_scanner._run_scanner_process",
        lambda _command: ScannerProcessResult(
            returncode=0,
            stdout=json.dumps({**_base_report([]), "temp_dir": str(temp_dir)}),
            stderr="",
        ),
    )

    report = run_archive_scan(tmp_path / "sample.zip", ArchiveScanOptions(keep_temp=True))

    assert report["_scanner_temp_root"] == str(temp_root)
    assert temp_dir.exists()
    status = cleanup_scan_temp(report)
    assert status["deleted"] is True
    assert status["temp_root_deleted"] is True
    assert not temp_root.exists()


def test_run_archive_scan_truncates_failed_scanner_output(monkeypatch, tmp_path):
    scanner = tmp_path / "scanner.exe"
    scanner.write_text("fake", encoding="utf-8")
    huge_stdout = "o" * (MAX_SCANNER_ERROR_CHARS + 100)
    huge_stderr = "e" * (MAX_SCANNER_ERROR_CHARS + 200)

    monkeypatch.setattr("archive_scanner.resolve_scanner_binary", lambda _explicit=None: scanner)
    monkeypatch.setattr(
        "archive_scanner._run_scanner_process",
        lambda _command: ScannerProcessResult(returncode=1, stdout=huge_stdout, stderr=huge_stderr),
    )

    with pytest.raises(RuntimeError) as exc:
        run_archive_scan(tmp_path / "sample.zip")

    message = str(exc.value)
    assert "<truncated" in message
    assert len(message) < (MAX_SCANNER_ERROR_CHARS * 3)


def test_run_archive_scan_rejects_oversized_stdout_before_json_parse(monkeypatch, tmp_path):
    scanner = tmp_path / "scanner.exe"
    scanner.write_text("fake", encoding="utf-8")

    monkeypatch.setattr("archive_scanner.resolve_scanner_binary", lambda _explicit=None: scanner)
    monkeypatch.setattr(
        "archive_scanner._run_scanner_process",
        lambda _command: ScannerProcessResult(
            returncode=0,
            stdout="{" + '"entries":[],' + '"pad":"' + ("x" * MAX_SCANNER_OUTPUT_CHARS) + '"}',
            stderr="",
            stdout_exceeded=True,
        ),
    )

    with pytest.raises(ValueError, match="JSON output exceeded limit"):
        run_archive_scan(tmp_path / "sample.zip")


def test_run_archive_scan_rejects_options_above_python_hard_limit(monkeypatch, tmp_path):
    def fail_if_called(_explicit=None):
        raise AssertionError("scanner binary should not be resolved for invalid options")

    monkeypatch.setattr("archive_scanner.resolve_scanner_binary", fail_if_called)

    with pytest.raises(ValueError, match="max_files"):
        run_archive_scan(
            tmp_path / "sample.zip",
            ArchiveScanOptions(max_files=HARD_MAX_ARCHIVE_FILES + 1),
        )


def test_run_scanner_process_caps_stdout_without_full_capture():
    result = _run_scanner_process([
        sys.executable,
        "-c",
        (
            "import sys; "
            f"sys.stdout.buffer.write(b'x' * ({MAX_SCANNER_OUTPUT_CHARS} + 1)); "
            "sys.stdout.flush()"
        ),
    ])

    assert result.stdout_exceeded is True
    assert len(result.stdout) <= MAX_SCANNER_OUTPUT_CHARS
