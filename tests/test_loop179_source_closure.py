"""Loop179 source_closure 源码闭包验证测试。"""

from __future__ import annotations

import pytest

from src.loop179.source_closure import (
    assert_phase0_closure,
    build_current_manifest,
    scan_source_closure,
)


def test_phase0_closure_passes_on_current_tree() -> None:
    """当前 src/loop179/ 树必须通过 Phase 0 闭包。"""

    report = scan_source_closure()
    if not report.passed:
        for violation in report.violations:
            print(f"  [{violation.kind}] {violation.file}: {violation.detail}")
    assert report.passed, "Phase 0 source closure must pass on current tree"


def test_assert_phase0_closure_does_not_raise() -> None:
    """assert_phase0_closure 必须不抛异常。"""

    report = assert_phase0_closure()
    assert report.passed


def test_closure_detects_sha_drift(tmp_path) -> None:
    """闭包必须检测到 SHA drift（通过 expected manifest）。"""

    # 构建一个错误的 manifest，所有 SHA 都设为全 0
    report = scan_source_closure()
    fake_manifest = {path: "0" * 64 for path in report.manifest}
    drift_report = scan_source_closure(expected_manifest=fake_manifest)
    assert not drift_report.passed
    # 每个文件都应该报 sha_drift
    drift_kinds = {v.kind for v in drift_report.violations}
    assert "sha_drift" in drift_kinds


def test_closure_scans_seven_files() -> None:
    """闭包必须扫描到 7 个 Phase 0 文件。"""

    report = scan_source_closure()
    assert len(report.scanned_files) == 7


def test_build_current_manifest_returns_sha256_hex() -> None:
    """build_current_manifest 必须返回 64 字符 hex SHA256。"""

    manifest = build_current_manifest()
    assert len(manifest) == 7
    for path, sha in manifest.items():
        assert len(sha) == 64, f"{path} sha256 must be 64 hex chars"
        int(sha, 16)  # 必须是合法 hex


def test_closure_rejects_missing_whitelist_file(tmp_path, monkeypatch) -> None:
    """白名单中的文件若不存在，闭包必须报 missing_whitelist。"""

    # monkeypatch source_closure 模块中的 PHASE0_SOURCE_WHITELIST 引用
    from src.loop179 import source_closure as closure_module

    original = closure_module.PHASE0_SOURCE_WHITELIST
    fake_whitelist = original + ("src/loop179/nonexistent.py",)
    monkeypatch.setattr(closure_module, "PHASE0_SOURCE_WHITELIST", fake_whitelist)

    report = scan_source_closure()
    assert not report.passed
    kinds = {v.kind for v in report.violations}
    assert "missing_whitelist" in kinds
