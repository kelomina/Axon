"""Loop182 源码闭包验证器（与 Loop179 一致，仅修改扫描路径）。"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .contracts import (
    FORBIDDEN_IMPORT_PATTERNS,
    PHASE0_SOURCE_WHITELIST,
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_LOOP182_DIR: Final[Path] = _PROJECT_ROOT / "src" / "loop182"


@dataclass(frozen=True)
class ClosureViolation:
    """单条闭包违规记录。"""

    file: str
    kind: str
    detail: str


@dataclass(frozen=True)
class ClosureReport:
    """闭包验证报告。"""

    passed: bool
    violations: tuple[ClosureViolation, ...]
    scanned_files: tuple[str, ...]
    manifest: dict[str, str]

    def assert_passed(self) -> None:
        if not self.passed:
            lines = ["Loop182 source closure failed:"]
            for violation in self.violations:
                lines.append(f"  [{violation.kind}] {violation.file}: {violation.detail}")
            raise AssertionError("\n".join(lines))


def _relative_to_project(path: Path) -> str:
    return path.resolve().relative_to(_PROJECT_ROOT).as_posix()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_imports(tree: ast.AST) -> tuple[str, ...]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
    return tuple(modules)


def scan_source_closure(
    *,
    expected_manifest: dict[str, str] | None = None,
) -> ClosureReport:
    violations: list[ClosureViolation] = []
    scanned: list[str] = []
    actual_manifest: dict[str, str] = {}

    actual_files: list[Path] = []
    for path in _LOOP182_DIR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        actual_files.append(path)

    whitelist_set = set(PHASE0_SOURCE_WHITELIST)

    for path in sorted(actual_files):
        rel = _relative_to_project(path)
        scanned.append(rel)
        sha = _file_sha256(path)
        actual_manifest[rel] = sha

        if rel not in whitelist_set:
            violations.append(ClosureViolation(
                file=rel,
                kind="missing_whitelist",
                detail=f"file not in PHASE0_SOURCE_WHITELIST (sha={sha})",
            ))

        if expected_manifest is not None and rel in expected_manifest:
            if expected_manifest[rel] != sha:
                violations.append(ClosureViolation(
                    file=rel,
                    kind="sha_drift",
                    detail=(
                        f"expected {expected_manifest[rel][:16]}... "
                        f"got {sha[:16]}..."
                    ),
                ))

        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            violations.append(ClosureViolation(
                file=rel,
                kind="parse_error",
                detail=str(exc),
            ))
            continue

        imports = _extract_imports(tree)
        for module in imports:
            for forbidden in FORBIDDEN_IMPORT_PATTERNS:
                if module == forbidden or module.startswith(forbidden.rstrip(".") + "."):
                    violations.append(ClosureViolation(
                        file=rel,
                        kind="forbidden_import",
                        detail=f"imports '{module}' matching forbidden pattern '{forbidden}'",
                    ))

    for whitelist_entry in PHASE0_SOURCE_WHITELIST:
        if whitelist_entry not in actual_manifest:
            violations.append(ClosureViolation(
                file=whitelist_entry,
                kind="missing_whitelist",
                detail="whitelisted file does not exist on disk",
            ))

    passed = len(violations) == 0
    return ClosureReport(
        passed=passed,
        violations=tuple(violations),
        scanned_files=tuple(scanned),
        manifest=actual_manifest,
    )


def build_current_manifest() -> dict[str, str]:
    report = scan_source_closure()
    return dict(report.manifest)


def assert_phase0_closure() -> ClosureReport:
    report = scan_source_closure()
    report.assert_passed()
    return report
