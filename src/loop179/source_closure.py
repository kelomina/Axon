"""Loop179 源码闭包验证器。

在 Phase 0 和 Phase A preflight 时调用，确保：
1. src/loop179/ 下所有 .py 文件都在 PHASE0_SOURCE_WHITELIST 内。
2. 没有任何文件导入 FORBIDDEN_IMPORT_PATTERNS 中的符号。
3. 文件 SHA256 与冻结 manifest 一致（如果提供了 manifest）。

不访问真实数据，不导入 torch，只做静态文本和哈希检查。
"""

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

# 项目根目录（src/loop179/source_closure.py 的上上上级）
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_LOOP179_DIR: Final[Path] = _PROJECT_ROOT / "src" / "loop179"


@dataclass(frozen=True)
class ClosureViolation:
    """单条闭包违规记录。"""

    file: str  # 相对项目根的路径
    kind: str  # "missing_whitelist" | "forbidden_import" | "sha_drift" | "parse_error"
    detail: str


@dataclass(frozen=True)
class ClosureReport:
    """闭包验证报告。"""

    passed: bool
    violations: tuple[ClosureViolation, ...]
    scanned_files: tuple[str, ...]
    manifest: dict[str, str]  # file -> sha256

    def assert_passed(self) -> None:
        """若未通过，抛出 AssertionError 并列出所有违规。"""

        if not self.passed:
            lines = ["Loop179 source closure failed:"]
            for violation in self.violations:
                lines.append(f"  [{violation.kind}] {violation.file}: {violation.detail}")
            raise AssertionError("\n".join(lines))


def _relative_to_project(path: Path) -> str:
    """返回相对项目根的 posix 路径。"""

    return path.resolve().relative_to(_PROJECT_ROOT).as_posix()


def _file_sha256(path: Path) -> str:
    """计算文件 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_imports(tree: ast.AST) -> tuple[str, ...]:
    """从 AST 中提取所有 import 的模块名。"""

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
    """扫描 src/loop179/ 下所有 .py 文件，验证闭包完整性。

    Args:
        expected_manifest: 若提供，则验证每个文件的 SHA256 与 manifest 一致。
                           manifest key 为 posix 相对路径，value 为 sha256 hex。

    Returns:
        ClosureReport: 包含 passed、violations、scanned_files、manifest。
    """

    violations: list[ClosureViolation] = []
    scanned: list[str] = []
    actual_manifest: dict[str, str] = {}

    # 收集 src/loop179/ 下所有 .py 文件（排除 __pycache__）
    actual_files: list[Path] = []
    for path in _LOOP179_DIR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        actual_files.append(path)

    whitelist_set = set(PHASE0_SOURCE_WHITELIST)

    for path in sorted(actual_files):
        rel = _relative_to_project(path)
        scanned.append(rel)
        sha = _file_sha256(path)
        actual_manifest[rel] = sha

        # 检查白名单
        if rel not in whitelist_set:
            violations.append(ClosureViolation(
                file=rel,
                kind="missing_whitelist",
                detail=f"file not in PHASE0_SOURCE_WHITELIST (sha={sha})",
            ))

        # 检查 SHA256 drift
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

        # 解析 AST 检查禁止导入
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

    # 检查白名单中有但实际不存在的文件
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
    """构建当前 src/loop179/ 的 SHA256 manifest，用于冻结和后续 drift 检测。"""

    report = scan_source_closure()
    return dict(report.manifest)


def assert_phase0_closure() -> ClosureReport:
    """Phase 0 专用闭包断言，必须通过才能进入 Phase A。"""

    report = scan_source_closure()
    report.assert_passed()
    return report
