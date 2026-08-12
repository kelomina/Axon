from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_kvd_dll_abi import (  # noqa: E402
    EXPECTED_EXPORTS,
    AbiVerificationError,
    parse_dumpbin_exports,
    verify_exports,
    verify_header,
)


def test_header_freezes_existing_cdecl_config_layout_and_exports() -> None:
    verify_header(PROJECT_ROOT / "tools" / "axon_onnx_dll" / "include" / "axon_onnx_predict.h")


def test_dumpbin_parser_requires_exact_names_and_ordinals() -> None:
    fixture = "\n".join(
        f" {ordinal:10d} {ordinal - 1:4X} 0002A000 {name}"
        for ordinal, name in enumerate(EXPECTED_EXPORTS, start=1)
    )
    exports = parse_dumpbin_exports(fixture)

    verify_exports(exports)
    with pytest.raises(AbiVerificationError, match="ordinals"):
        verify_exports((*exports[:-1], (18, "unexpected")))
