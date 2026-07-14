from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def test_package_import_defers_raw_context_and_external_runtime_modules() -> None:
    program = f"""
import json
import sys
sys.path.insert(0, {str(SRC_DIR)!r})
import loop167_phase_b
before = {{
    'raw_context': 'loop167_phase_b.raw_context' in sys.modules,
    'pefile': 'pefile' in sys.modules,
    'numpy': 'numpy' in sys.modules,
}}
context = loop167_phase_b.RawFeatureContext
after = {{
    'raw_context': 'loop167_phase_b.raw_context' in sys.modules,
    'context_name': context.__name__,
}}
print(json.dumps({{'before': before, 'after': after}}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)

    assert observed["before"] == {"numpy": False, "pefile": False, "raw_context": False}
    assert observed["after"] == {"context_name": "RawFeatureContext", "raw_context": True}
