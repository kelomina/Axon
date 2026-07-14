from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
CONTROLLER_PATH = PROJECT_ROOT / "scripts" / "run_loop167_phase_b_controller_v4.py"


def _load_controller():
    spec = importlib.util.spec_from_file_location("loop167_phase_b_controller_v4_test", CONTROLLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_controller_import_keeps_numerical_and_raw_modules_unloaded() -> None:
    program = f"""
import importlib.util
import json
import sys
sys.path.insert(0, {str(SRC_DIR)!r})
spec = importlib.util.spec_from_file_location('controller_v4', {str(CONTROLLER_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps({{
    'numpy': 'numpy' in sys.modules,
    'pefile': 'pefile' in sys.modules,
    'raw_worker': 'loop167_phase_b.raw_worker' in sys.modules,
    'fit_worker': 'loop167_phase_b.fit_worker' in sys.modules,
}}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "fit_worker": False,
        "numpy": False,
        "pefile": False,
        "raw_worker": False,
    }


def test_controller_orders_authorization_root_job_lease_then_raw_execution() -> None:
    controller = _load_controller()
    execute_source = inspect.getsource(controller.run_execute)
    assert execute_source.index("validate_execution_authorization_v4") < execute_source.index(
        "_resolve_fixed_raw_root_adapter"
    )
    assert execute_source.index("_resolve_fixed_raw_root_adapter") < execute_source.index(
        "_assign_current_process_to_job"
    )
    assert execute_source.index("_assign_current_process_to_job") < execute_source.index(
        "consumed_lease = consume_execution_lease_v4"
    )
    assert execute_source.index("consumed_lease = consume_execution_lease_v4") < execute_source.index(
        "lease = verify_consumed_execution_lease_v4"
    )
    assert execute_source.index("lease = verify_consumed_execution_lease_v4") < execute_source.index(
        "_execute_after_lease"
    )

    raw_execution_source = inspect.getsource(controller._execute_after_lease)
    assert "from loop167_phase_b.raw_worker import" in raw_execution_source
    assert "from loop167_phase_b.fit_worker import" in raw_execution_source
