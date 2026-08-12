from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
CONTROLLER_PATH = PROJECT_ROOT / "scripts" / "run_loop167_phase_b_controller_v10.py"


def _load_controller():
    specification = importlib.util.spec_from_file_location("loop167_phase_b_controller_v7_test", CONTROLLER_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_v7_controller_import_keeps_numerical_and_raw_modules_unloaded() -> None:
    program = f"""
import importlib.util
import json
import sys
sys.path.insert(0, {str(SRC_DIR)!r})
specification = importlib.util.spec_from_file_location('controller_v10', {str(CONTROLLER_PATH)!r})
module = importlib.util.module_from_spec(specification)
specification.loader.exec_module(module)
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


def test_v7_controller_orders_containment_attestation_lease_then_raw_execution() -> None:
    controller = _load_controller()
    execute_source = inspect.getsource(controller.run_execute)

    bootstrap = execute_source.index("bootstrap_thread_environment_v10()")
    runtime_invocation = execute_source.index("validate_current_runtime_invocation_v10")
    launch_id = execute_source.index("launch_id = _launch_id()")
    build_attestation = execute_source.index("child_attestation = build_child_job_attestation_payload_v10")
    write_attestation = execute_source.index("write_child_job_attestation_v10")
    static_preflight = execute_source.index("static_receipt = validate_static_preflight_v10")
    attested_authorization = execute_source.index("authorization = validate_execution_authorization_v10")
    consume_lease = execute_source.index("consumed_lease = consume_execution_lease_v10")
    verify_lease = execute_source.index("lease = verify_consumed_execution_lease_v10")
    post_lease_preflight = execute_source.index("post_lease_static_receipt = validate_static_preflight_v10")
    raw_root = execute_source.index("raw_root = _resolve_fixed_raw_root_adapter")
    runtime_import = execute_source.index("from loop167_phase_b.runtime_lock_v10 import validate_runtime_lock_v10")
    raw_execution = execute_source.index("receipt = _execute_after_lease")

    assert (
        bootstrap
        < runtime_invocation
        < launch_id
        < build_attestation
        < write_attestation
        < static_preflight
        < attested_authorization
        < consume_lease
        < verify_lease
        < post_lease_preflight
        < raw_root
        < runtime_import
        < raw_execution
    )

    raw_execution_source = inspect.getsource(controller._execute_after_lease)
    assert "from loop167_phase_b.raw_worker import" in raw_execution_source
    assert "from loop167_phase_b.fit_worker import" in raw_execution_source
    assert "from loop167_phase_b.raw_manifest_adapter_v4 import" in raw_execution_source
