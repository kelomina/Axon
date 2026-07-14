from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for search_path in (SRC_DIR, SCRIPTS_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_loop166_phase_b1_full_outer_resource_cell as b1  # noqa: E402
import run_loop166_phase_b1_step4096_recovery as recovery  # noqa: E402


def _canonical_args():
    return recovery._normalize_args(recovery.build_parser().parse_args([]))


def _handoff() -> b1.RunHandoff:
    return b1.RunHandoff(
        authorization_sha256="1" * 64,
        marker_sha256="2" * 64,
        handoff_nonce="ab" * 32,
        parent_pid=1234,
        canonical_parent_argv_sha256="3" * 64,
    )


def _control_closure() -> recovery.RecoveryControlClosure:
    return recovery.RecoveryControlClosure(
        contract={},
        contract_sha256="4" * 64,
        authorization={},
        authorization_sha256="1" * 64,
        runtime={},
    )


def _original_contract() -> dict:
    return json.loads(b1.DEFAULT_CONTRACT.read_text(encoding="utf-8"))


def _recovery_closure() -> recovery.RecoveryClosure:
    original_run_context = {"bound": "original-run-context"}
    return recovery.RecoveryClosure(
        **_control_closure().__dict__,
        original_contract=_original_contract(),
        original_bindings={"bound": {"path": "source", "sha256": "a" * 64}},
        source_payload={"run_context": original_run_context},
    )


def _scan_accounting() -> dict:
    return {
        "fit_metadata_rows": 16000,
        "fit_raw_open_attempts": 15988,
        "fit_raw_open_successes": 15988,
        "fit_raw_bytes_actually_read": 19239582561,
        "source_unavailable": 12,
        "outer_holdout_raw_opens": 0,
        "outer_holdout_raw_bytes": 0,
        "outer_fit_corpus_commitment_sha256": recovery.EXPECTED_COMMITMENTS[
            "outer_fit_corpus_commitment_sha256"
        ],
    }


def _compact_accounting() -> dict:
    return {
        "compact_corpus_commitment_sha256": recovery.EXPECTED_COMMITMENTS[
            "compact_corpus_commitment_sha256"
        ],
        "prepared_sequence_count": 115072,
        "original_window_bytes": recovery.EXPECTED_TOTAL_ORIGINAL_BYTES,
        "prepared_original_bytes": recovery.EXPECTED_TOTAL_ORIGINAL_BYTES,
        "dropped_content_tokens": 0,
        "dropped_original_bytes": 0,
        "overlength_windows_excluded": 0,
    }


def _final_payload(closure: recovery.RecoveryClosure) -> dict:
    scan = _scan_accounting()
    compact = _compact_accounting()
    return {
        "tokenizer_sha256": recovery.EXPECTED_ORIGINAL_HASHES["tokenizer"],
        "outer_fit_corpus_commitment_sha256": recovery.EXPECTED_COMMITMENTS[
            "outer_fit_corpus_commitment_sha256"
        ],
        "compact_corpus_commitment_sha256": recovery.EXPECTED_COMMITMENTS[
            "compact_corpus_commitment_sha256"
        ],
        "shuffle_commitment_sha256": recovery.EXPECTED_COMMITMENTS[
            "shuffle_commitment_sha256"
        ],
        "permutation_prefix_original_bytes": recovery.EXPECTED_TOTAL_ORIGINAL_BYTES,
        "completed_optimizer_steps": 28768,
        "completed_sequence_count": 115072,
        "next_permutation_cursor": 115072,
        "training_state": {
            "training_original_bytes": recovery.EXPECTED_TOTAL_ORIGINAL_BYTES,
        },
        "run_context": {
            "prepared_sequence_count": 115072,
            "total_optimizer_steps": 28768,
            "incident_sha256": recovery.EXPECTED_ORIGINAL_HASHES["incident"],
            "original_run_context": closure.source_payload["run_context"],
            "original_input_bindings": closure.original_bindings,
            "recovery_scan_accounting": scan,
            "recovery_scan_accounting_sha256": recovery._canonical_json_sha256(scan),
            "recovery_compact_accounting": compact,
            "recovery_compact_accounting_sha256": recovery._canonical_json_sha256(
                compact
            ),
        },
    }


def _receipt_resource_state() -> dict:
    return {
        "peak_process_rss_bytes": 1_000_000,
        "peak_cuda_allocated_bytes": 2_000_000,
        "peak_cuda_reserved_bytes": 3_000_000,
        "minimum_free_disk_bytes": 2_000_000_000,
    }


def test_pending_recovery_contract_fails_before_marker_or_raw(
    tmp_path: Path,
    monkeypatch,
):
    contract = json.loads(recovery.DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    contract["status"] = "draft_not_authorized_pending_recovery_source_closure"
    pending_contract = tmp_path / "pending-recovery.json"
    pending_contract.write_text(json.dumps(contract), encoding="utf-8")
    args = _canonical_args()
    args.contract = pending_contract
    raw_calls = 0
    marker_before = (
        recovery.DEFAULT_MARKER.read_bytes() if recovery.DEFAULT_MARKER.exists() else None
    )

    def unexpected_raw(*_args, **_kwargs):
        nonlocal raw_calls
        raw_calls += 1
        raise AssertionError("pending recovery preflight attempted raw access")

    monkeypatch.setattr(b1, "read_verified_outer_fit_source", unexpected_raw)
    with pytest.raises(recovery.RecoveryFatalError, match="contract drifted"):
        recovery.validate_control_preflight(args, mode="parent")

    assert raw_calls == 0
    assert (
        recovery.DEFAULT_MARKER.read_bytes() if recovery.DEFAULT_MARKER.exists() else None
    ) == marker_before


def test_canonical_source_closure_and_origin_checkpoint_preflight_is_pure(monkeypatch):
    args = _canonical_args()
    raw_calls = 0
    marker_before = (
        recovery.DEFAULT_MARKER.read_bytes() if recovery.DEFAULT_MARKER.exists() else None
    )

    def unexpected_raw(*_args, **_kwargs):
        nonlocal raw_calls
        raw_calls += 1
        raise AssertionError("source closure preflight attempted raw access")

    monkeypatch.setattr(b1, "read_verified_outer_fit_source", unexpected_raw)
    contract, contract_raw = recovery._read_json(
        args.contract, "synthetic recovery contract audit"
    )
    recovery._validate_recovery_contract(contract)
    authorization, _authorization_raw = recovery._read_json(
        args.authorization, "synthetic recovery authorization audit"
    )
    runtime = recovery._validate_recovery_authorization(
        args,
        contract,
        hashlib.sha256(contract_raw).hexdigest(),
        authorization,
    )
    original_contract, original_bindings, source_payload = (
        recovery._validate_original_evidence(args, contract)
    )

    assert runtime["python_executable"] == str(Path(sys.executable).resolve(strict=True))
    assert source_payload["completed_optimizer_steps"] == 4096
    assert source_payload["next_permutation_cursor"] == 16384
    assert source_payload["resume_pid"] == 0
    assert source_payload["run_context"]["input_bindings"] == original_bindings
    assert original_contract["data_scope"]["outer_holdout_raw_opens_allowed"] == 0
    assert raw_calls == 0
    assert (
        recovery.DEFAULT_MARKER.read_bytes() if recovery.DEFAULT_MARKER.exists() else None
    ) == marker_before


def test_mode_output_state_is_fail_closed(tmp_path: Path, monkeypatch):
    marker = tmp_path / "recovery-marker.json"
    receipt = tmp_path / "recovery-receipt.json"
    checkpoint = tmp_path / "recovery.pt"
    report = tmp_path / "recovery-report.json"
    args = _canonical_args()
    args.checkpoint_output = checkpoint
    args.report_output = report
    monkeypatch.setattr(recovery, "DEFAULT_MARKER", marker)
    monkeypatch.setattr(recovery, "DEFAULT_FINAL_RECEIPT", receipt)

    recovery._validate_mode_outputs(args, "parent")
    marker.write_text("{}", encoding="utf-8")
    recovery._validate_mode_outputs(args, "recovery")
    with pytest.raises(recovery.RecoveryFatalError, match="to exist"):
        recovery._validate_mode_outputs(args, "final_verify")
    checkpoint.write_bytes(b"checkpoint")
    recovery._validate_mode_outputs(args, "final_verify")
    with pytest.raises(recovery.RecoveryFatalError, match="be absent"):
        recovery._validate_mode_outputs(args, "recovery")


def test_recovery_marker_is_exclusive_and_never_persists_nonce(
    tmp_path: Path,
    monkeypatch,
):
    marker = tmp_path / "recovery-marker.json"
    args = _canonical_args()
    closure = _control_closure()
    nonce = "ab" * 32
    monkeypatch.setattr(recovery, "DEFAULT_MARKER", marker)
    monkeypatch.setattr(recovery.secrets, "token_hex", lambda _size: nonce)
    monkeypatch.setattr(b1, "_fsync_parent_directory", lambda _path: False)

    handoff = recovery.consume_recovery_authorization(
        closure,
        cumulative_wall_seconds_at_consumption=(
            recovery.RECOVERY_BUDGET_BASE_SECONDS + 1.0
        ),
        args=args,
    )
    marker_raw = marker.read_bytes()
    marker_payload = json.loads(marker_raw)

    assert nonce.encode("ascii") not in marker_raw
    assert "new_handoff_nonce" not in marker_payload
    assert marker_payload["new_handoff_nonce_sha256"] == hashlib.sha256(
        nonce.encode("ascii")
    ).hexdigest()
    assert handoff.marker_sha256 == hashlib.sha256(marker_raw).hexdigest()
    assert marker_payload["cumulative_wall_seconds_at_consumption"] == (
        recovery.RECOVERY_BUDGET_BASE_SECONDS + 1.0
    )
    assert recovery._validate_recovery_marker(handoff, closure, args) == marker_payload
    with pytest.raises(recovery.RecoveryFatalError, match="already exists"):
        recovery.consume_recovery_authorization(
            closure,
            cumulative_wall_seconds_at_consumption=(
                recovery.RECOVERY_BUDGET_BASE_SECONDS + 2.0
            ),
            args=args,
        )


def test_recovery_handoff_environment_rejects_invalid_nonce(monkeypatch):
    handoff = _handoff()
    environment = recovery._recovery_environment(
        handoff,
        mode="recovery",
        worker_pid=0,
        cumulative_base_seconds=recovery.RECOVERY_BUDGET_BASE_SECONDS,
    )
    for name, value in environment.items():
        if name.startswith("AXON_B1_RECOVERY_"):
            monkeypatch.setenv(name, value)

    restored, worker_pid, cumulative = recovery._handoff_from_environment("recovery")
    assert restored == handoff
    assert worker_pid == 0
    assert cumulative == recovery.RECOVERY_BUDGET_BASE_SECONDS

    monkeypatch.setenv("AXON_B1_RECOVERY_NONCE", "g" * 64)
    with pytest.raises(recovery.RecoveryFatalError, match="no valid in-memory nonce"):
        recovery._handoff_from_environment("recovery")


def test_worker_lineage_uses_authorization_frozen_executables(monkeypatch):
    observed = {}
    runtime = {
        "python_executable": r"E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe",
        "base_executable": (
            r"C:\Users\Saika\AppData\Local\Python\pythoncore-3.14-64\python.exe"
        ),
    }

    def fake_lineage(expected_parent_pid, *, launcher_executable, base_executable):
        observed.update(
            expected_parent_pid=expected_parent_pid,
            launcher_executable=launcher_executable,
            base_executable=base_executable,
        )
        return {"mode": "windows_venv_redirector"}

    monkeypatch.setattr(recovery, "validate_spawn_lineage", fake_lineage)
    audit = recovery._validate_worker_lineage(1234, runtime)

    assert audit["mode"] == "windows_venv_redirector"
    assert observed == {
        "expected_parent_pid": 1234,
        "launcher_executable": runtime["python_executable"],
        "base_executable": runtime["base_executable"],
    }


def test_final_receipt_requires_exact_independent_recovery_lineage(
    tmp_path: Path,
    monkeypatch,
):
    checkpoint = tmp_path / "recovered.pt"
    checkpoint.write_bytes(b"recovered-checkpoint")
    receipt_path = tmp_path / "recovery-receipt.json"
    args = _canonical_args()
    args.checkpoint_output = checkpoint
    handoff = _handoff()
    closure = _recovery_closure()
    scan_sha256 = recovery._canonical_json_sha256(_scan_accounting())
    compact_sha256 = recovery._canonical_json_sha256(_compact_accounting())
    receipt = {
        "schema": recovery.RECEIPT_SCHEMA,
        "decision": "phase_b1_recovery_final_checkpoint_verified",
        "recovery_authorization_sha256": closure.authorization_sha256,
        "recovery_contract_sha256": closure.contract_sha256,
        "recovery_marker_sha256": handoff.marker_sha256,
        "new_handoff_nonce_sha256": handoff.handoff_nonce_sha256,
        "source_checkpoint_sha256": recovery.EXPECTED_ORIGINAL_HASHES["checkpoint"],
        "original_marker_sha256": recovery.EXPECTED_ORIGINAL_HASHES["marker"],
        "old_handoff_nonce_sha256": recovery.EXPECTED_COMMITMENTS[
            "old_handoff_nonce_sha256"
        ],
        "old_handoff_nonce_possession": False,
        "incident_sha256": recovery.EXPECTED_ORIGINAL_HASHES["incident"],
        "tokenizer_sha256": recovery.EXPECTED_ORIGINAL_HASHES["tokenizer"],
        "outer_fit_corpus_commitment_sha256": recovery.EXPECTED_COMMITMENTS[
            "outer_fit_corpus_commitment_sha256"
        ],
        "compact_corpus_commitment_sha256": recovery.EXPECTED_COMMITMENTS[
            "compact_corpus_commitment_sha256"
        ],
        "shuffle_commitment_sha256": recovery.EXPECTED_COMMITMENTS[
            "shuffle_commitment_sha256"
        ],
        "total_original_bytes": recovery.EXPECTED_TOTAL_ORIGINAL_BYTES,
        "recovery_scan_accounting_sha256": scan_sha256,
        "recovery_compact_accounting_sha256": compact_sha256,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_cumulative_wall_seconds": (
            recovery.RECOVERY_BUDGET_BASE_SECONDS + 5.0
        ),
        "completed_optimizer_steps": 28768,
        "completed_sequence_count": 115072,
        "next_permutation_cursor": 115072,
        "recovery_parent_pid": handoff.parent_pid,
        "recovery_worker_pid": 2345,
        "verifier_pid": 3456,
        "resource_state": _receipt_resource_state(),
        "model_tensors_finite": True,
        "optimizer_tensors_finite": True,
        "rng_state_validated": True,
        "synthetic_logits_bit_exact": True,
        "outer_holdout_raw_opens": 0,
        "outer_holdout_raw_bytes": 0,
        "raw_access_performed": False,
        "quality_metrics_computed": False,
        "threshold_operations_performed": False,
        "cumulative_wall_seconds": recovery.RECOVERY_BUDGET_BASE_SECONDS + 10.0,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(recovery, "DEFAULT_FINAL_RECEIPT", receipt_path)

    assert (
        recovery._load_final_receipt(
            args,
            closure,
            handoff,
            2345,
            not_after_cumulative_wall_seconds=(
                recovery.RECOVERY_BUDGET_BASE_SECONDS + 11.0
            ),
        )
        == receipt
    )
    for name, invalid in (
        ("verifier_pid", 0),
        ("rng_state_validated", False),
        ("raw_access_performed", True),
        ("tokenizer_sha256", "f" * 64),
        ("total_original_bytes", recovery.EXPECTED_TOTAL_ORIGINAL_BYTES - 1),
    ):
        tampered = dict(receipt)
        tampered[name] = invalid
        receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(recovery.RecoveryFatalError, match="receipt drifted"):
            recovery._load_final_receipt(
                args,
                closure,
                handoff,
                2345,
                not_after_cumulative_wall_seconds=(
                    recovery.RECOVERY_BUDGET_BASE_SECONDS + 11.0
                ),
            )

    late = dict(receipt)
    late["cumulative_wall_seconds"] = recovery.RECOVERY_BUDGET_BASE_SECONDS + 12.0
    receipt_path.write_text(json.dumps(late), encoding="utf-8")
    with pytest.raises(recovery.RecoveryFatalError, match="receipt drifted"):
        recovery._load_final_receipt(
            args,
            closure,
            handoff,
            2345,
            not_after_cumulative_wall_seconds=recovery.RECOVERY_BUDGET_BASE_SECONDS + 11.0,
        )


def test_final_checkpoint_commitments_and_ledgers_fail_closed():
    closure = _recovery_closure()
    payload = _final_payload(closure)

    audit = recovery._validate_final_checkpoint_commitments(payload, closure)
    assert audit["total_original_bytes"] == recovery.EXPECTED_TOTAL_ORIGINAL_BYTES
    assert audit["incident_sha256"] == recovery.EXPECTED_ORIGINAL_HASHES["incident"]

    wrong_tokenizer = copy.deepcopy(payload)
    wrong_tokenizer["tokenizer_sha256"] = "f" * 64
    with pytest.raises(recovery.RecoveryFatalError, match="commitment drifted"):
        recovery._validate_final_checkpoint_commitments(wrong_tokenizer, closure)

    wrong_scan = copy.deepcopy(payload)
    wrong_scan["run_context"]["recovery_scan_accounting"][
        "fit_raw_bytes_actually_read"
    ] -= 1
    wrong_scan["run_context"]["recovery_scan_accounting_sha256"] = (
        recovery._canonical_json_sha256(
            wrong_scan["run_context"]["recovery_scan_accounting"]
        )
    )
    with pytest.raises(recovery.RecoveryFatalError, match="scan ledger drifted"):
        recovery._validate_final_checkpoint_commitments(wrong_scan, closure)

    wrong_total = copy.deepcopy(payload)
    wrong_total["permutation_prefix_original_bytes"] -= 1
    with pytest.raises(recovery.RecoveryFatalError, match="commitment drifted"):
        recovery._validate_final_checkpoint_commitments(wrong_total, closure)


def test_final_checkpoint_sha_must_remain_stable_during_verification(tmp_path: Path):
    checkpoint = tmp_path / "recovered.pt"
    checkpoint.write_bytes(b"verified-payload")
    before_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    assert (
        recovery._assert_stable_artifact_sha(
            checkpoint,
            before_sha256,
            "Recovery final checkpoint",
        )
        == before_sha256
    )

    checkpoint.write_bytes(b"replacement-payload")
    with pytest.raises(recovery.RecoveryFatalError, match="changed during verification"):
        recovery._assert_stable_artifact_sha(
            checkpoint,
            before_sha256,
            "Recovery final checkpoint",
        )


def test_final_verifier_resource_peaks_are_merged_and_guarded():
    contract = _original_contract()
    state = {
        "peak_process_rss_bytes": 100,
        "peak_cuda_allocated_bytes": 200,
        "peak_cuda_reserved_bytes": 300,
        "minimum_free_disk_bytes": 3_000_000_000,
    }
    receipt = {"resource_state": _receipt_resource_state()}

    recovery._merge_final_verifier_resources(state, receipt, contract)

    assert state == _receipt_resource_state()
    over_cap = copy.deepcopy(receipt)
    over_cap["resource_state"]["peak_cuda_reserved_bytes"] = contract["resource_gates"][
        "maximum_cuda_reserved_bytes_exclusive"
    ]
    with pytest.raises(recovery.RecoveryFatalError, match="crossed a frozen gate"):
        recovery._merge_final_verifier_resources(state, over_cap, contract)


def test_pre_marker_guard_rechecks_sources_outputs_disk_and_elapsed(monkeypatch):
    args = _canonical_args()
    closure = _recovery_closure()
    calls = []

    monkeypatch.setattr(
        recovery,
        "_assert_control_closure_unchanged",
        lambda *_args: calls.append("control"),
    )
    monkeypatch.setattr(
        recovery,
        "_assert_immutable_originals",
        lambda *_args: calls.append("originals"),
    )
    monkeypatch.setattr(
        recovery,
        "_validate_mode_outputs",
        lambda *_args: calls.append("outputs"),
    )
    monkeypatch.setattr(
        b1,
        "_free_disk_bytes",
        lambda *_args: calls.append("disk") or 2_000_000_000,
    )

    def fake_guard(*_args, **kwargs):
        calls.append("guard")
        assert kwargs["cumulative_wall_seconds"] == (
            recovery.RECOVERY_BUDGET_BASE_SECONDS + 1.0
        )
        return {"process_rss_bytes": 1, "free_disk_bytes": 2_000_000_000}

    monkeypatch.setattr(b1, "guard_non_cuda_phase", fake_guard)
    monkeypatch.setattr(recovery.time, "perf_counter", lambda: 101.0)

    cumulative = recovery._guard_before_marker_consumption(
        args,
        closure,
        parent_started=100.0,
    )

    assert cumulative == recovery.RECOVERY_BUDGET_BASE_SECONDS + 1.0
    assert calls == ["control", "originals", "outputs", "disk", "guard"]


def test_pre_raw_guard_rechecks_control_and_original_closures(monkeypatch):
    args = _canonical_args()
    closure = _recovery_closure()
    state = {"peak_process_rss_bytes": 0}
    calls = []

    monkeypatch.setattr(
        recovery,
        "_assert_control_closure_unchanged",
        lambda *_args: calls.append("control"),
    )
    monkeypatch.setattr(
        recovery,
        "_assert_immutable_originals",
        lambda *_args: calls.append("originals"),
    )

    def fake_guard(*_args, **kwargs):
        calls.append("resource")
        assert kwargs["cumulative_wall_seconds"] == (
            recovery.RECOVERY_BUDGET_BASE_SECONDS + 2.0
        )
        assert kwargs["state"] is state
        return {"process_rss_bytes": 1, "free_disk_bytes": 2_000_000_000}

    monkeypatch.setattr(b1, "guard_non_cuda_phase", fake_guard)

    audit = recovery._guard_before_recovery_raw_access(
        args,
        closure,
        cumulative_wall_seconds=recovery.RECOVERY_BUDGET_BASE_SECONDS + 2.0,
        state=state,
    )

    assert calls == ["control", "originals", "resource"]
    assert audit["free_disk_bytes"] == 2_000_000_000


def test_final_worker_validates_control_marker_lineage_before_checkpoint(
    tmp_path: Path,
    monkeypatch,
):
    args = _canonical_args()
    args.checkpoint_output = tmp_path / "recovered.pt"
    args.checkpoint_output.write_bytes(b"checkpoint-boundary")
    args.report_output = tmp_path / "report.json"
    handoff = _handoff()
    control = _control_closure()
    closure = _recovery_closure()
    calls = []

    class CheckpointBoundaryError(RuntimeError):
        pass

    monkeypatch.setattr(
        recovery,
        "_handoff_from_environment",
        lambda _mode: (
            handoff,
            2345,
            recovery.RECOVERY_BUDGET_BASE_SECONDS + 1.0,
        ),
    )
    monkeypatch.setattr(
        recovery,
        "validate_control_preflight",
        lambda *_args, **_kwargs: calls.append("control") or control,
    )
    monkeypatch.setattr(
        recovery,
        "_validate_recovery_marker",
        lambda *_args: calls.append("marker")
        or {
            "cumulative_wall_seconds_at_consumption": (
                recovery.RECOVERY_BUDGET_BASE_SECONDS
            )
        },
    )
    monkeypatch.setattr(
        recovery,
        "_validate_worker_lineage",
        lambda *_args: calls.append("lineage") or {"mode": "direct_parent"},
    )
    monkeypatch.setattr(
        recovery,
        "complete_preflight",
        lambda *_args: calls.append("complete") or closure,
    )

    def checkpoint_boundary(*_args, **_kwargs):
        calls.append("checkpoint")
        raise CheckpointBoundaryError

    monkeypatch.setattr(b1, "_load_checkpoint_weights_only", checkpoint_boundary)
    monkeypatch.setattr(
        recovery,
        "DEFAULT_FINAL_RECEIPT",
        tmp_path / "receipt.json",
    )

    with pytest.raises(CheckpointBoundaryError):
        recovery.run_final_verify_worker(args)

    assert calls == ["control", "marker", "lineage", "complete", "checkpoint"]


def test_training_continuation_binds_source_cursor_time_and_final_argv(monkeypatch):
    args = _canonical_args()
    closure = _recovery_closure()
    handoff = _handoff()
    state = {
        "completed_optimizer_steps": 4096,
        "completed_sequence_count": 16384,
        "next_permutation_cursor": 16384,
        "training_original_bytes": 8200700,
    }
    captured = {}

    def fake_train_segment(**kwargs):
        captured.update(kwargs)
        return kwargs["state"]

    monkeypatch.setattr(b1, "train_segment", fake_train_segment)
    compact = SimpleNamespace(corpus=object())
    recovery_context = {"bound": "recovery"}
    result = recovery._run_bound_training_continuation(
        args=args,
        closure=closure,
        handoff=handoff,
        recovery_context=recovery_context,
        torch_module=object(),
        model=object(),
        optimizer=object(),
        model_config=object(),
        scaler=object(),
        mask_generator=object(),
        tokenizer=object(),
        compact=compact,
        permutation=(0, 1),
        schedule=object(),
        state=state,
        cumulative_base=recovery.RECOVERY_BUDGET_BASE_SECONDS + 3.0,
        worker_started=123.0,
    )

    assert result is state
    assert captured["state"] is state
    assert captured["checkpoint_path"] == args.checkpoint_output
    assert captured["cumulative_wall_seconds_before"] == (
        recovery.RECOVERY_BUDGET_BASE_SECONDS + 3.0
    )
    assert captured["phase_started"] == 123.0
    assert captured["parent_pid"] == handoff.parent_pid
    assert captured["resume_pid"] == os.getpid()
    assert captured["run_context"] is recovery_context
    assert captured["canonical_child_argv_sha256"] == recovery._argv_sha256(
        recovery.canonical_command("final_verify", args)
    )

    bad_state = dict(state, next_permutation_cursor=16380)
    with pytest.raises(recovery.RecoveryFatalError, match="bound cursor"):
        recovery._run_bound_training_continuation(
            args=args,
            closure=closure,
            handoff=handoff,
            recovery_context=recovery_context,
            torch_module=object(),
            model=object(),
            optimizer=object(),
            model_config=object(),
            scaler=object(),
            mask_generator=object(),
            tokenizer=object(),
            compact=compact,
            permutation=(0, 1),
            schedule=object(),
            state=bad_state,
            cumulative_base=recovery.RECOVERY_BUDGET_BASE_SECONDS,
            worker_started=123.0,
        )


def test_parent_report_requires_every_gate(monkeypatch):
    args = _canonical_args()
    closure = _recovery_closure()
    handoff = _handoff()
    report = {
        "schema": recovery.REPORT_SCHEMA,
        "decision": recovery.PASS_DECISION,
        "recovery_authorization_sha256": closure.authorization_sha256,
        "recovery_contract_sha256": closure.contract_sha256,
        "recovery_marker_sha256": handoff.marker_sha256,
        "gates": {"exact_once": True, "receipt": False},
    }
    monkeypatch.setattr(recovery, "_read_json", lambda *_args: (report, b"{}"))

    with pytest.raises(recovery.RecoveryFatalError, match="report closure drifted"):
        recovery._validate_parent_report_closure(
            args,
            closure,
            handoff,
            recovery.RECOVERY_BUDGET_BASE_SECONDS + 10.0,
        )


def test_recovery_contract_preserves_original_failure_and_scope():
    contract = json.loads(recovery.DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    authorization = json.loads(
        recovery.DEFAULT_AUTHORIZATION.read_text(encoding="utf-8")
    )

    assert contract["status"] == "source_closure_complete"
    assert contract["raw_pass_accounting"]["maximum_total_full_fit_raw_passes"] == 2
    assert contract["raw_scope"]["outer_holdout_raw_opens_allowed"] == 0
    assert contract["continuation"]["resume_cursor"] == 16384
    assert contract["continuation"]["final_sequence_cursor"] == 115072
    assert contract["authority"]["public_key_required"] is False
    assert contract["ready_for"]["recovery_execution"] is True
    assert contract["ready_for"]["blocked_by"] == []
    assert contract["required_success_invariants"]["pass_decision"] == (
        recovery.PASS_DECISION
    )
    assert contract["required_success_invariants"]["failure_decision"] == (
        recovery.FAILURE_DECISION
    )
    assert contract["resource_gates"]["any_gate_failure_action"] == (
        recovery.FAILURE_DECISION
    )
    assert "lineage_tests" in contract["recovery_execution_closure"]
    assert authorization["authorization_granted"] is True
    assert authorization["runtime_source_closure"]["pending_bindings"] == []
    assert authorization["ready_for"]["recovery_execution"] is True
    assert authorization["forbidden"]["public_key_dependency"] is True
    assert "lineage_tests" in authorization["canonical_paths"]
    assert "lineage_tests" in authorization["bindings"]


def test_contract_validator_rejects_decision_drift():
    contract = json.loads(recovery.DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    contract["required_success_invariants"]["pass_decision"] = "wrong-pass"
    with pytest.raises(recovery.RecoveryFatalError, match="contract drifted"):
        recovery._validate_recovery_contract(contract)


def test_contract_validator_never_treats_pending_bindings_as_authorized():
    contract = json.loads(recovery.DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    contract["recovery_execution_closure"]["run_allowed_with_pending_binding"] = True
    with pytest.raises(recovery.RecoveryFatalError, match="closure remains pending"):
        recovery._validate_recovery_contract(contract)
