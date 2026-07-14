from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONTRACT_PATH = (
    PROJECT_ROOT
    / "manifests"
    / "roadmap_9997"
    / "loop166_code_section_foundation"
    / "phase_b1_full_outer_resource_cell.json"
)
AUTHORIZATION_PATH = CONTRACT_PATH.with_name("phase_b1_run_authorization.json")
for search_path in (SRC_DIR, SCRIPTS_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_loop166_phase_b1_full_outer_resource_cell as b1_controller  # noqa: E402
from run_loop166_phase_b1_full_outer_resource_cell import (  # noqa: E402
    B1FatalError,
    _atomic_torch_save,
    assert_report_has_no_quality_metrics,
    build_checkpoint_payload,
    read_verified_outer_fit_source,
    select_outer_fit_records,
    take_step_indices,
    validate_static_preflight,
    verify_checkpoint_payload,
)

from loop164.local_oof import LocalOOFRecord  # noqa: E402
from loop166.b1_schedule import (  # noqa: E402
    deterministic_permutation,
    iter_optimizer_groups,
    permutation_commitment_sha256,
    validate_exact_once_schedule,
)
from loop166.byte_bpe import LosslessTokenChunk, train_byte_bpe_tokenizer  # noqa: E402
from loop166.compact_corpus import (  # noqa: E402
    CompactSequenceCorpus,
    materialize_framed_batch,
)
from loop166.mlm_model import TinyMaskedLanguageModel, TinyMLMConfig  # noqa: E402


def _compact_corpus() -> CompactSequenceCorpus:
    corpus = CompactSequenceCorpus(vocab_size=1029, max_content_tokens=510)
    corpus.append(LosslessTokenChunk((5, 6, 7), 3))
    corpus.append(LosslessTokenChunk((8, 9), 2))
    return corpus


def _synthetic_pending_contract(tmp_path: Path) -> Path:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["execution_closure"]["phase_b1_controller_sha256"] = (
        "pending_before_phase_b1_execution"
    )
    contract["execution_closure"]["run_allowed_with_pending_binding"] = True
    output = tmp_path / "phase-b1-pending.json"
    output.write_text(json.dumps(contract), encoding="utf-8")
    return output


def _canonical_args():
    return b1_controller._normalize_cli_paths(b1_controller.build_parser().parse_args([]))


def _synthetic_handoff(*, nonce: str = "ab" * 32) -> b1_controller.RunHandoff:
    return b1_controller.RunHandoff(
        authorization_sha256="1" * 64,
        marker_sha256="2" * 64,
        handoff_nonce=nonce,
        parent_pid=1234,
        canonical_parent_argv_sha256="3" * 64,
    )


def _record(
    train_row_index: int,
    *,
    fold: int,
    label: int,
    source_path: Path,
    source_size_bytes: int | None = 1,
    availability: str = "supported",
    missing_reason: str | None = None,
    source_sha256: str | None = None,
) -> LocalOOFRecord:
    return LocalOOFRecord(
        train_row_index=train_row_index,
        sample_index=train_row_index,
        source_path=source_path,
        source_sha256=source_sha256 or f"{train_row_index:064x}",
        source_size_bytes=source_size_bytes,
        label=label,
        availability=availability,
        missing_reason=missing_reason,
        component_id=f"component-{train_row_index}",
        component_size=1,
        fold=fold,
    )


class _SyntheticScaler:
    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"scale": torch.tensor(128.0)}


def test_compact_corpus_uses_uint_arrays_without_persistent_padding():
    corpus = _compact_corpus()

    assert corpus._flat_token_ids.typecode == "H"
    assert corpus._offsets.typecode == "Q"
    assert corpus._lengths.typecode == "H"
    assert corpus._original_byte_lengths.typecode == "H"
    assert list(corpus._flat_token_ids) == [5, 6, 7, 8, 9]
    assert list(corpus._offsets) == [0, 3, 5]
    assert list(corpus._lengths) == [3, 2]
    assert corpus.total_tokens == 5
    assert corpus.total_original_bytes == 5
    assert corpus.estimated_storage_bytes == 5 * 2 + 3 * 8 + 2 * 2 + 2 * 2
    assert corpus[0] == LosslessTokenChunk((5, 6, 7), 3)

    compact_lengths_before = tuple(
        len(values)
        for values in (
            corpus._flat_token_ids,
            corpus._offsets,
            corpus._lengths,
            corpus._original_byte_lengths,
        )
    )
    batch = materialize_framed_batch(
        corpus,
        [0, 1],
        pad_id=0,
        cls_id=1,
        sep_id=2,
        sequence_tokens=512,
        torch_module=torch,
    )
    compact_lengths_after = tuple(
        len(values)
        for values in (
            corpus._flat_token_ids,
            corpus._offsets,
            corpus._lengths,
            corpus._original_byte_lengths,
        )
    )

    assert compact_lengths_after == compact_lengths_before
    assert batch.input_ids.shape == (2, 512)
    assert batch.input_ids[0, :5].tolist() == [1, 5, 6, 7, 2]
    assert batch.input_ids[1, :4].tolist() == [1, 8, 9, 2]
    assert bool(batch.input_ids[0, 5:].eq(0).all())
    assert bool(batch.input_ids[1, 4:].eq(0).all())
    assert batch.attention_mask[0, :6].tolist() == [True, True, True, True, True, False]
    assert batch.attention_mask[1, :5].tolist() == [True, True, True, True, False]
    assert batch.original_byte_lengths.tolist() == [3, 2]


def test_compact_corpus_commitment_is_deterministic_and_order_sensitive():
    first = _compact_corpus()
    same = _compact_corpus()
    reordered = CompactSequenceCorpus(vocab_size=1029, max_content_tokens=510)
    reordered.append(LosslessTokenChunk((8, 9), 2))
    reordered.append(LosslessTokenChunk((5, 6, 7), 3))

    assert first.commitment_sha256() == same.commitment_sha256()
    assert first.commitment_sha256() != reordered.commitment_sha256()


@pytest.mark.parametrize(
    "chunk, error",
    [
        (LosslessTokenChunk((), 1), "cannot be empty"),
        (LosslessTokenChunk((True,), 1), "cannot be boolean"),
        (LosslessTokenChunk((1029,), 1), "outside vocabulary"),
        (LosslessTokenChunk(tuple(range(511)), 511), "exceeds max_content_tokens"),
        (LosslessTokenChunk((5,), 513), "must be in \\[1, 512\\]"),
    ],
)
def test_compact_corpus_rejects_invalid_chunks_without_partial_append(chunk, error):
    corpus = CompactSequenceCorpus(vocab_size=1029, max_content_tokens=510)

    with pytest.raises(ValueError, match=error):
        corpus.append(chunk)

    assert len(corpus) == 0
    assert corpus.total_tokens == 0
    assert list(corpus._offsets) == [0]


def test_batch_materialization_rejects_content_framing_collision():
    corpus = CompactSequenceCorpus(vocab_size=1029, max_content_tokens=510)
    corpus.append(LosslessTokenChunk((1, 5), 2))

    with pytest.raises(ValueError, match="framing special token"):
        materialize_framed_batch(
            corpus,
            [0],
            pad_id=0,
            cls_id=1,
            sep_id=2,
            sequence_tokens=512,
            torch_module=torch,
        )


def test_deterministic_schedule_visits_partial_final_step_exactly_once():
    permutation = deterministic_permutation(sequence_count=11, seed=166)
    repeated = deterministic_permutation(sequence_count=11, seed=166)
    different_seed = deterministic_permutation(sequence_count=11, seed=167)

    assert permutation == repeated
    assert sorted(permutation) == list(range(11))
    assert permutation != different_seed
    assert permutation_commitment_sha256(permutation) == permutation_commitment_sha256(repeated)

    groups = list(
        iter_optimizer_groups(
            permutation,
            microbatch_size=2,
            gradient_accumulation_steps=2,
        )
    )
    validate_exact_once_schedule(permutation, groups, sequence_count=11)

    assert [group.sequence_count for group in groups] == [4, 4, 3]
    assert [group.cursor_start for group in groups] == [0, 4, 8]
    assert [group.cursor_end for group in groups] == [4, 8, 11]
    assert [batch.sequence_count for batch in groups[-1].microbatches] == [2, 1]
    assert [batch.loss_weight for batch in groups[-1].microbatches] == pytest.approx(
        [2 / 3, 1 / 3]
    )
    assert tuple(index for group in groups for index in group.indices) == permutation

    resumed_groups = list(
        iter_optimizer_groups(
            permutation,
            start_cursor=4,
            microbatch_size=2,
            gradient_accumulation_steps=2,
        )
    )
    assert [group.optimizer_step_index for group in resumed_groups] == [1, 2]
    assert tuple(index for group in resumed_groups for index in group.indices) == permutation[4:]

    first_indices, first_cursor = take_step_indices(permutation, 0)
    final_indices, final_cursor = take_step_indices(permutation, 8)
    exhausted_indices, exhausted_cursor = take_step_indices(permutation, 11)
    assert first_indices == permutation[:4]
    assert first_cursor == 4
    assert final_indices == permutation[8:]
    assert final_cursor == 11
    assert exhausted_indices == ()
    assert exhausted_cursor == 11


def test_pending_controller_binding_fails_before_any_raw_open(tmp_path: Path, monkeypatch):
    pending_contract = _synthetic_pending_contract(tmp_path)
    raw_open_calls = 0

    def unexpected_raw_open(*_args, **_kwargs):
        nonlocal raw_open_calls
        raw_open_calls += 1
        raise AssertionError("static preflight attempted raw access")

    monkeypatch.setattr(
        b1_controller,
        "read_verified_outer_fit_source",
        unexpected_raw_open,
    )

    with pytest.raises(B1FatalError, match="controller binding is still pending"):
        validate_static_preflight(
            pending_contract,
            controller_path=Path(b1_controller.__file__),
        )

    assert raw_open_calls == 0


def test_canonical_source_and_authorization_preflight_is_pure(monkeypatch):
    marker = b1_controller.DEFAULT_MARKER
    marker_before = marker.read_bytes() if marker.exists() else None
    raw_open_calls = 0

    def unexpected_raw_open(*_args, **_kwargs):
        nonlocal raw_open_calls
        raw_open_calls += 1
        raise AssertionError("pure preflight attempted raw access")

    monkeypatch.setattr(
        b1_controller,
        "read_verified_outer_fit_source",
        unexpected_raw_open,
    )
    contract, bindings = validate_static_preflight(CONTRACT_PATH)
    args = _canonical_args()
    b1_controller._validate_runtime_paths(args, bindings)
    authorization, authorization_sha = b1_controller.validate_run_authorization(
        contract,
        bindings,
        authorization_path=AUTHORIZATION_PATH,
    )

    assert authorization["authorization_granted"] is True
    assert authorization["ready_for"]["parent_execution"] is True
    assert bindings["contract_tests"]["path"] == str(Path(__file__).resolve(strict=True))
    assert bindings["phase_b1_controller"]["path"] == str(
        Path(b1_controller.__file__).resolve(strict=True)
    )
    assert len(authorization_sha) == 64
    assert raw_open_calls == 0
    assert (marker.read_bytes() if marker.exists() else None) == marker_before


@pytest.mark.parametrize("path_name", ["contract", "folds", "data_root"])
def test_noncanonical_runtime_path_fails_before_raw_open(
    tmp_path: Path,
    monkeypatch,
    path_name: str,
):
    args = _canonical_args()
    setattr(args, path_name, tmp_path / f"alternate-{path_name}")
    raw_open_calls = 0

    def unexpected_raw_open(*_args, **_kwargs):
        nonlocal raw_open_calls
        raw_open_calls += 1
        raise AssertionError("path validation attempted raw access")

    monkeypatch.setattr(
        b1_controller,
        "read_verified_outer_fit_source",
        unexpected_raw_open,
    )
    with pytest.raises(B1FatalError, match=f"runtime path is not canonical: {path_name}"):
        b1_controller._normalize_cli_paths(args)

    assert raw_open_calls == 0


def test_noncanonical_working_directory_fails_authorization_before_raw(
    tmp_path: Path,
    monkeypatch,
):
    contract, bindings = validate_static_preflight(CONTRACT_PATH)
    raw_open_calls = 0

    def unexpected_raw_open(*_args, **_kwargs):
        nonlocal raw_open_calls
        raw_open_calls += 1
        raise AssertionError("authorization validation attempted raw access")

    monkeypatch.setattr(
        b1_controller,
        "read_verified_outer_fit_source",
        unexpected_raw_open,
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(B1FatalError, match="working directory is not canonical"):
        b1_controller.validate_run_authorization(contract, bindings)

    assert raw_open_calls == 0


def test_one_shot_marker_is_exclusive_and_persists_only_nonce_commitment(
    tmp_path: Path,
    monkeypatch,
):
    marker = tmp_path / "phase-b1-consumed.json"
    nonce = "ab" * 32
    directory_flushes = 0

    def best_effort_directory_flush(_path: Path) -> bool:
        nonlocal directory_flushes
        directory_flushes += 1
        return False

    monkeypatch.setattr(b1_controller, "DEFAULT_MARKER", marker)
    monkeypatch.setattr(b1_controller, "_fsync_parent_directory", best_effort_directory_flush)
    monkeypatch.setattr(b1_controller.secrets, "token_hex", lambda _size: nonce)
    authorization = {
        "one_shot_lease": {"lease_id": "loop166-b1-outer0-resource-cell-v1"}
    }

    handoff = b1_controller.consume_run_authorization(authorization, "1" * 64)
    marker_raw = marker.read_bytes()
    marker_payload = json.loads(marker_raw)

    assert directory_flushes == 1
    assert nonce.encode("ascii") not in marker_raw
    assert "handoff_nonce" not in marker_payload
    assert marker_payload["handoff_nonce_sha256"] == hashlib.sha256(
        nonce.encode("ascii")
    ).hexdigest()
    assert handoff.handoff_nonce == nonce
    assert handoff.marker_sha256 == hashlib.sha256(marker_raw).hexdigest()
    assert b1_controller.validate_consumption_marker(handoff) == marker_payload
    monkeypatch.setattr(b1_controller.os, "getppid", lambda: handoff.parent_pid + 1)
    with pytest.raises(B1FatalError, match="not spawned by its bound parent"):
        b1_controller.validate_consumption_marker(
            handoff,
            require_direct_parent_pid=handoff.parent_pid,
        )
    with pytest.raises(B1FatalError, match="marker already exists"):
        b1_controller.consume_run_authorization(authorization, "1" * 64)


def test_windows_parent_directory_flush_failure_is_best_effort(tmp_path: Path, monkeypatch):
    class FakeCall:
        def __init__(self, result):
            self.result = result
            self.calls = []

        def __call__(self, *args):
            self.calls.append(args)
            return self.result

    class FakeKernel32:
        def __init__(self):
            self.CreateFileW = FakeCall(1234)
            self.FlushFileBuffers = FakeCall(0)
            self.CloseHandle = FakeCall(1)

    kernel32 = FakeKernel32()

    class FakeWindll:
        pass

    windll = FakeWindll()
    windll.kernel32 = kernel32
    monkeypatch.setattr(b1_controller.platform, "system", lambda: "Windows")
    monkeypatch.setattr(b1_controller.ctypes, "windll", windll)

    assert b1_controller._fsync_parent_directory(tmp_path) is False
    assert kernel32.FlushFileBuffers.calls == [(1234,)]
    assert kernel32.CloseHandle.calls == [(1234,)]
    assert kernel32.FlushFileBuffers.argtypes == [b1_controller.ctypes.c_void_p]


def test_windows_peak_rss_preserves_pointer_sized_process_handle(monkeypatch):
    process_handle = (1 << 48) + 123

    class FakeCall:
        def __init__(self, result):
            self.result = result

        def __call__(self, *_args):
            return self.result

    class FakeMemoryInfoCall:
        def __init__(self):
            self.process = None

        def __call__(self, process, counters, _size):
            self.process = process
            counters._obj.PeakWorkingSetSize = 987654321
            return 1

    class FakeKernel32:
        def __init__(self):
            self.GetCurrentProcess = FakeCall(process_handle)

    class FakePsapi:
        def __init__(self):
            self.GetProcessMemoryInfo = FakeMemoryInfoCall()

    class FakeWindll:
        def __init__(self):
            self.kernel32 = FakeKernel32()
            self.psapi = FakePsapi()

    windll = FakeWindll()
    monkeypatch.setattr(b1_controller.platform, "system", lambda: "Windows")
    monkeypatch.setattr(b1_controller.ctypes, "windll", windll)

    assert b1_controller._peak_process_rss_bytes() == 987654321
    assert windll.psapi.GetProcessMemoryInfo.process == process_handle
    assert windll.kernel32.GetCurrentProcess.restype is b1_controller.ctypes.c_void_p


def test_handoff_environment_rejects_non_hex_nonce(monkeypatch):
    handoff = _synthetic_handoff()
    environment = b1_controller._handoff_environment(handoff, resume_pid=5678)
    for name, value in environment.items():
        if name.startswith("AXON_B1_"):
            monkeypatch.setenv(name, value)

    restored, resume_pid = b1_controller._handoff_from_environment()
    assert restored == handoff
    assert resume_pid == 5678

    monkeypatch.setenv("AXON_B1_HANDOFF_NONCE", "g" * 64)
    with pytest.raises(B1FatalError, match="no valid in-memory handoff nonce"):
        b1_controller._handoff_from_environment()


def test_outer_fit_selection_excludes_fold_zero_and_keeps_all_16000_metadata_rows():
    contract_path = (
        PROJECT_ROOT
        / "manifests"
        / "roadmap_9997"
        / "loop166_code_section_foundation"
        / "phase_b1_full_outer_resource_cell.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    records = []
    for fold in range(5):
        for offset in range(4000):
            train_row_index = fold * 4000 + offset
            label = 0 if offset < 2000 else 1
            fit_position = (fold - 1) * 4000 + offset
            source_size_bytes = 1
            availability = "supported"
            missing_reason = None
            if fold > 0 and fit_position < 12:
                source_size_bytes = None
                availability = "read_failure"
                missing_reason = "read_failure"
            elif fold > 0 and fit_position < 351:
                source_size_bytes = 10 * 1024 * 1024
                availability = "oversize"
                missing_reason = "oversize"
            records.append(
                _record(
                    train_row_index,
                    fold=fold,
                    label=label,
                    source_path=Path("synthetic") / f"{train_row_index}.exe",
                    source_size_bytes=source_size_bytes,
                    availability=availability,
                    missing_reason=missing_reason,
                )
            )

    scope = select_outer_fit_records(list(reversed(records)), contract)

    assert len(scope.records) == 16000
    assert scope.records[0].train_row_index == 4000
    assert scope.records[-1].train_row_index == 19999
    assert {record.fold for record in scope.records} == {1, 2, 3, 4}
    assert scope.audit["fit_label_counts"] == {"0": 8000, "1": 8000}
    assert scope.audit["outer_holdout_metadata_rows"] == 4000
    assert scope.audit["known_size_records"] == 15988
    assert scope.audit["prior_source_unavailable_records"] == 12
    assert scope.audit["prior_oversize_records"] == 339
    assert scope.audit["outer_holdout_raw_opens"] == 0
    assert scope.audit["outer_holdout_raw_bytes"] == 0


def test_source_unavailable_is_missing_but_sha_drift_is_fatal(tmp_path: Path):
    missing = _record(
        1,
        fold=1,
        label=0,
        source_path=tmp_path / "missing.exe",
        source_size_bytes=None,
        availability="read_failure",
        missing_reason="read_failure",
    )
    assert (
        read_verified_outer_fit_source(
            missing,
            data_root=tmp_path,
            maximum_source_bytes=1024,
        )
        is None
    )

    payload = b"synthetic executable bytes"
    source_path = tmp_path / "present.exe"
    source_path.write_bytes(payload)
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    valid = _record(
        2,
        fold=1,
        label=1,
        source_path=source_path,
        source_size_bytes=len(payload),
        source_sha256=expected_sha256,
    )
    verified = read_verified_outer_fit_source(
        valid,
        data_root=tmp_path,
        maximum_source_bytes=1024,
    )
    assert verified is not None
    assert verified.raw_bytes == payload
    assert verified.sha256 == expected_sha256

    sha_drift = _record(
        3,
        fold=1,
        label=1,
        source_path=source_path,
        source_size_bytes=len(payload),
        source_sha256="0" * 64,
    )
    with pytest.raises(B1FatalError, match="SHA-256 drifted"):
        read_verified_outer_fit_source(
            sha_drift,
            data_root=tmp_path,
            maximum_source_bytes=1024,
        )


def test_weights_only_checkpoint_preserves_resume_state_and_exact_logits(
    tmp_path: Path,
    monkeypatch,
):
    training_bytes = bytes(range(256)) * 4
    tokenizer = train_byte_bpe_tokenizer([training_bytes], vocab_size=261)
    config = TinyMLMConfig(
        vocab_size=261,
        sequence_tokens=64,
        layers=1,
        hidden_dim=16,
        heads=4,
        ffn_dim=32,
        local_attention_window=8,
        dropout=0.0,
        gradient_checkpointing=False,
    )
    torch.manual_seed(166)
    model = TinyMaskedLanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    sample = torch.tensor([[1, 5, 6, 7, 8, 9, 10, 2] + [0] * 56], dtype=torch.long)
    attention = sample.ne(0)
    labels = sample.clone()
    labels[~attention] = -100
    model(input_ids=sample, attention_mask=attention, labels=labels)["loss"].backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    mask_generator = torch.Generator(device="cpu")
    mask_generator.manual_seed(166)
    state = {
        "completed_optimizer_steps": 3,
        "completed_sequence_count": 11,
        "next_permutation_cursor": 11,
    }
    handoff = _synthetic_handoff()
    child_argv_sha256 = "4" * 64
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda _device=None: torch.tensor([7, 8, 9], dtype=torch.uint8),
    )
    payload = build_checkpoint_payload(
        torch_module=torch,
        model=model,
        optimizer=optimizer,
        scaler=_SyntheticScaler(),
        model_config=config,
        tokenizer=tokenizer,
        tokenizer_sha256="1" * 64,
        mask_generator=mask_generator,
        permutation_commitment="2" * 64,
        corpus_commitment="3" * 64,
        compact_commitment="4" * 64,
        state=state,
        cumulative_wall_seconds=12.5,
        parent_pid=1234,
        run_context={
            "synthetic": True,
            "prepared_sequence_count": 11,
            "total_optimizer_steps": 3,
        },
        handoff=handoff,
        canonical_child_argv_sha256=child_argv_sha256,
    )

    checkpoint_path = tmp_path / "phase-b1-checkpoint.pt"
    _atomic_torch_save(torch, payload, checkpoint_path)
    assert [path.name for path in tmp_path.iterdir()] == [checkpoint_path.name]
    assert handoff.handoff_nonce.encode("ascii") not in checkpoint_path.read_bytes()
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    contract_path = (
        PROJECT_ROOT
        / "manifests"
        / "roadmap_9997"
        / "loop166_code_section_foundation"
        / "phase_b1_full_outer_resource_cell.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["tokenizer"]["expected_total_vocabulary"] = config.vocab_size
    for field in (
        "sequence_tokens",
        "layers",
        "hidden_dim",
        "heads",
        "ffn_dim",
        "local_attention_window",
        "global_token_index",
        "dropout",
        "activation",
        "gradient_checkpointing",
        "tied_input_output_embeddings",
    ):
        contract["model"][field] = getattr(config, field)
    verified = verify_checkpoint_payload(
        loaded,
        contract,
        expected_handoff=handoff,
        expected_child_argv_sha256=child_argv_sha256,
        require_final=True,
    )

    assert verified["completed_optimizer_steps"] == 3
    assert verified["completed_sequence_count"] == 11
    assert verified["next_permutation_cursor"] == 11
    assert verified["shuffle_commitment_sha256"] == "2" * 64
    assert verified["outer_fit_corpus_commitment_sha256"] == "3" * 64
    assert verified["compact_corpus_commitment_sha256"] == "4" * 64
    assert torch.equal(verified["mask_generator_state"], mask_generator.get_state())
    assert verified["scaler_state_dict"]["scale"].item() == 128.0
    assert verified["handoff_nonce_sha256"] == handoff.handoff_nonce_sha256
    assert "handoff_nonce" not in verified

    restored_model = TinyMaskedLanguageModel(TinyMLMConfig(**verified["model_config"])).eval()
    restored_model.load_state_dict(verified["model_state_dict"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1.0e-3)
    restored_optimizer.load_state_dict(verified["optimizer_state_dict"])
    assert len(restored_optimizer.state) > 0
    with torch.inference_mode():
        restored_logits = restored_model(
            verified["synthetic_input_ids"],
            attention_mask=verified["synthetic_attention_mask"],
        )["logits"]
    assert torch.equal(restored_logits, verified["synthetic_logits"])

    invalid_optimizer = dict(loaded)
    invalid_optimizer["optimizer_state_dict"] = {
        "state": {0: {"exp_avg": torch.tensor(float("inf"))}},
        "param_groups": [],
    }
    with pytest.raises(B1FatalError, match="optimizer_state_dict"):
        verify_checkpoint_payload(invalid_optimizer, contract)

    invalid_scaler = dict(loaded)
    invalid_scaler["scaler_state_dict"] = {"scale": float("nan")}
    with pytest.raises(B1FatalError, match="scaler_state_dict"):
        verify_checkpoint_payload(invalid_scaler, contract)

    invalid_training_state = dict(loaded)
    invalid_training_state["training_state"] = []
    with pytest.raises(B1FatalError, match="deep training state"):
        verify_checkpoint_payload(invalid_training_state, contract)


def test_final_receipt_requires_independent_finite_exact_verifier(
    tmp_path: Path,
    monkeypatch,
):
    handoff = _synthetic_handoff()
    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"synthetic-final-checkpoint")
    receipt_path = tmp_path / "final-receipt.json"
    receipt = {
        "schema": "axon_loop166_phase_b1_final_checkpoint_verification_v1",
        "loop_id": "loop166_code_section_foundation",
        "authorization_sha256": handoff.authorization_sha256,
        "marker_sha256": handoff.marker_sha256,
        "handoff_nonce_sha256": handoff.handoff_nonce_sha256,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "model_tensors_finite": True,
        "optimizer_tensors_finite": True,
        "rng_state_validated": True,
        "synthetic_logits_bit_exact": True,
        "parent_pid": handoff.parent_pid,
        "resume_pid": 2345,
        "verifier_pid": 3456,
        "quality_metrics_computed": False,
        "threshold_operations_performed": False,
        "decision": "phase_b1_final_checkpoint_verification_pass",
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(b1_controller, "DEFAULT_FINAL_VERIFY_RECEIPT", receipt_path)

    loaded = b1_controller._load_final_verify_receipt(
        handoff,
        checkpoint_path=checkpoint,
        resume_pid=2345,
    )
    assert loaded == receipt

    for field, invalid_value in (
        ("rng_state_validated", False),
        ("verifier_pid", 0),
        ("synthetic_logits_bit_exact", False),
    ):
        invalid = dict(receipt)
        invalid[field] = invalid_value
        receipt_path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(B1FatalError, match="receipt drifted"):
            b1_controller._load_final_verify_receipt(
                handoff,
                checkpoint_path=checkpoint,
                resume_pid=2345,
            )


def test_report_guard_allows_resource_fields_and_rejects_quality_or_threshold_fields():
    safe_report = {
        "training": {
            "quality_metrics_computed": False,
            "threshold_operations_performed": False,
            "completed_optimizer_steps": 4096,
        },
        "resources": {"peak_process_rss_bytes": 1024},
    }
    assert_report_has_no_quality_metrics(safe_report)

    for forbidden in (
        {"training": {"f1": 0.9}},
        {"nested": [{"threshold": 0.5}]},
        {"metrics": {"loss_curve": [1.0]}},
    ):
        with pytest.raises(B1FatalError, match="Forbidden quality-result field"):
            assert_report_has_no_quality_metrics(forbidden)


def test_phase_b1_contract_is_bound_resource_only_and_full_outer_scoped():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert contract["execution_closure"]["binding_drift_action"] == "fail_before_any_raw_open"
    assert contract["execution_closure"]["run_allowed_with_pending_binding"] is True
    assert contract["execution_closure"]["phase_b1_controller_sha256"] == hashlib.sha256(
        Path(b1_controller.__file__).read_bytes()
    ).hexdigest()
    assert contract["static_bindings"]["contract_tests"] == {
        "path": "tests/test_loop166_phase_b1.py",
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    assert contract["data_scope"]["outer_holdout_fold"] == 0
    assert contract["data_scope"]["outer_fit_folds"] == [1, 2, 3, 4]
    assert contract["data_scope"]["outer_holdout_metadata_rows"] == 4000
    assert contract["data_scope"]["outer_fit_metadata_rows"] == 16000
    assert contract["data_scope"]["outer_holdout_raw_opens_allowed"] == 0
    assert contract["compact_sequence_storage"]["content_token_dtype"] == "uint16"
    assert contract["compact_sequence_storage"]["python_int_tuple_per_padded_sequence_forbidden"] is True
    assert contract["compact_sequence_storage"]["durable_token_cache_allowed"] is False
    assert contract["compact_sequence_storage"]["padding_materialized_only_for_current_microbatch"] is True
    assert contract["training"]["quality_metric_allowed"] is False
    assert contract["training"]["training_loss_reporting_allowed"] is False
    assert contract["training"]["threshold_operation_allowed"] is False
    assert "f1" in contract["forbidden"]["quality_metrics"]
    assert contract["forbidden"]["threshold_search_or_sweep"] is True
    assert {
        "compact_corpus_commitment_sha256",
        "training_state",
        "synthetic_input_ids",
        "synthetic_attention_mask",
        "synthetic_logits",
        "parent_pid",
        "resume_pid",
        "authorization_sha256",
        "marker_sha256",
        "handoff_nonce_sha256",
        "canonical_parent_argv_sha256",
        "canonical_child_argv_sha256",
        "permutation_prefix_original_bytes",
        "run_context",
    } <= set(contract["checkpoint_and_resume"]["payload_requires"])
    assert contract["ready_for"]["phase_b1_execution"] is True
    assert contract["ready_for"]["blocked_by"] == []
    assert contract["decision"] == "contract_source_closure_complete_allow_authorized_phase_b1"
