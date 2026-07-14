from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for search_path in (SRC_DIR, SCRIPTS_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from run_loop166_phase_b0_resource_smoke import (  # noqa: E402
    _atomic_tokenizer_save,
    _atomic_torch_save,
    _join_outer_fit_records,
    _prepare_sequences,
)

from loop166.byte_bpe import (  # noqa: E402
    SPECIAL_TOKENS,
    bytes_to_unicode,
    chunk_token_ids_losslessly,
    decode_byte_string,
    decode_token_ids,
    encode_byte_string,
    encode_bytes,
    select_even_window_indices,
    select_even_windows,
    token_ids_original_byte_length,
    tokenizer_vocab_size,
    train_byte_bpe_tokenizer,
    unicode_to_bytes,
)
from loop166.mlm_model import (  # noqa: E402
    TinyMaskedLanguageModel,
    TinyMLMConfig,
    build_cls_global_local_attention_mask,
    count_parameters,
    count_trainable_parameters,
)


def _small_config() -> TinyMLMConfig:
    return TinyMLMConfig(
        vocab_size=261,
        sequence_tokens=8,
        layers=1,
        hidden_dim=16,
        heads=4,
        ffn_dim=32,
        local_attention_window=2,
        dropout=0.0,
        gradient_checkpointing=False,
    )


def test_byte_unicode_and_bpe_roundtrip_are_bijective():
    byte_alphabet = bytes_to_unicode()
    unicode_alphabet = unicode_to_bytes()
    all_bytes = bytes(range(256))

    assert set(byte_alphabet) == set(range(256))
    assert len(set(byte_alphabet.values())) == 256
    assert {unicode_alphabet[value] for value in byte_alphabet.values()} == set(range(256))
    assert decode_byte_string(encode_byte_string(all_bytes)) == all_bytes

    tokenizer = train_byte_bpe_tokenizer(
        [all_bytes, all_bytes[::-1], all_bytes],
        vocab_size=261,
    )
    token_ids = encode_bytes(tokenizer, all_bytes)
    special_literal_bytes = b"[PAD][CLS][MASK][UNK][SEP]"
    special_literal_ids = encode_bytes(tokenizer, special_literal_bytes)

    assert tokenizer_vocab_size(tokenizer) == 261
    assert decode_token_ids(tokenizer, token_ids) == all_bytes
    assert all(tokenizer.id_to_token(token_id) not in SPECIAL_TOKENS for token_id in special_literal_ids)
    assert decode_token_ids(tokenizer, special_literal_ids) == special_literal_bytes


def test_window_selection_is_deterministic_nonoverlapping_and_endpoint_preserving():
    payload = bytes(range(40))

    assert select_even_window_indices(10, 4) == [0, 3, 6, 9]
    assert select_even_window_indices(10, 1) == [5]
    assert select_even_windows(payload, window_bytes=4, max_windows=4) == [
        payload[0:4],
        payload[12:16],
        payload[24:28],
        payload[36:40],
    ]
    assert select_even_windows(payload, window_bytes=4, max_windows=4) == select_even_windows(
        payload,
        window_bytes=4,
        max_windows=4,
    )


def test_lossless_token_chunking_splits_unmerged_512_bytes_into_510_plus_2():
    payload = bytes(range(256)) * 2
    tokenizer = train_byte_bpe_tokenizer([payload], vocab_size=261)

    assert len(encode_bytes(tokenizer, payload)) == 512
    chunks = chunk_token_ids_losslessly(tokenizer, payload, max_content_tokens=510)

    assert [len(chunk.token_ids) for chunk in chunks] == [510, 2]
    assert sum(chunk.original_byte_length for chunk in chunks) == 512
    assert b"".join(decode_token_ids(tokenizer, chunk.token_ids) for chunk in chunks) == payload


def test_content_token_accounting_rejects_empty_special_and_unknown_ids():
    tokenizer = train_byte_bpe_tokenizer([bytes(range(256))], vocab_size=261)
    pad_token_id = tokenizer.token_to_id("[PAD]")
    assert pad_token_id is not None

    with pytest.raises(ValueError, match="cannot be empty"):
        chunk_token_ids_losslessly(tokenizer, b"", max_content_tokens=510)
    with pytest.raises(ValueError, match="cannot be empty"):
        token_ids_original_byte_length(tokenizer, [])
    with pytest.raises(ValueError, match="Special token id is forbidden"):
        token_ids_original_byte_length(tokenizer, [pad_token_id])
    with pytest.raises(ValueError, match="Unknown tokenizer id"):
        token_ids_original_byte_length(tokenizer, [tokenizer_vocab_size(tokenizer) + 100])


def test_prepare_sequences_conserves_bytes_and_expands_without_exclusion():
    unmerged_window = bytes(range(256)) * 2
    short_window = b"abc"
    tokenizer = train_byte_bpe_tokenizer([unmerged_window], vocab_size=261)

    prepared, counts = _prepare_sequences(
        tokenizer,
        [unmerged_window, short_window],
        sequence_tokens=512,
    )

    assert counts == {
        "prepared_sequences": 3,
        "original_window_bytes": 515,
        "prepared_original_bytes": 515,
        "split_window_count": 1,
        "sequence_expansion_count": 1,
        "overlength_windows_excluded": 0,
    }
    assert sum(row.original_bytes for row in prepared) == 515
    assert all(len(row.input_ids) == 512 for row in prepared)
    reconstructed = b"".join(
        decode_token_ids(tokenizer, row.input_ids[1 : row.valid_tokens - 1])
        for row in prepared
    )
    assert reconstructed == unmerged_window + short_window


def test_outer_fit_join_excludes_holdout_and_preserves_bundle_order_and_cap():
    bundle_records = [
        SimpleNamespace(
            source_path=Path(f"sample-{index}.exe"),
            source_sha256=f"{index:064x}",
            source_size_bytes=100 + index,
            label=index % 2,
        )
        for index in range(5)
    ]
    folds = [0, 2, 1, 0, 3]
    fold_records = [
        SimpleNamespace(
            source_path=record.source_path,
            source_sha256=record.source_sha256,
            source_size_bytes=record.source_size_bytes,
            label=record.label,
            availability="supported",
            missing_reason=None,
            fold=fold,
        )
        for record, fold in zip(bundle_records, folds)
    ]

    selected, counts = _join_outer_fit_records(
        bundle_records,
        fold_records,
        outer_holdout_fold=0,
        maximum_fit_records=2,
    )

    assert [record.source_sha256 for record in selected] == [
        bundle_records[1].source_sha256,
        bundle_records[2].source_sha256,
    ]
    assert [record.diagnostic_fold for record in selected] == [2, 1]
    assert counts == {
        "bundle_rows": 5,
        "fold_rows": 5,
        "outer_holdout_metadata_rows": 2,
        "outer_fit_metadata_rows": 3,
        "selected_outer_fit_rows": 2,
    }


def test_cls_global_local_attention_mask_has_only_contract_visibility():
    visibility = build_cls_global_local_attention_mask(
        sequence_length=8,
        local_window=2,
        global_token_index=0,
    )

    assert visibility.dtype == torch.bool
    assert visibility.shape == (8, 8)
    assert bool(visibility[0, :].all())
    assert bool(visibility[:, 0].all())
    assert torch.equal(visibility, visibility.T)
    assert bool(visibility[4, 6]) is True
    assert bool(visibility[4, 7]) is False
    assert bool(visibility[7, 4]) is False


def test_frozen_model_size_tied_weights_and_forward_contract():
    frozen_model = TinyMaskedLanguageModel(TinyMLMConfig())
    parameter_count = count_parameters(frozen_model)

    assert 10_000_000 <= parameter_count <= 15_000_000
    assert count_trainable_parameters(frozen_model) == parameter_count
    assert frozen_model.lm_head.weight is frozen_model.token_embeddings.weight

    config = _small_config()
    model = TinyMaskedLanguageModel(config)
    input_ids = torch.tensor(
        [
            [1, 5, 6, 7, 8, 9, 10, 2],
            [1, 11, 12, 13, 2, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = input_ids.ne(config.pad_token_id)
    labels = input_ids.clone()
    labels[~attention_mask] = -100
    output = model(input_ids, attention_mask=attention_mask, labels=labels)

    assert output["logits"].shape == (2, 8, config.vocab_size)
    assert output["loss"].ndim == 0
    assert bool(torch.isfinite(output["loss"]))


def test_weights_only_checkpoint_roundtrip_preserves_exact_logits(tmp_path: Path):
    torch.manual_seed(166)
    config = _small_config()
    model = TinyMaskedLanguageModel(config).eval()
    input_ids = torch.tensor([[1, 5, 6, 7, 8, 9, 10, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    with torch.no_grad():
        expected_logits = model(input_ids, attention_mask=attention_mask)["logits"]

    checkpoint_path = tmp_path / "phase_b0_checkpoint.pt"
    assert _atomic_torch_save(
        torch,
        {
            "schema": "axon_loop166_phase_b0_tiny_mlm_checkpoint_v1",
            "model_config": asdict(config),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    ) is True
    assert [path.name for path in tmp_path.iterdir()] == [checkpoint_path.name]
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restored = TinyMaskedLanguageModel(TinyMLMConfig(**loaded["model_config"])).eval()
    restored.load_state_dict(loaded["model_state_dict"], strict=True)

    with torch.no_grad():
        restored_logits = restored(input_ids, attention_mask=attention_mask)["logits"]

    assert torch.equal(restored_logits, expected_logits)


def test_atomic_tokenizer_save_roundtrip_leaves_no_temporary_file(tmp_path: Path):
    training_bytes = bytes(range(256)) * 3
    tokenizer = train_byte_bpe_tokenizer([training_bytes], vocab_size=261)
    tokenizer_path = tmp_path / "phase_b0_tokenizer.json"

    assert _atomic_tokenizer_save(tokenizer, tokenizer_path) is True
    assert [path.name for path in tmp_path.iterdir()] == [tokenizer_path.name]

    restored = Tokenizer.from_file(str(tokenizer_path))
    payload = b"[PAD][CLS][MASK][UNK][SEP]"
    restored_ids = encode_bytes(restored, payload)
    assert all(restored.id_to_token(token_id) not in SPECIAL_TOKENS for token_id in restored_ids)
    assert decode_token_ids(restored, restored_ids) == payload


def test_phase_b0_manifest_forbids_holdout_metrics_thresholds_and_public_key():
    contract_path = (
        PROJECT_ROOT
        / "manifests"
        / "roadmap_9997"
        / "loop166_code_section_foundation"
        / "phase_b0_resource_smoke.json"
    )
    proposal_path = contract_path.with_name("proposal.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))

    assert contract["authority"]["public_key_required"] is False
    assert contract["authority"]["a2_or_a3_authority"] is False
    assert contract["data_scope"]["holdout_raw_opens_allowed"] == 0
    assert contract["data_scope"]["sequence_overflow_policy"] == "lossless_bpe_token_chunking"
    assert contract["tokenizer"]["fit_scope"] == "selected_outer_fit_subset_only"
    assert contract["training"]["gradient_scaler_initial_scale"] == 128.0
    assert contract["training"]["gradient_scaler_growth_interval"] == 1000
    assert contract["training"]["quality_metric_allowed"] is False
    assert contract["training"]["threshold_operation_allowed"] is False
    assert contract["forbidden"]["val_test_or_full_access"] is True
    assert contract["forbidden"]["outer_holdout_raw_access"] is True
    assert contract["forbidden"]["quality_or_f1_claim"] is True
    assert contract["forbidden"]["promotion_claim"] is True
    assert {
        "source_sha256",
        "sample_index",
        "fold",
        "pe_features",
        "stat_features",
        "content_sidecars",
        "loop151_scores",
    }.issubset(set(proposal["forbidden_model_inputs"]))


def test_canonical_phase_b0_report_proves_resource_only_raw_access_boundary():
    report_path = (
        PROJECT_ROOT
        / "reports"
        / "roadmap_9997"
        / "loop166"
        / "phase_b0_resource_smoke.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["decision"] == "phase_b0_resource_gate_pass"
    assert report["selection"]["selected_outer_fit_rows"] == 64
    assert report["raw_access"]["fit_raw_opens"] == 64
    assert report["raw_access"]["outer_holdout_raw_opens"] == 0
    assert report["training"]["labels_used_as_model_inputs"] is False
    assert report["training"]["identity_used_as_model_inputs"] is False
    assert report["training"]["quality_metrics_computed"] is False
    assert report["training"]["threshold_operations"] is False
    assert report["artifacts"]["raw_code_artifact_bytes"] == 0
    assert report["artifacts"]["tokenizer"]["atomic"] is True
    assert report["artifacts"]["checkpoint"]["atomic"] is True
    assert report["artifacts"]["checkpoint"]["loaded_with_weights_only"] is True
    assert report["artifacts"]["checkpoint"]["roundtrip_exact_eval_logits"] is True
    assert report["gates"]["outer_holdout_raw_opens_zero"] is True
    assert report["gates"]["quality_metrics_not_computed"] is True
    assert report["gates"]["threshold_operations_not_performed"] is True
    assert report["ready_for"]["five_fold_oof"] is False
    assert report["ready_for"]["val_test_or_full"] is False
    assert report["ready_for"]["promotion"] is False
