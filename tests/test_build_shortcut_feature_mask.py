import json
import subprocess
import sys
from pathlib import Path

import torch

from config import AxonExperimentConfig
from feature_mask import apply_feature_mask_to_tensors, load_feature_mask_tensors
from kvd_features.schema_names import fixed_v3_feature_names, stat_feature_names

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER = PROJECT_ROOT / "scripts" / "build_shortcut_feature_mask.py"

SHORTCUT_PE_FEATURES = {
    "fixed_v3_section_entropy_max",
    "fixed_v3_section_entropy_min",
    "fixed_v3_section_entropy_avg",
    "fixed_v3_section_entropy_std",
    "fixed_v3_section_high_entropy_ratio",
    "fixed_v3_packer_keyword_hits_count",
    "fixed_v3_packer_keyword_hits_ratio",
}


def _build(tmp_path: Path, *extra: str) -> dict:
    output = tmp_path / "mask.json"
    subprocess.run(
        [sys.executable, str(BUILDER), "--schema", "fixed_v3", "--output", str(output), *extra],
        check=True,
        cwd=PROJECT_ROOT,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def _config() -> AxonExperimentConfig:
    return AxonExperimentConfig(
        pe_schema_version="fixed_v3",
        pe_feature_dim=256,
        pe_fixed_section_slots=32,
        stat_feature_dim=49,
    )


def test_mask_drops_exactly_the_shortcut_pe_columns(tmp_path):
    payload = _build(tmp_path)

    assert payload["type"] == "axon_feature_mask"
    assert payload["mask_spec"]["pe_search_dim"] == 142
    assert payload["mask_spec"]["search_dim"] == 142 + 49
    assert set(payload["dropped_pe_features"]) == SHORTCUT_PE_FEATURES
    assert payload["dropped_stat_features"] == []
    assert payload["kept_pe"] == 142 - 7
    assert payload["kept_stat"] == 49


def test_mask_loads_and_zeroes_only_those_columns(tmp_path):
    output = tmp_path / "mask.json"
    subprocess.run(
        [sys.executable, str(BUILDER), "--schema", "fixed_v3", "--output", str(output)],
        check=True,
        cwd=PROJECT_ROOT,
    )
    pe_mask, stat_mask, _payload = load_feature_mask_tensors(output, _config(), "cpu")

    pe_names = fixed_v3_feature_names(32)
    zeroed = {pe_names[index] for index in range(len(pe_names)) if pe_mask[index] == 0}
    assert zeroed == SHORTCUT_PE_FEATURES
    assert stat_mask.sum().item() == 49
    # reserved padding past the used dimension never passes through
    assert pe_mask[142:].sum().item() == 0


def test_mask_zeroes_planted_entropy_but_preserves_neighbours(tmp_path):
    output = tmp_path / "mask.json"
    subprocess.run(
        [sys.executable, str(BUILDER), "--schema", "fixed_v3", "--output", str(output)],
        check=True,
        cwd=PROJECT_ROOT,
    )
    feature_mask = load_feature_mask_tensors(output, _config(), "cpu")

    pe_row = torch.arange(256, dtype=torch.float32).unsqueeze(0)
    stat_row = torch.ones(1, 49)
    pe_out, stat_out = apply_feature_mask_to_tensors(pe_row, stat_row, feature_mask)

    assert pe_out[0, 113:118].abs().sum().item() == 0
    assert pe_out[0, 140:142].abs().sum().item() == 0
    assert pe_out[0, 112].item() == pe_row[0, 112].item()
    assert pe_out[0, 118].item() == pe_row[0, 118].item()
    assert torch.equal(stat_out, stat_row)


def test_strict_variant_also_drops_window_entropy(tmp_path):
    payload = _build(tmp_path, "--strict")

    assert payload["dropped_stat_features"] == [
        "stat_global_entropy_normalized",
        "stat_segment_0_entropy_normalized",
        "stat_segment_1_entropy_normalized",
        "stat_segment_2_entropy_normalized",
    ]
    assert payload["kept_stat"] == 45


def test_stat_feature_names_match_loop167_registry():
    """Guard against the two stat-name registries drifting apart."""
    from loop167.semantic_mapping import _stat_feature_names

    assert stat_feature_names(3, 10) == list(_stat_feature_names())


def test_stat_feature_names_match_config_expected_dim():
    config = AxonExperimentConfig()
    names = stat_feature_names(config.stat_segment_count, config.stat_chunk_count)
    assert len(names) == config.expected_stat_feature_dim()
