from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "hard_family_finetune"
    / "clean_hyperparam_search"
    / "run_f1_hyperparam_probe.py"
)


def load_probe_module():
    spec = importlib.util.spec_from_file_location("run_f1_hyperparam_probe", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_set_toml_value_updates_only_exact_key_in_section():
    module = load_probe_module()
    text = """[experiment]
seed_extra = 99
seed = 1

[training]
seed = 2
"""

    updated = module.set_toml_value(text, "experiment", "seed", "42")

    assert "seed_extra = 99" in updated
    assert "[experiment]\nseed_extra = 99\nseed = 42" in updated
    assert "[training]\nseed = 2" in updated


def test_set_toml_value_rejects_missing_key():
    module = load_probe_module()
    text = """[training]
learning_rate = 0.0001
"""

    try:
        module.set_toml_value(text, "training", "label_smoothing", "0")
    except ValueError as exc:
        assert "[training].label_smoothing" in str(exc)
    else:
        raise AssertionError("Expected missing key to raise ValueError")


def test_variant_names_are_unique_and_include_probe_base():
    module = load_probe_module()
    names = [variant.name for variant in module.VARIANTS]

    assert len(names) == len(set(names))
    assert "base_lr1e4" in names
    assert "lr12e4" in names


def test_manifest_items_include_config_path():
    module = load_probe_module()
    variant = module.VARIANTS[0]

    assert variant.config_path.name == f"probe_{variant.name}.toml"
