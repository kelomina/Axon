import csv
import json
import os
import pytest
import sys
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_generalization_group_split import (  # noqa: E402
    build_cache_matched_split,
    build_seed_plan,
    _cache_manifest_path,
    parse_seed_list,
    summarize_multiseed_results,
    set_toml_value,
    validate_manifest_config_compatibility,
    write_seed_config,
)


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_parse_seed_list_rejects_duplicates():
    assert parse_seed_list("42,43,44") == [42, 43, 44]
    try:
        parse_seed_list("42,42")
    except ValueError as exc:
        assert "Duplicate seeds" in str(exc)
    else:
        raise AssertionError("Expected duplicate seeds to fail")


def test_set_toml_value_updates_only_experiment_seed():
    text = """[experiment]
seed_extra = 99
seed = 1

[training]
seed = 2
"""

    updated = set_toml_value(text, "experiment", "seed", "43")

    assert "seed_extra = 99" in updated
    assert "[experiment]\nseed_extra = 99\nseed = 43" in updated
    assert "[training]\nseed = 2" in updated


def test_write_seed_config_and_plan_use_seed_specific_names():
    with _case_dir("generalization_seed_plan") as tmp_path:
        base_config = tmp_path / "base.toml"
        seed_config = tmp_path / "seed_43.toml"
        base_config.write_text("[experiment]\nseed = 42\n", encoding="utf-8")

        write_seed_config(base_config, seed_config, seed=43)
        seed_text = seed_config.read_text(encoding="utf-8")
        plan = build_seed_plan(tmp_path / "out", [42, 43])

    assert "seed = 43" in seed_text
    assert len(plan) == 6
    assert plan[0]["name"] == "seed_42_exp0_baseline"
    assert plan[-1]["name"] == "seed_43_exp4_near_threshold"


def test_cache_manifest_path_prefers_explicit_manifest():
    with _case_dir("explicit_cache_manifest") as tmp_path:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        old_manifest = cache_dir / "manifest_old.json"
        new_manifest = cache_dir / "manifest_new.json"
        old_manifest.write_text('{"samples": []}', encoding="utf-8")
        new_manifest.write_text('{"samples": []}', encoding="utf-8")

        selected = _cache_manifest_path(cache_dir, old_manifest)

    assert selected.name == "manifest_old.json"


def test_cache_manifest_path_falls_back_to_latest_manifest():
    with _case_dir("latest_cache_manifest") as tmp_path:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        old_manifest = cache_dir / "manifest_old.json"
        new_manifest = cache_dir / "manifest_new.json"
        old_manifest.write_text('{"samples": []}', encoding="utf-8")
        new_manifest.write_text('{"samples": []}', encoding="utf-8")
        old_time = 1_700_000_000
        new_time = 1_700_000_100
        os.utime(old_manifest, (old_time, old_time))
        os.utime(new_manifest, (new_time, new_time))

        selected = _cache_manifest_path(cache_dir)

    assert selected.name == "manifest_new.json"


def test_validate_manifest_config_compatibility_rejects_wrong_byte_length():
    with _case_dir("manifest_config_compat") as tmp_path:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            """[model]
max_byte_length = 512
pe_feature_dim = 256
stat_feature_dim = 49
lightweight_feature_dim = 256

[data]
strict_pe_parsing = true
allow_pe_fallback = false
pe_schema_version = "fixed_v2"
pe_fixed_section_slots = 32
""",
            encoding="utf-8",
        )
        manifest = {
            "max_byte_length": 64,
            "pe_feature_dim": 256,
            "stat_feature_dim": 49,
            "lightweight_feature_dim": 256,
            "strict_pe_parsing": True,
            "allow_pe_fallback": False,
            "pe_schema_version": "fixed_v2",
            "pe_fixed_section_slots": 32,
        }

        try:
            validate_manifest_config_compatibility(manifest, [config_path])
        except ValueError as exc:
            assert "max_byte_length=64" in str(exc)
            assert "max_byte_length=512" in str(exc)
        else:
            raise AssertionError("Expected incompatible manifest to fail")


def test_build_cache_matched_split_writes_source_path_for_dataset_matching():
    with _case_dir("cache_matched_split_source_path") as tmp_path:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        raw_split = tmp_path / "raw_split.csv"
        output_split = tmp_path / "split.csv"
        source_sha = "a" * 64
        raw_source = tmp_path / f"{source_sha}.exe"
        manifest_source = tmp_path / "data" / f"{source_sha}.exe"
        manifest_cache = cache_dir / "sample_ee122d6c.npz"
        manifest = {
            "samples": [
                {
                    "source_path": str(manifest_source),
                    "cache_path": str(manifest_cache),
                    "label": 1,
                    "source_sha256": source_sha,
                }
            ]
        }
        manifest_path = cache_dir / "manifest_ee122d6c.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with raw_split.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source_path", "split", "label"])
            writer.writeheader()
            writer.writerow({"source_path": str(raw_source), "split": "train", "label": "1"})
            writer.writerow({"source_path": str(raw_source), "split": "val", "label": "1"})
            writer.writerow({"source_path": str(raw_source), "split": "test", "label": "1"})

        summary = build_cache_matched_split(raw_split, output_split, cache_dir, manifest_path)

        rows = list(csv.DictReader(output_split.open("r", encoding="utf-8-sig", newline="")))

    assert summary["matched"] == 3
    assert rows[0]["source_path"] == str(manifest_source)
    assert rows[0]["cache_path"] == str(manifest_cache)


def test_summarize_multiseed_results_compares_against_same_seed_baseline():
    results = [
        {
            "experiment": "seed_42_exp0_baseline",
            "status": "success",
            "val_f1": 0.90,
            "test_f1": 0.80,
            "test": {"false_positive": 10, "false_negative": 20},
        },
        {
            "experiment": "seed_42_exp1_byte_noise",
            "status": "success",
            "val_f1": 0.91,
            "test_f1": 0.81,
            "test": {"false_positive": 11, "false_negative": 18},
        },
        {
            "experiment": "seed_43_exp0_baseline",
            "status": "success",
            "val_f1": 0.70,
            "test_f1": 0.60,
            "test": {"false_positive": 30, "false_negative": 40},
        },
        {
            "experiment": "seed_43_exp1_byte_noise",
            "status": "success",
            "val_f1": 0.72,
            "test_f1": 0.50,
            "test": {"false_positive": 28, "false_negative": 55},
        },
    ]

    summary = summarize_multiseed_results(results)
    deltas = {
        (row["seed"], row["base_experiment"]): row
        for row in summary["per_seed_delta"]
    }

    assert deltas[(42, "exp1_byte_noise")]["delta_test_f1_vs_seed_baseline"] == pytest.approx(0.01)
    assert deltas[(43, "exp1_byte_noise")]["delta_test_f1_vs_seed_baseline"] == pytest.approx(-0.10)
    assert deltas[(43, "exp1_byte_noise")]["delta_test_fn_vs_seed_baseline"] == 15
