#!/usr/bin/env python3
"""Group-isolated generalization comparison runner.

这个脚本用于真正按 group-isolated split 评估泛化能力。

和 scripts/run_comparison_from_cache.py 不同：
- 不再从 cache 随机抽样。
- 不再假设前半是良性、后半是恶意。
- 直接使用 reports/raw_group_diagnostics/group_isolated_split.csv。
- 只跑 baseline、byte_noise、near_threshold 三个当前最有价值的候选。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
import tomllib
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


DEFAULT_SPLIT_FILE = PROJECT_ROOT / "reports" / "raw_group_diagnostics" / "group_isolated_split.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "generalization_group_isolated"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / ".cache"

CACHE_PROTOCOL_FIELDS = [
    ("model", "max_byte_length", "max_byte_length"),
    ("model", "pe_feature_dim", "pe_feature_dim"),
    ("model", "stat_feature_dim", "stat_feature_dim"),
    ("model", "lightweight_feature_dim", "lightweight_feature_dim"),
    ("data", "strict_pe_parsing", "strict_pe_parsing"),
    ("data", "allow_pe_fallback", "allow_pe_fallback"),
    ("data", "pe_schema_version", "pe_schema_version"),
    ("data", "pe_fixed_section_slots", "pe_fixed_section_slots"),
]


EXPERIMENTS = [
    {
        "name": "exp0_baseline",
        "config": PROJECT_ROOT / "config" / "default_config.toml",
        "why": "当前可用主线，作为泛化基准。",
    },
    {
        "name": "exp1_byte_noise",
        "config": PROJECT_ROOT / "config" / "exp1_byte_noise.toml",
        "why": "上一轮最接近 baseline 的轻量增强。",
    },
    {
        "name": "exp4_near_threshold",
        "config": PROJECT_ROOT / "config" / "exp4_near_threshold.toml",
        "why": "上一轮第二接近 baseline，目标是改善边界样本。",
    },
]


def parse_seed_list(seed_text: str) -> list[int]:
    seeds = [int(item.strip()) for item in seed_text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds are not allowed: {seed_text}")
    return seeds


def set_toml_value(text: str, section: str, key: str, value: str) -> str:
    """Update one exact TOML key inside one exact section."""
    lines = text.splitlines()
    section_header = f"[{section}]"
    in_section = False
    changed = False
    key_pattern = re.compile(rf"^(\s*{re.escape(key)}\s*)=(.*)$")
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped == section_header
            continue
        if not in_section:
            continue
        match = key_pattern.match(line)
        if match:
            lines[index] = f"{match.group(1)}= {value}"
            changed = True
            break
    if not changed:
        raise ValueError(f"Config key not found: [{section}].{key}")
    return "\n".join(lines) + "\n"


def write_seed_config(base_config: Path, output_path: Path, seed: int) -> None:
    text = base_config.read_text(encoding="utf-8")
    text = set_toml_value(text, "experiment", "seed", str(seed))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def build_seed_plan(output_dir: Path, seeds: list[int]) -> list[dict[str, Any]]:
    plan = []
    config_dir = output_dir / "seed_configs"
    for seed in seeds:
        for exp in EXPERIMENTS:
            config_path = config_dir / f"seed_{seed}_{exp['name']}.toml"
            plan.append(
                {
                    "seed": seed,
                    "name": f"seed_{seed}_{exp['name']}",
                    "base_experiment": exp["name"],
                    "base_config": str(exp["config"]),
                    "config": str(config_path),
                    "why": exp["why"],
                }
            )
    return plan


def write_seed_configs(plan: list[dict[str, Any]]) -> None:
    for item in plan:
        write_seed_config(
            base_config=Path(item["base_config"]),
            output_path=Path(item["config"]),
            seed=int(item["seed"]),
        )


def summarize_split(split_file: Path) -> dict[str, Any]:
    """读取 split CSV，只统计元数据，不加载样本内容。"""
    split_counts: Counter[str] = Counter()
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rare_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_counts: dict[str, set[str]] = defaultdict(set)

    with split_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"source_path", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Split file missing required columns: {sorted(missing)}")

        for row in reader:
            split = row["split"]
            split_counts[split] += 1

            label = row.get("label")
            if label not in (None, ""):
                label_counts[split][str(label)] += 1

            rare = row.get("is_rare_group")
            if rare not in (None, ""):
                rare_counts[split][str(rare)] += 1

            group_id = row.get("group_id")
            if group_id:
                group_counts[split].add(group_id)

    return {
        "path": str(split_file),
        "split_counts": dict(split_counts),
        "label_counts": {split: dict(counts) for split, counts in label_counts.items()},
        "rare_counts": {split: dict(counts) for split, counts in rare_counts.items()},
        "group_counts": {split: len(groups) for split, groups in group_counts.items()},
    }


def _cache_manifest_path(cache_dir: Path, cache_manifest: Path | None = None) -> Path:
    """找到当前 cache 目录下的 manifest。

    当前项目的 cache 文件名带配置哈希，例如 manifest_6ea52de6.json。
    正式复验时应显式传入 cache_manifest，避免目录里新生成的 manifest
    因修改时间更晚而改变实验口径。
    """
    if cache_manifest is not None:
        manifest_path = cache_manifest
        if not manifest_path.is_absolute():
            manifest_path = PROJECT_ROOT / manifest_path
        manifest_path = manifest_path.resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")
        return manifest_path

    manifests = sorted(cache_dir.glob("manifest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not manifests:
        raise FileNotFoundError(f"No cache manifest found in {cache_dir}")
    return manifests[0]


def _cache_protocol_from_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("rb") as f:
        config = tomllib.load(f)

    values: dict[str, Any] = {}
    for section, config_key, manifest_key in CACHE_PROTOCOL_FIELDS:
        section_values = config.get(section, {})
        if config_key in section_values:
            values[manifest_key] = section_values[config_key]
    return values


def validate_manifest_config_compatibility(
    manifest: dict[str, Any],
    config_paths: list[Path],
) -> None:
    """确认 cache manifest 与所有待跑实验的输入规格一致。"""
    expected_by_key: dict[str, tuple[Any, Path]] = {}
    errors: list[str] = []

    for config_path in config_paths:
        config_values = _cache_protocol_from_config(config_path)
        for key, value in config_values.items():
            if key in expected_by_key and expected_by_key[key][0] != value:
                prev_value, prev_path = expected_by_key[key]
                errors.append(
                    f"{config_path} has {key}={value}, but {prev_path} has {key}={prev_value}"
                )
            else:
                expected_by_key.setdefault(key, (value, config_path))

            if key in manifest and manifest.get(key) != value:
                errors.append(
                    f"manifest {key}={manifest.get(key)} does not match {config_path} {key}={value}"
                )

    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(f"Cache manifest is incompatible with experiment configs:\n  - {joined}")


def _sha_from_raw_source(source_path: str) -> str:
    """从 raw split 的原始 PE 路径提取 SHA256 文件名。"""
    return Path(source_path).stem.casefold()


def build_cache_matched_split(
    raw_split_file: Path,
    output_path: Path,
    cache_dir: Path,
    cache_manifest: Path | None = None,
    experiment_config_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """把 raw group split 转成 FeatureCacheDataset 可以匹配的 split。

    raw group split 的 source_path 指向原始 .exe；当前 cache manifest 的
    source_path 指向 .npz。两者通过 raw 文件名 SHA256 和 manifest 的
    source_sha256 对齐。
    """
    manifest_path = _cache_manifest_path(cache_dir, cache_manifest)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if experiment_config_paths:
        validate_manifest_config_compatibility(manifest, experiment_config_paths)

    cache_by_sha: dict[str, dict[str, Any]] = {}
    duplicate_sha = 0
    for sample in manifest.get("samples", []):
        sha = str(sample.get("source_sha256") or "").casefold()
        if not sha:
            continue
        if sha in cache_by_sha:
            duplicate_sha += 1
            continue
        cache_by_sha[sha] = sample

    matched = 0
    missing = 0
    label_mismatch = 0
    converted_counts: Counter[str] = Counter()
    converted_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    converted_rare_counts: dict[str, Counter[str]] = defaultdict(Counter)
    converted_group_counts: dict[str, set[str]] = defaultdict(set)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "split",
        "label",
        "cache_path",
        "source_sha256",
        "raw_source_path",
        "group_id",
        "group_size",
        "is_rare_group",
        "group_source",
    ]

    with raw_split_file.open("r", encoding="utf-8-sig", newline="") as src, output_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            sha = _sha_from_raw_source(row.get("source_path", ""))
            sample = cache_by_sha.get(sha)
            if sample is None:
                missing += 1
                continue

            raw_label = str(row.get("label", "")).strip()
            cache_label = str(int(sample["label"]))
            if raw_label and raw_label != cache_label:
                label_mismatch += 1
                continue

            split = row["split"]
            cache_path = sample.get("cache_path") or sample.get("source_path")
            source_path = sample.get("source_path") or cache_path
            writer.writerow(
                {
                    "source_path": source_path,
                    "split": split,
                    "label": cache_label,
                    "cache_path": cache_path,
                    "source_sha256": sha,
                    "raw_source_path": row.get("source_path", ""),
                    "group_id": row.get("group_id", ""),
                    "group_size": row.get("group_size", ""),
                    "is_rare_group": row.get("is_rare_group", ""),
                    "group_source": row.get("group_source", ""),
                }
            )

            matched += 1
            converted_counts[split] += 1
            converted_label_counts[split][cache_label] += 1
            rare = row.get("is_rare_group", "")
            if rare:
                converted_rare_counts[split][rare] += 1
            group_id = row.get("group_id")
            if group_id:
                converted_group_counts[split].add(group_id)

    if not all(converted_counts.get(split, 0) > 0 for split in ("train", "val", "test")):
        raise ValueError(f"Converted split has empty split: {dict(converted_counts)}")

    return {
        "raw_split_file": str(raw_split_file),
        "converted_split_file": str(output_path),
        "manifest_path": str(manifest_path),
        "manifest_samples": len(manifest.get("samples", [])),
        "manifest_samples_with_source_sha256": len(cache_by_sha),
        "duplicate_source_sha256": duplicate_sha,
        "matched": matched,
        "missing": missing,
        "label_mismatch": label_mismatch,
        "split_counts": dict(converted_counts),
        "label_counts": {split: dict(counts) for split, counts in converted_label_counts.items()},
        "rare_counts": {split: dict(counts) for split, counts in converted_rare_counts.items()},
        "group_counts": {split: len(groups) for split, groups in converted_group_counts.items()},
    }


def metric_to_dict(metric: Any) -> dict[str, Any]:
    """把 TrainingMetrics dataclass 转成可写入 JSON 的普通 dict。"""
    if is_dataclass(metric):
        data = asdict(metric)
    elif hasattr(metric, "__dict__"):
        data = dict(metric.__dict__)
    else:
        return {"value": str(metric)}

    clean: dict[str, Any] = {}
    for key, value in data.items():
        if hasattr(value, "item"):
            value = value.item()
        clean[key] = value
    return clean


def best_metric(metrics: list[Any], key: str = "f1") -> dict[str, Any] | None:
    """从某个阶段的指标列表中选出 F1 最高的一条。"""
    if not metrics:
        return None
    best = max(metrics, key=lambda m: getattr(m, key))
    return metric_to_dict(best)


def parse_result_identity(experiment_name: str) -> tuple[int | None, str]:
    """从 seed_42_exp0_baseline 这类名字里提取 seed 和基础实验名。"""
    match = re.match(r"^seed_(\d+)_(.+)$", experiment_name)
    if not match:
        return None, experiment_name
    return int(match.group(1)), match.group(2)


def run_experiment(
    experiment_name: str,
    config_path: Path,
    data_dir: Path,
    split_file: Path,
    output_dir: Path,
    epochs: int,
    skip_test_eval: bool,
) -> dict[str, Any]:
    """运行单个实验，并保存 val/test 的完整指标。"""
    from main import train_command

    print("\n" + "=" * 80)
    print(f"实验: {experiment_name}")
    print(f"配置: {config_path}")
    print("=" * 80)

    start_time = time.time()

    try:
        args = argparse.Namespace()
        args.command = "train"
        args.config = str(config_path)
        args.data_dir = str(data_dir)
        args.split_file = str(split_file)
        args.output_dir = str(output_dir / experiment_name)
        args.fast = False
        args.batch_size = None
        args.device = None
        args.lr = None
        args.fp16 = False
        args.enable_swanlab = False
        args.epochs = epochs
        args.samples_per_class = None
        args.extract_workers = 1
        args.extract_backend = "thread"
        args.rare_group_weighting = False
        args.singleton_group_weight = None
        args.rare_group_weight = None
        args.medium_group_weight = None
        args.skip_test_eval = skip_test_eval
        args.resume = None
        args.no_resume = True
        args.init_checkpoint = None
        args.partial_init = False

        print(f"\n开始训练 {experiment_name}...")
        result = train_command(args)

        elapsed = time.time() - start_time
        best_val = best_metric(result.get("val", [])) if result else None
        test_metrics = metric_to_dict(result["test"][-1]) if result and result.get("test") else None

        val_f1 = float(best_val["f1"]) if best_val else 0.0
        test_f1 = float(test_metrics["f1"]) if test_metrics else None

        print(f"\n[OK] {experiment_name} 完成:")
        print(f"  Best Val F1: {val_f1:.4f}")
        if test_f1 is not None:
            print(f"  Test F1: {test_f1:.4f}")
        print(f"  用时: {elapsed:.1f}秒")

        return {
            "experiment": experiment_name,
            "seed": parse_result_identity(experiment_name)[0],
            "base_experiment": parse_result_identity(experiment_name)[1],
            "config": str(config_path),
            "best_val": best_val,
            "test": test_metrics,
            "val_f1": val_f1,
            "test_f1": test_f1,
            "elapsed": elapsed,
            "status": "success",
        }
    except Exception as exc:
        elapsed = time.time() - start_time
        print(f"\n[FAIL] {experiment_name} 失败: {exc}")
        import traceback

        traceback.print_exc()
        return {
            "experiment": experiment_name,
            "seed": parse_result_identity(experiment_name)[0],
            "base_experiment": parse_result_identity(experiment_name)[1],
            "config": str(config_path),
            "best_val": None,
            "test": None,
            "val_f1": 0.0,
            "test_f1": None,
            "elapsed": elapsed,
            "status": f"failed: {exc}",
        }


def _metric_stdev(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def summarize_multiseed_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """按基础实验聚合多 seed 结果，并计算相对同 seed baseline 的差值。"""
    successful = [row for row in results if row.get("status") == "success" and row.get("test")]
    baseline_by_seed: dict[int | None, dict[str, Any]] = {}
    for row in successful:
        seed, base_experiment = parse_result_identity(row["experiment"])
        row.setdefault("seed", seed)
        row.setdefault("base_experiment", base_experiment)
        if base_experiment == "exp0_baseline":
            baseline_by_seed[seed] = row

    per_seed_delta: list[dict[str, Any]] = []
    for row in successful:
        seed, base_experiment = parse_result_identity(row["experiment"])
        row.setdefault("seed", seed)
        row.setdefault("base_experiment", base_experiment)
        baseline = baseline_by_seed.get(seed)
        if baseline is None:
            continue
        test = row.get("test") or {}
        baseline_test = baseline.get("test") or {}
        per_seed_delta.append(
            {
                "seed": seed,
                "base_experiment": base_experiment,
                "experiment": row["experiment"],
                "val_f1": row.get("val_f1"),
                "test_f1": row.get("test_f1"),
                "delta_val_f1_vs_seed_baseline": float(row.get("val_f1", 0.0)) - float(baseline.get("val_f1", 0.0)),
                "delta_test_f1_vs_seed_baseline": float(row.get("test_f1", 0.0)) - float(baseline.get("test_f1", 0.0)),
                "delta_test_fp_vs_seed_baseline": int(test.get("false_positive", 0))
                - int(baseline_test.get("false_positive", 0)),
                "delta_test_fn_vs_seed_baseline": int(test.get("false_negative", 0))
                - int(baseline_test.get("false_negative", 0)),
            }
        )

    rows_by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        _seed, base_experiment = parse_result_identity(row["experiment"])
        rows_by_base[base_experiment].append(row)

    aggregate: dict[str, dict[str, Any]] = {}
    baseline_aggregate: dict[str, Any] | None = None
    for base_experiment, rows in rows_by_base.items():
        val_f1 = [float(row["val_f1"]) for row in rows]
        test_f1 = [float(row["test_f1"]) for row in rows]
        test_fp = [int((row.get("test") or {}).get("false_positive", 0)) for row in rows]
        test_fn = [int((row.get("test") or {}).get("false_negative", 0)) for row in rows]
        data = {
            "runs": len(rows),
            "seeds": [parse_result_identity(row["experiment"])[0] for row in rows],
            "val_f1_mean": mean(val_f1),
            "val_f1_stdev": _metric_stdev(val_f1),
            "test_f1_mean": mean(test_f1),
            "test_f1_stdev": _metric_stdev(test_f1),
            "test_fp_mean": mean(test_fp),
            "test_fn_mean": mean(test_fn),
            "test_fp_values": test_fp,
            "test_fn_values": test_fn,
        }
        aggregate[base_experiment] = data
        if base_experiment == "exp0_baseline":
            baseline_aggregate = data

    if baseline_aggregate is not None:
        for base_experiment, data in aggregate.items():
            data["delta_test_f1_mean_vs_baseline"] = (
                data["test_f1_mean"] - baseline_aggregate["test_f1_mean"]
            )
            data["delta_val_f1_mean_vs_baseline"] = (
                data["val_f1_mean"] - baseline_aggregate["val_f1_mean"]
            )
            data["delta_test_fp_mean_vs_baseline"] = (
                data["test_fp_mean"] - baseline_aggregate["test_fp_mean"]
            )
            data["delta_test_fn_mean_vs_baseline"] = (
                data["test_fn_mean"] - baseline_aggregate["test_fn_mean"]
            )

    return {
        "aggregate_by_base_experiment": aggregate,
        "per_seed_delta": per_seed_delta,
    }


def save_results(output_dir: Path, split_summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    """保存 JSON 和 Markdown，方便训练中途也能看阶段结果。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    multiseed_summary = summarize_multiseed_results(results)

    payload = {
        "protocol": "group_isolated_split",
        "split": split_summary,
        "multiseed_summary": multiseed_summary,
        "results": results,
    }

    results_file = output_dir / "results.json"
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    summary_file = output_dir / "summary.md"
    baseline_by_seed: dict[int | None, dict[str, Any]] = {}
    for row in results:
        seed, base_experiment = parse_result_identity(row["experiment"])
        row.setdefault("seed", seed)
        row.setdefault("base_experiment", base_experiment)
        if base_experiment == "exp0_baseline":
            baseline_by_seed[seed] = row

    with summary_file.open("w", encoding="utf-8") as f:
        f.write("# Group-Isolated 泛化对比实验\n\n")
        f.write("## Split\n\n")
        if "converted" in split_summary:
            raw_summary = split_summary["raw"]
            converted_summary = split_summary["converted"]
            f.write(f"- Raw split file: `{raw_summary['path']}`\n")
            f.write(f"- Converted split file: `{converted_summary['converted_split_file']}`\n")
            f.write(f"- Manifest: `{converted_summary['manifest_path']}`\n")
            f.write(f"- Raw split counts: `{raw_summary['split_counts']}`\n")
            f.write(f"- Converted split counts: `{converted_summary['split_counts']}`\n")
            f.write(f"- Converted label counts: `{converted_summary['label_counts']}`\n")
            f.write(f"- Converted rare group counts: `{converted_summary['rare_counts']}`\n")
            f.write(f"- Converted group counts: `{converted_summary['group_counts']}`\n")
            f.write(
                "- Conversion misses: "
                f"missing={converted_summary['missing']}, "
                f"label_mismatch={converted_summary['label_mismatch']}, "
                f"duplicate_source_sha256={converted_summary['duplicate_source_sha256']}\n\n"
            )
        else:
            f.write(f"- Split file: `{split_summary['path']}`\n")
            f.write(f"- Split counts: `{split_summary['split_counts']}`\n")
            f.write(f"- Label counts: `{split_summary['label_counts']}`\n")
            f.write(f"- Rare group counts: `{split_summary['rare_counts']}`\n")
            f.write(f"- Group counts: `{split_summary['group_counts']}`\n\n")

        f.write("## Results\n\n")
        f.write("| Seed | 实验 | Best Val F1 | Val Δ vs seed baseline | Test F1 | Test Δ vs seed baseline | Test FP | Test FN | FP Δ | FN Δ | 用时(秒) | 状态 |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in results:
            test = row.get("test") or {}
            seed, _base_experiment = parse_result_identity(row["experiment"])
            baseline = baseline_by_seed.get(seed)
            baseline_val = baseline["val_f1"] if baseline else row["val_f1"]
            baseline_test = baseline["test_f1"] if baseline and baseline.get("test_f1") is not None else None
            baseline_test_metrics = baseline.get("test") if baseline else {}
            val_delta = row["val_f1"] - baseline_val if baseline_val else 0.0
            test_f1 = row.get("test_f1")
            test_delta = (
                test_f1 - baseline_test
                if test_f1 is not None and baseline_test is not None
                else None
            )
            fp_delta = (
                int(test.get("false_positive", 0)) - int(baseline_test_metrics.get("false_positive", 0))
                if baseline_test_metrics and test
                else None
            )
            fn_delta = (
                int(test.get("false_negative", 0)) - int(baseline_test_metrics.get("false_negative", 0))
                if baseline_test_metrics and test
                else None
            )
            f.write(
                "| {seed} | {experiment} | {val_f1:.4f} | {val_delta:+.4f} | {test_f1} | {test_delta} | "
                "{fp} | {fn} | {fp_delta} | {fn_delta} | {elapsed:.1f} | {status} |\n".format(
                    seed=seed if seed is not None else "N/A",
                    experiment=row["experiment"],
                    val_f1=row["val_f1"],
                    val_delta=val_delta,
                    test_f1=f"{test_f1:.4f}" if test_f1 is not None else "N/A",
                    test_delta=f"{test_delta:+.4f}" if test_delta is not None else "N/A",
                    fp=test.get("false_positive", "N/A"),
                    fn=test.get("false_negative", "N/A"),
                    fp_delta=f"{fp_delta:+d}" if fp_delta is not None else "N/A",
                    fn_delta=f"{fn_delta:+d}" if fn_delta is not None else "N/A",
                    elapsed=row["elapsed"],
                    status=row["status"],
                )
            )

        aggregate = multiseed_summary["aggregate_by_base_experiment"]
        if aggregate:
            f.write("\n## Multi-Seed Aggregate\n\n")
            f.write(
                "| 基础实验 | Runs | Val F1 mean | Val F1 std | Test F1 mean | Test F1 std | "
                "Test Δ mean | Test FP mean | Test FN mean | FP Δ mean | FN Δ mean |\n"
            )
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for base_experiment, data in aggregate.items():
                f.write(
                    "| {base} | {runs} | {val_mean:.4f} | {val_std:.4f} | {test_mean:.4f} | "
                    "{test_std:.4f} | {test_delta:+.4f} | {fp_mean:.1f} | {fn_mean:.1f} | "
                    "{fp_delta:+.1f} | {fn_delta:+.1f} |\n".format(
                        base=base_experiment,
                        runs=data["runs"],
                        val_mean=data["val_f1_mean"],
                        val_std=data["val_f1_stdev"],
                        test_mean=data["test_f1_mean"],
                        test_std=data["test_f1_stdev"],
                        test_delta=data.get("delta_test_f1_mean_vs_baseline", 0.0),
                        fp_mean=data["test_fp_mean"],
                        fn_mean=data["test_fn_mean"],
                        fp_delta=data.get("delta_test_fp_mean_vs_baseline", 0.0),
                        fn_delta=data.get("delta_test_fn_mean_vs_baseline", 0.0),
                    )
                )

    print(f"\n[OK] 结果已保存到:")
    print(f"  JSON: {results_file}")
    print(f"  Markdown: {summary_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run group-isolated generalization comparison.")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated experiment seeds. Use 42,43,44 for multi-seed confirmation.",
    )
    parser.add_argument("--skip-test-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=None,
        help="Pin a specific data/.cache/manifest_*.json for reproducible split conversion.",
    )
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data"
    raw_split_file = args.split_file.resolve()
    output_dir = args.output_dir.resolve()
    seeds = parse_seed_list(args.seeds)

    if not raw_split_file.exists():
        raise FileNotFoundError(f"Split file not found: {raw_split_file}")

    print("=" * 100)
    print("Axon v2.6 Group-Isolated 泛化对比实验")
    print("=" * 100)
    print(f"Raw split: {raw_split_file}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Seeds: {seeds}")
    print(f"Skip test eval: {args.skip_test_eval}")
    print(f"Cache manifest: {args.cache_manifest or 'latest manifest in data/.cache'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_split_file, output_dir / "raw_group_isolated_split.csv")

    raw_split_summary = summarize_split(raw_split_file)
    converted_split_file = output_dir / "split.csv"
    conversion_summary = build_cache_matched_split(
        raw_split_file=raw_split_file,
        output_path=converted_split_file,
        cache_dir=DEFAULT_CACHE_DIR,
        cache_manifest=args.cache_manifest,
        experiment_config_paths=[Path(exp["config"]) for exp in EXPERIMENTS],
    )
    split_summary = {
        "raw": raw_split_summary,
        "converted": conversion_summary,
    }
    print("\nSplit summary:")
    print(json.dumps(split_summary, ensure_ascii=False, indent=2))

    seed_plan = build_seed_plan(output_dir, seeds)
    write_seed_configs(seed_plan)
    seed_plan_path = output_dir / "seed_plan.json"
    seed_plan_path.write_text(
        json.dumps(
            {
                "schema": "axon_group_isolated_multiseed_plan_v1",
                "seeds": seeds,
                "experiments": seed_plan,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n将运行实验:")
    for item in seed_plan:
        print(f"  - {item['name']}: {item['why']}")
    print(f"\nSeed plan: {seed_plan_path}")

    if args.dry_run:
        save_results(output_dir, split_summary, [])
        print("\nDry run complete. 未启动训练。")
        return

    results: list[dict[str, Any]] = []
    for exp in seed_plan:
        result = run_experiment(
            experiment_name=exp["name"],
            config_path=Path(exp["config"]),
            data_dir=data_dir,
            split_file=converted_split_file,
            output_dir=output_dir,
            epochs=args.epochs,
            skip_test_eval=args.skip_test_eval,
        )
        results.append(result)
        save_results(output_dir, split_summary, results)

    print("\n" + "=" * 100)
    print("Group-Isolated 泛化对比实验完成")
    print("=" * 100)


if __name__ == "__main__":
    main()
