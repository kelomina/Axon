#!/usr/bin/env python3
"""Search compact PE/stat feature subsets with a genetic algorithm.

这个脚本是一个离线实验工具，不修改模型结构，也不改缓存数据。
它做的事情很简单：给 PE 特征和统计特征各准备一张 0/1 开关表，
开关为 1 的特征照常喂给模型，开关为 0 的特征在输入端置零。
遗传算法会反复尝试不同开关组合，目标是在验证集指标尽量好的同时，
尽可能少保留特征，帮助我们发现噪声和冗余维度。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AxonExperimentConfig, TrainingConfig  # noqa: E402
from dataset import FeatureCacheDataset, NPZDataLoader, create_split_from_file, create_stratified_split  # noqa: E402
from model import AxonMalwareModel  # noqa: E402
from security import load_safe_checkpoint  # noqa: E402


Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
MAX_JSON_INPUT_BYTES = 8 * 1024 * 1024


def load_json_object(path: Path) -> dict:
    if path.stat().st_size > MAX_JSON_INPUT_BYTES:
        raise ValueError(f"JSON input is too large: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = handle.read(8388609)
    if len(raw.encode("utf-8")) > MAX_JSON_INPUT_BYTES:
        raise ValueError(f"JSON input exceeded bounded read limit: {path}")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return payload


@dataclass(frozen=True)
class FeatureMaskSpec:
    """描述 GA 个体如何映射到模型输入 mask。

    GA 个体只搜索有意义的 PE 维度和 stat 维度。以默认 fixed_v2 配置为例，
    pe_feature_dim 是 256，但实际写入只有 143 维，后面的补零维度默认不参与搜索。
    """

    pe_feature_dim: int
    stat_feature_dim: int
    pe_search_dim: int

    @property
    def search_dim(self) -> int:
        return self.pe_search_dim + self.stat_feature_dim

    @property
    def ignored_pe_dim(self) -> int:
        return max(0, self.pe_feature_dim - self.pe_search_dim)

    def split_individual(self, individual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if individual.shape[0] != self.search_dim:
            raise ValueError(
                f"individual length {individual.shape[0]} does not match search_dim {self.search_dim}"
            )
        pe_bits = individual[: self.pe_search_dim]
        stat_bits = individual[self.pe_search_dim :]
        return pe_bits.astype(bool), stat_bits.astype(bool)

    def to_model_masks(self, individual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """把短的 GA 个体展开成模型需要的 PE/stat 输入 mask。"""
        pe_bits, stat_bits = self.split_individual(individual)
        pe_mask = np.zeros(self.pe_feature_dim, dtype=np.float32)
        pe_mask[: self.pe_search_dim] = pe_bits.astype(np.float32)
        stat_mask = stat_bits.astype(np.float32)
        return pe_mask, stat_mask

    def selected_indices(self, individual: np.ndarray) -> tuple[list[int], list[int]]:
        pe_bits, stat_bits = self.split_individual(individual)
        pe_indices = np.flatnonzero(pe_bits).astype(int).tolist()
        stat_indices = np.flatnonzero(stat_bits).astype(int).tolist()
        return pe_indices, stat_indices


@dataclass(frozen=True)
class FitnessConfig:
    objective: str = "f1"
    feature_penalty: float = 0.01
    fp_penalty: float = 0.0
    fn_penalty: float = 0.0
    min_objective: Optional[float] = None
    below_min_penalty: float = 2.0
    min_recall: Optional[float] = None
    max_fn_rate: Optional[float] = None
    recall_guard_penalty: float = 3.0
    fn_guard_penalty: float = 3.0


@dataclass(frozen=True)
class GeneticConfig:
    population_size: int = 24
    generations: int = 12
    elite_size: int = 3
    tournament_size: int = 3
    mutation_rate: float = 0.02
    crossover_rate: float = 0.85
    target_keep_ratio: float = 0.45
    min_pe_features: int = 1
    min_stat_features: int = 1
    max_leaderboard_size: int = 100


def parse_thresholds(text: str) -> list[float]:
    thresholds = []
    for item in text.split(","):
        item = item.strip()
        if item:
            thresholds.append(float(item))
    if not thresholds:
        raise ValueError("at least one threshold is required")
    for threshold in thresholds:
        if not 0 < threshold < 1:
            raise ValueError(f"threshold must be in (0, 1): {threshold}")
    return thresholds


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> Optional[float]:
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return None


def compute_threshold_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> dict:
    preds = (probs >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)

    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())

    return {
        "threshold": float(threshold),
        "sample_count": int(labels.shape[0]),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc": _safe_auc(labels, probs),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "errors": int(fp + fn),
        "prob_mean": float(probs.mean()) if probs.size else 0.0,
        "prob_std": float(probs.std()) if probs.size else 0.0,
    }


def choose_best_metrics(labels: np.ndarray, probs: np.ndarray, thresholds: Sequence[float]) -> dict:
    rows = [compute_threshold_metrics(labels, probs, threshold) for threshold in thresholds]
    return max(
        rows,
        key=lambda item: (
            item["f1"],
            -item["errors"],
            item["precision"],
            item["recall"],
            item["accuracy"],
        ),
    )


def compute_fitness(metrics: dict, kept_ratio: float, fitness_config: FitnessConfig) -> float:
    objective = fitness_config.objective
    if objective == "auc":
        objective_value = metrics["auc"] if metrics["auc"] is not None else 0.0
    elif objective in {"accuracy", "precision", "recall", "f1"}:
        objective_value = float(metrics[objective])
    else:
        raise ValueError(f"unsupported objective: {objective}")

    sample_count = max(1, int(metrics.get("sample_count", 0)))
    fp_rate = float(metrics.get("fp", 0)) / sample_count
    fn_rate = float(metrics.get("fn", 0)) / sample_count
    recall_gap = 0.0
    if fitness_config.min_recall is not None:
        recall_gap = max(0.0, float(fitness_config.min_recall) - float(metrics.get("recall", 0.0)))
    fn_gap = 0.0
    if fitness_config.max_fn_rate is not None:
        fn_gap = max(0.0, fn_rate - float(fitness_config.max_fn_rate))

    below_min = 0.0
    if fitness_config.min_objective is not None:
        below_min = max(0.0, float(fitness_config.min_objective) - objective_value)

    # 这里的惩罚是业务含义：指标相近时，优先选择更短的特征清单；
    # 如果误报或漏报更敏感，也可以通过 fp/fn penalty 明确压低这类候选。
    # min_objective 则是保护线：候选低于完整特征基线太多时，不让“少特征”
    # 掩盖“准确率已明显变差”这个事实。
    return float(
        objective_value
        - fitness_config.feature_penalty * kept_ratio
        - fitness_config.fp_penalty * fp_rate
        - fitness_config.fn_penalty * fn_rate
        - fitness_config.below_min_penalty * below_min
        - fitness_config.recall_guard_penalty * recall_gap
        - fitness_config.fn_guard_penalty * fn_gap
    )


def _candidate_key(individual: np.ndarray) -> str:
    return "".join("1" if value else "0" for value in individual.astype(bool).tolist())


def repair_individual(
    individual: np.ndarray,
    spec: FeatureMaskSpec,
    rng: random.Random,
    min_pe_features: int,
    min_stat_features: int,
) -> np.ndarray:
    """确保候选不会变成“几乎什么都不看”的无效清单。"""
    fixed = individual.astype(bool).copy()
    pe_bits, stat_bits = spec.split_individual(fixed)

    def activate_random(bits: np.ndarray, count: int) -> None:
        missing = max(0, count - int(bits.sum()))
        if missing == 0 or bits.shape[0] == 0:
            return
        off_indices = np.flatnonzero(~bits).astype(int).tolist()
        rng.shuffle(off_indices)
        for index in off_indices[:missing]:
            bits[index] = True

    activate_random(pe_bits, min(min_pe_features, spec.pe_search_dim))
    activate_random(stat_bits, min(min_stat_features, spec.stat_feature_dim))
    fixed[: spec.pe_search_dim] = pe_bits
    fixed[spec.pe_search_dim :] = stat_bits
    return fixed


def make_random_individual(
    spec: FeatureMaskSpec,
    rng: random.Random,
    target_keep_ratio: float,
    min_pe_features: int,
    min_stat_features: int,
) -> np.ndarray:
    keep_ratio = min(0.98, max(0.01, rng.gauss(target_keep_ratio, 0.15)))
    bits = np.array([rng.random() < keep_ratio for _ in range(spec.search_dim)], dtype=bool)
    return repair_individual(bits, spec, rng, min_pe_features, min_stat_features)


def make_initial_population(
    spec: FeatureMaskSpec,
    ga_config: GeneticConfig,
    rng: random.Random,
    warm_start_masks: Optional[Sequence[np.ndarray]] = None,
) -> list[np.ndarray]:
    population = [np.ones(spec.search_dim, dtype=bool)]
    if warm_start_masks:
        for mask in warm_start_masks:
            population.append(
                repair_individual(
                    mask,
                    spec,
                    rng,
                    ga_config.min_pe_features,
                    ga_config.min_stat_features,
                )
            )

    while len(population) < ga_config.population_size:
        population.append(
            make_random_individual(
                spec,
                rng,
                ga_config.target_keep_ratio,
                ga_config.min_pe_features,
                ga_config.min_stat_features,
            )
        )
    return population[: ga_config.population_size]


def crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: random.Random) -> np.ndarray:
    if parent_a.shape != parent_b.shape:
        raise ValueError("parents must have the same shape")
    child = parent_a.copy()
    for index in range(child.shape[0]):
        if rng.random() < 0.5:
            child[index] = parent_b[index]
    return child


def mutate(individual: np.ndarray, rng: random.Random, mutation_rate: float) -> np.ndarray:
    mutated = individual.copy()
    for index in range(mutated.shape[0]):
        if rng.random() < mutation_rate:
            mutated[index] = not bool(mutated[index])
    return mutated


def tournament_select(results: Sequence[dict], rng: random.Random, tournament_size: int) -> np.ndarray:
    size = min(len(results), max(1, tournament_size))
    contenders = rng.sample(list(results), size)
    winner = max(contenders, key=lambda item: item["fitness"])
    return np.array(winner["individual"], dtype=bool)


def summarize_individual(
    individual: np.ndarray,
    spec: FeatureMaskSpec,
    metrics: dict,
    fitness_config: FitnessConfig,
) -> dict:
    kept_total = int(individual.sum())
    kept_ratio = kept_total / max(1, spec.search_dim)
    pe_indices, stat_indices = spec.selected_indices(individual)
    return {
        "key": _candidate_key(individual),
        "individual": individual.astype(bool).tolist(),
        "fitness": compute_fitness(metrics, kept_ratio, fitness_config),
        "kept_total": kept_total,
        "kept_ratio": float(kept_ratio),
        "kept_pe": len(pe_indices),
        "kept_stat": len(stat_indices),
        "selected_pe_indices": pe_indices,
        "selected_stat_indices": stat_indices,
        "metrics": metrics,
    }


def run_genetic_search(
    spec: FeatureMaskSpec,
    ga_config: GeneticConfig,
    fitness_config: FitnessConfig,
    evaluate_individual: Callable[[np.ndarray], dict],
    seed: int = 42,
    warm_start_masks: Optional[Sequence[np.ndarray]] = None,
) -> dict:
    rng = random.Random(seed)
    population = make_initial_population(spec, ga_config, rng, warm_start_masks)
    cache: dict[str, dict] = {}
    generation_summaries = []
    best_result: Optional[dict] = None

    def evaluate_cached(individual: np.ndarray) -> dict:
        key = _candidate_key(individual)
        if key not in cache:
            metrics = evaluate_individual(individual)
            cache[key] = summarize_individual(individual, spec, metrics, fitness_config)
        return cache[key]

    for generation in range(ga_config.generations + 1):
        results = [evaluate_cached(individual) for individual in population]
        results = sorted(
            results,
            key=lambda item: (item["fitness"], item["metrics"]["f1"], -item["kept_total"]),
            reverse=True,
        )
        if best_result is None or results[0]["fitness"] > best_result["fitness"]:
            best_result = results[0]

        generation_summaries.append(
            {
                "generation": generation,
                "best_fitness": float(results[0]["fitness"]),
                "best_f1": float(results[0]["metrics"]["f1"]),
                "best_auc": results[0]["metrics"]["auc"],
                "best_kept_total": int(results[0]["kept_total"]),
                "best_kept_pe": int(results[0]["kept_pe"]),
                "best_kept_stat": int(results[0]["kept_stat"]),
                "unique_evaluated": int(len(cache)),
            }
        )

        if generation == ga_config.generations:
            break

        next_population = [
            np.array(item["individual"], dtype=bool)
            for item in results[: max(1, ga_config.elite_size)]
        ]

        while len(next_population) < ga_config.population_size:
            parent_a = tournament_select(results, rng, ga_config.tournament_size)
            if rng.random() < ga_config.crossover_rate:
                parent_b = tournament_select(results, rng, ga_config.tournament_size)
                child = crossover(parent_a, parent_b, rng)
            else:
                child = parent_a.copy()
            child = mutate(child, rng, ga_config.mutation_rate)
            child = repair_individual(
                child,
                spec,
                rng,
                ga_config.min_pe_features,
                ga_config.min_stat_features,
            )
            next_population.append(child)
        population = next_population

    leaderboard = sorted(
        cache.values(),
        key=lambda item: (item["fitness"], item["metrics"]["f1"], -item["kept_total"]),
        reverse=True,
    )[: max(1, int(ga_config.max_leaderboard_size))]
    return {
        "best": best_result,
        "leaderboard": leaderboard,
        "generations": generation_summaries,
        "evaluated_candidates": int(len(cache)),
    }


def build_feature_mask_spec(
    config: AxonExperimentConfig,
    include_pe_padding: bool = False,
    pe_search_dim: Optional[int] = None,
) -> FeatureMaskSpec:
    if pe_search_dim is not None:
        search_dim = pe_search_dim
    elif include_pe_padding:
        search_dim = config.pe_feature_dim
    elif getattr(config, "pe_schema_version", "legacy_dynamic") == "fixed_v2":
        search_dim = min(config.pe_feature_dim, config.fixed_pe_schema_used_dim())
    else:
        search_dim = config.pe_feature_dim

    if search_dim <= 0 or search_dim > config.pe_feature_dim:
        raise ValueError(
            f"pe_search_dim must be in [1, {config.pe_feature_dim}], got {search_dim}"
        )

    return FeatureMaskSpec(
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        pe_search_dim=search_dim,
    )


def build_eval_loader(
    args: argparse.Namespace,
    config: AxonExperimentConfig,
    train_config: TrainingConfig,
):
    data_dir = Path(args.data_dir or config.data_dir or "")
    if not data_dir:
        raise ValueError("data dir is required; pass --data-dir or use checkpoint config")

    npz_split_dir = data_dir / args.split
    if args.split != "all" and npz_split_dir.exists():
        data_loader = NPZDataLoader(
            data_dir=str(data_dir),
            batch_size=args.batch_size,
            max_byte_length=config.max_byte_length,
            pe_feature_dim=config.pe_feature_dim,
            stat_feature_dim=config.stat_feature_dim,
            num_workers=args.num_workers,
            pin_memory=args.device == "cuda",
            shuffle=False,
            max_samples_per_class=args.samples_per_class,
            allow_raw_fallback=False,
        )
        return data_loader.create_dataloader(args.split)

    dataset = FeatureCacheDataset(
        data_dir=str(data_dir),
        max_byte_length=config.max_byte_length,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        cache_dir=args.cache_dir,
        max_samples_per_class=args.samples_per_class,
        axon_config=config,
    )
    if args.split == "all":
        selected_dataset = dataset
    elif args.split_file:
        train_dataset, val_dataset, test_dataset = create_split_from_file(dataset, Path(args.split_file))
        selected_dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}[args.split]
    else:
        train_dataset, val_dataset, test_dataset = create_stratified_split(dataset, axon_config=config)
        selected_dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}[args.split]

    return torch.utils.data.DataLoader(
        selected_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )


def collect_eval_batches(
    loader,
    device: torch.device,
    max_batches: Optional[int],
) -> tuple[list[Batch], np.ndarray]:
    """Collect bounded eval batches on CPU.

    GA evaluates many masks against the same samples. Keeping those samples on
    CUDA would make memory grow with the loaded eval set, so tensors are cached
    on CPU and moved per batch inside ``predict_with_masks``.
    """
    batches = []
    labels_all = []
    del device
    for batch_idx, (byte_seq, pe_features, stat_features, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batches.append(
            (
                byte_seq.detach().cpu(),
                pe_features.detach().cpu(),
                stat_features.detach().cpu(),
            )
        )
        labels_all.extend(labels.detach().cpu().numpy().astype(np.int64).tolist())
    if not batches:
        raise ValueError("no evaluation batches were available")
    return batches, np.asarray(labels_all, dtype=np.int64)


def _select_batches_by_global_indices(
    batches: Sequence[Batch],
    indices: Sequence[int],
) -> list[Batch]:
    selected: list[Batch] = []
    sorted_indices = np.asarray(sorted(int(index) for index in indices), dtype=np.int64)
    if sorted_indices.size == 0:
        return selected

    cursor = 0
    start = 0
    for byte_seq, pe_features, stat_features in batches:
        batch_size = int(byte_seq.shape[0])
        end_cursor = cursor + batch_size
        while start < sorted_indices.size and sorted_indices[start] < cursor:
            start += 1
        stop = start
        while stop < sorted_indices.size and sorted_indices[stop] < end_cursor:
            stop += 1
        if stop > start:
            local_indices = torch.from_numpy(sorted_indices[start:stop] - cursor).to(dtype=torch.long)
            selected.append(
                (
                    byte_seq.index_select(0, local_indices),
                    pe_features.index_select(0, local_indices),
                    stat_features.index_select(0, local_indices),
                )
            )
        cursor = end_cursor
        start = stop
    return selected


def split_batches_stratified(
    batches: Sequence[Batch],
    labels: np.ndarray,
    holdout_ratio: float,
    seed: int,
) -> tuple[list[Batch], np.ndarray, list[Batch], np.ndarray]:
    """按标签均衡切出 search/holdout，避免 GA 在同一批样本上自证成功。

    DataLoader 已经把样本分成 batch 了。这里按全局样本下标做选择，但不把
    所有 batch 先拼成一个巨型 tensor，避免 holdout 切分时出现 2-3 份数据峰值。
    """
    if not 0 < holdout_ratio < 1:
        raise ValueError("holdout_ratio must be in (0, 1)")

    labels = np.asarray(labels, dtype=np.int64)
    sample_count = sum(int(batch[0].shape[0]) for batch in batches)
    if labels.shape[0] != sample_count:
        raise ValueError("labels and batches have different sample counts")

    rng = random.Random(seed)
    search_indices: list[int] = []
    holdout_indices: list[int] = []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label).astype(int).tolist()
        rng.shuffle(indices)
        holdout_count = int(round(len(indices) * holdout_ratio))
        if len(indices) > 1:
            holdout_count = min(len(indices) - 1, max(1, holdout_count))
        holdout_indices.extend(indices[:holdout_count])
        search_indices.extend(indices[holdout_count:])

    if not search_indices or not holdout_indices:
        raise ValueError("holdout split produced an empty search or holdout set")

    search_indices.sort()
    holdout_indices.sort()
    search_batches = _select_batches_by_global_indices(batches, search_indices)
    holdout_batches = _select_batches_by_global_indices(batches, holdout_indices)
    return (
        search_batches,
        labels[np.asarray(search_indices, dtype=np.int64)],
        holdout_batches,
        labels[np.asarray(holdout_indices, dtype=np.int64)],
    )


@torch.no_grad()
def predict_with_masks(
    model: AxonMalwareModel,
    batches: Sequence[Batch],
    spec: FeatureMaskSpec,
    individual: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    pe_mask_np, stat_mask_np = spec.to_model_masks(individual)
    pe_mask = torch.from_numpy(pe_mask_np).to(device=device).view(1, -1)
    stat_mask = torch.from_numpy(stat_mask_np).to(device=device).view(1, -1)

    probs = []
    model.eval()
    for byte_seq, pe_features, stat_features in batches:
        byte_seq = byte_seq.to(device, non_blocking=True)
        pe_features = pe_features.to(device, non_blocking=True)
        stat_features = stat_features.to(device, non_blocking=True)
        pe_mask = pe_mask.to(dtype=pe_features.dtype)
        stat_mask = stat_mask.to(dtype=stat_features.dtype)
        outputs = model(
            byte_seq,
            pe_features * pe_mask,
            stat_features=stat_features * stat_mask,
        )
        batch_probs = torch.softmax(outputs["logits"], dim=1)[:, 1]
        probs.extend(batch_probs.detach().cpu().numpy().astype(np.float64).tolist())
    return np.asarray(probs, dtype=np.float64)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_leaderboard_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "rank",
        "fitness",
        "kept_total",
        "kept_pe",
        "kept_stat",
        "kept_ratio",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "fp",
        "fn",
        "errors",
        "selected_pe_indices",
        "selected_stat_indices",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            metrics = row["metrics"]
            writer.writerow(
                {
                    "rank": rank,
                    "fitness": row["fitness"],
                    "kept_total": row["kept_total"],
                    "kept_pe": row["kept_pe"],
                    "kept_stat": row["kept_stat"],
                    "kept_ratio": row["kept_ratio"],
                    "threshold": metrics["threshold"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "auc": metrics["auc"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "errors": metrics["errors"],
                    "selected_pe_indices": " ".join(str(v) for v in row["selected_pe_indices"]),
                    "selected_stat_indices": " ".join(str(v) for v in row["selected_stat_indices"]),
                }
            )


def individual_from_selected_indices(
    spec: FeatureMaskSpec,
    pe_indices: Sequence[int],
    stat_indices: Sequence[int],
) -> np.ndarray:
    individual = np.zeros(spec.search_dim, dtype=bool)
    for index in pe_indices:
        index = int(index)
        if index < 0 or index >= spec.pe_search_dim:
            raise ValueError(f"PE feature index out of searched range: {index}")
        individual[index] = True
    for index in stat_indices:
        index = int(index)
        if index < 0 or index >= spec.stat_feature_dim:
            raise ValueError(f"stat feature index out of range: {index}")
        individual[spec.pe_search_dim + index] = True
    return individual


def load_feature_mask(path: Path, spec: FeatureMaskSpec) -> np.ndarray:
    payload = load_json_object(path)
    mask_spec = payload.get("mask_spec", {})
    expected = {
        "pe_feature_dim": spec.pe_feature_dim,
        "pe_search_dim": spec.pe_search_dim,
        "stat_feature_dim": spec.stat_feature_dim,
    }
    for key, value in expected.items():
        if key in mask_spec and int(mask_spec[key]) != value:
            raise ValueError(
                f"Feature mask {path} was built for {key}={mask_spec[key]}, "
                f"but current config expects {value}"
            )

    if "individual" in payload:
        individual = np.asarray(payload["individual"], dtype=bool)
        if individual.shape[0] != spec.search_dim:
            raise ValueError(
                f"Feature mask individual has length {individual.shape[0]}, "
                f"expected {spec.search_dim}"
            )
        return individual

    return individual_from_selected_indices(
        spec,
        payload.get("selected_pe_indices", []),
        payload.get("selected_stat_indices", []),
    )


def build_feature_mask_payload(
    *,
    spec: FeatureMaskSpec,
    individual: np.ndarray,
    source_report: Optional[str],
    checkpoint: str,
    search_metrics: Optional[dict],
    holdout_metrics: Optional[dict],
    baseline_metrics: Optional[dict],
    note: str,
) -> dict:
    pe_indices, stat_indices = spec.selected_indices(individual)
    return {
        "version": 1,
        "type": "axon_feature_mask",
        "source_report": source_report,
        "checkpoint": checkpoint,
        "note": note,
        "mask_spec": {
            "pe_feature_dim": spec.pe_feature_dim,
            "pe_search_dim": spec.pe_search_dim,
            "ignored_pe_dim": spec.ignored_pe_dim,
            "stat_feature_dim": spec.stat_feature_dim,
            "search_dim": spec.search_dim,
        },
        "kept_total": int(individual.sum()),
        "kept_pe": len(pe_indices),
        "kept_stat": len(stat_indices),
        "selected_pe_indices": pe_indices,
        "selected_stat_indices": stat_indices,
        "individual": individual.astype(bool).tolist(),
        "search_metrics": search_metrics,
        "holdout_metrics": holdout_metrics,
        "baseline_metrics": baseline_metrics,
    }


def export_mask_from_report(
    report_path: Path,
    output_path: Path,
    note: str = "",
) -> dict:
    report = load_json_object(report_path)
    best = report["best"]
    mask_spec = report["mask_spec"]
    spec = FeatureMaskSpec(
        pe_feature_dim=int(mask_spec["pe_feature_dim"]),
        stat_feature_dim=int(mask_spec["stat_feature_dim"]),
        pe_search_dim=int(mask_spec["pe_search_dim"]),
    )
    individual = np.asarray(best["individual"], dtype=bool)
    payload = build_feature_mask_payload(
        spec=spec,
        individual=individual,
        source_report=str(report_path),
        checkpoint=str(report.get("checkpoint", "")),
        search_metrics=best.get("metrics"),
        holdout_metrics=(report.get("best_holdout") or {}).get("metrics"),
        baseline_metrics=(report.get("holdout_full_features") or report.get("baseline_full_features") or {}).get("metrics"),
        note=note,
    )
    write_json(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline GA search for compact PE/stat feature subsets. "
            "The byte sequence branch is kept unchanged."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--split-file", type=str, default=None)
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=20, help="0 means no limit")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--feature-mask",
        type=Path,
        default=None,
        help="Evaluate this exported feature mask instead of searching from scratch.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="Only evaluate --feature-mask and full-feature baseline; do not run GA.",
    )
    parser.add_argument(
        "--export-mask",
        type=Path,
        default=None,
        help="Write the best GA mask, or the evaluated --feature-mask, to this JSON path.",
    )
    parser.add_argument(
        "--export-mask-from-report",
        type=Path,
        default=None,
        help="Convert an existing GA report JSON into a reusable feature mask JSON.",
    )
    parser.add_argument("--mask-note", type=str, default="")
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.0,
        help=(
            "Optional stratified holdout ratio carved from the loaded evaluation samples. "
            "GA searches on the remaining samples and reports final confirmation on holdout."
        ),
    )

    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--elite-size", type=int, default=3)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--mutation-rate", type=float, default=0.02)
    parser.add_argument("--crossover-rate", type=float, default=0.85)
    parser.add_argument("--target-keep-ratio", type=float, default=0.45)
    parser.add_argument("--min-pe-features", type=int, default=1)
    parser.add_argument("--min-stat-features", type=int, default=1)
    parser.add_argument("--leaderboard-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated thresholds. Defaults to checkpoint train_config decision_threshold.",
    )
    parser.add_argument(
        "--objective",
        choices=["f1", "auc", "accuracy", "precision", "recall"],
        default="f1",
    )
    parser.add_argument("--feature-penalty", type=float, default=0.01)
    parser.add_argument("--fp-penalty", type=float, default=0.0)
    parser.add_argument("--fn-penalty", type=float, default=0.0)
    parser.add_argument(
        "--max-objective-drop",
        type=float,
        default=None,
        help=(
            "Protect accuracy before compactness. When set, candidates below "
            "baseline objective minus this drop receive an extra penalty."
        ),
    )
    parser.add_argument("--below-min-penalty", type=float, default=2.0)
    parser.add_argument(
        "--max-recall-drop",
        type=float,
        default=None,
        help="Extra guard: penalize candidates whose recall is below baseline recall minus this value.",
    )
    parser.add_argument(
        "--max-fn-increase-rate",
        type=float,
        default=None,
        help=(
            "Extra guard: penalize candidates whose FN/sample rate exceeds baseline "
            "FN/sample rate by more than this value."
        ),
    )
    parser.add_argument("--recall-guard-penalty", type=float, default=3.0)
    parser.add_argument("--fn-guard-penalty", type=float, default=3.0)

    parser.add_argument("--include-pe-padding", action="store_true", default=False)
    parser.add_argument(
        "--pe-search-dim",
        type=int,
        default=None,
        help="Override searched PE dimensions. By default fixed_v2 excludes known zero padding.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/feature_subset_ga.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/feature_subset_ga.csv"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.export_mask_from_report:
        if args.export_mask is None:
            raise ValueError("--export-mask is required with --export-mask-from-report")
        payload = export_mask_from_report(
            args.export_mask_from_report,
            args.export_mask,
            note=args.mask_note,
        )
        print(
            f"Exported feature mask: {args.export_mask} "
            f"(kept={payload['kept_total']}, pe={payload['kept_pe']}, stat={payload['kept_stat']})"
        )
        return 0

    if args.population_size <= 1:
        raise ValueError("--population-size must be greater than 1")
    if args.generations < 0:
        raise ValueError("--generations must be non-negative")
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --export-mask-from-report is used")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = load_safe_checkpoint(args.checkpoint, map_location="cpu")
    config = AxonExperimentConfig.from_dict(checkpoint["config"])
    train_config_dict = checkpoint.get("train_config", {})
    train_config = TrainingConfig(**train_config_dict) if train_config_dict else TrainingConfig()
    train_config.batch_size = args.batch_size
    train_config.num_workers = args.num_workers

    thresholds = (
        parse_thresholds(args.thresholds)
        if args.thresholds
        else [float(getattr(train_config, "decision_threshold", 0.5))]
    )

    model = AxonMalwareModel(config)
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)
    del checkpoint, model_state_dict
    model.to(device)
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    model.eval()

    spec = build_feature_mask_spec(
        config,
        include_pe_padding=args.include_pe_padding,
        pe_search_dim=args.pe_search_dim,
    )
    loader = build_eval_loader(args, config, train_config)
    max_batches = None if args.max_batches == 0 else args.max_batches
    batches, labels = collect_eval_batches(loader, device, max_batches)
    loaded_sample_count = int(labels.shape[0])
    holdout_batches: list[Batch] = []
    holdout_labels = np.asarray([], dtype=np.int64)
    if args.holdout_ratio > 0:
        batches, labels, holdout_batches, holdout_labels = split_batches_stratified(
            batches,
            labels,
            args.holdout_ratio,
            args.seed,
        )

    ga_config = GeneticConfig(
        population_size=args.population_size,
        generations=args.generations,
        elite_size=args.elite_size,
        tournament_size=args.tournament_size,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        target_keep_ratio=args.target_keep_ratio,
        min_pe_features=args.min_pe_features,
        min_stat_features=args.min_stat_features,
        max_leaderboard_size=max(int(args.leaderboard_size), int(args.top_k)),
    )

    def evaluate(individual: np.ndarray) -> dict:
        probs = predict_with_masks(model, batches, spec, individual, device)
        return choose_best_metrics(labels, probs, thresholds)

    full_individual = np.ones(spec.search_dim, dtype=bool)
    full_metrics = evaluate(full_individual)
    baseline_objective = (
        full_metrics["auc"]
        if args.objective == "auc" and full_metrics["auc"] is not None
        else full_metrics[args.objective]
    )
    min_objective = None
    if args.max_objective_drop is not None:
        min_objective = max(0.0, float(baseline_objective) - float(args.max_objective_drop))
    min_recall = None
    if args.max_recall_drop is not None:
        min_recall = max(0.0, float(full_metrics["recall"]) - float(args.max_recall_drop))
    max_fn_rate = None
    if args.max_fn_increase_rate is not None:
        baseline_fn_rate = float(full_metrics["fn"]) / max(1, int(full_metrics["sample_count"]))
        max_fn_rate = baseline_fn_rate + float(args.max_fn_increase_rate)

    fitness_config = FitnessConfig(
        objective=args.objective,
        feature_penalty=args.feature_penalty,
        fp_penalty=args.fp_penalty,
        fn_penalty=args.fn_penalty,
        min_objective=min_objective,
        below_min_penalty=args.below_min_penalty,
        min_recall=min_recall,
        max_fn_rate=max_fn_rate,
        recall_guard_penalty=args.recall_guard_penalty,
        fn_guard_penalty=args.fn_guard_penalty,
    )
    holdout_full_metrics = None
    if holdout_batches:
        holdout_full_probs = predict_with_masks(model, holdout_batches, spec, full_individual, device)
        holdout_full_metrics = choose_best_metrics(holdout_labels, holdout_full_probs, thresholds)

    mask_individual = load_feature_mask(args.feature_mask, spec) if args.feature_mask else None
    mask_metrics = evaluate(mask_individual) if mask_individual is not None else None
    mask_holdout = None
    if mask_individual is not None and holdout_batches:
        mask_holdout_probs = predict_with_masks(model, holdout_batches, spec, mask_individual, device)
        mask_holdout_metrics = choose_best_metrics(holdout_labels, mask_holdout_probs, thresholds)
        mask_holdout = summarize_individual(mask_individual, spec, mask_holdout_metrics, fitness_config)

    print("Axon feature subset GA")
    print(f"  device: {device}")
    print(f"  loaded samples: {loaded_sample_count}")
    print(f"  search samples: {labels.shape[0]}")
    if holdout_batches:
        print(f"  holdout samples: {holdout_labels.shape[0]}")
    print(f"  pe search dim: {spec.pe_search_dim}/{spec.pe_feature_dim}")
    print(f"  ignored PE padding dim: {spec.ignored_pe_dim}")
    print(f"  stat search dim: {spec.stat_feature_dim}")
    print(f"  baseline f1: {full_metrics['f1']:.6f}, auc: {full_metrics['auc']}")
    if mask_metrics is not None:
        mask_summary = summarize_individual(mask_individual, spec, mask_metrics, fitness_config)
        print(
            f"  mask f1: {mask_metrics['f1']:.6f}, auc: {mask_metrics['auc']}, "
            f"kept={mask_summary['kept_total']}"
        )
        if mask_holdout is not None:
            print(
                f"  mask holdout f1: {mask_holdout['metrics']['f1']:.6f}, "
                f"fp={mask_holdout['metrics']['fp']}, fn={mask_holdout['metrics']['fn']}"
            )

    if args.eval_only:
        if mask_individual is None:
            raise ValueError("--eval-only requires --feature-mask")
        payload = {
            "checkpoint": str(args.checkpoint),
            "feature_mask": str(args.feature_mask),
            "data_dir": args.data_dir,
            "split": args.split,
            "split_file": args.split_file,
            "device": str(device),
            "loaded_samples": loaded_sample_count,
            "search_samples": int(labels.shape[0]),
            "holdout_samples": int(holdout_labels.shape[0]),
            "thresholds": thresholds,
            "mask_spec": {
                "pe_feature_dim": spec.pe_feature_dim,
                "pe_search_dim": spec.pe_search_dim,
                "ignored_pe_dim": spec.ignored_pe_dim,
                "stat_feature_dim": spec.stat_feature_dim,
                "search_dim": spec.search_dim,
            },
            "baseline_full_features": {
                "kept_pe": spec.pe_search_dim,
                "kept_stat": spec.stat_feature_dim,
                "metrics": full_metrics,
            },
            "holdout_full_features": (
                {
                    "kept_pe": spec.pe_search_dim,
                    "kept_stat": spec.stat_feature_dim,
                    "metrics": holdout_full_metrics,
                }
                if holdout_full_metrics is not None
                else None
            ),
            "mask_result": summarize_individual(mask_individual, spec, mask_metrics, fitness_config),
            "mask_holdout": mask_holdout,
        }
        write_json(args.output_json, payload)
        if args.export_mask:
            write_json(
                args.export_mask,
                build_feature_mask_payload(
                    spec=spec,
                    individual=mask_individual,
                    source_report=str(args.output_json),
                    checkpoint=str(args.checkpoint),
                    search_metrics=mask_metrics,
                    holdout_metrics=mask_holdout["metrics"] if mask_holdout else None,
                    baseline_metrics=holdout_full_metrics or full_metrics,
                    note=args.mask_note,
                ),
            )
        print(f"  JSON: {args.output_json}")
        return 0

    result = run_genetic_search(
        spec=spec,
        ga_config=ga_config,
        fitness_config=fitness_config,
        evaluate_individual=evaluate,
        seed=args.seed,
    )

    for row in result["generations"]:
        print(
            f"  gen {row['generation']:03d} | fitness={row['best_fitness']:.6f} "
            f"f1={row['best_f1']:.6f} kept={row['best_kept_total']} "
            f"unique={row['unique_evaluated']}"
        )

    best_holdout = None
    if holdout_batches:
        best_individual = np.asarray(result["best"]["individual"], dtype=bool)
        holdout_probs = predict_with_masks(model, holdout_batches, spec, best_individual, device)
        holdout_metrics = choose_best_metrics(holdout_labels, holdout_probs, thresholds)
        best_holdout = summarize_individual(best_individual, spec, holdout_metrics, fitness_config)

    payload = {
        "checkpoint": str(args.checkpoint),
        "data_dir": args.data_dir,
        "split": args.split,
        "split_file": args.split_file,
        "device": str(device),
        "loaded_samples": loaded_sample_count,
        "samples": int(labels.shape[0]),
        "search_samples": int(labels.shape[0]),
        "holdout_samples": int(holdout_labels.shape[0]),
        "holdout_ratio": float(args.holdout_ratio),
        "thresholds": thresholds,
        "mask_spec": {
            "pe_feature_dim": spec.pe_feature_dim,
            "pe_search_dim": spec.pe_search_dim,
            "ignored_pe_dim": spec.ignored_pe_dim,
            "stat_feature_dim": spec.stat_feature_dim,
            "search_dim": spec.search_dim,
            "note": "Byte sequence branch is not searched; GA only masks PE/stat scalar features.",
        },
        "fitness_config": fitness_config.__dict__,
        "ga_config": ga_config.__dict__,
        "baseline_full_features": {
            "kept_pe": spec.pe_search_dim,
            "kept_stat": spec.stat_feature_dim,
            "metrics": full_metrics,
        },
        "input_feature_mask": str(args.feature_mask) if args.feature_mask else None,
        "input_feature_mask_result": (
            summarize_individual(mask_individual, spec, mask_metrics, fitness_config)
            if mask_individual is not None and mask_metrics is not None
            else None
        ),
        "input_feature_mask_holdout": mask_holdout,
        "holdout_full_features": (
            {
                "kept_pe": spec.pe_search_dim,
                "kept_stat": spec.stat_feature_dim,
                "metrics": holdout_full_metrics,
            }
            if holdout_full_metrics is not None
            else None
        ),
        "best": result["best"],
        "best_holdout": best_holdout,
        "generations": result["generations"],
        "leaderboard": result["leaderboard"][: max(1, args.top_k)],
        "evaluated_candidates": result["evaluated_candidates"],
    }

    write_json(args.output_json, payload)
    write_leaderboard_csv(args.output_csv, payload["leaderboard"])
    if args.export_mask:
        write_json(
            args.export_mask,
            build_feature_mask_payload(
                spec=spec,
                individual=np.asarray(payload["best"]["individual"], dtype=bool),
                source_report=str(args.output_json),
                checkpoint=str(args.checkpoint),
                search_metrics=payload["best"]["metrics"],
                holdout_metrics=(payload["best_holdout"] or {}).get("metrics"),
                baseline_metrics=holdout_full_metrics or full_metrics,
                note=args.mask_note,
            ),
        )

    best = payload["best"]
    print("\nBest candidate")
    print(
        f"  fitness={best['fitness']:.6f}, f1={best['metrics']['f1']:.6f}, "
        f"auc={best['metrics']['auc']}, kept={best['kept_total']} "
        f"(pe={best['kept_pe']}, stat={best['kept_stat']})"
    )
    if best_holdout is not None:
        print(
            "  holdout "
            f"f1={best_holdout['metrics']['f1']:.6f}, "
            f"auc={best_holdout['metrics']['auc']}, "
            f"fp={best_holdout['metrics']['fp']}, fn={best_holdout['metrics']['fn']}"
        )
    print(f"  JSON: {args.output_json}")
    print(f"  CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
