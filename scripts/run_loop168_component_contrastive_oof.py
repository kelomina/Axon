#!/usr/bin/env python3
"""Train-only five-fold comparison for component-aware contrastive features."""

from __future__ import annotations

import json
import random
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports/roadmap_9997/loop168_component_contrastive/oof_report.json"
TRUSTED_SIGNER_TERMS = (
    "Microsoft Corporation", "Microsoft Windows", "Seagate Technology", "FinalWire",
    "NetEase", "Beijing Sogou", "Beijing Kingsoft", "Beijing Qihu", "Wondershare",
    "IObit", "Yozosoft", "Huya",
)


class CacheClassifier(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 96),
        )
        self.classifier = nn.Linear(96, 2)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(values)
        return self.classifier(embedding), embedding


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    prediction = scores >= 0.5
    true_positive = int(np.sum((prediction == 1) & (labels == 1)))
    true_negative = int(np.sum((prediction == 0) & (labels == 0)))
    false_positive = int(np.sum((prediction == 1) & (labels == 0)))
    false_negative = int(np.sum((prediction == 0) & (labels == 1)))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "errors": false_positive + false_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
    }


def component_ids(values: tuple[str, ...]) -> np.ndarray:
    mapping = {value: index for index, value in enumerate(sorted(set(values)))}
    return np.asarray([mapping[value] for value in values], dtype=np.int64)


def authenticode_features(path: Path, rows: int) -> np.ndarray:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        evidence = list(csv.DictReader(handle))
    if len(evidence) != rows:
        raise ValueError("Authenticode evidence denominator drifted")
    by_index = {int(row["sample_index"]): row for row in evidence}
    if set(by_index) != set(range(rows)):
        raise ValueError("Authenticode evidence indices drifted")
    statuses = ("Valid", "NotSigned", "HashMismatch", "UnknownError", "CollectionError")
    matrix = np.zeros((rows, len(statuses) + len(TRUSTED_SIGNER_TERMS)), dtype=np.float32)
    for index in range(rows):
        row = by_index[index]
        status = str(row.get("auth_status") or "CollectionError")
        if status in statuses:
            matrix[index, statuses.index(status)] = 1.0
        subject = str(row.get("signer_subject") or "").casefold()
        for offset, term in enumerate(TRUSTED_SIGNER_TERMS, start=len(statuses)):
            matrix[index, offset] = float(term.casefold() in subject)
    return matrix


def fit_fold(
    features: np.ndarray,
    labels: np.ndarray,
    components: np.ndarray,
    train_index: np.ndarray,
    holdout_index: np.ndarray,
    *,
    contrastive_weight: float,
    focal_gamma: float,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    mean = features[train_index].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features[train_index].std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    train_x = torch.from_numpy((features[train_index] - mean) / scale)
    train_y = torch.from_numpy(labels[train_index].astype(np.int64))
    train_components = torch.from_numpy(components[train_index])
    model = CacheClassifier(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(8):
        order = torch.randperm(train_x.shape[0], generator=generator)
        model.train()
        for offset in range(0, order.numel(), 256):
            rows = order[offset : offset + 256]
            values = train_x[rows].to(device)
            target = train_y[rows].to(device)
            batch_components = train_components[rows].to(device)
            logits, embedding = model(values)
            per_sample_loss = functional.cross_entropy(logits, target, reduction="none")
            if focal_gamma:
                confidence = torch.exp(-per_sample_loss)
                loss = ((1.0 - confidence).pow(focal_gamma) * per_sample_loss).mean()
            else:
                loss = per_sample_loss.mean()
            if contrastive_weight:
                from src.component_contrastive import component_supervised_contrastive_loss

                loss = loss + contrastive_weight * component_supervised_contrastive_loss(
                    embedding, target, batch_components, temperature=0.15
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        values = torch.from_numpy((features[holdout_index] - mean) / scale).to(device)
        return torch.softmax(model(values)[0], dim=1)[:, 1].cpu().numpy()


def run_arm(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    components: np.ndarray,
    *,
    contrastive_weight: float,
    focal_gamma: float,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    scores = np.empty(labels.shape[0], dtype=np.float32)
    for fold in range(5):
        holdout = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        scores[holdout] = fit_fold(
            features, labels, components, train, holdout,
            contrastive_weight=contrastive_weight,
            focal_gamma=focal_gamma,
            seed=seed * 100 + fold,
            device=device,
        )
    return {"metrics": metrics(labels, scores), "scores": scores}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contrastive-weight", type=float, default=0.08)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--authenticode-csv", type=Path)
    parser.add_argument("--seed", type=int, default=168)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    arguments = parser.parse_args()
    if not 0.0 <= arguments.contrastive_weight <= 0.1:
        raise ValueError("contrastive weight must be in [0, 0.1]")
    if not 0.0 <= arguments.focal_gamma <= 3.0:
        raise ValueError("focal gamma must be in [0, 3]")
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.loop167_phase_b.fit_cache_loader_v13 import load_verified_v12_cache_for_v13
    from src.loop167_phase_b.fit_targets_adapter_v13 import load_fit_targets_v13

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    cache = load_verified_v12_cache_for_v13(PROJECT_ROOT)
    from src.loop167_phase_b.contracts import sha256_file

    protocol = PROJECT_ROOT / "manifests/roadmap_9997/loop167_ember_v3_novel_delta/phase_b_protocol.json"
    targets = load_fit_targets_v13(PROJECT_ROOT, protocol_sha256=sha256_file(protocol))
    values = cache.loaded_cache.cache
    features = np.concatenate(
        [values.b0_values, values.b0_missing_indicators, values.b1_values,
         values.b1_missing_indicators, values.novel_values, values.novel_complete[:, None]],
        axis=1,
    ).astype(np.float32, copy=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    component_vector = component_ids(targets.component_ids)
    baseline = run_arm(features, targets.labels, targets.folds, component_vector, contrastive_weight=0.0, focal_gamma=0.0, device=device, seed=arguments.seed)
    candidate_features = features
    if arguments.authenticode_csv is not None:
        candidate_features = np.concatenate(
            [features, authenticode_features(arguments.authenticode_csv, features.shape[0])], axis=1
        )
    candidate = run_arm(
        candidate_features,
        targets.labels,
        targets.folds,
        component_vector,
        contrastive_weight=arguments.contrastive_weight,
        focal_gamma=arguments.focal_gamma,
        device=device,
        seed=arguments.seed,
    )
    report = {
        "schema": "axon_loop168_component_contrastive_train_only_oof_v1",
        "scope": {"train_only": True, "raw_access": False, "heldout_access": False, "threshold_selection": False},
        "device": str(device), "rows": int(targets.labels.shape[0]),
        "features": int(features.shape[1]), "folds": 5,
        "contrastive_weight": arguments.contrastive_weight,
        "focal_gamma": arguments.focal_gamma,
        "authenticode_features": arguments.authenticode_csv is not None,
        "seed": arguments.seed,
        "baseline": baseline["metrics"], "component_contrastive": candidate["metrics"],
        "net_error_reduction": int(baseline["metrics"]["errors"]) - int(candidate["metrics"]["errors"]),
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
