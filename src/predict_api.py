"""Stable JSON prediction API used by the Axon DLL wrapper.

这个模块给外部程序调用，不负责训练，也不打印人类可读报告。
输入是文件路径、checkpoint 路径和少量选项；输出是稳定 JSON 字典。
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from archive_scanner import (
    ArchiveScanOptions,
    cleanup_scan_temp,
    iter_pe_prediction_targets,
    run_archive_scan,
)
from config import AxonExperimentConfig
from kvd_features import ExtractionConfig, extract_all_features
from kvd_features.content_pe_v1 import _content_pe_features_from_path
from model import AxonMalwareModel
from security import load_safe_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOOP28_STAGE2_MODEL = (
    PROJECT_ROOT
    / "reports"
    / "random_20w_split"
    / "stage2_loop28_content_pe_valonly"
    / "stage2_selected_model.pkl"
)
DEFAULT_FAMILY_CLASSIFIER = PROJECT_ROOT / "resources" / "axon_family" / "family_classifier.json"
STAGE2_MODEL_METADATA_SCHEMA = "axon_stage2_model_metadata_v1"
NESTED_SCAN_ENTRY_RESPONSE_LIMIT = 256
NESTED_PREDICTION_RESPONSE_LIMIT = 1024
MAX_STAGE2_CHUNK_COUNT = 4096
MAX_STAGE2_PICKLE_BYTES = 64 * 1024 * 1024
DEFAULT_STAGE2_TRUST_MANIFEST = (
    PROJECT_ROOT / "manifests" / "roadmap_9997" / "p0_raw_replay" / "pickle_sha256_allowlist.json"
)


def _stage2_metadata_path(model_path: Path) -> Path:
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}.metadata.json")


def _read_stage2_metadata(
    model_path: Path,
    metadata_path: Optional[Path] = None,
    expected_metadata_sha256: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    metadata_path = (
        Path(metadata_path) if metadata_path is not None else _stage2_metadata_path(model_path)
    )
    if not metadata_path.exists():
        return None
    try:
        metadata_bytes = metadata_path.read_bytes()
        if expected_metadata_sha256 is not None:
            expected_sha = str(expected_metadata_sha256).strip().casefold()
            if len(expected_sha) != 64 or any(
                char not in "0123456789abcdef" for char in expected_sha
            ):
                raise ValueError("Stage2 expected metadata SHA-256 is invalid")
            actual_sha = hashlib.sha256(metadata_bytes).hexdigest()
            if actual_sha != expected_sha:
                raise ValueError(
                    f"Stage2 metadata SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
                )
        metadata = json.loads(metadata_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Stage2 metadata JSON: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Stage2 metadata must be a JSON object: {metadata_path}")
    schema = metadata.get("schema")
    if schema != STAGE2_MODEL_METADATA_SCHEMA:
        raise ValueError(f"Unsupported Stage2 metadata schema: {schema}")
    return metadata


def _reject_unsupported_stage2_from_metadata(
    model_path: Path,
    *,
    metadata_path: Optional[Path] = None,
    expected_model_sha256: Optional[str] = None,
    expected_metadata_sha256: Optional[str] = None,
) -> dict[str, Any]:
    metadata = _read_stage2_metadata(
        model_path,
        metadata_path,
        expected_metadata_sha256,
    )
    if metadata is None:
        required_path = (
            Path(metadata_path) if metadata_path is not None else _stage2_metadata_path(model_path)
        )
        raise ValueError(f"Stage2 metadata sidecar is required before unpickling: {required_path}")
    declared_sha = str(metadata.get("model_sha256") or "").strip().casefold()
    expected_sha = str(expected_model_sha256 or declared_sha).strip().casefold()
    if not expected_sha:
        raise ValueError("Stage2 metadata must bind an immutable model_sha256")
    if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
        raise ValueError("Stage2 expected model SHA-256 is invalid")
    if declared_sha and declared_sha != expected_sha:
        raise ValueError(
            f"Stage2 metadata model SHA-256 mismatch: expected {declared_sha}, got {expected_sha}"
        )
    knn = metadata.get("knn") or {}
    if not isinstance(knn, dict):
        raise ValueError("Stage2 metadata knn field must be an object")
    if knn.get("enabled"):
        raise ValueError("Stage2 models with frozen kNN memory are not supported by predict_api")
    return metadata


@dataclass(frozen=True)
class PredictRequest:
    file: str
    checkpoint: str
    device: str = "cpu"
    scan_nested: bool = False
    archive_scanner: Optional[str] = None
    archive_max_depth: int = 4
    archive_max_files: int = 4096
    archive_max_total_bytes: int = 512 * 1024 * 1024
    archive_max_file_bytes: int = 128 * 1024 * 1024
    threshold: Optional[float] = None
    stage2_model: Optional[str] = None
    family_classifier: Optional[str] = None


@dataclass(frozen=True)
class Stage2FeatureConfig:
    prefix_len: int
    chunk_count: int
    include_pe: bool
    include_stat: bool
    include_lightweight: bool
    include_byte_summary: bool
    include_content_pe: bool = False
    content_cache_dir: Optional[str] = None
    include_content_pe_v2: bool = False
    content_pe_v2_cache_dir: Optional[str] = None
    content_pe_v2_groups: tuple[str, ...] = ("all",)
    include_content_string: bool = False
    content_string_cache_dir: Optional[str] = None
    include_content_cert: bool = False
    content_cert_cache_dir: Optional[str] = None

    @classmethod
    def from_payload(cls, value: Any) -> "Stage2FeatureConfig":
        if isinstance(value, cls):
            return value
        if hasattr(value, "__dict__"):
            value = dict(value.__dict__)
        if not isinstance(value, dict):
            raise ValueError("stage2 feature_config must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        cleaned = {key: value[key] for key in value if key in allowed}
        if "content_pe_v2_groups" in cleaned and not isinstance(
            cleaned["content_pe_v2_groups"], tuple
        ):
            cleaned["content_pe_v2_groups"] = tuple(cleaned["content_pe_v2_groups"])
        return cls(**cleaned)


class _Stage2PayloadUnpickler(pickle.Unpickler):
    """兼容旧训练脚本以 __main__.FeatureConfig 保存的 pkl。"""

    def find_class(self, module: str, name: str):  # noqa: D102
        if module == "__main__" and name == "FeatureConfig":
            return Stage2FeatureConfig
        return super().find_class(module, name)


class Stage2ModelBundle:
    def __init__(self, payload: dict[str, Any], path: Path):
        self.path = Path(path)
        self.model = payload["model"]
        self.threshold = float(payload.get("threshold", 0.5))
        self.selected = payload.get("selected")
        self.checkpoint_config = AxonExperimentConfig.from_dict(dict(payload["checkpoint_config"]))
        self.feature_config = Stage2FeatureConfig.from_payload(payload["feature_config"])
        self._validate_feature_config()
        self.knn = payload.get("knn") or {}
        if self.knn.get("enabled"):
            raise ValueError(
                "Stage2 models with frozen kNN memory are not supported by predict_api"
            )

    def _validate_feature_config(self) -> None:
        prefix_len = int(self.feature_config.prefix_len)
        chunk_count = int(self.feature_config.chunk_count)
        max_byte_length = int(getattr(self.checkpoint_config, "max_byte_length", 0))
        if prefix_len < 0 or prefix_len > max_byte_length:
            raise ValueError(
                f"Stage2 prefix_len {prefix_len} exceeds checkpoint max_byte_length {max_byte_length}"
            )
        if chunk_count < 1 or chunk_count > MAX_STAGE2_CHUNK_COUNT:
            raise ValueError(
                f"Stage2 chunk_count {chunk_count} must be in [1, {MAX_STAGE2_CHUNK_COUNT}]"
            )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        metadata_path: Optional[Path] = None,
        expected_model_sha256: Optional[str] = None,
        expected_metadata_sha256: Optional[str] = None,
    ) -> "Stage2ModelBundle":
        path = Path(path)
        metadata = _reject_unsupported_stage2_from_metadata(
            path,
            metadata_path=metadata_path,
            expected_model_sha256=expected_model_sha256,
            expected_metadata_sha256=expected_metadata_sha256,
        )
        try:
            payload_size = path.stat().st_size
        except OSError as exc:
            raise ValueError(f"Stage2 pickle is not readable: {path}") from exc
        if payload_size > MAX_STAGE2_PICKLE_BYTES:
            raise ValueError(
                f"Stage2 pickle exceeds {MAX_STAGE2_PICKLE_BYTES} byte predict_api limit"
            )
        payload_bytes = path.read_bytes()
        if len(payload_bytes) > MAX_STAGE2_PICKLE_BYTES:
            raise ValueError(
                f"Stage2 pickle exceeds {MAX_STAGE2_PICKLE_BYTES} byte predict_api limit"
            )
        expected_sha = (
            str(expected_model_sha256 or metadata.get("model_sha256") or "").strip().casefold()
        )
        actual_sha = hashlib.sha256(payload_bytes).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"Stage2 model SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
        payload = _Stage2PayloadUnpickler(io.BytesIO(payload_bytes)).load()
        if not isinstance(payload, dict):
            raise ValueError("stage2 model payload must be a dict")
        for key in ("model", "feature_config", "checkpoint_config"):
            if key not in payload:
                raise ValueError(f"stage2 model missing key: {key}")
        return cls(payload, path)

    def predict_probability(self, features: np.ndarray) -> float:
        matrix = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(matrix)[0, 1])
        scores = np.asarray(self.model.decision_function(matrix), dtype=np.float32)
        score = float(np.clip(scores.reshape(-1)[0], -50.0, 50.0))
        return float(1.0 / (1.0 + math.exp(-score)))


class FamilyClassifier:
    def __init__(self, payload: dict[str, Any], path: Path):
        if payload.get("schema") != "axon_family_classifier_v1":
            raise ValueError(f"Unsupported family classifier schema: {payload.get('schema')}")
        self.path = Path(path)
        self.feature_dim = int(payload["feature_dim"])
        self.pe_feature_dim = int(payload["pe_feature_dim"])
        self.stat_feature_dim = int(payload["stat_feature_dim"])
        self.cluster_ids = [int(value) for value in payload["cluster_ids"]]
        self.family_names = [str(value) for value in payload["family_names"]]
        self.thresholds = np.asarray(payload["thresholds"], dtype=np.float32)
        self.centroids = np.asarray(payload["centroids"], dtype=np.float32)
        self.scaler_mean = np.asarray(payload["scaler_mean"], dtype=np.float32)
        self.scaler_scale = np.asarray(payload["scaler_scale"], dtype=np.float32)

        if self.centroids.ndim != 2 or self.centroids.shape[1] != self.feature_dim:
            raise ValueError("family classifier centroids shape does not match feature_dim")
        if self.scaler_mean.shape != (self.feature_dim,) or self.scaler_scale.shape != (
            self.feature_dim,
        ):
            raise ValueError("family classifier scaler shape does not match feature_dim")
        if (
            len(self.cluster_ids) != self.centroids.shape[0]
            or len(self.family_names) != self.centroids.shape[0]
        ):
            raise ValueError("family classifier family count mismatch")

    @classmethod
    def load(cls, path: Path) -> "FamilyClassifier":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("family classifier JSON must be an object")
        return cls(payload, Path(path))

    def predict(
        self, pe_features: np.ndarray, stat_features: np.ndarray
    ) -> Optional[dict[str, Any]]:
        if (
            pe_features.shape[0] != self.pe_feature_dim
            or stat_features.shape[0] != self.stat_feature_dim
        ):
            return None
        features = np.concatenate(
            [
                np.nan_to_num(pe_features.astype(np.float32, copy=False), copy=False),
                np.nan_to_num(stat_features.astype(np.float32, copy=False), copy=False),
            ]
        )
        scale = np.where(np.abs(self.scaler_scale) < 1.0e-12, 1.0, self.scaler_scale)
        scaled = (features - self.scaler_mean) / scale
        deltas = self.centroids - scaled.reshape(1, -1)
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        best_index = int(np.argmin(distances))
        distance = float(distances[best_index])
        threshold = float(self.thresholds[best_index])
        return {
            "family_name": self.family_names[best_index],
            "cluster_id": self.cluster_ids[best_index],
            "is_new_family": bool(distance > threshold),
            "distance": distance,
            "threshold": threshold,
        }


@dataclass
class ExtractedFeatures:
    byte_seq: np.ndarray
    pe_features: np.ndarray
    stat_features: np.ndarray
    lightweight_features: np.ndarray
    orig_len: int


@dataclass
class PredictionContext:
    request: PredictRequest
    checkpoint_path: Path
    device: str
    model: AxonMalwareModel
    config: AxonExperimentConfig
    stage2: Optional[Stage2ModelBundle]
    family_classifier: Optional[FamilyClassifier]


def _ok(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, **payload}


def _error(code: str, message: str, *, details: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": False, "error_code": code, "error": message}
    if details:
        result["details"] = details
    return result


def _scan_response_fields(report: dict[str, Any]) -> dict[str, Any]:
    entries = report.get("entries")
    if not isinstance(entries, list):
        entries = []
    visible_entries = entries[:NESTED_SCAN_ENTRY_RESPONSE_LIMIT]
    return {
        "scan_summary": report.get("summary", {}),
        "scan_entry_count": len(entries),
        "scan_entry_response_limit": NESTED_SCAN_ENTRY_RESPONSE_LIMIT,
        "scan_entries_truncated": len(entries) > len(visible_entries),
        "scan_entries": visible_entries,
    }


def _append_visible_prediction(
    visible_predictions: list[dict[str, Any]],
    prediction: dict[str, Any],
) -> bool:
    if len(visible_predictions) < NESTED_PREDICTION_RESPONSE_LIMIT:
        visible_predictions.append(prediction)
        return False
    return True


def _resolve_device(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_optional_path(raw_path: Optional[str], default: Path) -> Optional[Path]:
    if raw_path is None:
        return default if default.exists() else None
    if raw_path == "":
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_trusted_project_path(raw_path: object, *, field: str) -> Path:
    path = Path(str(raw_path or ""))
    resolved_root = PROJECT_ROOT.resolve()
    resolved = (path if path.is_absolute() else resolved_root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Trusted Stage2 {field} escapes project root: {resolved}") from exc
    return resolved


def _trusted_stage2_binding(model_path: Path) -> dict[str, Any]:
    """Resolve model trust from the server-owned registry, never from request JSON."""

    try:
        payload = json.loads(DEFAULT_STAGE2_TRUST_MANIFEST.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"Stage2 trust manifest is missing: {DEFAULT_STAGE2_TRUST_MANIFEST}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Stage2 trust manifest: {DEFAULT_STAGE2_TRUST_MANIFEST}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "axon_pickle_sha256_allowlist_v1":
        raise ValueError("Unsupported Stage2 trust manifest schema")
    resolved_model = _resolve_trusted_project_path(model_path, field="model path")
    model_key = os.path.normcase(str(resolved_model))
    for entry in payload.get("entries", []):
        if not isinstance(entry, dict):
            continue
        trusted_model = _resolve_trusted_project_path(entry.get("path"), field="model path")
        if os.path.normcase(str(trusted_model)) != model_key:
            continue
        if entry.get("load_authorized") is not True:
            raise ValueError(f"Stage2 model is inventory-only: {resolved_model}")
        model_sha = str(entry.get("sha256") or "").strip().casefold()
        metadata_sha = str(entry.get("metadata_sha256") or "").strip().casefold()
        if len(model_sha) != 64 or len(metadata_sha) != 64:
            raise ValueError("Trusted Stage2 binding is missing SHA-256 values")
        return {
            "model_path": trusted_model,
            "model_sha256": model_sha,
            "metadata_path": _resolve_trusted_project_path(
                entry.get("metadata_path"),
                field="metadata path",
            ),
            "metadata_sha256": metadata_sha,
            "trust_manifest_path": DEFAULT_STAGE2_TRUST_MANIFEST,
        }
    raise ValueError(f"Stage2 model is not present in the server trust manifest: {resolved_model}")


def _load_prediction_model(checkpoint_path: Path, device: str):
    checkpoint = load_safe_checkpoint(checkpoint_path, map_location="cpu")
    config = AxonExperimentConfig.from_dict(dict(checkpoint["config"]))

    model = AxonMalwareModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    model.to(device)
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    model.eval()
    return model, config


def _load_prediction_context(
    request: PredictRequest,
    checkpoint_path: Path,
    device: str,
) -> PredictionContext:
    model, config = _load_prediction_model(checkpoint_path, device)

    stage2_path = _resolve_optional_path(request.stage2_model, DEFAULT_LOOP28_STAGE2_MODEL)
    if stage2_path is not None:
        binding = _trusted_stage2_binding(stage2_path)
        stage2 = Stage2ModelBundle.load(
            binding["model_path"],
            metadata_path=binding["metadata_path"],
            expected_model_sha256=binding["model_sha256"],
            expected_metadata_sha256=binding["metadata_sha256"],
        )
    else:
        stage2 = None
    if stage2 is not None:
        _assert_stage2_matches_checkpoint(stage2, config)

    family_path = _resolve_optional_path(request.family_classifier, DEFAULT_FAMILY_CLASSIFIER)
    family_classifier = FamilyClassifier.load(family_path) if family_path is not None else None

    return PredictionContext(
        request=request,
        checkpoint_path=checkpoint_path,
        device=device,
        model=model,
        config=config,
        stage2=stage2,
        family_classifier=family_classifier,
    )


def _assert_stage2_matches_checkpoint(
    stage2: Stage2ModelBundle, config: AxonExperimentConfig
) -> None:
    fields = ("max_byte_length", "pe_feature_dim", "stat_feature_dim", "lightweight_feature_dim")
    mismatches = {
        field: {
            "checkpoint": getattr(config, field),
            "stage2": getattr(stage2.checkpoint_config, field),
        }
        for field in fields
        if getattr(config, field) != getattr(stage2.checkpoint_config, field)
    }
    if mismatches:
        raise ValueError(f"Stage2 model does not match checkpoint config: {mismatches}")


def _extract_features(config: AxonExperimentConfig, file_path: Path) -> Optional[ExtractedFeatures]:
    extraction_config = ExtractionConfig.from_axon_config(
        config,
        max_file_size=config.max_byte_length,
        pe_feature_dim=config.pe_feature_dim,
    )

    byte_seq, pe_features, stat_features, lightweight_features, orig_len = extract_all_features(
        str(file_path),
        extraction_config,
        axon_config=config,
        allow_pe_fallback=config.allow_pe_fallback,
    )

    if (
        byte_seq is None
        or pe_features is None
        or stat_features is None
        or lightweight_features is None
    ):
        return None
    return ExtractedFeatures(
        byte_seq=byte_seq,
        pe_features=pe_features,
        stat_features=stat_features,
        lightweight_features=lightweight_features,
        orig_len=int(orig_len),
    )


def _base_model_probability(
    model: AxonMalwareModel,
    features: ExtractedFeatures,
    device: str,
) -> tuple[int, float, float, float]:
    byte_tensor = torch.from_numpy(features.byte_seq).long().unsqueeze(0).to(device)
    pe_tensor = torch.from_numpy(features.pe_features).float().unsqueeze(0).to(device)
    stat_tensor = torch.from_numpy(features.stat_features).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(byte_tensor, pe_tensor, stat_features=stat_tensor)["logits"]
        probs = torch.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred].item())
        prob_benign = float(probs[0, 0].item())
        prob_malicious = float(probs[0, 1].item())
    return pred, confidence, prob_benign, prob_malicious


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-(probs * np.log2(probs)).sum() / 8.0)


def _safe_logit(probability: float) -> float:
    clipped = min(max(float(probability), 1.0e-6), 1.0 - 1.0e-6)
    return float(math.log(clipped / (1.0 - clipped)))


def _byte_summary_features(byte_seq: np.ndarray, prefix_len: int, chunk_count: int) -> np.ndarray:
    byte_values = byte_seq.astype(np.uint8, copy=False)
    counts = np.bincount(byte_values, minlength=256).astype(np.float32)
    hist = counts / max(float(byte_values.shape[0]), 1.0)
    log_hist = np.log1p(counts) / np.log1p(max(float(byte_values.shape[0]), 1.0))

    prefix = byte_values[:prefix_len].astype(np.float32) / 255.0
    if prefix.shape[0] < prefix_len:
        prefix = np.pad(prefix, (0, prefix_len - prefix.shape[0]))

    chunk_features = []
    for chunk in np.array_split(byte_values, max(1, chunk_count)):
        if chunk.size == 0:
            chunk_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        chunk_counts = np.bincount(chunk, minlength=256).astype(np.float32)
        chunk_features.extend(
            [
                float(np.mean(chunk) / 255.0),
                float(np.std(chunk) / 255.0),
                _entropy_from_counts(chunk_counts),
                float(np.count_nonzero(chunk) / max(chunk.size, 1)),
                float(np.max(chunk_counts) / max(chunk.size, 1)),
            ]
        )

    scalar = np.asarray(
        [
            _entropy_from_counts(counts),
            float(np.count_nonzero(byte_values) / max(byte_values.shape[0], 1)),
            float(np.mean(byte_values) / 255.0),
            float(np.std(byte_values) / 255.0),
            float(np.max(counts) / max(byte_values.shape[0], 1)),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [hist, log_hist, prefix, np.asarray(chunk_features, dtype=np.float32), scalar]
    )


def _stage2_feature_vector(
    file_path: Path,
    features: ExtractedFeatures,
    base_prob_malicious: float,
    stage2_config: Stage2FeatureConfig,
) -> np.ndarray:
    prob = float(base_prob_malicious)
    parts = [
        np.asarray(
            [
                prob,
                prob * prob,
                abs(prob - 0.5),
                math.log(max(prob, 1.0e-6)),
                math.log(max(1.0 - prob, 1.0e-6)),
                _safe_logit(prob),
            ],
            dtype=np.float32,
        )
    ]
    if stage2_config.include_stat:
        parts.append(
            np.nan_to_num(features.stat_features.astype(np.float32, copy=False), copy=False)
        )
    if stage2_config.include_pe:
        parts.append(np.nan_to_num(features.pe_features.astype(np.float32, copy=False), copy=False))
    if stage2_config.include_lightweight:
        parts.append(
            np.nan_to_num(features.lightweight_features.astype(np.float32, copy=False), copy=False)
        )
    if stage2_config.include_byte_summary:
        parts.append(
            _byte_summary_features(
                features.byte_seq,
                int(stage2_config.prefix_len),
                int(stage2_config.chunk_count),
            )
        )
    if stage2_config.include_content_pe:
        parts.append(_content_pe_features_from_path(file_path))
    if stage2_config.include_content_pe_v2:
        raise ValueError("Stage2 content_pe_v2 features are not supported by predict_api")
    if stage2_config.include_content_string:
        raise ValueError("Stage2 content_string features are not supported by predict_api")
    if stage2_config.include_content_cert:
        raise ValueError("Stage2 content_cert features are not supported by predict_api")
    return np.concatenate(parts).astype(np.float32, copy=False)


def _predict_pe_file(context: PredictionContext, file_path: Path) -> Optional[dict[str, Any]]:
    features = _extract_features(context.config, file_path)
    if features is None:
        return None

    base_pred, base_confidence, base_prob_benign, base_prob_malicious = _base_model_probability(
        context.model,
        features,
        context.device,
    )

    result: dict[str, Any] = {
        "status": "predicted",
        "prediction": base_pred,
        "label": "malicious" if base_pred == 1 else "benign",
        "confidence": base_confidence,
        "prob_benign": base_prob_benign,
        "prob_malicious": base_prob_malicious,
        "original_length": int(features.orig_len),
        "base_model": {
            "prediction": base_pred,
            "label": "malicious" if base_pred == 1 else "benign",
            "confidence": base_confidence,
            "prob_benign": base_prob_benign,
            "prob_malicious": base_prob_malicious,
        },
    }

    if context.stage2 is not None:
        vector = _stage2_feature_vector(
            file_path,
            features,
            base_prob_malicious,
            context.stage2.feature_config,
        )
        stage2_prob = context.stage2.predict_probability(vector)
        threshold = float(
            context.request.threshold
            if context.request.threshold is not None
            else context.stage2.threshold
        )
        stage2_pred = int(stage2_prob >= threshold)
        result.update(
            {
                "prediction": stage2_pred,
                "label": "malicious" if stage2_pred == 1 else "benign",
                "confidence": stage2_prob if stage2_pred == 1 else 1.0 - stage2_prob,
                "prob_benign": 1.0 - stage2_prob,
                "prob_malicious": stage2_prob,
                "stage2": {
                    "enabled": True,
                    "model": str(context.stage2.path),
                    "threshold": threshold,
                    "feature_dim": int(vector.shape[0]),
                    "selected": context.stage2.selected,
                    "prob_malicious": stage2_prob,
                },
            }
        )
    else:
        threshold = float(
            context.request.threshold if context.request.threshold is not None else 0.5
        )
        if context.request.threshold is not None:
            pred = int(base_prob_malicious >= threshold)
            result.update(
                {
                    "prediction": pred,
                    "label": "malicious" if pred == 1 else "benign",
                    "confidence": base_prob_malicious if pred == 1 else base_prob_benign,
                }
            )
        result["threshold"] = threshold

    if result["prediction"] == 1 and context.family_classifier is not None:
        family = context.family_classifier.predict(features.pe_features, features.stat_features)
        if family is not None:
            result["malware_family"] = family
            result["family_classifier"] = str(context.family_classifier.path)

    return result


def predict_file(request: PredictRequest) -> dict[str, Any]:
    file_path = Path(request.file)
    checkpoint_path = Path(request.checkpoint)

    if not file_path.exists():
        return _error("file_not_found", f"Input file not found: {file_path}")
    if not checkpoint_path.exists():
        return _error("checkpoint_not_found", f"Checkpoint not found: {checkpoint_path}")

    stage2_path = _resolve_optional_path(request.stage2_model, DEFAULT_LOOP28_STAGE2_MODEL)
    if stage2_path is not None and not stage2_path.exists():
        return _error("stage2_model_not_found", f"Stage2 model not found: {stage2_path}")
    family_path = _resolve_optional_path(request.family_classifier, DEFAULT_FAMILY_CLASSIFIER)
    if family_path is not None and not family_path.exists():
        return _error("family_classifier_not_found", f"Family classifier not found: {family_path}")

    device = _resolve_device(request.device)

    context = None
    try:
        if request.scan_nested:
            return _predict_nested(
                request, checkpoint_path, device, file_path, stage2_path, family_path
            )
        context = _load_prediction_context(request, checkpoint_path, device)
        prediction = _predict_pe_file(context, file_path)
        if prediction is None:
            return _error("feature_extraction_failed", "PE feature extraction failed")
        return _ok(
            {
                "mode": "single_pe",
                "file": str(file_path),
                "checkpoint": str(checkpoint_path),
                "device": device,
                "stage2_model": str(context.stage2.path) if context.stage2 else None,
                "family_classifier": str(context.family_classifier.path)
                if context.family_classifier
                else None,
                "result": prediction,
            }
        )
    except Exception as exc:  # noqa: BLE001 - DLL callers need JSON, not tracebacks.
        return _error(type(exc).__name__, str(exc))
    finally:
        context = None
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()


def _predict_nested(
    request: PredictRequest,
    checkpoint_path: Path,
    device: str,
    file_path: Path,
    stage2_path: Optional[Path],
    family_path: Optional[Path],
) -> dict[str, Any]:
    scan_options = ArchiveScanOptions(
        max_depth=request.archive_max_depth,
        max_files=request.archive_max_files,
        max_total_bytes=request.archive_max_total_bytes,
        max_file_bytes=request.archive_max_file_bytes,
        keep_temp=True,
        scanner_binary=Path(request.archive_scanner) if request.archive_scanner else None,
    )
    report = None
    context = None
    response = None
    try:
        report = run_archive_scan(file_path, scan_options)
        visible_predictions = []
        predictions_truncated = False
        pe_prediction_count = 0
        malicious_inner_count = 0

        pe_target_iter = iter_pe_prediction_targets(report)
        first_entry = next(pe_target_iter, None)
        if first_entry is not None:
            context = _load_prediction_context(request, checkpoint_path, device)

        for entry in [first_entry] if first_entry is not None else []:
            inner_path = Path(entry["extracted_path"])
            result = _predict_pe_file(context, inner_path)
            pe_prediction_count += 1
            if result is None:
                prediction = {
                    "logical_path": entry["logical_path"],
                    "sha256": entry.get("sha256"),
                    "status": "feature_extraction_failed",
                }
            else:
                prediction = {
                    "logical_path": entry["logical_path"],
                    "sha256": entry.get("sha256"),
                    **result,
                }
            if prediction.get("status") == "predicted" and prediction.get("prediction") == 1:
                malicious_inner_count += 1
            predictions_truncated = (
                _append_visible_prediction(visible_predictions, prediction) or predictions_truncated
            )

        for entry in pe_target_iter:
            inner_path = Path(entry["extracted_path"])
            result = _predict_pe_file(context, inner_path)
            pe_prediction_count += 1
            if result is None:
                prediction = {
                    "logical_path": entry["logical_path"],
                    "sha256": entry.get("sha256"),
                    "status": "feature_extraction_failed",
                }
            else:
                prediction = {
                    "logical_path": entry["logical_path"],
                    "sha256": entry.get("sha256"),
                    **result,
                }
            if prediction.get("status") == "predicted" and prediction.get("prediction") == 1:
                malicious_inner_count += 1
            predictions_truncated = (
                _append_visible_prediction(visible_predictions, prediction) or predictions_truncated
            )

        response_payload = {
            "mode": "nested_archive",
            "file": str(file_path),
            "checkpoint": str(context.checkpoint_path)
            if context is not None
            else str(checkpoint_path),
            "device": context.device if context is not None else device,
            "stage2_model": str(context.stage2.path)
            if context is not None and context.stage2
            else (str(stage2_path) if stage2_path is not None else None),
            "family_classifier": (
                str(context.family_classifier.path)
                if context is not None and context.family_classifier
                else (str(family_path) if family_path is not None else None)
            ),
            "parent_verdict": "malicious"
            if malicious_inner_count
            else "benign_or_no_malicious_inner_pe",
            "runtime_rule": "any malicious inner PE triggers parent alert",
            "training_label_policy": "unknown_training_label: do not inherit parent archive/MSI label",
            **_scan_response_fields(report),
            "pe_prediction_count": pe_prediction_count,
            "malicious_inner_count": malicious_inner_count,
            "prediction_response_limit": NESTED_PREDICTION_RESPONSE_LIMIT,
            "predictions_truncated": predictions_truncated,
            "predictions": visible_predictions,
        }
        response = _ok(response_payload)
        return response
    finally:
        context = None
        if report is not None:
            cleanup_status = cleanup_scan_temp(report)
            if response is not None:
                response["archive_cleanup"] = cleanup_status
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()


def predict_json(request_json: str) -> str:
    try:
        payload = json.loads(request_json)
        if not isinstance(payload, dict):
            return json.dumps(
                _error("invalid_request", "Request JSON must be an object"), ensure_ascii=False
            )
        request = PredictRequest(**payload)
    except Exception as exc:  # noqa: BLE001
        return json.dumps(_error("invalid_request", str(exc)), ensure_ascii=False)
    return json.dumps(predict_file(request), ensure_ascii=False)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Axon stable JSON prediction API")
    parser.add_argument(
        "--request-json", required=True, help="JSON object accepted by PredictRequest"
    )
    args = parser.parse_args()
    print(predict_json(args.request_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
