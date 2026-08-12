"""Raw-file reconstruction of the frozen Loop151 research champion."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for search_path in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from config import AxonExperimentConfig
from evaluate_stage2_oof_stacker import score_oof_payload
from model import AxonMalwareModel
from security import load_safe_checkpoint
from train_loop135_pairwise_selector import CONTENT_FEATURE_NAMES, SCORE_FEATURE_NAMES, predict_scores as selector_scores
from train_loop43_content_cross import content_cross_features_from_arrays
from train_stage2_cache_matrix import (
    CONTENT_PE_V2_FEATURE_NAMES,
    CONTENT_STRING_FEATURE_NAMES,
    FeatureConfig,
    _byte_summary_features,
    _content_pe_v2_features_from_path,
    _content_string_features_from_path,
    predict_scores,
)
from kvd_features.content_pe_v1 import CONTENT_PE_V1_FEATURE_NAMES, extract_content_pe_v1_features
from kvd_features.extractor import ExtractionConfig, extract_all_features

from .trusted_signer import TrustedSignerDecision, apply_trusted_signer_guard


class Loop151RuntimeError(RuntimeError):
    """Raised when the frozen champion cannot be reproduced safely."""


@dataclass(frozen=True)
class RawFeatures:
    byte_seq: np.ndarray
    pe_features: np.ndarray
    stat_features: np.ndarray
    lightweight_features: np.ndarray
    content_pe_v1: np.ndarray
    content_pe_v2: np.ndarray
    content_string: np.ndarray


@dataclass(frozen=True)
class Loop151Prediction:
    prediction: int
    probability: float
    primary_probability: float
    conservative_probability: float
    content_cross_probability: float
    loop130_prediction: int
    loop134_probability: float
    loop136_prediction: int
    selector_score: float | None
    signer: TrustedSignerDecision


@dataclass(frozen=True)
class FieldAblationScores:
    loop28_probability: float
    primary_probability: float
    conservative_probability: float
    content_cross_probability: float
    noise_probability: float
    selector_score: float | None


@dataclass(frozen=True)
class FieldAblationResult:
    loop28_score: float
    arm_a: int
    arm_b: int
    arm_c: int
    arm_d: int
    arm_e: int
    scores: FieldAblationScores


class Loop151FieldAblationRuntime:
    def __init__(self, *, device: str = "cpu", artifact_manifest: Path | None = None) -> None:
        self._runtime = Loop151Runtime(device=device, artifact_manifest=artifact_manifest)
        self._loop28 = load_loop28_stage2()

    def predict_one(self, path: Path) -> FieldAblationResult:
        file_path = path.resolve()
        if not file_path.is_file():
            raise Loop151RuntimeError(f"File does not exist: {file_path}")
        raw = self._runtime.extract(file_path)
        base_prob = _single_probability(self._runtime.model, raw, self._runtime.device)

        primary_feat = _stage2_vector(raw, base_prob, _feature_config(self._runtime.primary)).reshape(1, -1)
        conservative_feat = _stage2_vector(raw, base_prob, _feature_config(self._runtime.conservative)).reshape(1, -1)
        cross_feat = _stage2_vector(raw, base_prob, _feature_config(self._runtime.content_cross))
        cross_feat = np.concatenate([cross_feat, content_cross_features_from_arrays(raw.content_pe_v1, raw.content_pe_v2)]).astype(np.float32, copy=False).reshape(1, -1)
        noise_feat = _stage2_vector(raw, base_prob, _feature_config(self._runtime.noise)).reshape(1, -1)
        loop28_feat = _stage2_vector(raw, base_prob, self._loop28.feature_config).reshape(1, -1)

        loop28_prob = self._loop28.predict_probability(loop28_feat[0])
        arm_a = _prediction(loop28_prob, self._loop28.threshold)

        primary_score, _ = score_oof_payload(self._runtime.primary, primary_feat)
        primary_prob = float(primary_score[0])
        arm_b = _prediction(primary_prob, self._runtime.primary["threshold"])

        conservative_score, _ = score_oof_payload(self._runtime.conservative, conservative_feat)
        conservative_prob = float(conservative_score[0])
        cross_score = float(predict_scores(self._runtime.content_cross["model"], cross_feat)[0])
        noise_score, _ = score_oof_payload(self._runtime.noise, noise_feat)
        noise_prob = float(noise_score[0])

        primary_pred = arm_b
        conservative_pred = _prediction(conservative_prob, self._runtime.conservative["threshold"])
        cross_pred = _prediction(cross_score, self._runtime.content_cross["threshold"])
        possible = primary_pred == 1 and (conservative_pred == 0 or cross_pred == 0)
        v1 = {name: raw.content_pe_v1[index] for index, name in enumerate(CONTENT_PE_V1_FEATURE_NAMES)}
        v2 = {name: raw.content_pe_v2[index] for index, name in enumerate(CONTENT_PE_V2_FEATURE_NAMES)}
        strings = {name: raw.content_string[index] for index, name in enumerate(CONTENT_STRING_FEATURE_NAMES)}
        r4 = (possible and primary_prob <= 0.65 and v2["v2_resource_data_entry_count_log"] >= 2.0
              and v2["v2_resource_type_icon_count_log"] >= 1.5 and v1["content_dir_resource_size_ratio"] >= 0.001)
        r5_flip = r4 or (possible and not r4 and strings["string_benign_vendor_count_log"] >= 3.0)
        arm_c = 0 if r5_flip else primary_pred

        noise_pred = _prediction(noise_prob, self._runtime.noise["threshold"])
        selector_score: float | None = None
        arm_d = arm_c
        if arm_c != noise_pred:
            selector_values = [
                primary_prob, noise_prob, noise_prob - primary_prob,
                abs(primary_prob - 0.5), abs(noise_prob - 0.5),
                abs(noise_prob - 0.5) - abs(primary_prob - 0.5),
                float(arm_c), float(noise_pred),
                float(arm_c == 0 and noise_pred == 1),
                float(arm_c == 1 and noise_pred == 0),
            ]
            for name in SCORE_FEATURE_NAMES + CONTENT_FEATURE_NAMES:
                if len(selector_values) >= 32:
                    break
                selector_values.append(float(v1[name] if name in v1 else v2[name] if name in v2 else strings[name]))
            selector_matrix = np.asarray(selector_values, dtype=np.float32).reshape(1, -1)
            selector_score = float(selector_scores(self._runtime.selector["model"], selector_matrix)[0])
            if selector_score >= float(self._runtime.selector["selected"]["threshold"]):
                arm_d = noise_pred

        auth_status, signer_subject = _authenticode(file_path) if arm_d == 1 else ("", "")
        signer = apply_trusted_signer_guard(loop136_prediction=arm_d, authenticode_status=auth_status, signer_subject=signer_subject)
        arm_e = signer.prediction

        return FieldAblationResult(
            loop28_score=loop28_prob, arm_a=arm_a, arm_b=arm_b, arm_c=arm_c, arm_d=arm_d, arm_e=arm_e,
            scores=FieldAblationScores(
                loop28_probability=loop28_prob, primary_probability=primary_prob,
                conservative_probability=conservative_prob, content_cross_probability=cross_score,
                noise_probability=noise_prob, selector_score=selector_score,
            ),
        )


@dataclass(frozen=True)
class FieldAblationPrediction:
    loop28_prediction: int
    loop28_probability: float
    primary_prediction: int
    primary_probability: float
    loop130_prediction: int
    loop136_prediction: int
    final_prediction: int
    conservative_probability: float
    content_cross_probability: float
    noise_probability: float
    selector_score: float | None
    r5_flip: bool
    signer_downgraded: bool

    @property
    def arm_predictions(self) -> dict[str, int]:
        return {
            "A": self.loop28_prediction,
            "B": self.primary_prediction,
            "C": self.loop130_prediction,
            "D": self.loop136_prediction,
            "E": self.final_prediction,
        }

    @classmethod
    def from_result(cls, result: FieldAblationResult) -> FieldAblationPrediction:
        return cls(
            loop28_prediction=result.arm_a, loop28_probability=result.loop28_score,
            primary_prediction=result.arm_b, primary_probability=result.scores.primary_probability,
            loop130_prediction=result.arm_c, loop136_prediction=result.arm_d,
            final_prediction=result.arm_e,
            conservative_probability=result.scores.conservative_probability,
            content_cross_probability=result.scores.content_cross_probability,
            noise_probability=result.scores.noise_probability,
            selector_score=result.scores.selector_score,
            r5_flip=(result.arm_a == 1 and result.arm_c == 0),
            signer_downgraded=(result.arm_d == 1 and result.arm_e == 0),
        )


@dataclass(frozen=True)
class Loop28Stage2Bundle:
    payload: dict[str, Any]
    threshold: float
    feature_config: FeatureConfig

    def predict_probability(self, features: np.ndarray) -> float:
        row = np.asarray(features, dtype=np.float32).reshape(-1)
        if row.shape[0] != int(self.payload["n_features"]):
            raise ValueError("Loop28 Stage-2 feature dimension mismatch")
        score = float(self.payload["baseline_prediction"])
        for tree in self.payload["trees"]:
            nodes = tree["nodes"]
            node_index = 0
            while True:
                node = nodes[node_index]
                if bool(node["is_leaf"]):
                    score += float(node["value"])
                    break
                value = float(row[int(node["feature_idx"])])
                if math.isnan(value):
                    go_left = bool(node["missing_go_to_left"])
                else:
                    go_left = value <= float(node["num_threshold"])
                node_index = int(node["left"] if go_left else node["right"])
        clipped = min(max(score, -50.0), 50.0)
        return float(1.0 / (1.0 + math.exp(-clipped)))


def build_field_ablation_prediction(
    *,
    loop28_prediction: int,
    loop28_probability: float,
    primary_prediction: int,
    primary_probability: float,
    loop130_prediction: int,
    loop136_prediction: int,
    final_prediction: int,
    conservative_probability: float,
    content_cross_probability: float,
    noise_probability: float,
    selector_score: float | None,
    r5_flip: bool,
    signer_downgraded: bool,
) -> FieldAblationPrediction:
    predictions = (
        loop28_prediction,
        primary_prediction,
        loop130_prediction,
        loop136_prediction,
        final_prediction,
    )
    if any(value not in (0, 1) for value in predictions):
        raise ValueError("Field ablation predictions must be binary")
    probabilities = (
        loop28_probability,
        primary_probability,
        conservative_probability,
        content_cross_probability,
        noise_probability,
    )
    if not all(math.isfinite(float(value)) for value in probabilities):
        raise ValueError("Field ablation probabilities must be finite")
    if selector_score is not None and not math.isfinite(float(selector_score)):
        raise ValueError("Field ablation selector score must be finite")
    return FieldAblationPrediction(
        loop28_prediction=loop28_prediction,
        loop28_probability=float(loop28_probability),
        primary_prediction=primary_prediction,
        primary_probability=float(primary_probability),
        loop130_prediction=loop130_prediction,
        loop136_prediction=loop136_prediction,
        final_prediction=final_prediction,
        conservative_probability=float(conservative_probability),
        content_cross_probability=float(content_cross_probability),
        noise_probability=float(noise_probability),
        selector_score=None if selector_score is None else float(selector_score),
        r5_flip=bool(r5_flip),
        signer_downgraded=bool(signer_downgraded),
    )


def load_loop28_stage2(metadata_path: Path | None = None) -> Loop28Stage2Bundle:
    resolved_metadata = (
        Path(metadata_path)
        if metadata_path is not None
        else PROJECT_ROOT / "manifests/roadmap_9997/p0_raw_replay/loop28_stage2.metadata.json"
    ).resolve()
    try:
        metadata = json.loads(resolved_metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Loop151RuntimeError(f"Cannot read frozen Loop28 metadata: {resolved_metadata}") from exc
    if metadata.get("schema") != "axon_stage2_model_metadata_v1":
        raise ValueError("Frozen Loop28 metadata schema changed")
    if float(metadata.get("threshold", -1.0)) != 0.5:
        raise ValueError("Frozen Loop28 threshold changed")
    if int(metadata.get("feature_dim", -1)) != 1520:
        raise ValueError("Frozen Loop28 feature dimension changed")
    if str(metadata.get("checkpoint_sha256") or "").casefold() != str(
        _load_artifact_manifest(None)["checkpoint"]["sha256"]
    ).casefold():
        raise ValueError("Frozen Loop28 checkpoint binding changed")
    json_path = PROJECT_ROOT / "models/random_20w_8192/loop28_stage2_hgb.json"
    if _sha256(json_path) != "c2c0cb0f39d12892891f9949e6f765e03fb1188f8fa3f3e574c0c4f73c63c648":
        raise ValueError("Frozen Loop28 JSON model SHA-256 mismatch")
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Loop151RuntimeError(f"Cannot read frozen Loop28 JSON model: {json_path}") from exc
    if payload.get("schema") != "axon_stage2_hgb_json_v1":
        raise ValueError("Frozen Loop28 JSON model schema changed")
    if float(payload.get("threshold", -1.0)) != 0.5:
        raise ValueError("Frozen Loop28 JSON model threshold changed")
    if int(payload.get("n_features", -1)) != 1520:
        raise ValueError("Frozen Loop28 JSON model feature dimension changed")
    feature_config = _feature_config(payload)
    return Loop28Stage2Bundle(payload=payload, threshold=0.5, feature_config=feature_config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_artifact_manifest(path: Path | None) -> dict[str, dict[str, str]]:
    manifest_path = path or PROJECT_ROOT / "manifests/roadmap_9997/loop151_runtime/frozen_artifacts.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Loop151RuntimeError(f"Cannot read frozen artifact manifest: {manifest_path}") from exc
    if payload.get("schema") != "axon_loop151_raw_runtime_artifacts_v1":
        raise Loop151RuntimeError("Unexpected frozen artifact manifest schema")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise Loop151RuntimeError("Frozen artifact manifest has no artifact mapping")
    required = {"checkpoint", "primary", "conservative", "content_cross", "noise", "selector"}
    if set(artifacts) != required:
        raise Loop151RuntimeError("Frozen artifact manifest names do not match the Loop151 chain")
    resolved: dict[str, dict[str, str]] = {}
    root = PROJECT_ROOT.resolve()
    for name, item in artifacts.items():
        if not isinstance(item, dict):
            raise Loop151RuntimeError(f"Artifact {name} metadata is invalid")
        relative_path = str(item.get("path") or "")
        expected_sha = str(item.get("sha256") or "").casefold()
        artifact_path = (root / relative_path).resolve()
        if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
            raise Loop151RuntimeError(f"Artifact {name} has an invalid SHA-256")
        try:
            artifact_path.relative_to(root)
        except ValueError as exc:
            raise Loop151RuntimeError(f"Artifact {name} escapes the project root") from exc
        if not artifact_path.is_file():
            raise Loop151RuntimeError(f"Frozen artifact is missing: {artifact_path}")
        actual_sha = _sha256(artifact_path)
        if actual_sha != expected_sha:
            raise Loop151RuntimeError(f"Frozen artifact SHA-256 mismatch: {name}")
        resolved[name] = {"path": str(artifact_path), "sha256": expected_sha}
    return resolved


def _load_frozen_pickle(path: Path) -> dict[str, Any]:
    # SHA verification occurs before deserialization; arbitrary paths are never accepted.
    import warnings
    from sklearn.base import InconsistentVersionWarning

    class _SklearnCompatibleUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> object:
            if module == "_loss":
                module = "sklearn._loss._loss"
            if module not in sys.modules:
                try:
                    __import__(module)
                except ImportError:
                    pass
            return super().find_class(module, name)

    try:
        scripts_dir = str(PROJECT_ROOT / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        with path.open("rb") as handle:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                payload = _SklearnCompatibleUnpickler(handle).load()
    except Exception as exc:
        raise Loop151RuntimeError(f"Frozen sklearn artifact cannot be loaded: {path.name}") from exc
    if not isinstance(payload, dict):
        raise Loop151RuntimeError(f"Frozen sklearn artifact has invalid payload: {path.name}")
    return payload


def _feature_config(payload: dict[str, Any]) -> FeatureConfig:
    value = payload.get("feature_config")
    if isinstance(value, FeatureConfig):
        return value
    if hasattr(value, "__dict__"):
        value = value.__dict__
    if not isinstance(value, dict):
        raise Loop151RuntimeError("Frozen artifact does not contain a feature configuration")
    return FeatureConfig(**value)


def _single_probability(model: AxonMalwareModel, features: RawFeatures, device: str) -> float:
    byte_tensor = torch.from_numpy(features.byte_seq).long().unsqueeze(0).to(device)
    pe_tensor = torch.from_numpy(features.pe_features).float().unsqueeze(0).to(device)
    stat_tensor = torch.from_numpy(features.stat_features).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(byte_tensor, pe_tensor, stat_features=stat_tensor)["logits"]
        return float(torch.softmax(logits, dim=1)[0, 1].item())


def _probability_columns(probability: float) -> np.ndarray:
    clipped = min(max(float(probability), 1.0e-6), 1.0 - 1.0e-6)
    return np.asarray(
        [
            probability,
            probability * probability,
            abs(probability - 0.5),
            math.log(clipped),
            math.log(1.0 - clipped),
            math.log(clipped / (1.0 - clipped)),
        ],
        dtype=np.float32,
    )


def _stage2_vector(features: RawFeatures, probability: float, config: FeatureConfig) -> np.ndarray:
    parts: list[np.ndarray] = [_probability_columns(probability)]
    if config.include_stat:
        parts.append(features.stat_features)
    if config.include_pe:
        parts.append(features.pe_features)
    if config.include_lightweight:
        parts.append(features.lightweight_features)
    if config.include_byte_summary:
        parts.append(_byte_summary_features(features.byte_seq, config.prefix_len, config.chunk_count))
    if config.include_content_pe:
        parts.append(features.content_pe_v1)
    if config.include_content_pe_v2:
        groups = tuple(config.content_pe_v2_groups)
        if groups != ("all",):
            raise Loop151RuntimeError(f"Unsupported frozen content PE v2 groups: {groups}")
        parts.append(features.content_pe_v2)
    if config.include_content_string:
        parts.append(features.content_string)
    if config.include_content_cert:
        raise Loop151RuntimeError("Loop151 champion does not permit a certificate feature fallback")
    vector = np.concatenate(parts).astype(np.float32, copy=False)
    if not np.isfinite(vector).all():
        raise Loop151RuntimeError("Raw feature vector contains non-finite values")
    return np.nan_to_num(vector, copy=False)


def _prediction(score: float, threshold: float) -> int:
    return int(float(score) >= float(threshold))


def _authenticode(path: Path) -> tuple[str, str]:
    if os.name != "nt":
        return "Unavailable", ""
    script = (
        "$sig=Get-AuthenticodeSignature -LiteralPath $env:AXON_LOOP151_SCAN_PATH;"
        "[pscustomobject]@{status=[string]$sig.Status;subject="
        "$(if($sig.SignerCertificate){[string]$sig.SignerCertificate.Subject}else{''})}"
        "|ConvertTo-Json -Compress"
    )
    environment = os.environ.copy()
    environment["AXON_LOOP151_SCAN_PATH"] = str(path)
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
            timeout=30,
        )
        payload = json.loads(result.stdout)
        return str(payload.get("status") or "Unavailable"), str(payload.get("subject") or "")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return "Unavailable", ""


class Loop151Runtime:
    """Loads only frozen assets and reproduces the Loop151 decision DAG for a file."""

    def __init__(self, *, device: str = "cpu", artifact_manifest: Path | None = None) -> None:
        self.device = device
        artifacts = _load_artifact_manifest(artifact_manifest)
        self._artifacts = artifacts
        checkpoint = load_safe_checkpoint(Path(artifacts["checkpoint"]["path"]), map_location="cpu")
        self.config = AxonExperimentConfig.from_dict(dict(checkpoint["config"]))
        self.model = AxonMalwareModel(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        self.primary = _load_frozen_pickle(Path(artifacts["primary"]["path"]))
        self.conservative = _load_frozen_pickle(Path(artifacts["conservative"]["path"]))
        self.content_cross = _load_frozen_pickle(Path(artifacts["content_cross"]["path"]))
        self.noise = _load_frozen_pickle(Path(artifacts["noise"]["path"]))
        self.selector = _load_frozen_pickle(Path(artifacts["selector"]["path"]))
        self._validate_frozen_payloads()

    def _validate_frozen_payloads(self) -> None:
        for name, payload, expected in (
            ("primary", self.primary, 0.31),
            ("conservative", self.conservative, 0.415),
            ("content_cross", self.content_cross, 0.4),
            ("noise", self.noise, 0.39),
        ):
            if abs(float(payload.get("threshold", -1.0)) - expected) > 1.0e-9:
                raise Loop151RuntimeError(f"Frozen {name} threshold changed")
        selected = self.selector.get("selected") or {}
        if abs(float(selected.get("threshold", -1.0)) - 0.79) > 1.0e-9:
            raise Loop151RuntimeError("Frozen Loop136 selector threshold changed")
        if list(self.selector.get("feature_names") or []) != SCORE_FEATURE_NAMES + CONTENT_FEATURE_NAMES:
            raise Loop151RuntimeError("Frozen Loop136 selector feature order changed")
        if self.selector.get("support_info") is not None:
            raise Loop151RuntimeError("Frozen Loop136 selector unexpectedly requires kNN support")

    def extract(self, path: Path) -> RawFeatures:
        extraction_config = ExtractionConfig.from_axon_config(
            self.config,
            max_file_size=self.config.max_byte_length,
            pe_feature_dim=self.config.pe_feature_dim,
        )
        byte_seq, pe_features, stat_features, lightweight_features, _orig_len = extract_all_features(
            str(path), extraction_config, axon_config=self.config, allow_pe_fallback=self.config.allow_pe_fallback
        )
        if any(item is None for item in (byte_seq, pe_features, stat_features, lightweight_features)):
            raise Loop151RuntimeError("Raw PE feature extraction failed")
        raw = RawFeatures(
            byte_seq=np.asarray(byte_seq, dtype=np.int64),
            pe_features=np.asarray(pe_features, dtype=np.float32),
            stat_features=np.asarray(stat_features, dtype=np.float32),
            lightweight_features=np.asarray(lightweight_features, dtype=np.float32),
            content_pe_v1=np.asarray(extract_content_pe_v1_features(path), dtype=np.float32),
            content_pe_v2=np.asarray(_content_pe_v2_features_from_path(path), dtype=np.float32),
            content_string=np.asarray(_content_string_features_from_path(path), dtype=np.float32),
        )
        expected = (
            self.config.max_byte_length,
            self.config.pe_feature_dim,
            self.config.stat_feature_dim,
            self.config.lightweight_feature_dim,
            len(CONTENT_PE_V1_FEATURE_NAMES),
            len(CONTENT_PE_V2_FEATURE_NAMES),
            len(CONTENT_STRING_FEATURE_NAMES),
        )
        actual = tuple(array.shape[0] for array in raw.__dict__.values())
        if actual != expected or not all(np.isfinite(array).all() for array in raw.__dict__.values()):
            raise Loop151RuntimeError(f"Raw feature shapes or values do not match Loop151: {actual}")
        return raw

    def predict_path(self, path: str | Path) -> Loop151Prediction:
        file_path = Path(path).resolve()
        if not file_path.is_file():
            raise Loop151RuntimeError(f"Input file does not exist: {file_path}")
        raw = self.extract(file_path)
        base_probability = _single_probability(self.model, raw, self.device)
        primary_vector = _stage2_vector(raw, base_probability, _feature_config(self.primary)).reshape(1, -1)
        conservative_vector = _stage2_vector(raw, base_probability, _feature_config(self.conservative)).reshape(1, -1)
        cross_vector = _stage2_vector(raw, base_probability, _feature_config(self.content_cross))
        cross_vector = np.concatenate(
            [cross_vector, content_cross_features_from_arrays(raw.content_pe_v1, raw.content_pe_v2)]
        ).astype(np.float32, copy=False).reshape(1, -1)
        noise_vector = _stage2_vector(raw, base_probability, _feature_config(self.noise)).reshape(1, -1)
        primary_score, _ = score_oof_payload(self.primary, primary_vector)
        conservative_score, _ = score_oof_payload(self.conservative, conservative_vector)
        cross_score = float(predict_scores(self.content_cross["model"], cross_vector)[0])
        noise_score, _ = score_oof_payload(self.noise, noise_vector)
        primary_probability = float(primary_score[0])
        conservative_probability = float(conservative_score[0])
        content_cross_probability = float(cross_score)
        noise_probability = float(noise_score[0])
        primary_prediction = _prediction(primary_probability, self.primary["threshold"])
        conservative_prediction = _prediction(conservative_probability, self.conservative["threshold"])
        cross_prediction = _prediction(content_cross_probability, self.content_cross["threshold"])
        possible = primary_prediction == 1 and (conservative_prediction == 0 or cross_prediction == 0)
        v1 = {name: raw.content_pe_v1[index] for index, name in enumerate(CONTENT_PE_V1_FEATURE_NAMES)}
        v2 = {name: raw.content_pe_v2[index] for index, name in enumerate(CONTENT_PE_V2_FEATURE_NAMES)}
        strings = {name: raw.content_string[index] for index, name in enumerate(CONTENT_STRING_FEATURE_NAMES)}
        r4 = (
            possible
            and primary_probability <= 0.65
            and v2["v2_resource_data_entry_count_log"] >= 2.0
            and v2["v2_resource_type_icon_count_log"] >= 1.5
            and v1["content_dir_resource_size_ratio"] >= 0.001
        )
        r5_flip = r4 or (possible and not r4 and strings["string_benign_vendor_count_log"] >= 3.0)
        loop130_prediction = 0 if r5_flip else primary_prediction
        noise_prediction = _prediction(noise_probability, self.noise["threshold"])
        selector_score: float | None = None
        loop136_prediction = loop130_prediction
        probability = primary_probability
        if loop130_prediction != noise_prediction:
            selector_values = [
                primary_probability,
                noise_probability,
                noise_probability - primary_probability,
                abs(primary_probability - 0.5),
                abs(noise_probability - 0.5),
                abs(noise_probability - 0.5) - abs(primary_probability - 0.5),
                float(loop130_prediction),
                float(noise_prediction),
                float(loop130_prediction == 0 and noise_prediction == 1),
                float(loop130_prediction == 1 and noise_prediction == 0),
            ]
            for name in CONTENT_FEATURE_NAMES:
                selector_values.append(float(v1[name] if name in v1 else v2[name] if name in v2 else strings[name]))
            selector_matrix = np.asarray(selector_values, dtype=np.float32).reshape(1, -1)
            selector_score = float(selector_scores(self.selector["model"], selector_matrix)[0])
            if selector_score >= float(self.selector["selected"]["threshold"]):
                loop136_prediction = noise_prediction
                probability = noise_probability
        auth_status, signer_subject = _authenticode(file_path) if loop136_prediction == 1 else ("", "")
        signer = apply_trusted_signer_guard(
            loop136_prediction=loop136_prediction,
            authenticode_status=auth_status,
            signer_subject=signer_subject,
        )
        return Loop151Prediction(
            prediction=signer.prediction,
            probability=probability,
            primary_probability=primary_probability,
            conservative_probability=conservative_probability,
            content_cross_probability=content_cross_probability,
            loop130_prediction=loop130_prediction,
            loop134_probability=noise_probability,
            loop136_prediction=loop136_prediction,
            selector_score=selector_score,
            signer=signer,
        )
