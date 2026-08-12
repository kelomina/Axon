"""Axon Flash + Pro Cascade Engine with Speakeasy-X Integration.

Combines:
1. Flash Model: Fast static PE + StreamGNN engine (< 20ms SLA)
2. Pro Model: Speakeasy-X dynamic emulation + API behavior classifier (for high uncertainty/OOD samples)
"""

from __future__ import annotations

import json
import os
import sys
import time
import tempfile
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEAKEASY_X_ROOT = Path("E:/Project/python/Speakeasy-X")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SPEAKEASY_X_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAKEASY_X_ROOT))

try:
    from ml_engine.feature_extractor import StaticExtractor, BehaviorExtractor
    from ml_engine.detection_engine import _simulate_worker
    SPEAKEASY_X_AVAILABLE = True
except Exception as e:
    SPEAKEASY_X_AVAILABLE = False
    print(f"[Warning] Speakeasy-X import issue: {e}")


class AxonCascadeEngine:
    def __init__(self, flash_model, pro_model=None, uncertainty_low: float = 0.20, uncertainty_high: float = 0.80):
        self.flash_model = flash_model
        self.pro_model = pro_model
        self.uncertainty_low = uncertainty_low
        self.uncertainty_high = uncertainty_high

    def predict_flash(self, feat: np.ndarray) -> float:
        """Flash static prediction (< 5ms)."""
        if feat.ndim == 1:
            feat = feat.reshape(1, -1)
        prob = float(self.flash_model.predict_proba(feat)[0, 1])
        return prob

    def should_escalate_to_pro(self, flash_prob: float, stat_features: Optional[np.ndarray] = None) -> bool:
        """Determines whether a sample should be escalated to Pro Speakeasy-X dynamic emulation."""
        # Condition 1: Uncertainty in [0.20, 0.80]
        if self.uncertainty_low <= flash_prob <= self.uncertainty_high:
            return True

        # Condition 2: Structural feature anomaly (if stat_features provided)
        if stat_features is not None:
            # Check Z-score of entropy / section anomalies
            entropy_stat = stat_features[15] if len(stat_features) > 15 else 0
            if entropy_stat > 80.0:  # High packing/obfuscation indicator
                return True

        return False

    def predict_pro_simulation(self, sample_path: str, timeout_sec: int = 15) -> Dict[str, Any]:
        """Run Speakeasy-X dynamic simulation in isolated process with timeout."""
        if not SPEAKEASY_X_AVAILABLE or not os.path.exists(sample_path):
            return {"status": "skipped", "prob": None}

        result_file = tempfile.mktemp(suffix=".json", prefix="speakeasy_axon_")
        proc = mp.Process(target=_simulate_worker, args=(sample_path, result_file))
        
        t0 = time.time()
        proc.start()
        proc.join(timeout=timeout_sec)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            if os.path.exists(result_file):
                try: os.remove(result_file)
                except Exception: pass
            return {"status": "timeout", "elapsed": time.time() - t0, "prob": 0.50}

        report_data = None
        if os.path.exists(result_file):
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
                os.remove(result_file)
            except Exception as e:
                report_data = {"status": "read_error", "error": str(e)}

        elapsed = time.time() - t0
        if report_data and report_data.get("status") == "success":
            res = report_data.get("result", {})
            apis = res.get("apis", [])
            # Simple heuristic / behavior score from API trace
            suspicious_apis = ["VirtualAlloc", "WriteProcessMemory", "CreateRemoteThread", "URLDownloadToFileA"]
            matched_apis = sum(1 for api in apis if any(s in str(api.get("api", "")) for s in suspicious_apis))
            pro_prob = min(1.0, 0.40 + matched_apis * 0.15)
            return {"status": "success", "elapsed": elapsed, "prob": pro_prob, "api_count": len(apis)}
        else:
            return {"status": "error", "elapsed": elapsed, "prob": 0.50}

    def predict(self, feat: np.ndarray, sample_path: Optional[str] = None) -> Tuple[float, str, float]:
        """Full Cascade Prediction: Flash first, escalate to Pro if needed."""
        t0 = time.time()
        flash_prob = self.predict_flash(feat)

        stat_feat = feat[256:305] if len(feat) >= 305 else None
        escalate = self.should_escalate_to_pro(flash_prob, stat_feat)

        if not escalate or not sample_path or not os.path.exists(sample_path):
            elapsed = time.time() - t0
            return flash_prob, "flash", elapsed

        # Escalate to Pro Engine
        pro_result = self.predict_pro_simulation(sample_path)
        pro_prob = pro_result.get("prob")
        if pro_prob is not None:
            final_prob = 0.3 * flash_prob + 0.7 * pro_prob
        else:
            final_prob = flash_prob

        elapsed = time.time() - t0
        return final_prob, "pro", elapsed
