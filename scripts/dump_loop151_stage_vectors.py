import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.loop151_runtime.raw_runtime import (
    Loop151Runtime,
    _feature_config,
    _single_probability,
    _stage2_vector,
)
from scripts.train_loop43_content_cross import content_cross_features_from_arrays
from scripts.train_stage2_cache_matrix import predict_scores

runtime = Loop151Runtime(device="cpu")
path = Path(sys.argv[1])
raw = runtime.extract(path)
base_probability = _single_probability(runtime.model, raw, "cpu")
primary = _stage2_vector(raw, base_probability, _feature_config(runtime.primary))
conservative = _stage2_vector(raw, base_probability, _feature_config(runtime.conservative))
cross = _stage2_vector(raw, base_probability, _feature_config(runtime.content_cross))
cross = np.concatenate([cross, content_cross_features_from_arrays(raw.content_pe_v1, raw.content_pe_v2)]).astype(np.float32)
noise = _stage2_vector(raw, base_probability, _feature_config(runtime.noise))
scores = {
    "primary": float(runtime.primary["model"] if False else 0.0),
    "base_probability": float(base_probability),
    "primary_vector": primary.tolist(),
    "conservative_vector": conservative.tolist(),
    "cross_vector": cross.tolist(),
    "noise_vector": noise.tolist(),
}
print(json.dumps(scores, separators=(",", ":")))
