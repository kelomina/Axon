import json
import sys
from pathlib import Path

import numpy as np

from train_stage2_cache_matrix import _content_pe_v2_features_from_path, _content_string_features_from_path

path = Path(sys.argv[1])
payload = {
    "v2": np.asarray(_content_pe_v2_features_from_path(path), dtype=np.float32).tolist(),
    "strings": np.asarray(_content_string_features_from_path(path), dtype=np.float32).tolist(),
}
print(json.dumps(payload, separators=(",", ":")))
