from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_loop164_local_oof_result import (  # noqa: E402
    DEFAULT_FOLDS,
    DEFAULT_PREDICTIONS,
    DEFAULT_REPORT,
    _max_target_errors,
    _roc_auc,
    analyze,
)


def test_auc_uses_average_ranks_for_ties():
    assert _roc_auc([0, 1, 0, 1], [0.1, 0.8, 0.8, 0.9]) == 0.875


def test_balanced_local_target_allows_only_five_supported_errors():
    assert _max_target_errors(9739) == 5


def test_completed_local_oof_analysis_is_bound_and_nonpromotable():
    payload = analyze(
        report_path=DEFAULT_REPORT,
        predictions_path=DEFAULT_PREDICTIONS,
        folds_path=DEFAULT_FOLDS,
    )

    assert payload["fixed_threshold_result"]["f1"] == 0.9620420176595961
    assert payload["fixed_threshold_result"]["errors"] == 748
    assert payload["target_gap"]["minimum_error_reduction_required"] == 743
    assert payload["ready_for"]["more_standalone_seeds_or_epochs"] is False
    assert payload["ready_for"]["candidate_promotion"] is False
    assert payload["decision"].startswith("stop_current_standalone_scale")
