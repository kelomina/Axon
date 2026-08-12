from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop171.capa_aggregate import CapaAggregateError, aggregate_capa_json  # noqa: E402


def test_aggregate_discards_match_locations() -> None:
    payload = {
        "meta": {},
        "rules": {
            "rule-a": {"meta": {"namespace": "collection"}, "source": "ignored", "matches": [{"address": 123}]},
            "rule-b": {"meta": {"namespace": "collection"}, "source": "ignored", "matches": [{"address": 456}]},
        },
    }

    assert aggregate_capa_json(payload).namespace_counts == (("collection", 2),)


def test_aggregate_rejects_schema_drift() -> None:
    with pytest.raises(CapaAggregateError):
        aggregate_capa_json({"rules": {}})


def test_aggregate_keeps_rules_without_optional_namespace() -> None:
    payload = {"meta": {}, "rules": {"rule-a": {"meta": {}, "source": "ignored", "matches": []}}}

    assert aggregate_capa_json(payload).namespace_counts == (("unscoped", 1),)
