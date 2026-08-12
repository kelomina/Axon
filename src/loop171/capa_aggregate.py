"""Parse capa JSON into aggregate-only capability evidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping


class CapaAggregateError(ValueError):
    """Raised when capa's machine-readable result no longer matches the contract."""


@dataclass(frozen=True)
class CapabilityAggregate:
    rule_count: int
    namespace_counts: tuple[tuple[str, int], ...]


def aggregate_capa_json(payload: Mapping[str, object]) -> CapabilityAggregate:
    """Retain only rule and namespace counts, never match locations or strings."""
    if set(payload) != {"meta", "rules"} or not isinstance(payload["rules"], Mapping):
        raise CapaAggregateError("capa JSON top-level schema drifted")
    namespaces: Counter[str] = Counter()
    for rule in payload["rules"].values():
        if not isinstance(rule, Mapping) or set(rule) != {"meta", "source", "matches"}:
            raise CapaAggregateError("capa rule schema drifted")
        metadata = rule["meta"]
        if not isinstance(metadata, Mapping):
            raise CapaAggregateError("capa rule metadata drifted")
        namespace = metadata.get("namespace", "unscoped")
        if not isinstance(namespace, str) or not namespace.strip():
            raise CapaAggregateError("capa rule namespace has an invalid type")
        namespaces[namespace.strip()] += 1
    return CapabilityAggregate(len(payload["rules"]), tuple(sorted(namespaces.items())))
