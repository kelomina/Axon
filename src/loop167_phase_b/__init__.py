"""Unexecuted Phase-B source components for Loop167.

The controller imports static governance modules before it holds the one-shot
lease.  Keep this package initializer free of PE and numerical-runtime imports
so that those static checks cannot accidentally initialize the raw feature path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .raw_context import RawFeatureContext

__all__ = ["RawFeatureContext"]


def __getattr__(name: str) -> object:
    """Preserve the public context export without eager raw-runtime imports."""

    if name != "RawFeatureContext":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .raw_context import RawFeatureContext

    globals()[name] = RawFeatureContext
    return RawFeatureContext


def __dir__() -> list[str]:
    return sorted((*globals(), "RawFeatureContext"))
