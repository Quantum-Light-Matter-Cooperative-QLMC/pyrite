"""Public beam and source-distribution descriptions.

This is a lazy canonical import facade. Existing root and implementation imports
remain valid and refer to the same objects; no configuration is converted here.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrite.campaign.beam import (
        Beam as Beam,
    )
    from pyrite.campaign.beam import (
        BunchFrontTilt as BunchFrontTilt,
    )
    from pyrite.campaign.beam import (
        BunchLength as BunchLength,
    )
    from pyrite.campaign.beam import (
        GdfSource as GdfSource,
    )
    from pyrite.campaign.beam import (
        Offsets as Offsets,
    )
    from pyrite.campaign.beam import (
        PointBunch as PointBunch,
    )
    from pyrite.campaign.beam import (
        PointSpot as PointSpot,
    )
    from pyrite.campaign.beam import (
        Spot as Spot,
    )
    from pyrite.campaign.beam import (
        TransverseSpot as TransverseSpot,
    )
    from pyrite.campaign.longitudinal import LongitudinalDistribution as LongitudinalDistribution

_EXPORTS = {
    "Beam": ("pyrite.campaign.beam", "Beam"),
    "BunchFrontTilt": ("pyrite.campaign.beam", "BunchFrontTilt"),
    "BunchLength": ("pyrite.campaign.beam", "BunchLength"),
    "GdfSource": ("pyrite.campaign.beam", "GdfSource"),
    "LongitudinalDistribution": ("pyrite.campaign.longitudinal", "LongitudinalDistribution"),
    "Offsets": ("pyrite.campaign.beam", "Offsets"),
    "PointBunch": ("pyrite.campaign.beam", "PointBunch"),
    "PointSpot": ("pyrite.campaign.beam", "PointSpot"),
    "Spot": ("pyrite.campaign.beam", "Spot"),
    "TransverseSpot": ("pyrite.campaign.beam", "TransverseSpot"),
}

__all__ = [
    "Beam",
    "BunchFrontTilt",
    "BunchLength",
    "GdfSource",
    "LongitudinalDistribution",
    "Offsets",
    "PointBunch",
    "PointSpot",
    "Spot",
    "TransverseSpot",
]


def __getattr__(name: str) -> Any:
    """Resolve a canonical API name without eagerly loading numerical engines."""
    try:
        module, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
