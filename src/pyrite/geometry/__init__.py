"""Public sample geometry, targets, and physical bodies.

This is a lazy canonical import facade. Existing root and implementation imports
remain valid and refer to the same objects; no configuration is converted here.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrite.bodies import (
        RectangularBody as RectangularBody,
    )
    from pyrite.bodies import (
        SubstanceRef as SubstanceRef,
    )
    from pyrite.campaign.geometry import (
        BlazedGrooves as BlazedGrooves,
    )
    from pyrite.campaign.geometry import (
        CrystalCut as CrystalCut,
    )
    from pyrite.campaign.geometry import (
        Footprint as Footprint,
    )
    from pyrite.campaign.geometry import (
        Layer as Layer,
    )
    from pyrite.campaign.geometry import (
        Slab as Slab,
    )
    from pyrite.campaign.geometry import (
        Stack as Stack,
    )
    from pyrite.campaign.target import Target as Target
    from pyrite.instrument import Filter as Filter
    from pyrite.instrument.model import FilterPlate as FilterPlate
    from pyrite.materials._named_filters import load_filter as load_filter

_EXPORTS = {
    "BlazedGrooves": ("pyrite.campaign.geometry", "BlazedGrooves"),
    "CrystalCut": ("pyrite.campaign.geometry", "CrystalCut"),
    "Filter": ("pyrite.instrument", "Filter"),
    "FilterPlate": ("pyrite.instrument", "FilterPlate"),
    "Footprint": ("pyrite.campaign.geometry", "Footprint"),
    "Layer": ("pyrite.campaign.geometry", "Layer"),
    "load_filter": ("pyrite.materials._named_filters", "load_filter"),
    "RectangularBody": ("pyrite.bodies", "RectangularBody"),
    "Slab": ("pyrite.campaign.geometry", "Slab"),
    "Stack": ("pyrite.campaign.geometry", "Stack"),
    "SubstanceRef": ("pyrite.bodies", "SubstanceRef"),
    "Target": ("pyrite.campaign.target", "Target"),
}

__all__ = [
    "BlazedGrooves",
    "CrystalCut",
    "Filter",
    "FilterPlate",
    "Footprint",
    "Layer",
    "load_filter",
    "RectangularBody",
    "Slab",
    "Stack",
    "SubstanceRef",
    "Target",
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
