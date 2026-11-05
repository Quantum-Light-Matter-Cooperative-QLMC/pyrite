"""Typed downstream-filter loader above the material catalog boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrite.instrument.model import FilterPlate
    from pyrite.materials import MaterialCatalog


def load_filter(name: str, *, catalog: MaterialCatalog | None = None) -> FilterPlate:
    """Load a named, validated :class:`~pyrite.geometry.FilterPlate` value.

    Placements can override any field by attaching ``ref = NAME`` in an
    ordered profile filter entry. This loader returns the unplaced definition's
    declared pose, without assigning the catalog name to the plate's name.
    """
    if catalog is None:
        from pyrite.materials import CATALOG as catalog
    try:
        definition = catalog.named_filters[name]
    except KeyError:
        raise KeyError(
            f"unknown filter preset {name!r}; have {sorted(catalog.named_filters)}"
        ) from None
    from .observation import filter_from_config

    return filter_from_config(definition)
