"""Typed downstream-filter loader above the material catalog boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pyrite.instrument.model import FilterPlate
    from pyrite.materials import MaterialCatalog


def _definition(name: str, catalog: MaterialCatalog | None) -> Mapping[str, object]:
    if catalog is None:
        from pyrite.materials import CATALOG as catalog
    try:
        return catalog.named_filters[name]
    except KeyError:
        raise KeyError(
            f"unknown filter preset {name!r}; have {sorted(catalog.named_filters)}"
        ) from None


def load_filter(name: str, *, catalog: MaterialCatalog | None = None) -> FilterPlate:
    """Load a named, validated :class:`~pyrite.geometry.FilterPlate` value.

    Placements can override any field by attaching ``ref = NAME`` in an
    ordered profile filter entry. This loader returns the unplaced definition's
    declared pose, without assigning the catalog name to the plate's name.

    Raises
    ------
    KeyError
        If the preset does not exist.
    ValueError
        If the preset is a sequence; use :func:`load_filters`.
    """
    definition = _definition(name, catalog)
    if definition.get("kind") == "sequence":
        raise ValueError(
            f"filter preset {name!r} is a sequence; load it with pyrite.geometry.load_filters"
        )
    from .observation import filter_from_config

    return filter_from_config(definition)


def load_filters(name: str, *, catalog: MaterialCatalog | None = None) -> tuple[FilterPlate, ...]:
    """Load a named plate or sequence as an ordered tuple of ``FilterPlate`` values.

    A single plate yields a one-element tuple. Members keep their declared
    order and absolute poses; lengths are millimetres and angles degrees.
    """
    from .observation import filter_from_config

    definition = _definition(name, catalog)
    if definition.get("kind") == "sequence":
        plates = cast("tuple[Mapping[str, object], ...]", definition["plates"])
        return tuple(filter_from_config(plate) for plate in plates)
    return (filter_from_config(definition),)
