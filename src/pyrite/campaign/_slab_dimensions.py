"""Slab transverse dimension input and replacement helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Self, cast

from .._api_deprecations import warn_api
from .._spec_defaults import UNSET as _UNSET
from .._spec_defaults import _Unset
from .._units import _positive_length_quantity as _geometry_length
from ._geometry_values import _quantity_geometry

if TYPE_CHECKING:
    from pint import Quantity

    from .geometry import Footprint


def slab_footprint(
    footprint: Footprint | None | _Unset,
    width: object,
    height: object,
    infinite_via_dimensions: bool,
) -> tuple[Footprint | None, bool]:
    """Resolve explicit dimension pairs and the compatibility footprint form."""
    if width is not _UNSET or height is not _UNSET:
        if footprint is not _UNSET:
            raise TypeError("Slab footprint and width/height are mutually exclusive")
        if width is _UNSET or height is _UNSET:
            raise TypeError("Slab requires both width and height")
        if (width is None) != (height is None):
            raise ValueError("Slab width and height must both be None or both lengths")
        if width is None:
            return None, True
        from .geometry import Footprint

        return Footprint(
            _geometry_length(width, unit="mm", field="Slab.width"),
            _geometry_length(height, unit="mm", field="Slab.height"),
        ), False
    if footprint is _UNSET:
        raise TypeError("Slab requires width and height or footprint")
    if footprint is None and not infinite_via_dimensions:
        warn_api("Slab(footprint=None)", stacklevel=4)
    return footprint, infinite_via_dimensions and footprint is None


class SlabDimensions:
    """Quantity-valued transverse dimensions for slab geometry."""

    if TYPE_CHECKING:

        @property
        def footprint(self) -> Footprint | None: ...

    @property
    def width(self) -> Quantity | None:
        """Full transverse width in mm, or None for an infinite slab."""
        return None if self.footprint is None else _quantity_geometry(self.footprint.width_mm, "mm")

    @property
    def height(self) -> Quantity | None:
        """Full transverse height in mm, or None for an infinite slab."""
        return (
            None if self.footprint is None else _quantity_geometry(self.footprint.height_mm, "mm")
        )

    def with_width(self, width: object) -> Self:
        """Copy a finite slab with a new unit-bearing width."""
        if self.footprint is None:
            raise ValueError("Set both dimensions when changing an infinite slab")
        return cast(Self, replace(cast(Any, self), footprint=self.footprint.with_width(width)))

    def with_height(self, height: object) -> Self:
        """Copy a finite slab with a new unit-bearing height."""
        if self.footprint is None:
            raise ValueError("Set both dimensions when changing an infinite slab")
        return cast(Self, replace(cast(Any, self), footprint=self.footprint.with_height(height)))
