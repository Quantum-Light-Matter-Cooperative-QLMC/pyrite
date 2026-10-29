"""Source-target role adapter for a shared homogeneous rectangular body."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Real
from typing import TYPE_CHECKING, Literal, Self

from .._spec_defaults import UNSET as _UNSET
from .._spec_defaults import _Unset
from .._units import normalize_unit
from ..bodies import RectangularBody, SubstanceRef
from . import geometry
from ._geometry_values import _angle_spelling, _quantity_geometry, _scalar_angle_quantity
from .geometry import CrystalCut, Footprint, Slab

if TYPE_CHECKING:
    from pint import Quantity


@dataclass(frozen=True, init=False)
class Target:
    """Assign a finite homogeneous crystalline body to the source-target role.

    ``tilt_deg`` and ``tilt_azim_deg`` are scalar target angles in degrees;
    keyword-only ``tilt``/``tilt_azimuth`` take explicit scalar Pint plane
    angles instead, read back as fresh degree quantities and replaced through
    ``with_tilt``/``with_tilt_azimuth``;
    ``lattice_cut='catalog'`` retains the selected crystal's existing cut; a
    :class:`~pyrite.geometry.CrystalCut` orients the lattice relative to the entrance
    face instead.
    ``mosaic`` and ``allow_normal_incidence`` have the same meaning as on
    :class:`~pyrite.geometry.Slab`. The target reference remains the entrance origin.
    Arbitrary lab poses, amorphous entrances, layers and grooves are unsupported
    by this adapter; existing ``Slab``/``Stack`` constructors remain available.
    Interaction models and sampling belong to ``Scene.physics``/``numerics``.
    """

    body: RectangularBody
    tilt_deg: float = 30.0
    tilt_azim_deg: float = 0.0
    lattice_cut: Literal["catalog"] | CrystalCut = "catalog"
    mosaic: bool = False
    allow_normal_incidence: bool = False

    def __init__(
        self,
        body: RectangularBody,
        tilt_deg: float | _Unset = _UNSET,
        tilt_azim_deg: float | _Unset = _UNSET,
        lattice_cut: Literal["catalog"] | CrystalCut = "catalog",
        mosaic: bool = False,
        allow_normal_incidence: bool = False,
        *,
        tilt: object = _UNSET,
        tilt_azimuth: object = _UNSET,
    ) -> None:
        angles = {
            "tilt_deg": ("tilt", tilt_deg, tilt, 30.0),
            "tilt_azim_deg": ("tilt_azimuth", tilt_azim_deg, tilt_azimuth, 0.0),
        }
        for numeric_name, (semantic_name, numeric, semantic, default) in angles.items():
            value = _angle_spelling(
                "Target",
                numeric_name,
                numeric,
                semantic_name,
                semantic,
                unset=_UNSET,
                default=default,
                scalar=True,
            )
            object.__setattr__(self, numeric_name, value)
        for name, value in (
            ("body", body),
            ("lattice_cut", lattice_cut),
            ("mosaic", mosaic),
            ("allow_normal_incidence", allow_normal_incidence),
        ):
            object.__setattr__(self, name, value)
        self.__post_init__()

    @property
    def tilt(self) -> Quantity:
        """Fresh scalar degree quantity for the polar target tilt."""
        return _quantity_geometry(self.tilt_deg, "degree")

    def with_tilt(self, tilt: object) -> Self:
        """Copy with an explicit scalar plane-angle tilt; preserve other settings."""
        return replace(self, tilt_deg=_scalar_angle_quantity(tilt, field="Target.tilt"))

    @property
    def tilt_azimuth(self) -> Quantity:
        """Fresh scalar degree quantity for the target tilt azimuth."""
        return _quantity_geometry(self.tilt_azim_deg, "degree")

    def with_tilt_azimuth(self, tilt_azimuth: object) -> Self:
        """Copy with an explicit scalar plane-angle tilt azimuth; preserve other settings."""
        degrees = _scalar_angle_quantity(tilt_azimuth, field="Target.tilt_azimuth")
        return replace(self, tilt_azim_deg=degrees)

    def __post_init__(self) -> None:
        if not isinstance(self.body, RectangularBody):
            raise TypeError("Target.body must be a RectangularBody")
        if (
            not isinstance(self.body.substance, SubstanceRef)
            or self.body.substance.kind != "crystal"
        ):
            raise ValueError("Target requires an explicit crystalline SubstanceRef")
        if not isinstance(self.lattice_cut, CrystalCut) and self.lattice_cut != "catalog":
            raise ValueError("Target.lattice_cut must be 'catalog' or a CrystalCut")
        for name in ("tilt_deg", "tilt_azim_deg"):
            value = normalize_unit(
                getattr(self, name),
                "degree",
                field=f"Target.{name}",
                expected="angle",
                kind="angle",
            )
            object.__setattr__(self, name, value)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Target.{name} must be a scalar real angle")
            if not math.isfinite(float(value)):
                raise ValueError(f"Target.{name} must be finite")
            object.__setattr__(self, name, float(value))
        self.to_slab()

    def to_slab(self) -> Slab:
        """Lower to the existing finite slab using the selected target catalog.

        Thickness converts explicitly from mm to Å without rounding. A recipe
        key that redirects to another crystal is rejected rather than silently
        changing the explicit substance reference. No lab translation or new
        transport policy is introduced.
        """
        substance = self.body.substance
        assert isinstance(substance, SubstanceRef)
        substance.composition(catalog=geometry.CATALOG)
        recipe = geometry.CATALOG.materials.get(substance.key)
        if recipe is not None and recipe.crystal_key != substance.key:
            raise ValueError(
                f"target crystal {substance.key!r} conflicts with a runnable recipe's crystal; "
                "explicit substance lowering cannot redirect to another crystal"
            )
        return Slab(
            material=substance.key,
            thickness_ang=self.body.thickness_mm * 10_000_000.0,
            tilt_deg=self.tilt_deg,
            tilt_azim_deg=self.tilt_azim_deg,
            footprint=Footprint(*self.body.size_mm),
            mosaic=self.mosaic,
            allow_normal_incidence=self.allow_normal_incidence,
            cut=self.lattice_cut if isinstance(self.lattice_cut, CrystalCut) else None,
        )


__all__ = ["Target"]
