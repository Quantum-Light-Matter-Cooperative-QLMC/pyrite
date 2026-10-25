"""Source-target role adapter for a shared homogeneous rectangular body."""

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

from .._units import normalize_unit
from ..bodies import RectangularBody, SubstanceRef
from . import geometry
from .geometry import CrystalCut, Footprint, Slab


@dataclass(frozen=True)
class Target:
    """Assign a finite homogeneous crystalline body to the source-target role.

    ``tilt_deg`` and ``tilt_azim_deg`` are scalar target angles in degrees;
    ``lattice_cut='catalog'`` retains the selected crystal's existing cut; a
    :class:`~pyrite.CrystalCut` orients the lattice relative to the entrance
    face instead.
    ``mosaic`` and ``allow_normal_incidence`` have the same meaning as on
    :class:`~pyrite.Slab`. The target reference remains the entrance origin.
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
