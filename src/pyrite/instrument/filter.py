"""Downstream filter role adapter for a shared rectangular body."""

from dataclasses import dataclass
from typing import Literal

from .._planar_geometry import PlanarPose
from ..bodies import RectangularBody, SubstanceRef
from ..materials._composition import has_positive_number_densities, resolve_composition
from ..materials._schema import MaterialCatalog, MediumSpec
from . import model
from .model import FilterPlate


@dataclass(frozen=True)
class Filter:
    """Place a homogeneous finite body in the primary-attenuation filter role.

    ``pose`` is the box centre and local axes in lab mm; it is independent of
    substance and crystal orientation. ``name`` is optional display metadata.
    Only ``interaction='primary-attenuation-only'`` is supported. Electron
    transport, fluorescence, diffraction and secondary production are rejected
    when requested through this adapter. Scene validation still checks the
    full downstream volume and overlaps; it does not trace scattered electrons.
    """

    body: RectangularBody
    pose: PlanarPose
    name: str | None = None
    interaction: Literal["primary-attenuation-only"] = "primary-attenuation-only"

    def __post_init__(self) -> None:
        if not isinstance(self.body, RectangularBody):
            raise TypeError("Filter.body must be a RectangularBody")
        if not isinstance(self.pose, PlanarPose):
            raise TypeError("Filter.pose must be a PlanarPose")
        if self.name is not None and (not isinstance(self.name, str) or not self.name.strip()):
            raise ValueError("Filter.name must be a non-empty string or None")
        if self.interaction != "primary-attenuation-only":
            raise ValueError("Filter supports only interaction='primary-attenuation-only'")

    def to_plate(self, *, catalog: MaterialCatalog | None = None) -> FilterPlate:
        """Lower to a legacy plate, resolving against the selected catalog.

        An explicit ``catalog`` is used without a bundled fallback. Preserve a
        string material reference when legacy lookup selects the same substance
        in the plate's catalog. Otherwise use a resolved inline medium so an
        explicit namespace or external key cannot be reinterpreted downstream.
        Existing observation and trajectory encoders receive legacy values.
        """
        selected = model.CATALOG if catalog is None else catalog
        substance = self.body.substance
        material: str | MediumSpec
        if isinstance(substance, SubstanceRef):
            composition = substance.composition(catalog=selected)
            legacy_kind = "medium" if substance.key in selected.media else "crystal"
            if selected is model.CATALOG and substance.kind == legacy_kind:
                material = substance.key
            else:
                material = MediumSpec(substance.key, composition)
        else:
            material = substance
        # Retain the attenuation boundary's validation even when a catalog was
        # supplied explicitly and the plate uses an inline resolved medium.
        composition = resolve_composition(material, catalog=selected)
        if not has_positive_number_densities(composition):
            raise ValueError("Filter substance must contain positive number densities")
        return FilterPlate(
            material, self.body.thickness_mm, self.body.size_mm, self.pose, self.name
        )


__all__ = ["Filter"]
