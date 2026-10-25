"""Scalar homogeneous body descriptions shared by source and filter roles."""

from dataclasses import dataclass
from numbers import Real
from typing import Literal

from ._planar_geometry import _positive_pair, _positive_real
from ._units import normalize_unit
from .materials._composition import has_positive_number_densities, resolve_composition
from .materials._schema import MaterialCatalog, MediumSpec


@dataclass(frozen=True)
class SubstanceRef:
    """Reference one homogeneous crystal or medium in a selected catalog.

    ``kind`` selects the namespace explicitly; ``key`` is case-sensitive.
    Runnable material recipes are not substances. Crystal references retain
    their identity for target orientation; filters consume only composition.
    """

    kind: Literal["crystal", "medium"]
    key: str

    def __post_init__(self) -> None:
        if self.kind not in {"crystal", "medium"}:
            raise ValueError("SubstanceRef.kind must be 'crystal' or 'medium'")
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("SubstanceRef.key must be a non-empty catalog key")

    def composition(self, *, catalog: MaterialCatalog) -> tuple[tuple[str, float], ...]:
        """Resolve ordered number densities [atoms/Å³] from ``catalog``.

        Unknown references raise ``ValueError`` without consulting another
        namespace or the bundled catalog.
        """
        try:
            return resolve_composition(self.key, catalog=catalog, namespaces=(self.kind,))
        except KeyError:
            raise ValueError(f"unknown {self.kind} substance {self.key!r}") from None


@dataclass(frozen=True)
class RectangularBody:
    """Immutable homogeneous rectangular solid without a scene role or pose.

    Parameters
    ----------
    substance
        Explicit crystal/medium reference or inline number-density medium.
    size_mm
        Positive finite full width and height along the body's local x/y axes.
    thickness_mm
        Positive finite full thickness along its local normal, in mm.

    All dimensions are scalar. Use ``Sweep.axes`` to vary a scene's body;
    sequence-valued thicknesses, layers and infinite extent are unsupported.
    """

    substance: SubstanceRef | MediumSpec
    size_mm: tuple[float, float]
    thickness_mm: float

    def __post_init__(self) -> None:
        if not isinstance(self.substance, (SubstanceRef, MediumSpec)):
            raise TypeError("RectangularBody.substance must be a SubstanceRef or MediumSpec")
        if isinstance(self.substance, MediumSpec) and any(
            not isinstance(element, str) or not isinstance(density, Real)
            for element, density in self.substance.composition
        ):
            raise TypeError(
                "RectangularBody medium must contain element names and scalar densities"
            )
        if isinstance(self.substance, MediumSpec) and not has_positive_number_densities(
            self.substance.composition
        ):
            raise ValueError("RectangularBody medium must contain positive number densities")
        if isinstance(self.substance, MediumSpec):
            object.__setattr__(
                self,
                "substance",
                MediumSpec(
                    self.substance.key,
                    tuple((element, density) for element, density in self.substance.composition),
                ),
            )
        object.__setattr__(
            self,
            "size_mm",
            _positive_pair(
                "RectangularBody.size_mm",
                normalize_unit(
                    self.size_mm,
                    "mm",
                    field="RectangularBody.size_mm",
                    expected="length",
                ),
            ),
        )
        object.__setattr__(
            self,
            "thickness_mm",
            _positive_real(
                "RectangularBody.thickness_mm",
                normalize_unit(
                    self.thickness_mm,
                    "mm",
                    field="RectangularBody.thickness_mm",
                    expected="length",
                ),
            ),
        )


__all__ = ["RectangularBody", "SubstanceRef"]
