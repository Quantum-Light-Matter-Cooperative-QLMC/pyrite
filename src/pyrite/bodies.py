"""Scalar homogeneous body descriptions shared by source and filter roles."""

from dataclasses import dataclass, replace
from numbers import Real
from typing import TYPE_CHECKING, Literal, Self

from ._planar_geometry import _positive_pair, _positive_real
from ._spec_defaults import UNSET as _UNSET
from ._spec_defaults import _Unset
from ._units import _positive_length_quantity, normalize_unit
from .materials._composition import has_positive_number_densities, resolve_composition
from .materials._schema import MaterialCatalog, MediumSpec

if TYPE_CHECKING:
    from pint import Quantity


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


@dataclass(frozen=True, init=False)
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

    size, thickness
        Keyword-only Pint lengths, mutually exclusive with the corresponding
        numeric spelling. ``size`` has exactly two components; ``thickness`` is
        scalar. Read-back returns detached quantities in mm. ``with_size`` and
        ``with_thickness`` replace dimensions without changing the substance.
        ``dataclasses.replace`` uses numeric fields; semantic aliases conflict.

    All dimensions are scalar. Use ``Sweep.axes`` to vary a scene's body;
    sequence-valued thicknesses, layers and infinite extent are unsupported.
    """

    substance: SubstanceRef | MediumSpec
    size_mm: tuple[float, float]
    thickness_mm: float

    def __init__(
        self,
        substance: SubstanceRef | MediumSpec,
        size_mm: tuple[float, float] | _Unset = _UNSET,
        thickness_mm: float | _Unset = _UNSET,
        *,
        size: object = _UNSET,
        thickness: object = _UNSET,
    ) -> None:
        if size is not _UNSET:
            if size_mm is not _UNSET:
                raise TypeError("RectangularBody received both size_mm and size")
            size_values = _positive_length_quantity(
                size, unit="mm", field="RectangularBody.size", shape=(2,)
            )
            size_mm = _positive_pair("RectangularBody.size", size_values)
        elif size_mm is _UNSET:
            raise TypeError("RectangularBody requires size_mm or size")
        if thickness is not _UNSET:
            if thickness_mm is not _UNSET:
                raise TypeError("RectangularBody received both thickness_mm and thickness")
            thickness_mm = float(
                _positive_length_quantity(
                    thickness, unit="mm", field="RectangularBody.thickness", shape=()
                )
            )
        elif thickness_mm is _UNSET:
            raise TypeError("RectangularBody requires thickness_mm or thickness")
        object.__setattr__(self, "substance", substance)
        object.__setattr__(self, "size_mm", size_mm)
        object.__setattr__(self, "thickness_mm", thickness_mm)
        self.__post_init__()

    @property
    def size(self) -> "Quantity":
        """Full local x/y dimensions as a detached two-component mm quantity."""
        from .units import Q_

        return Q_(list(self.size_mm), "mm")

    @property
    def thickness(self) -> "Quantity":
        """Full normal thickness as a fresh scalar quantity in mm."""
        from .units import Q_

        return Q_(self.thickness_mm, "mm")

    def with_size(self, size: object) -> Self:
        """Replace full x/y dimensions using an explicit two-component length."""
        return replace(
            self,
            size_mm=_positive_length_quantity(
                size, unit="mm", field="RectangularBody.size", shape=(2,)
            ),
        )

    def with_thickness(self, thickness: object) -> Self:
        """Replace the normal thickness using an explicit scalar length."""
        return replace(
            self,
            thickness_mm=_positive_length_quantity(
                thickness, unit="mm", field="RectangularBody.thickness", shape=()
            ),
        )

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
