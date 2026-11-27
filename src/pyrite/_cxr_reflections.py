"""CXR reflection-domain truncation, mosaic and fiber-roll numerics.

Split from ``_numerics`` (which re-exports both names) so the reflection
controls own one module; ``pyrite._numerics.CxrReflections`` stays valid.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from ._fiber_average import FiberAverage

MosaicRoute = Literal["analytic", "mc", "mc-sampled"]


def fiber_average(value: object) -> FiberAverage | None:
    """A :class:`FiberAverage` from itself, a profile/preset table, or ``None``."""
    if value is None or isinstance(value, FiberAverage):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("cxr_fiber must be a table of FiberAverage fields")
    return FiberAverage(**dict(value))


def reflection_controls_from_values(values: Mapping[str, object]) -> "CxrReflections":
    """``CxrReflections`` from flat profile ``CONVERGENCE_KEYS`` values.

    The profile spelling of ``fiber`` is the ``cxr_fiber`` table (#1483).
    """
    kwargs: dict = {key: value for key, value in values.items() if key != "cxr_fiber"}
    return CxrReflections(**kwargs, fiber=fiber_average(values.get("cxr_fiber")))


@dataclass(frozen=True)
class CxrReflections:
    """Control CXR reflection-family truncation and mosaic quadrature.

    Parameters
    ----------
    n_families
        Explicit number of dominant reflection families. ``None`` (default)
        selects by measured line yield at tolerance 1e-3 on supported routes;
        unsupported routes keep four families. An explicit count overrides
        the automatic tolerance.
    max_reflections
        Optional cap on individual reflections after family expansion.
        Supplying a cap disables automatic yield selection.
    mosaic_nodes
        Number of quadrature nodes per tilt axis used for mosaic averaging
        (``nodes**2`` orientations); under ``"mc-sampled"``, the number of
        crystallite tilts drawn per electron history.
    mosaic_route
        Mosaic integration strategy: ``"analytic"`` detector broadening,
        ``"mc"`` Gauss-Hermite orientation quadrature, or ``"mc-sampled"``
        per-history sampled crystallites (incoherent emission only).
    line_yield_tolerance
        Explicit measured reflection selection (#1173). The spectrum phase
        scores incoherent lines in reciprocal shells ``|g|`` of 2 1/Angstrom up
        to the form-factor range (75.4 1/Angstrom) and stops where the
        estimated line yield beyond the kept shells is at most this fraction
        of the kept yield on the line axis. ``None`` uses the automatic
        policy unless a family count or reflection cap is given. An explicit
        tolerance refuses unsupported routes. Incoherent emission only;
        diffracted-brem and Kossel terms keep the ``n_families`` set.
    fiber
        Experimental (#1480): a :class:`FiberAverage` (reciprocal cutoff, roll
        nodes, batch size) for the full non-basal domain of a fiber-textured
        crystal. ``None`` (default) keeps the catalog's basal-only ``(00l)``
        domain. Campaign runs (#1483) score it for HOPG with incoherent
        emission, a single slab and BremsLib, on an explicit
        (``EnergyBins.cxr``) or automatic (#1489) line grid; other routes
        refuse it. They lower ``mosaic_route`` and
        ``mosaic_nodes`` into the case as basal runs do. A mosaic target needs
        ``mosaic_route="mc"``: ``"analytic"`` (its width reads the basal
        geometry) and ``"mc-sampled"`` are refused. Excludes ``n_families``,
        ``max_reflections`` and ``line_yield_tolerance``. The direct
        ``pyrite.montecarlo.spectrum.cxr.fiber.FiberCXRModel.from_numerics``
        reads this field only: the other controls here, mosaic ones included,
        do not reach that scorer.
    """

    n_families: int | None = None
    max_reflections: int | None = None
    mosaic_nodes: int = 5
    mosaic_route: MosaicRoute = "analytic"
    line_yield_tolerance: float | None = None
    fiber: FiberAverage | None = None

    def __post_init__(self) -> None:
        if self.fiber is not None and not isinstance(self.fiber, FiberAverage):
            raise TypeError("fiber must be a FiberAverage or None")
        if self.fiber is not None and (
            self.n_families is not None
            or self.max_reflections is not None
            or self.line_yield_tolerance is not None
        ):
            raise ValueError(
                "fiber owns the reflection domain; n_families, max_reflections and "
                "line_yield_tolerance must be None"
            )
        if self.n_families is not None and (
            type(self.n_families) is not int or self.n_families <= 0
        ):
            raise ValueError("n_families must be a positive integer or None")
        if self.max_reflections is not None and (
            type(self.max_reflections) is not int or self.max_reflections <= 0
        ):
            raise ValueError("max_reflections must be a positive integer or None")
        if type(self.mosaic_nodes) is not int or self.mosaic_nodes <= 0:
            raise ValueError("mosaic_nodes must be a positive integer")
        if self.mosaic_route not in {"analytic", "mc", "mc-sampled"}:
            raise ValueError("mosaic_route must be 'analytic', 'mc' or 'mc-sampled'")
        if self.line_yield_tolerance is not None and not (
            isinstance(self.line_yield_tolerance, int | float)
            and not isinstance(self.line_yield_tolerance, bool)
            and 0.0 < self.line_yield_tolerance < 1.0
        ):
            raise ValueError("line_yield_tolerance must lie in (0, 1) or be None")
