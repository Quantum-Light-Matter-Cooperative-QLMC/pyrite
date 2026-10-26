"""Quantum specifications, configuration, runtime states, and advanced diagnostics.

This import-light module is the canonical quantum facade. Portable specs and
configuration are advanced-stable; solver and projection diagnostics are
experimental. Runtime wavefunctions are nonportable. See the quantum API guide.

Only the established planar density-matrix zone can be lowered to production
cases. Unsupported recipes fail before case construction; FFT-grid escape is
never interpreted as physical dechanneling.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ._numerics import Numerics, Physics, _miller, validate_channeling_numerics
from ._units import normalize_unit

if TYPE_CHECKING:
    from pyrite.montecarlo.channeling.axial_diagnostic import (
        AxialDiagnosticConfig as AxialDiagnosticConfig,
    )
    from pyrite.montecarlo.channeling.axial_diagnostic import (
        AxialDiagnosticResult as AxialDiagnosticResult,
    )
    from pyrite.montecarlo.channeling.axial_diagnostic import (
        axial_diagnostic as axial_diagnostic,
    )
    from pyrite.montecarlo.channeling.axial_mcwf import (
        AxialMCWF as AxialMCWF,
    )
    from pyrite.montecarlo.channeling.axial_mcwf import (
        AxialMCWFResult as AxialMCWFResult,
    )
    from pyrite.montecarlo.channeling.density import (
        DensityMatrixResult as DensityMatrixResult,
    )
    from pyrite.montecarlo.channeling.density import (
        DensityMatrixSolver as DensityMatrixSolver,
    )
    from pyrite.montecarlo.channeling.mcwf import (
        MCWFResult as MCWFResult,
    )
    from pyrite.montecarlo.channeling.mcwf import (
        MCWFSolver as MCWFSolver,
    )
    from pyrite.montecarlo.channeling.operators import (
        ScatteringKernel as ScatteringKernel,
    )
    from pyrite.montecarlo.channeling.operators import (
        electronic_kernel as electronic_kernel,
    )
    from pyrite.montecarlo.channeling.operators import (
        tds_kernel as tds_kernel,
    )
    from pyrite.montecarlo.channeling.realspace import (
        RealSpaceGrid as RealSpaceGrid,
    )
    from pyrite.montecarlo.channeling.realspace import (
        SplitStepPropagator as SplitStepPropagator,
    )
    from pyrite.montecarlo.channeling.states import (
        EntryState as EntryState,
    )
    from pyrite.montecarlo.channeling.states import (
        MixedState as MixedState,
    )
    from pyrite.montecarlo.channeling.states import (
        SampledState as SampledState,
    )
    from pyrite.montecarlo.channeling.states import (
        Superposition as Superposition,
    )
    from pyrite.montecarlo.channeling.states import (
        Translated as Translated,
    )
    from pyrite.montecarlo.channeling.states import (
        TransverseState as TransverseState,
    )
    from pyrite.montecarlo.channeling.states import (
        airy as airy,
    )
    from pyrite.montecarlo.channeling.states import (
        aperture_probe as aperture_probe,
    )
    from pyrite.montecarlo.channeling.states import (
        bessel_gauss as bessel_gauss,
    )
    from pyrite.montecarlo.channeling.states import (
        donut as donut,
    )
    from pyrite.montecarlo.channeling.states import (
        from_abtem as from_abtem,
    )
    from pyrite.montecarlo.channeling.states import (
        from_array as from_array,
    )
    from pyrite.montecarlo.channeling.states import (
        gaussian as gaussian,
    )
    from pyrite.montecarlo.channeling.states import (
        hermite_gauss as hermite_gauss,
    )
    from pyrite.montecarlo.channeling.states import (
        laguerre_gauss as laguerre_gauss,
    )
    from pyrite.montecarlo.channeling.states import (
        mixture as mixture,
    )
    from pyrite.montecarlo.channeling.states import (
        superpose as superpose,
    )
    from pyrite.montecarlo.channeling.states import (
        translate as translate,
    )
    from pyrite.planar_source import (
        PlanarStateProjection as PlanarStateProjection,
    )
    from pyrite.planar_source import (
        project_planar_source as project_planar_source,
    )
    from pyrite.planar_source import (
        propagate_planar_source as propagate_planar_source,
    )
    from pyrite.planar_source import (
        score_planar_source as score_planar_source,
    )
    from pyrite.quantum_state import (
        TransverseStateSpec as TransverseStateSpec,
    )


@dataclass(frozen=True)
class QuantumRegion:
    """Physical region and handoff cap, not an FFT grid or numerical boundary."""

    kind: Literal["free", "planar", "axial"] = "free"
    plane_hkl: tuple[int, int, int] | None = None
    axis_uvw: tuple[int, int, int] | None = None
    zone_max_ang: float | None = None

    def __post_init__(self) -> None:
        if self.zone_max_ang is not None:
            object.__setattr__(
                self,
                "zone_max_ang",
                normalize_unit(
                    self.zone_max_ang,
                    "angstrom",
                    field="QuantumRegion.zone_max_ang",
                    expected="length",
                ),
            )
        if self.kind not in ("free", "planar", "axial"):
            raise ValueError("unknown quantum region kind")
        if self.kind == "planar":
            if self.plane_hkl is None:
                raise ValueError("planar quantum region requires plane_hkl")
            object.__setattr__(self, "plane_hkl", _miller(self.plane_hkl))
            if self.axis_uvw is not None:
                raise ValueError("planar quantum region cannot set axis_uvw")
        elif self.kind == "axial":
            if self.axis_uvw is None:
                raise ValueError("axial quantum region requires axis_uvw")
            object.__setattr__(self, "axis_uvw", _miller(self.axis_uvw))
            if self.plane_hkl is not None:
                raise ValueError("axial quantum region cannot set plane_hkl")
        elif any(v is not None for v in (self.plane_hkl, self.axis_uvw, self.zone_max_ang)):
            raise ValueError("free quantum region cannot set alignment or zone cap")
        if self.zone_max_ang is not None:
            validate_channeling_numerics(16, 16, 250.0, self.zone_max_ang)


@dataclass(frozen=True)
class QuantumDynamics:
    """Physical dynamics recipe, separate from discretization and source."""

    kind: Literal["free", "channeling"] = "free"

    def __post_init__(self) -> None:
        if self.kind not in ("free", "channeling"):
            raise ValueError("unknown quantum dynamics kind")


@dataclass(frozen=True)
class QuantumNumerics:
    """Propagation representation and numerical integration controls."""

    representation: Literal["bloch", "real-space"] = "bloch"
    method: Literal["coherent", "density-matrix", "mcwf"] = "density-matrix"
    basis_size: int = 16
    bloch_cells: int = 16
    step_ang: float = 250.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "step_ang",
            normalize_unit(
                self.step_ang,
                "angstrom",
                field="QuantumNumerics.step_ang",
                expected="length",
            ),
        )
        if self.representation not in ("bloch", "real-space"):
            raise ValueError("unknown quantum representation")
        if self.method not in ("coherent", "density-matrix", "mcwf"):
            raise ValueError("unknown quantum propagation method")
        validate_channeling_numerics(self.basis_size, self.bloch_cells, self.step_ang, None)


@dataclass(frozen=True)
class QuantumSetup:
    """Reusable quantum stage for simulation or legacy Scene construction.

    Pass independently as ``simulate(..., quantum=setup)``. Only planar Bloch
    density-matrix channeling is supported in production; other configurations
    explicitly refuse.
    """

    region: QuantumRegion = field(default_factory=QuantumRegion)
    dynamics: QuantumDynamics = field(default_factory=QuantumDynamics)
    numerics: QuantumNumerics = field(default_factory=QuantumNumerics)

    def __post_init__(self) -> None:
        for name, cls in (
            ("region", QuantumRegion),
            ("dynamics", QuantumDynamics),
            ("numerics", QuantumNumerics),
        ):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"QuantumSetup.{name} must be {cls.__name__}")

    def lower(self, physics: Physics, numerics: Numerics) -> tuple[Physics, Numerics]:
        """Adapt supported planar configuration to unchanged versioned case fields."""
        if self.region.kind == "axial":
            from .campaign.axial_spec import refuse_unverified_axial_run

            refuse_unverified_axial_run()
        if (
            self.region.kind != "planar"
            or self.dynamics.kind != "channeling"
            or self.numerics.representation != "bloch"
            or self.numerics.method != "density-matrix"
        ):
            raise ValueError(
                "quantum production supports only planar Bloch density-matrix "
                "channeling; other configurations are diagnostic-only"
            )
        if physics.channeling_model != "off" or physics.channeling_axial is not None:
            raise ValueError("Scene.quantum conflicts with legacy Physics.channeling_model")
        if numerics.precision is not None:
            raise ValueError("adaptive precision is unsupported by planar quantum channeling")
        values = dict(
            channeling_basis_size=self.numerics.basis_size,
            channeling_bloch_cells=self.numerics.bloch_cells,
            channeling_dz_ang=self.numerics.step_ang,
            channeling_zone_max_ang=self.region.zone_max_ang,
        )
        defaults = Numerics()
        for name, value in values.items():
            current = getattr(numerics, name)
            if current != getattr(defaults, name) and current != value:
                raise ValueError(f"Scene.quantum conflicts with legacy Numerics.{name}")
        return (
            replace(physics, channeling_model="planar", channeling_plane=self.region.plane_hkl),
            replace(numerics, **values),
        )

    def project_state(
        self,
        state: TransverseStateSpec,
        *,
        crystal: str,
        kinetic_energy_eV: float,
        incidence_rad: float = 0.0,
        impact: str = "averaged",
        temperature_K: float = 293.15,
        margin_eV: float | None = None,
        base_dir: str | Path | None = None,
    ) -> PlanarStateProjection:
        """Project a portable state for diagnostic planar density evolution.

        This does not admit production transport or persisted CR emission.
        The incidence is the *effective refracted* angle to the planes.
        Axial configurations use #624's shared physical-admission refusal.
        """
        from .planar_source import project_planar_source

        return project_planar_source(
            state,
            setup=self,
            crystal=crystal,
            kinetic_energy_eV=kinetic_energy_eV,
            incidence_rad=incidence_rad,
            impact=impact,
            temperature_K=temperature_K,
            margin_eV=margin_eV,
            base_dir=base_dir,
        )


_EXPORTS = {
    "AxialDiagnosticConfig": (
        "pyrite.montecarlo.channeling.axial_diagnostic",
        "AxialDiagnosticConfig",
    ),
    "AxialDiagnosticResult": (
        "pyrite.montecarlo.channeling.axial_diagnostic",
        "AxialDiagnosticResult",
    ),
    "AxialMCWF": ("pyrite.montecarlo.channeling.axial_mcwf", "AxialMCWF"),
    "AxialMCWFResult": ("pyrite.montecarlo.channeling.axial_mcwf", "AxialMCWFResult"),
    "DensityMatrixResult": ("pyrite.montecarlo.channeling.density", "DensityMatrixResult"),
    "DensityMatrixSolver": ("pyrite.montecarlo.channeling.density", "DensityMatrixSolver"),
    "EntryState": ("pyrite.montecarlo.channeling.states", "EntryState"),
    "MCWFResult": ("pyrite.montecarlo.channeling.mcwf", "MCWFResult"),
    "MCWFSolver": ("pyrite.montecarlo.channeling.mcwf", "MCWFSolver"),
    "MixedState": ("pyrite.montecarlo.channeling.states", "MixedState"),
    "PlanarStateProjection": ("pyrite.planar_source", "PlanarStateProjection"),
    "RealSpaceGrid": ("pyrite.montecarlo.channeling.realspace", "RealSpaceGrid"),
    "SampledState": ("pyrite.montecarlo.channeling.states", "SampledState"),
    "ScatteringKernel": ("pyrite.montecarlo.channeling.operators", "ScatteringKernel"),
    "SplitStepPropagator": ("pyrite.montecarlo.channeling.realspace", "SplitStepPropagator"),
    "Superposition": ("pyrite.montecarlo.channeling.states", "Superposition"),
    "Translated": ("pyrite.montecarlo.channeling.states", "Translated"),
    "TransverseState": ("pyrite.montecarlo.channeling.states", "TransverseState"),
    "TransverseStateSpec": ("pyrite.quantum_state", "TransverseStateSpec"),
    "airy": ("pyrite.montecarlo.channeling.states", "airy"),
    "aperture_probe": ("pyrite.montecarlo.channeling.states", "aperture_probe"),
    "axial_diagnostic": ("pyrite.montecarlo.channeling.axial_diagnostic", "axial_diagnostic"),
    "bessel_gauss": ("pyrite.montecarlo.channeling.states", "bessel_gauss"),
    "donut": ("pyrite.montecarlo.channeling.states", "donut"),
    "electronic_kernel": ("pyrite.montecarlo.channeling.operators", "electronic_kernel"),
    "from_abtem": ("pyrite.montecarlo.channeling.states", "from_abtem"),
    "from_array": ("pyrite.montecarlo.channeling.states", "from_array"),
    "gaussian": ("pyrite.montecarlo.channeling.states", "gaussian"),
    "hermite_gauss": ("pyrite.montecarlo.channeling.states", "hermite_gauss"),
    "laguerre_gauss": ("pyrite.montecarlo.channeling.states", "laguerre_gauss"),
    "mixture": ("pyrite.montecarlo.channeling.states", "mixture"),
    "project_planar_source": ("pyrite.planar_source", "project_planar_source"),
    "propagate_planar_source": ("pyrite.planar_source", "propagate_planar_source"),
    "score_planar_source": ("pyrite.planar_source", "score_planar_source"),
    "superpose": ("pyrite.montecarlo.channeling.states", "superpose"),
    "tds_kernel": ("pyrite.montecarlo.channeling.operators", "tds_kernel"),
    "translate": ("pyrite.montecarlo.channeling.states", "translate"),
}

__all__ = [
    "AxialDiagnosticConfig",
    "AxialDiagnosticResult",
    "AxialMCWF",
    "AxialMCWFResult",
    "DensityMatrixResult",
    "DensityMatrixSolver",
    "EntryState",
    "MCWFResult",
    "MCWFSolver",
    "MixedState",
    "PlanarStateProjection",
    "QuantumDynamics",
    "QuantumNumerics",
    "QuantumRegion",
    "QuantumSetup",
    "RealSpaceGrid",
    "SampledState",
    "ScatteringKernel",
    "SplitStepPropagator",
    "Superposition",
    "Translated",
    "TransverseState",
    "TransverseStateSpec",
    "airy",
    "aperture_probe",
    "axial_diagnostic",
    "bessel_gauss",
    "donut",
    "electronic_kernel",
    "from_abtem",
    "from_array",
    "gaussian",
    "hermite_gauss",
    "laguerre_gauss",
    "mixture",
    "project_planar_source",
    "propagate_planar_source",
    "score_planar_source",
    "superpose",
    "tds_kernel",
    "translate",
]


def __getattr__(name: str) -> Any:
    """Resolve an advanced contract only when requested, preserving type identity."""
    try:
        module, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the curated API without importing diagnostic backends."""
    return sorted(set(globals()) | set(__all__))
