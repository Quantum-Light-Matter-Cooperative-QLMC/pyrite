"""Quantum specifications, configuration, runtime states, and advanced diagnostics.

This import-light module is the canonical quantum facade. The whole
surface, including planar channeling, is experimental. Runtime wavefunctions are nonportable. See the quantum API guide.

:class:`Channeling` is the run section of the channeling zone, passed as
``simulate(..., channeling=...)``. Only the planar density-matrix zone and
repository-calibrated axial fibre cases lower to production cases.
Unsupported recipes fail before case construction; FFT-grid escape is never
interpreted as physical dechanneling.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from ._channeling import Channeling, ChannelingNumerics

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
    from pyrite.quantum.planar_source import (
        PlanarStateProjection as PlanarStateProjection,
    )
    from pyrite.quantum.planar_source import (
        project_planar_source as project_planar_source,
    )
    from pyrite.quantum.planar_source import (
        propagate_planar_source as propagate_planar_source,
    )
    from pyrite.quantum.planar_source import (
        score_planar_source as score_planar_source,
    )
    from pyrite.quantum.state import (
        TransverseStateSpec as TransverseStateSpec,
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
    "PlanarStateProjection": ("pyrite.quantum.planar_source", "PlanarStateProjection"),
    "RealSpaceGrid": ("pyrite.montecarlo.channeling.realspace", "RealSpaceGrid"),
    "SampledState": ("pyrite.montecarlo.channeling.states", "SampledState"),
    "ScatteringKernel": ("pyrite.montecarlo.channeling.operators", "ScatteringKernel"),
    "SplitStepPropagator": ("pyrite.montecarlo.channeling.realspace", "SplitStepPropagator"),
    "Superposition": ("pyrite.montecarlo.channeling.states", "Superposition"),
    "Translated": ("pyrite.montecarlo.channeling.states", "Translated"),
    "TransverseState": ("pyrite.montecarlo.channeling.states", "TransverseState"),
    "TransverseStateSpec": ("pyrite.quantum.state", "TransverseStateSpec"),
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
    "project_planar_source": ("pyrite.quantum.planar_source", "project_planar_source"),
    "propagate_planar_source": ("pyrite.quantum.planar_source", "propagate_planar_source"),
    "score_planar_source": ("pyrite.quantum.planar_source", "score_planar_source"),
    "superpose": ("pyrite.montecarlo.channeling.states", "superpose"),
    "tds_kernel": ("pyrite.montecarlo.channeling.operators", "tds_kernel"),
    "translate": ("pyrite.montecarlo.channeling.states", "translate"),
}

__all__ = [
    "AxialDiagnosticConfig",
    "AxialDiagnosticResult",
    "AxialMCWF",
    "AxialMCWFResult",
    "Channeling",
    "ChannelingNumerics",
    "DensityMatrixResult",
    "DensityMatrixSolver",
    "EntryState",
    "MCWFResult",
    "MCWFSolver",
    "MixedState",
    "PlanarStateProjection",
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


#: Classes removed by #1017 without a deprecation window (experimental surface).
_REMOVED = {
    "QuantumSetup": "Channeling",
    "QuantumRegion": "Channeling(model=..., plane=...)",
    "QuantumDynamics": "Channeling",
    "QuantumNumerics": "ChannelingNumerics",
}


def removed_quantum_name(name: str) -> AttributeError | None:
    """The error naming the replacement of a removed quantum setup class."""
    replacement = _REMOVED.get(name)
    if replacement is None:
        return None
    return AttributeError(
        f"pyrite.quantum.{name} was removed; use pyrite.quantum.{replacement} and pass it "
        "as simulate(..., channeling=...)"
    )


def __getattr__(name: str) -> Any:
    """Resolve an advanced contract only when requested, preserving type identity."""
    removed = removed_quantum_name(name)
    if removed is not None:
        raise removed
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
