"""Experimental channeling solvers and diagnostics; not production admission.

Importable, documented, provisional contract; existing domain imports remain valid.
Resolving an attribute imports the original implementation without wrapping it.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrite.montecarlo.channeling.axial_diagnostic import (
        AxialDiagnosticConfig as AxialDiagnosticConfig,
    )
    from pyrite.montecarlo.channeling.axial_diagnostic import (
        AxialDiagnosticResult as AxialDiagnosticResult,
    )
    from pyrite.montecarlo.channeling.axial_diagnostic import axial_diagnostic as axial_diagnostic
    from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF as AxialMCWF
    from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWFResult as AxialMCWFResult
    from pyrite.montecarlo.channeling.density import DensityMatrixResult as DensityMatrixResult
    from pyrite.montecarlo.channeling.density import DensityMatrixSolver as DensityMatrixSolver
    from pyrite.montecarlo.channeling.mcwf import MCWFResult as MCWFResult
    from pyrite.montecarlo.channeling.mcwf import MCWFSolver as MCWFSolver
    from pyrite.montecarlo.channeling.realspace import RealSpaceGrid as RealSpaceGrid
    from pyrite.montecarlo.channeling.realspace import SplitStepPropagator as SplitStepPropagator
    from pyrite.planar_source import PlanarStateProjection as PlanarStateProjection
    from pyrite.planar_source import project_planar_source as project_planar_source
    from pyrite.planar_source import propagate_planar_source as propagate_planar_source
    from pyrite.planar_source import score_planar_source as score_planar_source

_EXPORTS = {
    "DensityMatrixSolver": ("pyrite.montecarlo.channeling.density", "DensityMatrixSolver"),
    "DensityMatrixResult": ("pyrite.montecarlo.channeling.density", "DensityMatrixResult"),
    "MCWFSolver": ("pyrite.montecarlo.channeling.mcwf", "MCWFSolver"),
    "MCWFResult": ("pyrite.montecarlo.channeling.mcwf", "MCWFResult"),
    "AxialMCWF": ("pyrite.montecarlo.channeling.axial_mcwf", "AxialMCWF"),
    "AxialMCWFResult": ("pyrite.montecarlo.channeling.axial_mcwf", "AxialMCWFResult"),
    "AxialDiagnosticConfig": (
        "pyrite.montecarlo.channeling.axial_diagnostic",
        "AxialDiagnosticConfig",
    ),
    "AxialDiagnosticResult": (
        "pyrite.montecarlo.channeling.axial_diagnostic",
        "AxialDiagnosticResult",
    ),
    "axial_diagnostic": ("pyrite.montecarlo.channeling.axial_diagnostic", "axial_diagnostic"),
    "PlanarStateProjection": ("pyrite.planar_source", "PlanarStateProjection"),
    "project_planar_source": ("pyrite.planar_source", "project_planar_source"),
    "propagate_planar_source": ("pyrite.planar_source", "propagate_planar_source"),
    "score_planar_source": ("pyrite.planar_source", "score_planar_source"),
    "RealSpaceGrid": ("pyrite.montecarlo.channeling.realspace", "RealSpaceGrid"),
    "SplitStepPropagator": ("pyrite.montecarlo.channeling.realspace", "SplitStepPropagator"),
}

__all__ = [
    "DensityMatrixSolver",
    "DensityMatrixResult",
    "MCWFSolver",
    "MCWFResult",
    "AxialMCWF",
    "AxialMCWFResult",
    "AxialDiagnosticConfig",
    "AxialDiagnosticResult",
    "axial_diagnostic",
    "PlanarStateProjection",
    "project_planar_source",
    "propagate_planar_source",
    "score_planar_source",
    "RealSpaceGrid",
    "SplitStepPropagator",
]


def __getattr__(name: str) -> Any:
    try:
        module, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
