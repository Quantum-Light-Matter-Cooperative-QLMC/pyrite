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
    from pyrite.materials.channeling.axial import (
        axial_profile as axial_profile,
    )
    from pyrite.materials.channeling.potential import (
        planar_profile as planar_profile,
    )
    from pyrite.montecarlo.channeling.axial_bloch import (
        AxialStates as AxialStates,
    )
    from pyrite.montecarlo.channeling.axial_bloch import (
        axial_states as axial_states,
    )
    from pyrite.montecarlo.channeling.axial_emission import (
        axial_line_catalog as axial_line_catalog,
    )
    from pyrite.montecarlo.channeling.axial_fiber_unraveling import (
        unravel_fiber_outgoing as unravel_fiber_outgoing,
    )
    from pyrite.montecarlo.channeling.axial_mcwf import (
        AxialMCWF as AxialMCWF,
    )
    from pyrite.montecarlo.channeling.axial_mcwf import (
        AxialMCWFResult as AxialMCWFResult,
    )
    from pyrite.montecarlo.channeling.axial_propagation import (
        AxialPropagationConfig as AxialPropagationConfig,
    )
    from pyrite.montecarlo.channeling.axial_propagation import (
        AxialPropagationResult as AxialPropagationResult,
    )
    from pyrite.montecarlo.channeling.axial_propagation import (
        propagate_axial as propagate_axial,
    )
    from pyrite.montecarlo.channeling.bloch import (
        entry_populations as entry_populations,
    )
    from pyrite.montecarlo.channeling.bloch import (
        line_energy_eV as line_energy_eV,
    )
    from pyrite.montecarlo.channeling.bloch import (
        longitudinal_kinematics as longitudinal_kinematics,
    )
    from pyrite.montecarlo.channeling.bloch import (
        transverse_states as transverse_states,
    )
    from pyrite.montecarlo.channeling.density import (
        DensityMatrixResult as DensityMatrixResult,
    )
    from pyrite.montecarlo.channeling.density import (
        DensityMatrixSolver as DensityMatrixSolver,
    )
    from pyrite.montecarlo.channeling.emission import (
        channeling_spectrum as channeling_spectrum,
    )
    from pyrite.montecarlo.channeling.exit_angular import (
        classical_exit_angles as classical_exit_angles,
    )
    from pyrite.montecarlo.channeling.exit_angular import (
        coherent_exit_distribution as coherent_exit_distribution,
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
    from pyrite.montecarlo.channeling.planar_fiber import (
        planar_fiber_inputs as planar_fiber_inputs,
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

    #: Electron rest energy ``m_e c^2`` [eV], a plain float (no Quantity).
    M_E_EV: float


_EXPORTS = {
    "AxialPropagationConfig": (
        "pyrite.montecarlo.channeling.axial_propagation",
        "AxialPropagationConfig",
    ),
    "AxialPropagationResult": (
        "pyrite.montecarlo.channeling.axial_propagation",
        "AxialPropagationResult",
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
    "propagate_axial": ("pyrite.montecarlo.channeling.axial_propagation", "propagate_axial"),
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
    "AxialStates": ("pyrite.montecarlo.channeling.axial_bloch", "AxialStates"),
    "M_E_EV": ("pyrite.materials.crystal", "M_E_EV"),
    "axial_line_catalog": ("pyrite.montecarlo.channeling.axial_emission", "axial_line_catalog"),
    "axial_profile": ("pyrite.materials.channeling.axial", "axial_profile"),
    "axial_states": ("pyrite.montecarlo.channeling.axial_bloch", "axial_states"),
    "channeling_spectrum": ("pyrite.montecarlo.channeling.emission", "channeling_spectrum"),
    "classical_exit_angles": ("pyrite.montecarlo.channeling.exit_angular", "classical_exit_angles"),
    "coherent_exit_distribution": (
        "pyrite.montecarlo.channeling.exit_angular",
        "coherent_exit_distribution",
    ),
    "entry_populations": ("pyrite.montecarlo.channeling.bloch", "entry_populations"),
    "line_energy_eV": ("pyrite.montecarlo.channeling.bloch", "line_energy_eV"),
    "longitudinal_kinematics": ("pyrite.montecarlo.channeling.bloch", "longitudinal_kinematics"),
    "planar_fiber_inputs": ("pyrite.montecarlo.channeling.planar_fiber", "planar_fiber_inputs"),
    "planar_profile": ("pyrite.materials.channeling.potential", "planar_profile"),
    "transverse_states": ("pyrite.montecarlo.channeling.bloch", "transverse_states"),
    "unravel_fiber_outgoing": (
        "pyrite.montecarlo.channeling.axial_fiber_unraveling",
        "unravel_fiber_outgoing",
    ),
}

__all__ = [
    "AxialMCWF",
    "AxialMCWFResult",
    "AxialPropagationConfig",
    "AxialPropagationResult",
    "AxialStates",
    "Channeling",
    "ChannelingNumerics",
    "DensityMatrixResult",
    "DensityMatrixSolver",
    "EntryState",
    "MCWFResult",
    "MCWFSolver",
    "M_E_EV",
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
    "axial_line_catalog",
    "axial_profile",
    "axial_states",
    "bessel_gauss",
    "channeling_spectrum",
    "classical_exit_angles",
    "coherent_exit_distribution",
    "donut",
    "electronic_kernel",
    "entry_populations",
    "from_abtem",
    "from_array",
    "gaussian",
    "hermite_gauss",
    "laguerre_gauss",
    "line_energy_eV",
    "longitudinal_kinematics",
    "mixture",
    "planar_fiber_inputs",
    "planar_profile",
    "project_planar_source",
    "propagate_axial",
    "propagate_planar_source",
    "score_planar_source",
    "superpose",
    "tds_kernel",
    "translate",
    "transverse_states",
    "unravel_fiber_outgoing",
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
