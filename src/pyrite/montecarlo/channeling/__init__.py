"""Open-system depth evolution of planar-channeled electrons (ADR-0018, P2).

Incoherent scattering (thermal diffuse and electronic) acts on the transverse
state as a Lindblad dissipator. :mod:`.operators` builds its position-space
kernel ``C(x, x')`` and the decoherence diagnostic; :mod:`.density`
integrates the density matrix with a dechanneling sink, and :mod:`.mcwf`
unravels the same equation into per-electron quantum jumps. :mod:`.emission`
scores channeling radiation on ``rho(z)`` as a diagnostic spectrum (P3) from
the photon transition primitives of :mod:`.dipole`; :mod:`.transition_radiation`
bounds the excluded transition-radiation background. :mod:`.states` builds
user-definable entry wavefunctions (Gaussian, Hermite/Laguerre-Gauss vortex,
Bessel, Airy, aberrated aperture probes, sampled arrays, abTEM waves) and
projects them onto the planar Bloch basis. :mod:`.realspace` is the
experimental axial tier's periodic split-step backend. CPU only.
"""

from . import realspace, states
from .axial_energy_mcwf import (
    AxialEnergyMCWFCheckpoint,
    AxialEnergyMCWFOptions,
    AxialEnergyMCWFResult,
    axial_energy_mcwf,
    sample_electron_energies,
)
from .density import (
    DECHANNELING_MARGIN_EV,
    DECHANNELING_MARGIN_PROVENANCE,
    DensityMatrixResult,
    DensityMatrixSolver,
)
from .dipole import (
    FirstOrderTransition,
    boosted_dipole_pattern,
    dipole_far_field,
    dipole_rate_per_ang,
    dipole_stokes,
    first_order_transition,
    rest_frame_dipole_rate_inv_s,
    transition_operator,
)
from .emission import (
    KIND_CAPTURE,
    KIND_LINE,
    CaptureBound,
    ChannelingSpectrum,
    EmissionLines,
    LineGroup,
    capture_bound,
    channeling_spectrum,
    emission_lines,
    line_shape,
)
from .energy_ensemble import (
    AxialEnergyDiagnostic,
    AxialEnergySpectrum,
    EnergyQuadrature,
    IncidentEnergyEnsemble,
    PlanarEnergyDiagnostic,
    PlanarEnergySpectrum,
    axial_energy_diagnostic,
    axial_energy_spectrum,
    check_quadrature_convergence,
    planar_energy_diagnostic,
    planar_energy_spectrum,
    planar_solver_factory,
)
from .handoff import CHANNELING_MODEL
from .mcwf import MCWFResult, MCWFSolver, mcwf_stream_key
from .operators import (
    DecoherenceRate,
    ElectronicChannels,
    ScatteringKernel,
    decoherence_rate,
    electronic_channels,
    electronic_kernel,
    tds_kernel,
)
from .spatial import spatial_density
from .transition_radiation import (
    TransitionRadiationBound,
    transition_radiation,
    transition_radiation_bound,
    transition_radiation_per_eV,
)

__all__ = [
    "realspace",
    "states",
    "CHANNELING_MODEL",
    "DECHANNELING_MARGIN_EV",
    "DECHANNELING_MARGIN_PROVENANCE",
    "KIND_CAPTURE",
    "KIND_LINE",
    "AxialEnergyMCWFCheckpoint",
    "AxialEnergyMCWFOptions",
    "AxialEnergyMCWFResult",
    "AxialEnergyDiagnostic",
    "AxialEnergySpectrum",
    "CaptureBound",
    "ChannelingSpectrum",
    "DecoherenceRate",
    "DensityMatrixResult",
    "DensityMatrixSolver",
    "ElectronicChannels",
    "EmissionLines",
    "EnergyQuadrature",
    "IncidentEnergyEnsemble",
    "FirstOrderTransition",
    "LineGroup",
    "MCWFResult",
    "PlanarEnergyDiagnostic",
    "PlanarEnergySpectrum",
    "MCWFSolver",
    "ScatteringKernel",
    "TransitionRadiationBound",
    "axial_energy_mcwf",
    "axial_energy_diagnostic",
    "axial_energy_spectrum",
    "boosted_dipole_pattern",
    "capture_bound",
    "check_quadrature_convergence",
    "channeling_spectrum",
    "decoherence_rate",
    "dipole_rate_per_ang",
    "dipole_far_field",
    "dipole_stokes",
    "electronic_channels",
    "electronic_kernel",
    "emission_lines",
    "first_order_transition",
    "line_shape",
    "mcwf_stream_key",
    "planar_energy_diagnostic",
    "planar_energy_spectrum",
    "planar_solver_factory",
    "rest_frame_dipole_rate_inv_s",
    "sample_electron_energies",
    "spatial_density",
    "tds_kernel",
    "transition_operator",
    "transition_radiation",
    "transition_radiation_bound",
    "transition_radiation_per_eV",
]
