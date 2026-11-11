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
projects them onto the planar Bloch basis. :mod:`.bloch` solves the planar Bloch
eigenproblem at a beam energy and :mod:`.entry` maps sample tilt onto plane incidence. :mod:`.exit_angular` gives the
angular distribution of electrons leaving the crystal. :mod:`.realspace` is the
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
from .bloch import (
    TransverseStates,
    doppler_denominator,
    entry_populations,
    line_energy_eV,
    longitudinal_kinematics,
    transverse_states,
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
from .energy_time import (
    EnergyNodes,
    IncidentPhaseSpace,
    LinearEnergyChirp,
    TimeSlices,
    quantize_energies,
    sample_incident_phase_space,
    time_slices,
)
from .entry import PlaneIncidence, mean_inner_potential_eV, plane_incidence
from .exit_angular import (
    ExitAngularDistribution,
    classical_exit_angles,
    coherent_exit_distribution,
    density_exit_distribution,
    open_exit_distribution,
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
from .temporal_radiation import (
    AxialPhaseSpaceMCWFResult,
    TemporalSliceDiagnostic,
    TimeResolvedSpectrum,
    axial_phase_space_mcwf,
    axial_temporal_slices,
    axial_time_resolved_spectrum,
    planar_temporal_slices,
    planar_time_resolved_spectrum,
    projected_arrival_fs,
    retarded_delay_fs,
)
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
    "ExitAngularDistribution",
    "EmissionLines",
    "EnergyQuadrature",
    "IncidentEnergyEnsemble",
    "FirstOrderTransition",
    "LineGroup",
    "MCWFResult",
    "PlanarEnergyDiagnostic",
    "PlanarEnergySpectrum",
    "MCWFSolver",
    "PlaneIncidence",
    "TransverseStates",
    "ScatteringKernel",
    "TransitionRadiationBound",
    "axial_energy_mcwf",
    "axial_energy_diagnostic",
    "axial_energy_spectrum",
    "boosted_dipole_pattern",
    "capture_bound",
    "check_quadrature_convergence",
    "channeling_spectrum",
    "classical_exit_angles",
    "coherent_exit_distribution",
    "decoherence_rate",
    "density_exit_distribution",
    "dipole_rate_per_ang",
    "dipole_far_field",
    "dipole_stokes",
    "doppler_denominator",
    "electronic_channels",
    "electronic_kernel",
    "emission_lines",
    "entry_populations",
    "first_order_transition",
    "line_energy_eV",
    "line_shape",
    "longitudinal_kinematics",
    "mean_inner_potential_eV",
    "mcwf_stream_key",
    "open_exit_distribution",
    "planar_energy_diagnostic",
    "planar_energy_spectrum",
    "planar_solver_factory",
    "plane_incidence",
    "rest_frame_dipole_rate_inv_s",
    "sample_electron_energies",
    "spatial_density",
    "tds_kernel",
    "transition_operator",
    "transverse_states",
    "AxialPhaseSpaceMCWFResult",
    "EnergyNodes",
    "IncidentPhaseSpace",
    "LinearEnergyChirp",
    "TemporalSliceDiagnostic",
    "TimeResolvedSpectrum",
    "TimeSlices",
    "axial_phase_space_mcwf",
    "axial_temporal_slices",
    "axial_time_resolved_spectrum",
    "planar_temporal_slices",
    "planar_time_resolved_spectrum",
    "projected_arrival_fs",
    "quantize_energies",
    "retarded_delay_fs",
    "sample_incident_phase_space",
    "time_slices",
    "transition_radiation",
    "transition_radiation_bound",
    "transition_radiation_per_eV",
]
