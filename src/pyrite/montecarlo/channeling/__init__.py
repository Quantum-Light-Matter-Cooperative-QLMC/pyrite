"""Open-system depth evolution of planar-channeled electrons (ADR-0018, P2).

Incoherent scattering (thermal diffuse and electronic) acts on the transverse
state as a Lindblad dissipator. :mod:`.operators` builds its position-space
kernel ``C(x, x')`` and the decoherence diagnostic; :mod:`.density`
integrates the density matrix with a dechanneling sink, and :mod:`.mcwf`
unravels the same equation into per-electron quantum jumps. :mod:`.emission`
scores channeling radiation on ``rho(z)`` as a diagnostic spectrum (P3) from
the photon transition primitives of :mod:`.dipole`; :mod:`.transition_radiation`
bounds the excluded transition-radiation background. CPU only.
"""

from .density import (
    DECHANNELING_MARGIN_EV,
    DECHANNELING_MARGIN_PROVENANCE,
    DensityMatrixResult,
    DensityMatrixSolver,
)
from .dipole import (
    FirstOrderTransition,
    boosted_dipole_pattern,
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
from .transition_radiation import (
    TransitionRadiationBound,
    transition_radiation,
    transition_radiation_bound,
    transition_radiation_per_eV,
)

__all__ = [
    "CHANNELING_MODEL",
    "DECHANNELING_MARGIN_EV",
    "DECHANNELING_MARGIN_PROVENANCE",
    "KIND_CAPTURE",
    "KIND_LINE",
    "CaptureBound",
    "ChannelingSpectrum",
    "DecoherenceRate",
    "DensityMatrixResult",
    "DensityMatrixSolver",
    "ElectronicChannels",
    "EmissionLines",
    "FirstOrderTransition",
    "LineGroup",
    "MCWFResult",
    "MCWFSolver",
    "ScatteringKernel",
    "TransitionRadiationBound",
    "boosted_dipole_pattern",
    "capture_bound",
    "channeling_spectrum",
    "decoherence_rate",
    "dipole_rate_per_ang",
    "dipole_stokes",
    "electronic_channels",
    "electronic_kernel",
    "emission_lines",
    "first_order_transition",
    "line_shape",
    "mcwf_stream_key",
    "rest_frame_dipole_rate_inv_s",
    "tds_kernel",
    "transition_operator",
    "transition_radiation",
    "transition_radiation_bound",
    "transition_radiation_per_eV",
]
