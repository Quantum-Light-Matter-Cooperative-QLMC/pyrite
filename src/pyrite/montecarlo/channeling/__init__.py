"""Open-system depth evolution of planar-channeled electrons (ADR-0018, P2).

Incoherent scattering (thermal diffuse and electronic) acts on the transverse
state as a Lindblad dissipator. :mod:`.operators` builds its position-space
kernel ``C(x, x')`` and the decoherence diagnostic; :mod:`.density`
integrates the density matrix with a dechanneling sink, and :mod:`.mcwf`
unravels the same equation into per-electron quantum jumps. CPU only.
"""

from .density import (
    DECHANNELING_MARGIN_EV,
    DECHANNELING_MARGIN_PROVENANCE,
    DensityMatrixResult,
    DensityMatrixSolver,
)
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

__all__ = [
    "DECHANNELING_MARGIN_EV",
    "DECHANNELING_MARGIN_PROVENANCE",
    "DecoherenceRate",
    "DensityMatrixResult",
    "DensityMatrixSolver",
    "ElectronicChannels",
    "MCWFResult",
    "MCWFSolver",
    "ScatteringKernel",
    "decoherence_rate",
    "electronic_channels",
    "electronic_kernel",
    "mcwf_stream_key",
    "tds_kernel",
]
