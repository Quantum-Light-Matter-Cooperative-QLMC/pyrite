"""Open-system depth evolution of planar-channeled electrons (ADR-0018, P2).

Incoherent scattering (thermal diffuse and electronic) acts on the transverse
state as a Lindblad dissipator. :mod:`.operators` builds its position-space
kernel ``C(x, x')`` and the decoherence diagnostic; :mod:`.density`
integrates the density matrix with a dechanneling sink. CPU only.
"""

from .density import DensityMatrixResult, DensityMatrixSolver
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
    "DecoherenceRate",
    "DensityMatrixResult",
    "DensityMatrixSolver",
    "ElectronicChannels",
    "ScatteringKernel",
    "decoherence_rate",
    "electronic_channels",
    "electronic_kernel",
    "tds_kernel",
]
