"""Polarized optical spectral coherency shared by transport and detectors.

A leaf below :mod:`pyrite.montecarlo` and :mod:`pyrite.detectors`: the finite
PEC screen handoff builds it from complex screen fields and the Martin--Puplett
forward model consumes it, without Monte Carlo importing the detector package.
The public name stays importable from
:mod:`pyrite.detectors.optical_interferometer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._backend import array_namespace

__all__ = ["OpticalSpectralCoherency"]


@dataclass(frozen=True, eq=False)
class OpticalSpectralCoherency:
    """Hermitian positive-semidefinite 2x2 photon-density matrix per E/pulse.

    ``J[i,a,b] = <E_a E_b*>`` in a *declared Jones basis*, scaled so that
    ``trace(J[i])`` is already-collected photons/(eV pulse). It is not an
    electric field and contains no frequency-to-frequency mutual coherence.
    The matrix may encode partial polarization and polarization phase.

    NumPy or CuPy arrays select their own backend. Storage is copied; NumPy
    copies are read-only, CuPy copies remain writable. Validation synchronizes
    scalar predicates; no spectral matrix is downloaded for validation.

    Validation: optical-instrument-observable
    """

    energy_eV: Any
    photons_per_eV_per_pulse: Any
    provenance: str

    def __post_init__(self):
        xp = array_namespace(self.energy_eV, self.photons_per_eV_per_pulse)
        e = xp.array(self.energy_eV, dtype=float, copy=True)
        j = xp.array(self.photons_per_eV_per_pulse, dtype=complex, copy=True)
        if (
            e.ndim != 1
            or e.size < 2
            or not bool(xp.all(xp.isfinite(e)))
            or e[0] <= 0
            or not bool(xp.all(xp.diff(e) > 0))
        ):
            raise ValueError(
                "spectral energy grid must be positive, finite, increasing with >=2 nodes"
            )
        if j.shape != (len(e), 2, 2) or not bool(xp.all(xp.isfinite(j))):
            raise ValueError("spectral coherency must be finite (E,2,2)")
        if not self.provenance or not self.provenance.strip():
            raise ValueError("source polarization requires nonempty provenance")
        scale = xp.maximum(1.0, xp.max(xp.abs(j), axis=(1, 2)))
        if bool(
            xp.any(xp.max(xp.abs(j - j.conj().transpose(0, 2, 1)), axis=(1, 2)) > 1e-11 * scale)
        ):
            raise ValueError("spectral coherency must be Hermitian")
        # Reject non-physical matrices; symmetrize tiny floating-point asymmetry.
        j = (j + j.conj().transpose(0, 2, 1)) / 2
        if bool(xp.any(xp.linalg.eigvalsh(j)[:, 0] < -1e-11 * scale)):
            raise ValueError("spectral coherency must be positive semidefinite")
        if xp is np:
            e.setflags(write=False)
            j.setflags(write=False)
        object.__setattr__(self, "energy_eV", e)
        object.__setattr__(self, "photons_per_eV_per_pulse", j)

    @classmethod
    def linearly_polarized(cls, energy_eV, photon_density, *, angle_rad, provenance):
        """Explicit *assumption* of unit-degree linear polarization in xy.

        Use only if the source polarization is externally known. It must never
        be inferred from an intensity-only spectrum without such evidence.
        """
        if not np.isfinite(angle_rad):
            raise ValueError("polarization angle must be finite")
        xp = array_namespace(energy_eV, photon_density)
        vector = xp.asarray([np.cos(angle_rad), np.sin(angle_rad)])
        photons = xp.asarray(photon_density, dtype=float)
        if photons.ndim != 1 or not bool(xp.all(xp.isfinite(photons))) or bool(xp.any(photons < 0)):
            raise ValueError("photon density must be finite and nonnegative")
        return cls(energy_eV, photons[:, None, None] * xp.outer(vector, vector), provenance)

    @property
    def total_photons_per_eV_per_pulse(self):
        return (
            self.photons_per_eV_per_pulse[:, 0, 0].real
            + self.photons_per_eV_per_pulse[:, 1, 1].real
        )
