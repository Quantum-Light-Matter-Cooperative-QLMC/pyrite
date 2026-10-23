"""Polarized optical spectral coherency shared by transport and detectors.

A leaf below :mod:`pyrite.montecarlo` and :mod:`pyrite.detectors`: the finite
PEC screen handoff builds it from complex screen fields and the Martin--Puplett
forward model consumes it, without Monte Carlo importing the detector package.
The public name stays importable from
:mod:`pyrite.detectors.optical_interferometer`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["OpticalSpectralCoherency"]


@dataclass(frozen=True, eq=False)
class OpticalSpectralCoherency:
    """Hermitian positive-semidefinite 2x2 photon-density matrix per E/pulse.

    ``J[i,a,b] = <E_a E_b*>`` in a *declared Jones basis*, scaled so that
    ``trace(J[i])`` is already-collected photons/(eV pulse). It is not an
    electric field and contains no frequency-to-frequency mutual coherence.
    The matrix may encode partial polarization and polarization phase.
    """

    energy_eV: np.ndarray
    photons_per_eV_per_pulse: np.ndarray
    provenance: str

    def __post_init__(self):
        e = np.array(self.energy_eV, dtype=float, copy=True)
        j = np.array(self.photons_per_eV_per_pulse, dtype=complex, copy=True)
        if (
            e.ndim != 1
            or e.size < 2
            or not np.all(np.isfinite(e))
            or e[0] <= 0
            or not np.all(np.diff(e) > 0)
        ):
            raise ValueError(
                "spectral energy grid must be positive, finite, increasing with >=2 nodes"
            )
        if j.shape != (len(e), 2, 2) or not np.all(np.isfinite(j)):
            raise ValueError("spectral coherency must be finite (E,2,2)")
        if not self.provenance or not self.provenance.strip():
            raise ValueError("source polarization requires nonempty provenance")
        scale = np.maximum(1.0, np.max(np.abs(j), axis=(1, 2)))
        if np.any(np.max(np.abs(j - j.conj().transpose(0, 2, 1)), axis=(1, 2)) > 1e-11 * scale):
            raise ValueError("spectral coherency must be Hermitian")
        # Reject non-physical matrices; symmetrize tiny floating-point asymmetry.
        j = (j + j.conj().transpose(0, 2, 1)) / 2
        if np.any(np.linalg.eigvalsh(j)[:, 0] < -1e-11 * scale):
            raise ValueError("spectral coherency must be positive semidefinite")
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
        vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        photons = np.asarray(photon_density, dtype=float)
        if photons.ndim != 1 or not np.all(np.isfinite(photons)) or np.any(photons < 0):
            raise ValueError("photon density must be finite and nonnegative")
        return cls(energy_eV, photons[:, None, None] * np.outer(vector, vector), provenance)

    @property
    def total_photons_per_eV_per_pulse(self):
        return (
            self.photons_per_eV_per_pulse[:, 0, 0].real
            + self.photons_per_eV_per_pulse[:, 1, 1].real
        )
