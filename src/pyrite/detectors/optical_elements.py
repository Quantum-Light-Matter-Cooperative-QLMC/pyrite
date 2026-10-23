"""Measured passive optical elements acting on polarized spectral coherency (#616).

A window, mirror or polarizer is represented only by its *measured* 2x2 complex
Jones transfer ``T(E)`` in the same declared transverse basis as the source
coherency, on a bounded calibration energy grid, with a stated maximum ray
half-angle for which that measurement holds. Application is the linear
second-moment map

    J_out(E) = T(E) J_in(E) T(E)^H,

which preserves Hermitian positive semidefiniteness, and cannot increase the
photon density when every singular value of ``T`` is at most one (passivity).
Between calibration nodes ``T`` is interpolated linearly in its complex
entries; a convex combination of contractions is a contraction, so the
interpolant remains passive. There is no extrapolation outside the calibrated
band, no inferred optical constants, no thickness/etalon model, and no
angle-dependent transfer beyond the declared half-angle bound: the paraxial
coherency carries no per-ray angle, so the caller must state its collection
half-angle. Spatial phase (focusing, aberration) is outside this operator.

Validation: optical-wavefront-propagation
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .._optical_coherency import OpticalSpectralCoherency

__all__ = ["CalibratedJonesElement", "apply_optical_elements"]

_PASSIVITY_TOL = 1e-12


@dataclass(frozen=True, eq=False)
class CalibratedJonesElement:
    """Measured passive Jones transfer T(E), shape (n_energy, 2, 2).

    ``element_id`` and ``calibration_provenance`` identify the physical part
    and its measurement; neither may be empty. ``max_half_angle_rad`` is the
    largest collected-ray angle to the element normal covered by the data.
    """

    element_id: str
    energy_eV: np.ndarray
    jones: np.ndarray
    max_half_angle_rad: float
    calibration_provenance: str

    def __post_init__(self):
        e = np.array(self.energy_eV, dtype=float, copy=True)
        t = np.array(self.jones, dtype=complex, copy=True)
        if (
            e.ndim != 1
            or e.size < 2
            or not np.all(np.isfinite(e))
            or e[0] <= 0
            or not np.all(np.diff(e) > 0)
        ):
            raise ValueError("calibration energy grid must be positive, finite, increasing")
        if t.shape != (e.size, 2, 2) or not np.all(np.isfinite(t)):
            raise ValueError("jones must be a finite (n_energy, 2, 2) complex array")
        if np.any(np.linalg.svd(t, compute_uv=False)[:, 0] > 1 + _PASSIVITY_TOL):
            raise ValueError("measured Jones transfer is not passive (singular value > 1)")
        if not np.isfinite(self.max_half_angle_rad) or not (
            0 < self.max_half_angle_rad < np.pi / 2
        ):
            raise ValueError("max_half_angle_rad must lie in (0, pi/2)")
        for name in ("element_id", "calibration_provenance"):
            if not getattr(self, name) or not getattr(self, name).strip():
                raise ValueError(f"{name} is required")
        e.setflags(write=False)
        t.setflags(write=False)
        object.__setattr__(self, "energy_eV", e)
        object.__setattr__(self, "jones", t)

    @classmethod
    def from_intensity_transmission(
        cls, element_id, energy_eV, transmission, *, max_half_angle_rad, calibration_provenance
    ):
        """Polarization-independent element from a measured *intensity* T(E).

        Uses ``sqrt(T)`` times identity: the declared assumption is equal s/p
        transmission and no retardance. Spectral phase is dropped, which is
        exact for an energy-integrating detector and the interferometer's
        independent spectral bins, but not for a time-domain field.
        """
        transmission = np.asarray(transmission, dtype=float)
        if (
            transmission.ndim != 1
            or not np.all(np.isfinite(transmission))
            or np.any(transmission < 0)
            or np.any(transmission > 1)
        ):
            raise ValueError("intensity transmission must be finite and within [0, 1]")
        jones = np.sqrt(transmission)[:, None, None] * np.eye(2)[None]
        return cls(element_id, energy_eV, jones, max_half_angle_rad, calibration_provenance)

    @property
    def response_sha256(self) -> str:
        digest = hashlib.sha256()
        for array in (self.energy_eV, self.jones):
            digest.update(repr((array.shape, array.dtype.str)).encode("ascii"))
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(
            json.dumps(
                [self.element_id, self.max_half_angle_rad, self.calibration_provenance]
            ).encode()
        )
        return digest.hexdigest()

    def transfer(self, energy_eV) -> np.ndarray:
        """Interpolated passive T at ``energy_eV``; refuses extrapolation."""
        e = np.asarray(energy_eV, dtype=float)
        if e.ndim != 1 or np.any(e < self.energy_eV[0]) or np.any(e > self.energy_eV[-1]):
            raise ValueError(f"{self.element_id}: energies outside the calibrated band")
        out = np.empty((e.size, 2, 2), complex)
        for a in range(2):
            for b in range(2):
                out[:, a, b] = np.interp(e, self.energy_eV, self.jones[:, a, b].real) + (
                    1j * np.interp(e, self.energy_eV, self.jones[:, a, b].imag)
                )
        return out


def apply_optical_elements(
    source: OpticalSpectralCoherency,
    elements,
    *,
    collection_half_angle_rad: float,
) -> OpticalSpectralCoherency:
    """Apply measured elements in beam order (first element first).

    ``collection_half_angle_rad`` is the caller's stated largest ray angle at
    every element; each element must be calibrated for it.

    Validation: optical-wavefront-propagation
    """
    if not isinstance(source, OpticalSpectralCoherency):
        raise TypeError("source must be an OpticalSpectralCoherency")
    elements = tuple(elements)
    if not elements or not all(isinstance(x, CalibratedJonesElement) for x in elements):
        raise TypeError("supply at least one CalibratedJonesElement")
    if not np.isfinite(collection_half_angle_rad) or collection_half_angle_rad < 0:
        raise ValueError("collection_half_angle_rad must be finite and nonnegative")
    j = np.array(source.photons_per_eV_per_pulse)
    for element in elements:
        if collection_half_angle_rad > element.max_half_angle_rad:
            raise ValueError(
                f"{element.element_id}: collection angle exceeds its calibrated half-angle"
            )
        t = element.transfer(source.energy_eV)
        j = t @ j @ t.conj().transpose(0, 2, 1)
    chain = "; ".join(f"{x.element_id} sha256={x.response_sha256}" for x in elements)
    return OpticalSpectralCoherency(source.energy_eV, j, f"{source.provenance} -> [{chain}]")
