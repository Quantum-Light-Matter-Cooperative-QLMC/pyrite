"""Incident-normalized axial photon *assembly*, not a validated radiation model.

Issue #623: integrates a caller-supplied positive differential source with
geometry-dependent attenuation and checks explicit radiation ownership. It
neither derives the axial emitting subspace/currents nor licenses standard
PyRITE radiation scoring. The source producer must separately validate
scattering-sector closure, lab-frame polarization, line shape and normalization.

Validation scope: cr-axial-zone-handoff-accounting (unverified).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AxialRadiationOwnership:
    """Explicit process tags, with a disjoint native/axial partition.

    Tags must describe the same *physical* process, not merely function or
    feature names. A QRT source including free/free transitions cannot label
    itself purely ``channeling-radiation`` if native CBS covers those photons.
    The caller must demonstrate that its chosen tags are physically exhaustive.
    """

    axial_processes: frozenset[str]
    native_processes: frozenset[str]

    def __post_init__(self) -> None:
        if not self.axial_processes:
            raise ValueError("an axial source needs explicit owned radiation processes")
        if any(
            not isinstance(tag, str) or not tag.strip()
            for tag in self.axial_processes | self.native_processes
        ):
            raise ValueError("radiation process tags must be nonempty strings")
        duplicate = self.axial_processes & self.native_processes
        if duplicate:
            raise ValueError(f"duplicate axial/native radiation ownership: {sorted(duplicate)}")


def integrate_incident_axial_source(
    depths_ang,
    photon_energy_eV,
    rate_per_eV_sr_ang,
    *,
    absorption_inv_ang,
    path_to_exit_ang,
    ownership: AxialRadiationOwnership,
    max_working_bytes: int = 256 * 1024**2,
) -> np.ndarray:
    r"""Integrate *provided* incident-normalized photon production over depth.

    ``rate_per_eV_sr_ang[z,E,ray]`` is photons / (eV sr Å incident electron),
    NOT photons per surviving electron. ``absorption_inv_ang[E]`` and
    ``path_to_exit_ang[z,ray]`` are positive and in compatible units; the
    latter must come from physical lab geometry and vary with azimuth/ray.
    Returned ``spectrum[E,ray]`` is photons / (eV sr incident electron).

    The source must already include complete photon line profiles and the
    correct lab-direction/polarization Jacobian. Attenuation is a scalar
    Beer-Lambert approximation; multiple scattering and re-emission are not
    treated. There is deliberately no normalization by surviving weight and
    no reuse of native transport spectrum bins.
    """
    if not isinstance(ownership, AxialRadiationOwnership):
        raise ValueError("explicit radiation ownership is required")
    if (
        isinstance(max_working_bytes, bool)
        or not isinstance(max_working_bytes, int)
        or max_working_bytes < 1
    ):
        raise ValueError("max_working_bytes must be positive integer")
    z = np.asarray(depths_ang, dtype=float)
    e = np.asarray(photon_energy_eV, dtype=float)
    r = np.asarray(rate_per_eV_sr_ang, dtype=float)
    mu = np.asarray(absorption_inv_ang, dtype=float)
    path = np.asarray(path_to_exit_ang, dtype=float)
    if (
        z.ndim != 1
        or z.size < 2
        or not np.all(np.isfinite(z))
        or np.any(z < 0)
        or np.any(np.diff(z) <= 0)
    ):
        raise ValueError("depths_ang must have >=2 finite increasing nonnegative entries")
    if e.ndim != 1 or not e.size or not np.all(np.isfinite(e)) or np.any(e <= 0):
        raise ValueError("photon_energy_eV must be a nonempty positive energy grid")
    if r.ndim != 3 or r.shape[:2] != (z.size, e.size) or not r.shape[2]:
        raise ValueError("rates must be (depth, energy, ray) with nonempty ray axis")
    if mu.shape != e.shape or path.shape != (z.size, r.shape[2]):
        raise ValueError("absorption and exit path dimensions are incompatible with rate")
    if any(not np.all(np.isfinite(x)) or np.any(x < 0) for x in (r, mu, path)):
        raise ValueError("rates, absorption and paths must be finite and nonnegative")
    # At most one z x ray slice and the output spectrum. No full-size copy
    # of the potentially large z x E x ray rate array is needed.
    required = 8 * (2 * z.size * r.shape[2] + e.size * r.shape[2])
    if required > max_working_bytes:
        raise ValueError("axial source assembly exceeds working-memory budget")
    result = np.zeros((e.size, r.shape[2]), dtype=float)
    dz = np.diff(z)
    for j, attenuation in enumerate(mu):
        survival = np.exp(-attenuation * path)
        trapezoids = 0.5 * (r[1:, j] * survival[1:] + r[:-1, j] * survival[:-1])
        result[j] = np.sum(trapezoids * dz[:, None], axis=0)
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise FloatingPointError("axial photon result is not finite and nonnegative")
    return result
