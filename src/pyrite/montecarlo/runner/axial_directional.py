"""Directional fibre source of an unadmitted axial case (#1148).

An axial case lowered with the ``"directional"`` observable (off-axis angle,
finite aperture or a physical detector; runs only under
``Channeling(admission="accept-inaccurate")``) keeps the fibre's mean
two-current correlation ``G_ab(z_i, z_j)`` and its contiguous batch means from
the same histories that set the fates. The spectrum at a photon direction
``n`` is the trace of the anchored polarization matrix
(:func:`.axial_fiber_angular.polarization_spectrum`, ``cr-axial-fiber-angular``)
with ``(theta, phi)`` of ``n`` in the profile frame of the axis,

``dN/(dE dOmega)(n) = tr S(E, theta, phi)``,

averaged over the cone of half angle ``dtheta_obs_rad`` with the planar
source's nodes. The depth sum uses the exact linear-interpolation weights
(:func:`.axial_fiber_angular.linear_depth_weights`), so off-axis directions,
whose Doppler phase step exceeds the on-axis trapezoid cap, alias no line
replicas. The standard error follows from the batch spread (``S`` is linear
in ``G``). Directions at or beyond 90 degrees from the axis score zero: the
fibre source is forward emission only (resonances there lie at
``E <= Delta_fi``, below the keV grid; the angular density at 90 degrees is
about ``gamma^-4/4`` of the forward peak). Each line comes out low by about
``(Delta_fi dz / hbar v)^2 / 6`` at any angle (the linear quadrature's peak
factor ``sinc^4``): about 6% for an 80 eV transition at the default 15 A step.

Validation: cr-axial-directional-observation
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

#: Contiguous incident batches kept for the directional standard error.
DIRECTIONAL_BATCHES = 8
#: Gauss--Legendre radial order of the aperture cone (the planar source's).
_APERTURE_ORDER = 6


def is_directional(case) -> bool:
    """Whether ``case`` is an axial case lowered with the directional observable.

    Validation: cr-axial-directional-observation
    """
    record = case.get("channeling")
    return isinstance(record, Mapping) and record.get("observable") == "directional"


def directional_record(source, frame, kinetic_energy_eV, absorption: bool) -> dict[str, Any]:
    """Transport-side record of the fibre correlation for later directions.

    Validation: cr-axial-directional-observation
    """
    return {
        "depths_ang": np.asarray(source.correlation.depths_ang, dtype=float).copy(),
        "correlation": np.asarray(source.correlation.values, dtype=complex).copy(),
        "batch_correlations": np.asarray(source.batch_correlations, dtype=complex).copy(),
        "batch_sizes": np.asarray(source.batch_sizes, dtype=float).copy(),
        "frame": np.asarray(frame, dtype=float).copy(),
        "kinetic_energy_eV": float(kinetic_energy_eV),
        "absorption": bool(absorption),
    }


def _cone(n_hat: np.ndarray, aperture_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Directions and weights over the cone of half angle ``aperture_rad`` about ``n_hat``."""
    from ..channeling.emission import _direction_nodes

    radius, chi, weight = _direction_nodes(float(aperture_rad), _APERTURE_ORDER)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n_hat, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_hat, e1)
    directions = np.cos(radius)[:, None] * n_hat + np.sin(radius)[:, None] * (
        np.cos(chi)[:, None] * e1 + np.sin(chi)[:, None] * e2
    )
    return directions, weight


def _trace(values, record, energies, amplitude, theta, phi) -> np.ndarray:
    from ..channeling.axial_fiber_angular import polarization_spectrum

    s = polarization_spectrum(
        values,
        record["depths_ang"],
        energies,
        record["kinetic_energy_eV"],
        [theta],
        [phi],
        amplitude=amplitude,
        quadrature="linear",
    )
    return (s[0, 0, 0, 0] + s[0, 0, 1, 1]).real


def directional_spectrum(case, record, n_hat, energies_eV) -> tuple[np.ndarray, np.ndarray]:
    """``(spectrum, sem)`` [photons/(eV sr incident electron)] at ``n_hat`` on ``energies_eV``.

    Validation: cr-axial-directional-observation
    """
    from ...materials.attenuation import linear_attenuation_inv_mm
    from ..channeling.axial_fiber_angular import hermitian_complete

    energies = np.asarray(energies_eV, dtype=float)
    z = record["depths_ang"]
    amplitude = None
    if record["absorption"]:
        mu = linear_attenuation_inv_mm(case["crystal"], energies) * 1e-7
        amplitude = np.exp(-0.5 * mu[:, None] * (z[-1] - z)[None, :])
    batches = [hermitian_complete(g) for g in record["batch_correlations"]]
    sizes = record["batch_sizes"]
    directions, weights = _cone(np.asarray(n_hat, dtype=float), float(case["dtheta_obs_rad"]))
    mean = np.zeros(energies.size)
    per_batch = np.zeros((len(batches), energies.size))
    for direction, weight in zip(directions, weights, strict=True):
        local = record["frame"].T @ direction
        theta = float(np.arctan2(np.hypot(local[0], local[1]), local[2]))
        if theta >= 0.5 * np.pi:
            continue  # forward-only source
        phi = float(np.arctan2(local[1], local[0]))
        mean += weight * _trace(record["correlation"], record, energies, amplitude, theta, phi)
        for b, values in enumerate(batches):
            per_batch[b] += weight * _trace(values, record, energies, amplitude, theta, phi)
    count = float(sizes.sum())
    spread = np.sum(sizes[:, None] * (per_batch - mean) ** 2, axis=0) / (len(batches) - 1)
    return mean, np.sqrt(spread / count)


def cached_directional_spectrum(case, transport) -> tuple[np.ndarray, np.ndarray]:
    """:func:`directional_spectrum` at the transport's ``n_hat``/``E_grid``, memoized.

    The spectrum and the diagnostics phases both read it; the cache lives on
    the shared transport record and is keyed by direction and grid bytes.

    Validation: cr-axial-directional-observation
    """
    info = transport["segs"]["channeling"]
    record = info.get("directional")
    if record is None:
        from ...quantum._admission import AxialAdmissionError

        raise AxialAdmissionError("axial transport carries no directional fibre record (#1148)")
    n_hat = np.asarray(transport["n_hat"], dtype=float)
    grid = np.asarray(transport["E_grid"], dtype=float)
    key = (n_hat.tobytes(), grid.tobytes(), float(case["dtheta_obs_rad"]))
    cache = record.setdefault("_cache", {})
    if key not in cache:
        cache.clear()  # one direction at a time: bounded memory
        cache[key] = directional_spectrum(case, record, n_hat, grid)
    return cache[key]
