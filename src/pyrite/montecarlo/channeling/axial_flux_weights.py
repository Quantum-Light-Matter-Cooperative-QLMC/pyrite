"""Channeling flux weights of local incoherent emission on axial zone rows (#1148).

The axial counterpart of :mod:`.flux_weights`. A local incoherent process
(close-collision bremsstrahlung, inner-shell ionization) on a zone row sees
the retained transverse density ``rho(r, r; z)`` of the fibre ensemble
instead of a uniform one, so its yield relative to amorphous path is

``F_c(z) = <rho(r; z) K_c(r)>_cell / (<rho(r; z)>_cell <K_c>_cell)``,

now a mean over the rectangular 2D cell of the axis. With the fibre states
written as Bloch sums ``psi(r) = e^{i kappa.r} sum_G psi_G e^{i G.r}`` on the
window orders ``|m| <= M_x``, ``|n| <= M_y``, ``|psi|^2`` holds harmonics
``|Q_i| <= 2 M_i`` only, and

``F_c = Re sum_Q rho_Q conj(K_c,Q) / rho_0``,

with ``K_c,Q`` normalized to ``K_c,0 = 1``. The carrier drops out. Kernels
use the planar construction at each 2D ``|Q|``
(:func:`.flux_weights._shell_harmonics`, :func:`.flux_weights._brem_harmonics`)
with the thermally smeared per-element string density

``n_el(Q) = sum_strings lambda_el / A e^{-i Q.R} e^{-a_el^2 Q^2 / 2}``,

``a_el^2`` the in-plane mean-square displacement per axis. Summed over
elements this is the profile's ``nuclear_density``. Every assumption of
:mod:`.flux_weights` carries over (first Born, local diagonal approximation,
entrance-energy kernels, ``delta -> 0`` brem transfer). A uniform density, and
therefore an amorphous or misaligned path, gives ``F = 1`` exactly.

**Estimator.** Each fibre history reports its normalized retained state at
the recorded depths (no random draws); the weighted sums of ``F`` and of the
alive indicator give the ensemble ``F_c(z)`` of the retained density
``E[W 1(alive) |psi><psi|]``. Zone rows then carry the depth average of
``F_c`` over their own length (:func:`.flux_weights.row_flux_weights`). Since
the same histories set both the row lengths and the ensemble, the summed
weighted path equals ``sum_i int_0^{L_i} F_i(z) dz`` of the histories' own
states exactly.

Validation: cr-zone-incoherent-flux-weight
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .flux_weights import BREM_KEY, _brem_harmonics, _shell_harmonics

__all__ = ["AxialFluxKernels", "axial_flux_kernels", "cached_axial_flux_kernels"]


@dataclass(frozen=True)
class AxialFluxKernels:
    """Localization kernels on the 2D axial cell.

    ``harmonics[c]`` is the centred ``(4 M_x + 1, 4 M_y + 1)`` array of
    harmonics ``Q = (m g_x, n g_y)`` of kernel ``keys[c]``, ``|m| <= 2 M_x``,
    ``|n| <= 2 M_y``, normalized so ``Q = 0`` is 1. ``order_max`` is the state
    window ``(M_x, M_y)`` the kernels serve.
    """

    keys: tuple[str, ...]
    harmonics: np.ndarray
    order_max: tuple[int, int]

    def factors(self, orders: np.ndarray) -> np.ndarray:
        """``F_c`` of one state on the window orders (C order, x outer); shape ``(n_keys,)``.

        The state need not be normalized; a zero state gives ``F = 1``.

        Validation: cr-zone-incoherent-flux-weight
        """
        mx, my = self.order_max
        shape = (4 * mx + 2, 4 * my + 2)  # > 4M: the autocorrelation never aliases
        psi = np.asarray(orders, dtype=complex).reshape(2 * mx + 1, 2 * my + 1)
        layout = np.zeros(shape, dtype=complex)
        ix = np.arange(-mx, mx + 1) % shape[0]
        iy = np.arange(-my, my + 1) % shape[1]
        layout[np.ix_(ix, iy)] = psi
        density = np.abs(np.fft.ifft2(layout)) ** 2
        rho = np.fft.fft2(density)  # proportional to rho_Q; the constant cancels
        qx = np.arange(-2 * mx, 2 * mx + 1) % shape[0]
        qy = np.arange(-2 * my, 2 * my + 1) % shape[1]
        rho_q = rho[np.ix_(qx, qy)]
        total = rho[0, 0].real
        if total <= 0.0:
            return np.ones(len(self.keys))
        return (rho_q[None] * self.harmonics.conj()).real.sum(axis=(1, 2)) / total


def _string_densities(profile, q_vectors: np.ndarray) -> dict[str, np.ndarray]:
    """Thermally smeared per-element string densities ``n_el(Q)`` [1/angstrom^3]."""
    from .axial_dissipator import _site_displacements

    a2 = _site_displacements(profile)
    q2 = (q_vectors**2).sum(axis=-1)
    out: dict[str, np.ndarray] = {}
    for string in profile.strings:
        phase = np.exp(-1j * (q_vectors @ np.asarray(string.position_ang, dtype=float)))
        for element, lam in string.linear_density.items():
            value = lam / profile.area_ang2 * phase * np.exp(-0.5 * a2[element] * q2)
            out[element] = out.get(element, 0.0) + value
    return out


def axial_flux_kernels(
    profile, kinetic_eV: float, order_max: tuple[int, int], *, oscillators=None
) -> AxialFluxKernels:
    """Brem and inner-shell ionization kernels of an axial ``profile``.

    ``order_max`` is the fibre's state window ``(M_x, M_y)``; ``oscillators``
    defaults to the catalog shell oscillators of the crystal.

    Validation: cr-zone-incoherent-flux-weight
    """
    if oscillators is None:
        from ..transport.shell_rates import catalog_shell_oscillators

        oscillators = catalog_shell_oscillators(profile.crystal)
    mx, my = (int(v) for v in order_max)
    gx, gy = profile.g_inv_ang
    m, n = np.meshgrid(
        np.arange(-2 * mx, 2 * mx + 1), np.arange(-2 * my, 2 * my + 1), indexing="ij"
    )
    shape = m.shape
    # Flatten with Q = 0 first: the planar helpers treat index 0 as G = 0.
    order = np.argsort((m**2 + n**2).ravel(), kind="stable")
    vectors = np.stack([m.ravel() * gx, n.ravel() * gy], axis=-1)[order]
    g = np.hypot(vectors[:, 0], vectors[:, 1])
    densities = _string_densities(profile, vectors)
    kernels = {BREM_KEY: _brem_harmonics(None, densities, g)}
    kernels.update(_shell_harmonics(None, oscillators, float(kinetic_eV), densities, g))
    keys = tuple(kernels)
    out = np.empty((len(keys), *shape), dtype=complex)
    for index, key in enumerate(keys):
        flat = np.empty(g.size, dtype=complex)
        flat[order] = kernels[key] / kernels[key][0].real
        out[index] = flat.reshape(shape)
    return AxialFluxKernels(keys, out, (mx, my))


@lru_cache(maxsize=8)
def cached_axial_flux_kernels(
    crystal: str, axis_uvw, temperature_K: float, kinetic_eV: float, order_max
) -> AxialFluxKernels:
    """:func:`axial_flux_kernels` of the catalog axial profile, cached by its inputs.

    Validation: cr-zone-incoherent-flux-weight
    """
    from ...materials.channeling.axial import axial_profile

    profile = axial_profile(crystal, tuple(axis_uvw), temperature_K=temperature_K)
    return axial_flux_kernels(profile, kinetic_eV, tuple(order_max))
