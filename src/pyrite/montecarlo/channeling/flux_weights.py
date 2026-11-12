"""Channeling flux weights of local incoherent emission on planar zone rows (#1148).

A zone row is the straight mean segment of a channeled electron (ADR-0018).
Native scorers treat it as amorphous path, i.e. a transverse density that is
uniform over the planar cell. An incoherent process whose rate is local in the
transverse coordinate instead sees the channeled density ``rho(x, x; z)``.
Summed over final states, its rate per unit depth is a cell average against a
localization kernel ``K_c(x)`` of the process,

``dN_c/dz = n sigma_c F_c(z)``,
``F_c(z) = <rho(x; z) K_c(x)>_cell / (<rho(x; z)>_cell <K_c>_cell)``,

so ``F_c`` is the yield of process ``c`` relative to amorphous path. A uniform
density gives ``F_c = 1`` for every kernel. A zone row of length ``L`` carries
the depth average ``W_c(L) = (1/L) int_0^L F_c(z) dz`` over the conditional
(retained) ensemble at each depth; native rows carry ``1``.

**Kernels.** Both follow the local-density convention of the zone's own
electronic kernel (``operators``, maintainer decision #504): a transfer of
transverse momentum ``q`` localizes the interaction on a density Gaussian-
smoothed with ``sigma = 1/q``, which multiplies harmonic ``G`` by
``exp(-G^2 / 2 q^2)``.

- *Inner-shell ionization* (characteristic emission), per element and shell
  ``s``: ``K_s(G) ∝ n_el(G) e^{-B s_G^2} [f_s(G) T_s(G) + delta_{G0} sigma_t]``,
  where ``n_el`` is the static site density, ``f_s`` the shell charge form
  factor, ``T_s(G) = int d^2q w_s(q) e^{-G^2/2q^2}`` the shell's
  PENELOPE-2024 distant-longitudinal plus close GOS resolved in ``q_perp``
  (``operators.ElectronicChannels``), and ``sigma_t`` its distant transverse
  (photon-exchange) strength. Transverse distant collisions reach impact
  parameters ``~ gamma v / omega``, far beyond the cell, and are uniform.
- *Incoherent bremsstrahlung*: screened Bethe--Heitler momentum transfer,
  ``dsigma ∝ (q - delta)^2 / q^3 [1 - F(q)/Z]^2 dq`` for the nuclear field
  (Z^2) and ``[1 - (F/Z)^2]`` for the atomic-electron field (Z), with
  ``delta -> 0`` (photon energy far below the electron energy) and ``q`` up
  to the Compton wavenumber ``m c / hbar``. Nuclear terms localize on the
  thermally smeared nuclear density, electron terms on the electron density.

The shell form factor is that of a Slater-type orbital ``r^{n-1} e^{-zeta r}``
with ``zeta = sqrt(2 U_s / E_h) / a_0`` from the shell binding energy ``U_s``:
``f(G) = sin((N+2) t) cos^{N+3} t / ((N+2) sin t)``, ``tan t = G / 2 zeta``,
``N = 2n - 2`` (``N = 0`` is the hydrogenic ``(1 + (G/2zeta)^2)^-2``).

Assumptions: first Born; local (diagonal) approximation of the mixed dynamic
form factor; the kernels are evaluated at the entrance energy (the zone keeps
``S(E0)``); characteristic EEDL cross sections set the magnitude and only the
normalized localization comes from the GOS; brem from multi-element crystals
uses one ``Z(Z+1)``-weighted kernel; photon-energy dependence of the brem
kernel (``delta > 0``) is neglected.

Validation: cr-zone-incoherent-flux-weight
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from ...materials.atomic import Z_TABLE, cromer_mann_f0
from ...materials.channeling.potential import F0_S_MAX_INV_ANG, PlanarProfile
from ...materials.crystal import HBARC_EV_ANG, M_E_EV

#: Column key of the incoherent-bremsstrahlung weight.
BREM_KEY = "brem"
_HARTREE_EV = 27.211386245988
_BOHR_ANG = 0.529177210903
#: Compton wavenumber ``m c / hbar`` [1/angstrom], the upper screened-BH transfer.
_Q_COMPTON_INV_ANG = M_E_EV / HBARC_EV_ANG
_SHELL_N = {"K": 1, "L": 2, "M": 3, "N": 4, "O": 5, "P": 6, "Q": 7}

__all__ = [
    "BREM_KEY",
    "FluxKernels",
    "flux_factors",
    "flux_kernels",
    "row_flux_weights",
    "shell_key",
    "slater_form_factor",
]


def shell_key(atomic_number: int, label: str) -> str:
    """Column key ``"<Z>:<shell>"`` of an inner-shell ionization weight."""
    return f"{int(atomic_number)}:{label}"


@dataclass(frozen=True)
class FluxKernels:
    """Localization kernels on the true planar period.

    ``harmonics[c, m]`` is harmonic ``m = 0..M`` (units of
    ``profile.g_inv_ang``) of kernel ``keys[c]``, normalized so ``m = 0`` is 1.
    """

    keys: tuple[str, ...]
    harmonics: np.ndarray
    g_inv_ang: float

    def values(self, x_ang) -> np.ndarray:
        """Kernels ``K_c(x)`` with unit cell mean; shape ``(n_keys, n_x)``."""
        x = np.asarray(x_ang, dtype=float)
        m = np.arange(1, self.harmonics.shape[1])
        phase = np.exp(1j * self.g_inv_ang * np.multiply.outer(x, m))
        return self.harmonics[:, :1].real + 2.0 * (self.harmonics[:, 1:] @ phase.T).real


def slater_form_factor(g_inv_ang, binding_eV: float, n: int) -> np.ndarray:
    """Normalized charge form factor of a Slater ``r^{n-1} e^{-zeta r}`` orbital.

    ``zeta = sqrt(2 U / E_h) / a_0`` from the binding energy ``U`` [eV];
    ``f(0) = 1``. See the module docstring for the closed form.

    Validation: cr-zone-incoherent-flux-weight
    """
    zeta = np.sqrt(2.0 * float(binding_eV) / _HARTREE_EV) / _BOHR_ANG
    order = 2 * int(n)  # N + 2
    theta = np.arctan(np.asarray(g_inv_ang, dtype=float) / (2.0 * zeta))
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.sin(order * theta) * np.cos(theta) ** (order + 1) / (order * np.sin(theta))
    return np.where(theta == 0.0, 1.0, value)


def _element_densities(profile: PlanarProfile, max_harmonic: int) -> dict[str, np.ndarray]:
    """Thermally smeared site densities ``n_el(G_m)``, ``m = 0..M`` [1/angstrom^3]."""
    from ..channeling.operators import _site_densities

    sites = _site_densities(profile, max_harmonic)
    s = np.arange(max_harmonic + 1) * profile.g_inv_ang / (4.0 * np.pi)
    return {
        el: density[max_harmonic:] * np.exp(-profile.b_factors_ang2[el] * s**2)
        for el, density in sites.items()
    }


def _shell_harmonics(profile, oscillators, kinetic_eV, densities, g):
    """Per-shell ionization kernel harmonics (unnormalized), keyed by :func:`shell_key`."""
    from ..transport.shell_gos import shell_gos_moments
    from .operators import _electronic_transform, electronic_channels

    channels = electronic_channels(oscillators, kinetic_eV)
    moments = shell_gos_moments(oscillators, kinetic_eV)
    # Per formula unit [cm^2] -> per electron [angstrom^2], the channels' units.
    to_channel = 1e16 / oscillators.electrons_per_formula
    symbols = {int(Z_TABLE[el]): el for el in densities}
    out = {}
    for index, osc in enumerate(oscillators.oscillators):
        element = symbols.get(int(osc.atomic_number))
        if element is None or osc.ionization_energy_eV <= 0.0 or osc.label[0] not in _SHELL_N:
            continue
        transverse = float(moments.distant_transverse[index, 0]) * to_channel
        one = replace(
            channels,
            **{
                name: getattr(channels, name)[index : index + 1]
                for name in (
                    "distant_ang2",
                    "distant_qmin_inv_ang",
                    "distant_qmax_inv_ang",
                    "close_ang2_eV",
                    "close_lower_eV",
                    "close_upper_eV",
                    "close_prime_eV",
                )
            },
        )
        localized = _electronic_transform(one, g)
        if localized[0] + transverse <= 0.0:
            continue  # closed channel at this energy
        form = slater_form_factor(g, osc.ionization_energy_eV, _SHELL_N[osc.label[0]])
        kernel = densities[element] * form * localized
        kernel[0] += densities[element][0] * transverse
        out[shell_key(osc.atomic_number, osc.label)] = kernel
    return out


def _screened_transform(element: str, g: np.ndarray, electron_field: bool) -> np.ndarray:
    """``<exp(-G^2/2q^2)>`` over the screened-BH transfer ``w(q) dq`` (``delta -> 0``)."""
    from .operators import _geometric_edges, _panels

    q, w = _panels(_geometric_edges(1e-3, _Q_COMPTON_INV_ANG, 1.25), 8)
    z = float(Z_TABLE[element])
    f0 = np.where(
        q <= 4.0 * np.pi * F0_S_MAX_INV_ANG,
        cromer_mann_f0(element, np.minimum(q, 4.0 * np.pi * F0_S_MAX_INV_ANG)),
        0.0,
    )
    ratio = np.clip(f0 / z, 0.0, 1.0)
    weight = w * ((1.0 - ratio**2) if electron_field else (1.0 - ratio) ** 2) / q
    return (weight @ np.exp(-0.5 * (g[None, :] / q[:, None]) ** 2)) / weight.sum()


def _brem_harmonics(profile, densities, g):
    """Incoherent-brem kernel harmonics (unnormalized): nuclear Z^2 plus electron Z."""
    kernel = np.zeros(g.size, dtype=complex)
    for element, density in densities.items():
        z = float(Z_TABLE[element])
        f0 = np.empty(g.size)
        f0[0] = z
        f0[1:] = cromer_mann_f0(element, g[1:])
        kernel += z**2 * density * _screened_transform(element, g, False)
        kernel += z * density * (f0 / z) * _screened_transform(element, g, True)
    return kernel


@lru_cache(maxsize=16)
def _cached_kernels(crystal, plane, temperature_K, kinetic_eV, max_harmonic) -> FluxKernels:
    from ...materials.channeling.potential import planar_profile
    from ..transport.shell_rates import catalog_shell_oscillators

    profile = planar_profile(crystal, plane, temperature_K)
    return flux_kernels(
        profile, kinetic_eV, max_harmonic, oscillators=catalog_shell_oscillators(crystal)
    )


def flux_kernels(
    profile: PlanarProfile, kinetic_eV: float, max_harmonic: int, *, oscillators=None
) -> FluxKernels:
    """Brem and inner-shell ionization localization kernels of ``profile``.

    ``oscillators`` defaults to the catalog shell oscillators of the crystal.

    Validation: cr-zone-incoherent-flux-weight
    """
    if oscillators is None:
        from ..transport.shell_rates import catalog_shell_oscillators

        oscillators = catalog_shell_oscillators(profile.crystal)
    densities = _element_densities(profile, int(max_harmonic))
    g = np.arange(int(max_harmonic) + 1) * profile.g_inv_ang
    kernels = {BREM_KEY: _brem_harmonics(profile, densities, g)}
    kernels.update(_shell_harmonics(profile, oscillators, float(kinetic_eV), densities, g))
    keys = tuple(kernels)
    stacked = np.stack([kernels[key] / kernels[key][0].real for key in keys])
    return FluxKernels(keys, stacked, float(profile.g_inv_ang))


def flux_factors(
    kernels: FluxKernels, x_ang: np.ndarray, density_per_ang: np.ndarray
) -> np.ndarray:
    """Relative yields ``F_c(z)`` of the recorded cell densities; shape ``(n_z, n_keys)``.

    ``x_ang`` is a uniform periodic grid over one true period and
    ``density_per_ang`` the solver's ``(n_z, n_x)`` cell densities. A depth with
    no retained population gets ``F = 1``.

    Validation: cr-zone-incoherent-flux-weight
    """
    density = np.asarray(density_per_ang, dtype=float)
    weighted = density @ kernels.values(x_ang).T
    total = density.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        factors = weighted / total
    return np.where(total > 0.0, factors, 1.0)


def row_flux_weights(z_ang, factors, lengths_ang) -> np.ndarray:
    """Depth averages ``(1/L) int_0^L F_c dz`` of zone rows; ``F(0)`` at ``L = 0``.

    Trapezoid integration on the recorded depths, linear between them, and the
    last factor held beyond the last depth. Shape ``(n_rows, n_keys)``.

    Validation: cr-zone-incoherent-flux-weight
    """
    z = np.asarray(z_ang, dtype=float)
    f = np.asarray(factors, dtype=float)
    length = np.asarray(lengths_ang, dtype=float)
    cumulative = np.concatenate(
        [np.zeros((1, f.shape[1])), np.cumsum(0.5 * np.diff(z)[:, None] * (f[1:] + f[:-1]), 0)]
    )
    inside = np.minimum(length, z[-1])
    # Exact integral of the piecewise-linear F up to ``inside``.
    index = np.clip(np.searchsorted(z, inside, side="right") - 1, 0, z.size - 2)
    z0, z1 = z[index], z[index + 1]
    t = (inside - z0) / (z1 - z0)
    f0, f1 = f[index], f[index + 1]
    integral = cumulative[index] + (inside - z0)[:, None] * (f0 + 0.5 * t[:, None] * (f1 - f0))
    integral = integral + np.maximum(length - z[-1], 0.0)[:, None] * f[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        weights = integral / length[:, None]
    return np.where(length[:, None] > 0.0, weights, f[0][None, :])
