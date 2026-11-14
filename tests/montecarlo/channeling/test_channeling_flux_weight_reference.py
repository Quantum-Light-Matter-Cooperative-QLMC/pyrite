"""Independent real-space reference for planar zone-row flux weights (#1148).

Rebuilds the localization kernels ``K_c(x)`` of the ledger claim on the
admitted ``channeling_demo`` zone (Si(1-10), 17 MeV, 2.0027 um) directly in
position space across the plane cell, without the production Fourier
harmonics: each kernel is a sum over thermally smeared (110) planes of a
mixture of normalized Gaussians of variance ``u^2 + 1/q_perp^2`` (the real-space
form of ``exp(-B s^2) exp(-G^2/2q^2)``), weighted by the transfer distribution
of the process, convolved with the projected shell (Slater) or atomic
(Waasmaier--Kirfel Gaussian) charge density, plus the uniform
distant-transverse term for shells. ``F_c(z) = <rho K>/(<rho><K>)`` and
``W_c = (1/L) int F dz`` follow from the zone's retained density.

Inputs owned by other claims: the zone solve's retained density, plane period
and Debye--Waller ``B``, the PENELOPE GOS oscillators and zeroth moments, and
the catalog ``f0`` fit coefficients. ``f0`` is taken as zero beyond the fit
range ``s > 6 /A`` in the brem transfer weights; extrapolating the fit instead
lowers ``W_brem`` by 1.9e-4 relative (pinned below).

Reference quadrature (Gauss--Legendre in log transfer, periodic fine grid of
the shell convolution, Gaussian image sums) is converged to < 1e-8 in ``W``
(n_q 128..800, fine grid 4096..16384 points); production's own panel
quadrature from ``q = 1e-3 /A`` moves ``W_brem`` by 2.5e-6. Tolerance
``rel = 1e-5`` covers both with margin and is 19x below the extrapolation
convention effect.

Validation: cr-zone-incoherent-flux-weight
"""

import json

import numpy as np
import pytest
import xraydb
from scipy.special import gammaincc

from pyrite.montecarlo.channeling import flux_weights as fw
from pyrite.montecarlo.channeling.handoff import _cell_grid, _solve
from pyrite.montecarlo.transport.shell_gos import shell_gos_moments
from pyrite.montecarlo.transport.shell_rates import catalog_shell_oscillators

E_EV = 17e6
H = 16
PATH_ANG = 20027.44691995842
Z_SI = 14
MC2_EV = 510998.95
HBARC_EV_ANG = 1973.269804
HARTREE_EV = 27.211386245988
BOHR_ANG = 0.529177210903
S_FIT_MAX = 6.0
N_Q = 160
N_FINE = 128 * 32
RTOL = 1e-5

pytestmark = pytest.mark.slow  # ~5-6 s wall, dominated by the zone solve


@pytest.fixture(scope="module")
def zone():
    result, _, profile, _, _ = _solve(
        "silicon", (1, -1, 0), 293.15, E_EV, H, 16, 300.0, PATH_ANG, 250.0, 0.0, 0.0, 0.0409277
    )
    x = _cell_grid(profile, H)
    rho = np.asarray(result.spatial_density_per_ang)
    z = np.asarray(result.z_ang)
    return profile, x, rho, z


def _cp(t):
    return np.sqrt(t * (t + 2.0 * MC2_EV))


def _qmin_sq(w):
    """``(hbar c q_min)^2``: zero-angle (longitudinal) transfer for loss ``w`` at ``E``."""
    return (_cp(E_EV) - _cp(E_EV - w)) ** 2


def _log_nodes(lo, hi, n):
    t, w = np.polynomial.legendre.leggauss(n)
    a, b = np.log(lo), np.log(hi)
    return np.exp(0.5 * (b - a) * t + 0.5 * (a + b)), 0.5 * (b - a) * w


def _periodic_gauss(r, var, d):
    sigma = np.sqrt(var)
    if sigma > 3.0 * d:  # uniform to within 2 exp(-18 pi^2)
        return np.full_like(r, 1.0 / d)
    n = int(np.ceil(12.0 * sigma / d)) + 2
    y = r[..., None] - np.arange(-n, n + 1) * d
    return np.exp(-0.5 * y * y / var).sum(-1) / np.sqrt(2.0 * np.pi * var)


def _mixture(r, var0, q_perp, mu, d):
    """``sum_k mu_k`` periodic Gaussian of variance ``var0 + 1/q_k^2`` (``q = 0``: uniform)."""
    out = np.zeros_like(r)
    for q, m in zip(q_perp, mu, strict=True):
        out += m * (np.full_like(r, 1.0 / d) if q <= 0.0 else _periodic_gauss(r, var0 + q**-2, d))
    return out


def _brem_kernel(x, d, u2, s_cut=S_FIT_MAX):
    row = next(r for r in xraydb.get_xraydb().get_cache("Waasmaier") if r.ion == "Si")
    amp = np.array(json.loads(row.scale))
    expo = np.array(json.loads(row.exponents))
    q_top = MC2_EV / HBARC_EV_ANG  # mc/hbar
    q_cut = min(4.0 * np.pi * (s_cut or np.inf), q_top)
    parts = [_log_nodes(1e-6, q_cut, N_Q)] + (
        [_log_nodes(q_cut, q_top, N_Q)] if q_cut < q_top else []
    )
    q = np.concatenate([p[0] for p in parts])
    dlnq = np.concatenate([p[1] for p in parts])
    s = q / (4.0 * np.pi)
    ratio = ((amp * np.exp(-expo * s[:, None] ** 2)).sum(1) + row.offset) / Z_SI
    ratio = np.where(q <= q_cut, ratio, 0.0)
    w_nuc = (1.0 - ratio) ** 2 * dlnq  # [1 - F/Z]^2 dq/q
    w_el = (1.0 - ratio**2) * dlnq  # [1 - (F/Z)^2] dq/q
    kernel = Z_SI**2 * _mixture(x, u2, q, w_nuc / w_nuc.sum(), d)
    # Atomic electron density: f0 = sum a_i exp(-b_i s^2) + c  <->  Gaussians of
    # variance b_i/(8 pi^2) of weight a_i, plus c at the nucleus.
    for a, var in zip([*amp, row.offset], [*(expo / (8.0 * np.pi**2)), 0.0], strict=True):
        kernel += a * _mixture(x, u2 + var, q, w_el / w_el.sum(), d)
    return kernel


def _shell_kernel(x, d, u2, osc, dl, tr, close):
    u, w = osc.ionization_energy_eV, osc.resonance_energy_eV
    assert E_EV > 3.0 * w - 2.0 * u  # far above threshold: W' = W, Q' = U
    # Distant longitudinal: dQ/(Q(Q + 2mc^2)) on [Q_-, U], q_perp^2 = q^2 - q_min^2.
    qm2 = _qmin_sq(w)
    q_lo = np.sqrt(qm2 + MC2_EV**2) - MC2_EV
    big_q, dln = _log_nodes(q_lo, u, N_Q)
    mu_dl = dln / (big_q + 2.0 * MC2_EV)
    qp_dl = np.sqrt(np.maximum(big_q * (big_q + 2.0 * MC2_EV) - qm2, 0.0)) / HBARC_EV_ANG
    # Close (Moller, free target, Q = W): W^-2 F^-(E + U, W) on [U, (E + U)/2].
    e_prime, a = E_EV + u, (E_EV / (E_EV + MC2_EV)) ** 2
    loss, dlw = _log_nodes(u, 0.5 * (E_EV + u), N_Q)
    ratio = loss / (e_prime - loss)
    mu_c = dlw * (1.0 + ratio**2 - (1.0 - a) * ratio + a * (loss / e_prime) ** 2) / loss
    qp_c = np.sqrt(np.maximum(loss * (loss + 2.0 * MC2_EV) - _qmin_sq(loss), 0.0)) / HBARC_EV_ANG
    h = d / N_FINE
    r = np.arange(N_FINE) * h
    smear = _mixture(
        r,
        u2,
        np.concatenate([qp_dl, qp_c]),
        np.concatenate([dl * mu_dl / mu_dl.sum(), close * mu_c / mu_c.sum()]),
        d,
    )
    # Slater density r^N e^{-2 zeta r}, N = 2n - 2, projected on the normal:
    # P(y) = a Gamma(N + 2, a|y|) / (2 Gamma(N + 3)), a = 2 zeta.
    zeta = np.sqrt(2.0 * u / HARTREE_EV) / BOHR_ANG
    big_n = 2 * {"K": 1, "L": 2, "M": 3}[osc.label[0]] - 2
    images = int(np.ceil(30.0 / zeta / d)) + 1
    shell = sum(
        zeta / (big_n + 2) * gammaincc(big_n + 2, 2.0 * zeta * np.abs(r - j * d))
        for j in range(-images, images + 1)
    )
    stride = N_FINE // x.size
    local = np.array(
        [h * shell @ smear[(i * stride - np.arange(N_FINE)) % N_FINE] for i in range(x.size)]
    )
    return local + tr / d  # distant transverse: uniform


def _row_weight(rho, kernel, z):
    factor = (rho * kernel).mean(1) / (rho.mean(1) * kernel.mean())
    return float(np.trapezoid(factor, z) / z[-1])


def _reference(profile, x, rho, z):
    d = profile.period_ang
    u2 = profile.b_factors_ang2["Si"] / (8.0 * np.pi**2)
    out = {fw.BREM_KEY: _row_weight(rho, _brem_kernel(x, d, u2), z)}
    oscillators = catalog_shell_oscillators("silicon")
    moments = shell_gos_moments(oscillators, E_EV)
    for i, osc in enumerate(oscillators.oscillators):
        if osc.atomic_number == Z_SI:
            kernel = _shell_kernel(
                x,
                d,
                u2,
                osc,
                moments.distant_longitudinal[i, 0],
                moments.distant_transverse[i, 0],
                moments.close[i, 0],
            )
            out[f"{Z_SI}:{osc.label}"] = _row_weight(rho, kernel, z)
    return out


def test_production_row_weights_match_the_real_space_reference(zone):
    profile, x, rho, z = zone
    kernels = fw.flux_kernels(profile, E_EV, 2 * H)
    factors = fw.flux_factors(kernels, x, rho)
    production = dict(
        zip(kernels.keys, fw.row_flux_weights(z, factors, np.array([z[-1]]))[0], strict=True)
    )
    reference = _reference(profile, x, rho, z)
    assert set(reference) == {fw.BREM_KEY, "14:K", "14:L1", "14:L2", "14:L3"}
    for key, value in reference.items():
        assert production[key] == pytest.approx(value, rel=RTOL), key
    # Recorded reference values (2026-11-13; re-pinned 2026-11-14 for the
    # measured Si bonding charge, #1177, which reshapes the zone density).
    assert reference[fw.BREM_KEY] == pytest.approx(1.3418340, abs=1e-6)
    assert reference["14:K"] == pytest.approx(1.1762400, abs=1e-6)
    assert reference["14:L1"] == pytest.approx(1.0570164, abs=1e-6)
    assert reference["14:L2"] == pytest.approx(1.0417924, abs=1e-6)
    assert reference["14:L3"] == pytest.approx(1.0418096, abs=1e-6)


def test_brem_reference_resolves_the_form_factor_fit_range_convention(zone):
    profile, x, rho, z = zone
    d = profile.period_ang
    u2 = profile.b_factors_ang2["Si"] / (8.0 * np.pi**2)
    cut = _row_weight(rho, _brem_kernel(x, d, u2), z)
    extrapolated = _row_weight(rho, _brem_kernel(x, d, u2, s_cut=None), z)
    # Extrapolating the f0 fit past s = 6 /A lowers W_brem by ~1.9e-4: 19x RTOL.
    assert extrapolated / cut - 1.0 == pytest.approx(-1.9e-4, abs=0.2e-4)
