"""Scattering width of the sub-MeV W <001> 2p -> 1s line (#585). Local CPU, ~1 min.

Run with::

    uv run python checks/channeling_submev_w001_linewidth.py --energies-kev 30 100 300

The radiating coherence ``sigma = <1s|rho|2p>`` obeys the secular equation of
``cr-line-shape``. In the flat-band limit, with ``|w_n|^2`` the on-column
orbital density of level ``n`` cut to one string cell, it decays at

    kappa = (Gamma_2p + Gamma_1s) / 2 - J_tds - Re J_el,
    Gamma_n = int |w_n|^2 C(r, r),

where ``C(r, r)`` is the full no-jump loss rate of :func:`axial_dissipator`
(in-grid TDS, large-angle TDS sink, electronic). The gains are the coherence
the jumps carry over to the same band pair (a kick that acts alike on ket and
bra leaves no which-path information):

* TDS, from real phase jumps ``L_u`` of every string with rate measure
  ``lambda_s p(u) d^2u``: ``J_tds = sum_s lambda_s E_u[<L_u>_2p <L_u>_1s]``,
  by Gauss--Hermite quadrature in ``u``;
* electronic, from ``L_q = sqrt(w rho_q) e^{iq.r}`` at the transfer nodes
  with a uniform azimuth: ``J_el = sum_q W_q avg_phi A_2p(q) conj(A_1s(q))``,
  ``A_n(q) = int |w_n|^2 sqrt(rho_q) e^{iq.(r - R)}``.

The sink only removes. By Cauchy--Schwarz each gain is at most
``sqrt(Gamma_2p Gamma_1s)`` of its mechanism. The transverse Lorentzian half
width is ``hbar v kappa``; the forward photon width is that divided by
``1 - beta``. The record also gives the population decay length of each level,
``1 / (Gamma_n - J_nn)``, to compare with the occupation run.

The kernels are the MeV-calibrated ones. The record reports the in-grid phase
variance one atom of its own string imprints on 1s (density-weighted
``Var_u phi / lambda``) and each level's full per-atom TDS loss ``V_n``
(in-grid plus sink, all strings). The jumps linearize ``exp(i phi)``, so a
variance near 1 rad^2 is outside the kernel's small-phase range. As a sensitivity estimate only (not a kernel), the record
also gives ``kappa`` with each level's TDS loss scaled by the Gaussian
phase-object saturation ``(1 - e^{-V_n}) / V_n``, ``V_n`` its TDS loss per
atom. Flat-band orbitals ignore the 2p dispersion (2.6 eV at
30 keV); that is small next to the widths found. Not a validated linewidth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import channeling_submev_w001_occupation as occ  # noqa: E402

from pyrite.materials.channeling.axial import axial_profile  # noqa: E402
from pyrite.montecarlo.channeling import longitudinal_kinematics  # noqa: E402
from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator  # noqa: E402

LEVELS = ("1s", "2px", "2py")
CELLS = 4
POINTS = 25


def string_densities(catalog, grid, cells: int) -> tuple[np.ndarray, tuple[float, float]]:
    """Normalized ``|w_n|^2`` of 1s and the 2p doublet about an interior string."""
    period = catalog.profile.period_ang
    sx, sy = catalog._string_ang
    middle = cells // 2
    centre = (middle * period[0] + sx, middle * period[1] + sy)
    one_s, two_p = catalog.orbitals(grid, centre)
    stack = np.stack([np.abs(one_s[0]) ** 2, np.abs(two_p[0]) ** 2, np.abs(two_p[1]) ** 2])
    return stack / (stack.sum(axis=(1, 2), keepdims=True) * grid.cell_area_ang2), centre


def loss_rates(dissipator, densities) -> dict[str, np.ndarray]:
    """``Gamma_n`` per mechanism [1/angstrom], one entry per level.

    Validation: cr-submev-axial-coherence-width
    """
    area = dissipator.grid.cell_area_ang2
    fields = {
        "tds_grid": dissipator.tds_grid_rate,
        "tds_sink": dissipator.tds_sink_rate,
        "electronic": dissipator.electronic_rate,
    }
    return {key: np.einsum("nij,ij->n", densities, f) * area for key, f in fields.items()}


def tds_gain(dissipator, densities, n_gauss: int = 20) -> tuple[np.ndarray, float]:
    """``J_mn = sum_s lambda_s E_u[<L_u>_m <L_u>_n]`` [1/angstrom] and a quadrature check.

    The check is the largest relative error of ``lambda E_u[L_u^2]`` against
    ``tds_source_rate`` for source 0.

    Validation: cr-submev-axial-coherence-width
    """
    area = dissipator.grid.cell_area_ang2
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_gauss)
    unit = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), -1).reshape(-1, 2)
    w2 = np.outer(weights, weights).ravel() / (2.0 * np.pi)
    gain = np.zeros((len(densities), len(densities)))
    check = 0.0
    for s, src in enumerate(dissipator.tds_sources):
        phases = dissipator.tds_phase(s, np.sqrt(src.displacement_ang2) * unit)
        means = np.einsum("pij,nij->pn", phases, densities) * area
        gain += src.linear_density * np.einsum("p,pm,pn->mn", w2, means, means)
        if s == 0:
            rate = dissipator.tds_source_rate(0)
            quadrature = src.linear_density * np.einsum("p,pij->ij", w2, phases**2)
            scale = np.max(rate)
            check = float(np.max(np.abs(quadrature - rate)) / scale) if scale > 0 else 0.0
    return gain, check


def electronic_gain(dissipator, densities, centre, n_azimuth: int = 16) -> tuple[np.ndarray, float]:
    """``J_el`` matrix [1/angstrom] and the loss-consistency check.

    The check is the largest relative error of ``sum_q W_q <rho_q>_n`` against
    ``<C_el>_n``: the gain uses the same nodes as the loss.

    Validation: cr-submev-axial-coherence-width
    """
    grid = dissipator.grid
    area = grid.cell_area_ang2
    x = grid.x[:, None] - centre[0]
    y = grid.y[None, :] - centre[1]
    angles = (np.arange(n_azimuth) + 0.5) * 2.0 * np.pi / n_azimuth
    n = len(densities)
    gain = np.zeros((n, n), dtype=complex)
    expected = np.zeros(n)
    for q, weight in zip(
        dissipator.electronic_q_inv_ang, dissipator.electronic_weights, strict=True
    ):
        rho = dissipator.smoothed_density(float(q))
        root = np.sqrt(rho)
        expected += weight * np.einsum("nij,ij->n", densities, rho) * area
        for phi in angles:
            wave = np.exp(1j * q * (np.cos(phi) * x + np.sin(phi) * y))
            amp = np.einsum("nij,ij->n", densities, root * wave) * area
            gain += weight / n_azimuth * np.outer(amp, amp.conj())
    loss = np.einsum("nij,ij->n", densities, dissipator.electronic_rate) * area
    check = float(np.max(np.abs(expected - loss) / loss))
    return gain, check


def coherence_decay(losses, gain_tds, gain_el, initial: int, final: int) -> dict:
    """Decay rate ``kappa`` [1/angstrom] of ``<final|rho|initial>`` and its parts.

    Validation: cr-submev-axial-coherence-width
    """
    total = sum(losses.values())
    half_loss = 0.5 * (total[initial] + total[final])
    j_tds = float(gain_tds[initial, final])
    j_el = float(np.real(gain_el[initial, final]))
    tds = losses["tds_grid"] + losses["tds_sink"]
    el = losses["electronic"]
    return {
        "kappa_per_ang": half_loss - j_tds - j_el,
        "half_loss_per_ang": half_loss,
        "gain_tds_per_ang": j_tds,
        "gain_electronic_per_ang": j_el,
        "gain_tds_bound": float(np.sqrt(tds[initial] * tds[final])),
        "gain_electronic_bound": float(np.sqrt(el[initial] * el[final])),
        "kappa_tds_only_per_ang": 0.5 * (tds[initial] + tds[final]) - j_tds,
    }


def screen_energy(kinetic_keV: float, n_gauss: int, n_azimuth: int) -> dict:
    kinetic_eV = 1e3 * kinetic_keV
    profile = axial_profile("tungsten", (0, 0, 1), occ.TEMPERATURE_K)
    gamma, beta = longitudinal_kinematics(kinetic_eV)
    catalog = occ.two_level_catalog(profile, kinetic_eV)
    (line,) = catalog.lines
    dissipator = axial_dissipator(
        profile, kinetic_eV, (CELLS, CELLS), (CELLS * POINTS,) * 2, mechanisms=("tds", "electronic")
    )
    densities, centre = string_densities(catalog, dissipator.grid, CELLS)
    losses = loss_rates(dissipator, densities)
    gain_tds, quad_check = tds_gain(dissipator, densities, n_gauss)
    gain_el, el_check = electronic_gain(dissipator, densities, centre, n_azimuth)
    hbar_v = dissipator.hbar_v_eV_ang
    central = CELLS // 2 * CELLS + CELLS // 2  # source index of the interior string
    lam = dissipator.tds_sources[central].linear_density
    phase_var = np.einsum("ij,ij->", densities[0], dissipator.tds_source_rate(central))
    phase_var *= dissipator.grid.cell_area_ang2 / lam
    total = sum(losses.values())
    tds_loss = losses["tds_grid"] + losses["tds_sink"]
    per_atom = tds_loss / lam
    scaled = dict(losses)
    factor = -np.expm1(-per_atom) / per_atom
    scaled["tds_grid"] = losses["tds_grid"] * factor
    scaled["tds_sink"] = losses["tds_sink"] * factor
    lines = {}
    for upper in (1, 2):
        decay = coherence_decay(losses, gain_tds, gain_el, upper, 0)
        saturated = coherence_decay(scaled, gain_tds, gain_el, upper, 0)["kappa_per_ang"]
        hwhm = hbar_v * decay["kappa_per_ang"]
        decay |= {
            "saturated_transverse_fwhm_eV": 2.0 * hbar_v * saturated,
            "transverse_hwhm_eV": hwhm,
            "transverse_fwhm_over_gap": 2.0 * hwhm / line.delta_eV,
            "forward_photon_fwhm_eV": 2.0 * hwhm / (1.0 - beta),
            "tds_only_transverse_fwhm_eV": 2.0 * hbar_v * decay["kappa_tds_only_per_ang"],
        }
        lines[f"{LEVELS[upper]}->1s"] = decay
    return {
        "kinetic_keV": kinetic_keV,
        "beta": beta,
        "hbar_v_eV_ang": hbar_v,
        "k0_gap_eV": line.delta_eV,
        "forward_k0_line_eV": float(catalog.photon_energy_eV(line)),
        "loss_per_ang": {k: v.tolist() for k, v in losses.items()},
        "population_decay_length_ang": dict(
            zip(
                LEVELS,
                (1.0 / (total - np.diag(gain_tds) - np.diag(gain_el).real)).tolist(),
                strict=True,
            )
        ),
        "lines": lines,
        "per_atom_tds_phase_variance_1s_rad2": float(phase_var),
        "per_atom_tds_loss": dict(zip(LEVELS, per_atom.tolist(), strict=True)),
        "checks": {"tds_quadrature_rel": quad_check, "electronic_nodes_rel": el_check},
        "grid": {"cells": CELLS, "points_per_cell": POINTS, "gauss": n_gauss, "azimuth": n_azimuth},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--energies-kev", type=float, nargs="+", default=[30.0, 100.0, 300.0])
    parser.add_argument("--gauss", type=int, default=20, help="Gauss-Hermite nodes per axis")
    parser.add_argument("--azimuth", type=int, default=16, help="electronic azimuth nodes")
    args = parser.parse_args(argv)
    record = {
        "meta": {"issue": 585, "material": "tungsten", "axis_uvw": [0, 0, 1]},
        "energies": [screen_energy(e, args.gauss, args.azimuth) for e in args.energies_kev],
    }
    json.dump(record, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
