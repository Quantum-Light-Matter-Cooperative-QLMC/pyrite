"""Strong-phase check of the TDS kernel for the sub-MeV W <001> line (#585). Local CPU.

Run with::

    uv run python checks/channeling_submev_w001_tds_phase.py --energies-kev 30 100 300

The TDS jumps of :func:`axial_dissipator` expand one displaced atom's
transmission ``exp(i dphi_u)`` to second order in the phase fluctuation
``dphi_u = phi_u - <phi>`` (``cr-axial-realspace-dissipator``). This record
keeps the full phase object instead, with the same projected potential and
Einstein displacements. Per atom of string ``s`` (``lambda_s`` per angstrom),
with ``M_n(u) = int |w_n|^2 exp(i dphi_u)`` over the flat-band string-orbital
density of level ``n``:

* coherent-wave loss of ``n``: ``G_n = -2 sum_s lambda_s ln |E_u M_n|``;
* population retained in ``n``: ``P_n = -sum_s lambda_s ln E_u |M_n|^2``;
* 2p -> 1s coherence decay: ``kappa = -sum_s lambda_s ln |E_u M_1s conj(M_2p)|``;
  its phase is a gap shift beyond the mean field,
  ``hbar v sum_s lambda_s arg E_u[M_1s conj(M_2p)]`` (``phi = -V / hbar v``).

Each atom acts once and independently (the Markov assumption of the
dissipator). To second order in ``dphi`` these are the dissipator's
``Gamma_n``, ``Gamma_n - J_nn`` and the TDS part of ``kappa`` in
``channeling_submev_w001_linewidth.py``; the record gives both, on the same
grid. The phase is band-limited to the grid, so the grid is refined until the
large-angle TDS sink is small against the in-grid loss; half the remaining sink
is added to ``kappa`` as pure loss.

Not included: the eikonal (projection) assumption itself, which multislice
would test; electronic excitation (unchanged, small); the 2p dispersion.
Energy-domain widths use ``hbar v kappa`` as in the linewidth record.
Diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import channeling_submev_w001_linewidth as lw  # noqa: E402
import channeling_submev_w001_occupation as occ  # noqa: E402

from pyrite.materials.channeling.axial import axial_profile  # noqa: E402
from pyrite.montecarlo.channeling import longitudinal_kinematics  # noqa: E402
from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator  # noqa: E402

CHUNK = 64


def displacement_quadrature(a2: float, n_gauss: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes ``(n, 2)`` and weights of the isotropic 2D Gaussian ``N(0, a2 I)``."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_gauss)
    unit = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), -1).reshape(-1, 2)
    return np.sqrt(a2) * unit, np.outer(weights, weights).ravel() / (2.0 * np.pi)


def phase_moments(dissipator, densities, n_gauss: int) -> dict[str, np.ndarray]:
    """Per-source phase-object and second-order moments of the level overlaps.

    Returns, per source ``s`` (rows) and level ``n`` (columns): ``mean_m``
    ``E_u M_n``, ``mean_m2`` ``E_u |M_n|^2``, ``cross`` ``E_u M_0 conj(M_n)``,
    and the second-order counterparts ``var`` ``E_u <dphi^2>_n``, ``lin``
    ``E_u <dphi>_0 <dphi>_n``; plus ``linear_density``.

    Validation: cr-submev-tds-phase-object
    """
    area = dissipator.grid.cell_area_ang2
    n_src, n_lev = len(dissipator.tds_sources), len(densities)
    out = {
        k: np.zeros((n_src, n_lev), dtype=complex if k in ("mean_m", "cross") else float)
        for k in ("mean_m", "mean_m2", "cross", "var", "lin")
    }
    out["linear_density"] = np.array([s.linear_density for s in dissipator.tds_sources])
    for s, src in enumerate(dissipator.tds_sources):
        nodes, weights = displacement_quadrature(src.displacement_ang2, n_gauss)
        for start in range(0, len(nodes), CHUNK):
            u, w = nodes[start : start + CHUNK], weights[start : start + CHUNK]
            dphi = dissipator.tds_phase(s, u)
            m = np.einsum("pij,nij->pn", np.exp(1j * dphi), densities) * area
            first = np.einsum("pij,nij->pn", dphi, densities) * area
            second = np.einsum("pij,nij->pn", dphi**2, densities) * area
            out["mean_m"][s] += w @ m
            out["mean_m2"][s] += w @ np.abs(m) ** 2
            out["cross"][s] += w @ (m[:, :1] * m.conj())
            out["var"][s] += w @ second
            out["lin"][s] += w @ (first[:, :1] * first)
    return out


def rates(moments) -> dict[str, np.ndarray]:
    """Phase-object and second-order rates [1/angstrom] per level (and per line).

    Validation: cr-submev-tds-phase-object
    """
    lam = moments["linear_density"][:, None]
    exact_coherent = -2.0 * np.sum(lam * np.log(np.abs(moments["mean_m"])), axis=0)
    exact_retained = -np.sum(lam * np.log(moments["mean_m2"]), axis=0)
    exact_kappa = -np.sum(lam * np.log(np.abs(moments["cross"])), axis=0)
    # Gap shift beyond the mean field: with phi = -V / hbar v, level n moves by
    # -hbar v lambda <phi>_n, so Delta = E_2p - E_1s moves by hbar v sum lambda arg(cross).
    exact_shift = np.sum(lam * np.angle(moments["cross"]), axis=0)
    var = np.sum(lam * moments["var"], axis=0)
    lin = np.sum(lam * moments["lin"], axis=0)
    return {
        "exact_coherent_loss": exact_coherent,
        "exact_population_loss": exact_retained,
        "exact_kappa": exact_kappa,  # column 0 is 1s with itself: its population loss
        "exact_gap_shift": exact_shift,  # [1/angstrom]; times hbar v gives eV
        "linear_loss": var,
        "linear_kappa": 0.5 * (var[0] + var) - lin,
        "linear_population_loss_1s": float(var[0] - lin[0]),  # lin[0] = E <dphi>_1s^2
    }


def screen_energy(kinetic_keV: float, cells: int, points: int, n_gauss: int) -> dict:
    kinetic_eV = 1e3 * kinetic_keV
    profile = axial_profile("tungsten", (0, 0, 1), occ.TEMPERATURE_K)
    gamma, beta = longitudinal_kinematics(kinetic_eV)
    catalog = occ.two_level_catalog(profile, kinetic_eV)
    (line,) = catalog.lines
    dissipator = axial_dissipator(
        profile, kinetic_eV, (cells, cells), (cells * points,) * 2, mechanisms=("tds",)
    )
    densities, _ = lw.string_densities(catalog, dissipator.grid, cells)
    densities = densities[:2]  # 1s and 2px (2py is its C4 image)
    losses = lw.loss_rates(dissipator, densities)
    out = rates(phase_moments(dissipator, densities, n_gauss))
    hbar_v = dissipator.hbar_v_eV_ang
    grid_loss = losses["tds_grid"]
    return {
        "kinetic_keV": kinetic_keV,
        "cells": cells,
        "points_per_cell": points,
        "gauss": n_gauss,
        "k0_gap_eV": line.delta_eV,
        "hbar_v_eV_ang": hbar_v,
        "beta": beta,
        "dissipator_tds_grid_loss": grid_loss.tolist(),
        "dissipator_tds_sink_loss": losses["tds_sink"].tolist(),
        "sink_fraction": (losses["tds_sink"] / (grid_loss + losses["tds_sink"])).tolist(),
        "quadrature_vs_grid_loss_rel": float(np.max(np.abs(out["linear_loss"] / grid_loss - 1))),
        "levels": ["1s", "2px"],
        "exact_coherent_loss_per_ang": out["exact_coherent_loss"].tolist(),
        "linear_loss_per_ang": out["linear_loss"].tolist(),
        "exact_population_loss_1s_per_ang": float(out["exact_population_loss"][0]),
        "linear_population_loss_1s_per_ang": out["linear_population_loss_1s"],
        "exact_kappa_per_ang": float(out["exact_kappa"][1]),
        "linear_kappa_per_ang": float(out["linear_kappa"][1]),
        "kappa_ratio_exact_over_linear": float(out["exact_kappa"][1] / out["linear_kappa"][1]),
        "exact_tds_fwhm_eV": float(2.0 * hbar_v * out["exact_kappa"][1]),
        "linear_tds_fwhm_eV": float(2.0 * hbar_v * out["linear_kappa"][1]),
        "exact_gap_shift_eV": float(hbar_v * out["exact_gap_shift"][1]),
        # The large-angle sink is pure loss outside the grid band: add half of it.
        "sink_half_loss_per_ang": float(0.5 * (losses["tds_sink"][0] + losses["tds_sink"][1])),
        "exact_tds_fwhm_with_sink_eV": float(
            2.0 * hbar_v * (out["exact_kappa"][1] + 0.5 * np.sum(losses["tds_sink"]))
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--energies-kev", type=float, nargs="+", default=[30.0, 100.0, 300.0])
    parser.add_argument("--cells", type=int, default=3)
    parser.add_argument("--points", type=int, nargs="+", default=[96], help="per cell per axis")
    parser.add_argument("--gauss", type=int, nargs="+", default=[24], help="nodes per axis")
    args = parser.parse_args(argv)
    rows = [
        screen_energy(e, args.cells, p, g)
        for e in args.energies_kev
        for p in args.points
        for g in args.gauss
    ]
    json.dump(
        {"meta": {"issue": 585, "material": "tungsten", "axis_uvw": [0, 0, 1]}, "rows": rows},
        sys.stdout,
        indent=1,
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
