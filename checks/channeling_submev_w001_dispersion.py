"""Zone-resolved W <001> 2p -> 1s line record for sub-MeV axial CR (#585).

Run from the repository root with::

    PYRITE_MC_BACKEND=cpu OMP_NUM_THREADS=4 uv run python checks/channeling_submev_w001_dispersion.py

``axial_line_catalog`` uses ``k = 0`` levels and admits a level only when its
band spreads by less than 0.1 eV over the zone. Below 300 keV the W <001> 2p
band spreads by 1-3 eV, so the catalog refuses the line. Widening that gate
would assign the zone-centre energy to a dispersive band. This record instead
resolves the line over the Brillouin zone.

A forward photon has no transverse momentum, so the Bloch ``k`` is unchanged
and every transition is vertical in ``k``. At each zone node, band
``b in {1, 2}`` (the 2p doublet) emits to band 0 (1s) at the gap
``Delta_b(k)``, with the velocity-form dipole
``x_0b(k) = -i (hbar^2 / gamma m) <0|k + G|b> / (E_0 - E_b)``. The rate per
unit path is ``dipole_rate_per_ang`` (``cr-dipole-rate``).

Each band's weight is its static entry population: an incoherent, isotropic
2D Gaussian mixture of incident plane waves, ``P_b(k) = sum_G w(k + G)
|<G|b>|^2``. This is the zone integral of ``channeling_submev_w001_entry.py``.
Coherences between the two 2p bands are dropped. Where the bands split, they
emit at different frequencies and do not interfere in the spectrum. Where the
doublet is nearly degenerate, the record measures the omission:
``doublet_coherence_rel`` compares the dipole strength of the full entry
state with and without inter-band coherences. Midpoint nodes avoid the
degenerate zone centre, and the record gives the smallest and largest
doublet splittings on the grid.

From the weights ``P_b(k) * rate_b(k)`` over the zone, the record gives:

* the weighted forward vacuum line distribution (mean, rms width, 5/50/95 %
  quantiles, range) against the ``k = 0`` line that the catalog would give;
* the same mean shifted into the medium, ``Delta / [1 - beta (1 - delta)]``
  (``cr-submev-refractive-doppler``), with ``delta`` evaluated at the line;
* the total initial rate per unit path, a static entry quantity.

It compares two zone grids and two basis cutoffs. It omits occupation
evolution, TDS and electronic scattering, linewidths, capture, escape and
the finite-beta and spin terms (bounded separately). Nothing here is a yield,
a spectrum prediction or a support verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import channeling_submev_w001_entry as entry  # noqa: E402

from pyrite.materials.channeling.axial import axial_profile  # noqa: E402
from pyrite.materials.crystal import HBAR2_2M_EV_ANG2, HBARC_EV_ANG, M_E_EV  # noqa: E402
from pyrite.montecarlo.channeling import longitudinal_kinematics  # noqa: E402
from pyrite.montecarlo.channeling.axial_bloch import axial_states  # noqa: E402
from pyrite.montecarlo.channeling.dipole import dipole_rate_per_ang  # noqa: E402

DEFAULT_ENERGIES_KEV = (30.0, 50.0, 100.0, 200.0, 300.0)
DEFAULT_DIVERGENCE_FRACTIONS = (0.05, 0.1, 0.25)
QUANTILES = (0.05, 0.5, 0.95)
#: Medium-line fixed point, as in ``channeling_submev_optics.medium_line``.
MEDIUM_ITERATIONS = 50


def vertical_lines(states, gamma: float, bloch_k, g) -> tuple[np.ndarray, np.ndarray]:
    """Gaps ``E_b - E_0`` [eV] and ``|x_0b|`` [angstrom] for ``b = 1, 2`` at one ``k``."""
    kvec = np.asarray(bloch_k, dtype=float) + states.orders * np.asarray(g, dtype=float)
    c = states.coefficients
    momentum = np.einsum("g,ga,gb->ab", c[:, 0].conj(), kvec, c[:, 1:3])
    gaps = states.energies_eV[1:3] - states.energies_eV[0]
    dipole = (2.0 * HBAR2_2M_EV_ANG2 / gamma) * np.abs(momentum) / gaps
    return gaps, np.sqrt(np.sum(dipole**2, axis=0))


def doublet_coherence(states, gamma: float, bloch_k, g, sigmas_k, cell: float) -> np.ndarray:
    """Dipole strength from the entry state with and without 2p inter-band coherences.

    Returns ``(coherent, incoherent)``, each of shape ``(n_sigma,)``. Each
    incident order ``G`` puts amplitude ``<b|G>`` into band ``b``. The coherent
    form sums ``|sum_b x_0b <b|G>|^2``; the incoherent form sums
    ``sum_b |<b|G>|^2 |x_0b|^2``. Both are weighted by ``w(k + G)``.
    """
    kvec = np.asarray(bloch_k, dtype=float) + states.orders * np.asarray(g, dtype=float)
    c = states.coefficients
    momentum = np.einsum("g,ga,gb->ab", c[:, 0].conj(), kvec, c[:, 1:3])
    gaps = states.energies_eV[1:3] - states.energies_eV[0]
    x = (2.0 * HBAR2_2M_EV_ANG2 / gamma) * momentum / gaps
    weights = cell * entry._gaussian_density(np.sum(kvec**2, axis=1), sigmas_k)
    amplitude = np.einsum("ab,gb->ga", x, c[:, 1:3].conj())
    coherent = weights @ np.sum(np.abs(amplitude) ** 2, axis=1)
    incoherent = weights @ (np.abs(c[:, 1:3]) ** 2 @ np.sum(np.abs(x) ** 2, axis=0))
    return coherent, incoherent


def band_populations(states, bloch_k, g, sigmas_k, cell: float) -> np.ndarray:
    """Entry population of bands 1 and 2 at one zone node, shape ``(n_sigma, 2)``."""
    k2 = np.sum((np.asarray(bloch_k) + states.orders * np.asarray(g)) ** 2, axis=1)
    weights = cell * entry._gaussian_density(k2, sigmas_k)
    return weights @ (np.abs(states.coefficients[:, 1:3]) ** 2)


def weighted_summary(lines_eV: np.ndarray, weights: np.ndarray) -> dict:
    """Weighted mean, rms width, quantiles and range of a line distribution."""
    lines = np.ravel(lines_eV)
    w = np.ravel(weights)
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("line weights must have positive total")
    mean = float(np.sum(w * lines) / total)
    order = np.argsort(lines)
    cdf = np.cumsum(w[order]) / total
    quantiles = [float(lines[order][np.searchsorted(cdf, q)]) for q in QUANTILES]
    return {
        "mean_eV": mean,
        "rms_width_eV": float(np.sqrt(np.sum(w * (lines - mean) ** 2) / total)),
        "quantiles_eV": dict(zip((f"q{int(100 * q)}" for q in QUANTILES), quantiles, strict=True)),
        "range_eV": [float(lines.min()), float(lines.max())],
    }


def medium_mean(vacuum_eV: float, beta: float, delta_at) -> dict:
    """Forward line in the medium, ``delta`` at the shifted energy."""
    gap = vacuum_eV * (1.0 - beta)
    line = vacuum_eV
    for _ in range(MEDIUM_ITERATIONS):
        delta = float(delta_at(line))
        new = gap / (1.0 - beta * (1.0 - delta))
        if abs(new - line) < 1e-9 * line:
            return {"line_eV": new, "shift_eV": new - vacuum_eV, "delta_at_line": delta}
        line = new
    raise RuntimeError("medium line did not converge")


def zone_record(profile, gamma: float, beta: float, orders, sigmas_k, n_zone: int) -> dict:
    """Zone-resolved line distribution for every divergence."""
    if n_zone < 2 or n_zone % 2:
        raise ValueError("n_zone must be even so midpoint nodes avoid the zone centre")
    g = np.asarray(profile.g_inv_ang, dtype=float)
    cell = float(g[0] * g[1]) / n_zone**2
    nodes = (np.arange(n_zone) + 0.5) / n_zone - 0.5
    barrier = profile.extrema()[1]
    doppler = 1.0 - beta
    lines, rates, pops, splits = [], [], [], []
    coherent = np.zeros(len(sigmas_k))
    incoherent = np.zeros(len(sigmas_k))
    bound = True
    for fx in nodes:
        for fy in nodes:
            k = g * (fx, fy)
            states = axial_states(profile, gamma, tuple(k), max_order=orders)
            bound &= bool(np.all(states.energies_eV[:3] < barrier))
            gaps, dipole = vertical_lines(states, gamma, k, g)
            lines.append(gaps / doppler)
            rates.append(dipole_rate_per_ang(gaps, dipole, gamma))
            pops.append(band_populations(states, k, g, sigmas_k, cell))
            splits.append(float(abs(gaps[1] - gaps[0])))
            coh, inc = doublet_coherence(states, gamma, k, g, sigmas_k, cell)
            coherent += coh
            incoherent += inc
    lines_arr = np.asarray(lines)  # (nodes, 2)
    rates_arr = np.asarray(rates)
    pops_arr = np.asarray(pops)  # (nodes, sigma, 2)
    rows = []
    for i in range(pops_arr.shape[1]):
        weights = pops_arr[:, i, :] * rates_arr
        rows.append(
            {
                "population_2p": float(pops_arr[:, i, :].sum()),
                "initial_rate_per_ang": float(weights.sum()),
                "doublet_coherence_rel": float(coherent[i] / incoherent[i] - 1.0),
                "line": weighted_summary(lines_arr, weights),
            }
        )
    return {
        "n_zone": n_zone,
        "bands_0_to_2_bound_zone_wide": bound,
        "min_doublet_splitting_eV": float(min(splits)),
        "max_doublet_splitting_eV": float(max(splits)),
        "unweighted_line_range_eV": [float(lines_arr.min()), float(lines_arr.max())],
        "by_divergence": rows,
    }


def _compare(a: dict, b: dict) -> list[dict]:
    out = []
    for ra, rb in zip(a["by_divergence"], b["by_divergence"], strict=True):
        out.append(
            {
                "mean_eV": abs(ra["line"]["mean_eV"] - rb["line"]["mean_eV"]),
                "rms_width_eV": abs(ra["line"]["rms_width_eV"] - rb["line"]["rms_width_eV"]),
                "population_2p": abs(ra["population_2p"] - rb["population_2p"]),
                "initial_rate_rel": abs(
                    ra["initial_rate_per_ang"] / rb["initial_rate_per_ang"] - 1
                ),
            }
        )
    return out


def screen_energy(profile, kinetic_keV: float, fractions, n_zone: int, n_fine: int, delta_at):
    """Zone record at one energy, with zone-grid and cutoff comparisons."""
    gamma, beta = longitudinal_kinematics(1e3 * kinetic_keV)
    lo, barrier = profile.extrema()
    depth = barrier - lo
    orders = entry._orders(profile, gamma, depth, 2.0)
    fine_orders = entry._orders(profile, gamma, depth, 3.0)
    critical = float(np.sqrt(2.0 * depth / (gamma * beta**2 * M_E_EV)))
    p_inv_ang = gamma * beta * M_E_EV / HBARC_EV_ANG
    sigmas = p_inv_ang * critical * np.asarray(fractions, dtype=float)
    on_axis = axial_states(profile, gamma, max_order=orders)
    k0_line = float((on_axis.energies_eV[1] - on_axis.energies_eV[0]) / (1.0 - beta))
    coarse = zone_record(profile, gamma, beta, orders, sigmas, n_zone)
    fine = zone_record(profile, gamma, beta, orders, sigmas, n_fine)
    cutoff = zone_record(profile, gamma, beta, fine_orders, sigmas, n_zone)
    for fraction, row in zip(fractions, fine["by_divergence"], strict=True):
        row["divergence_over_critical"] = float(fraction)
        row["sigma_mrad"] = float(1e3 * fraction * critical)
        row["mean_minus_k0_line_eV"] = row["line"]["mean_eV"] - k0_line
        row["medium"] = medium_mean(row["line"]["mean_eV"], beta, delta_at)
    return {
        "kinetic_keV": kinetic_keV,
        "orders": list(orders),
        "refined_orders": list(fine_orders),
        "k0_line_eV": k0_line,
        "record": fine,
        "zone_grid_abs_delta": _compare(coarse, fine),
        "cutoff_abs_delta": _compare(coarse, cutoff),
    }


def _tungsten_delta(energy_eV: float) -> float:
    from pyrite.materials.attenuation import _resolve_composition
    from pyrite.materials.crystal import optical_constants

    return float(
        sum(
            optical_constants(el, energy_eV, rho)[0] for el, rho in _resolve_composition("tungsten")
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--energies-kev", type=float, nargs="+", default=DEFAULT_ENERGIES_KEV)
    parser.add_argument("--zone-nodes", type=int, default=12)
    parser.add_argument("--refined-zone-nodes", type=int, default=16)
    args = parser.parse_args(argv)
    if (
        not np.all(np.isfinite(args.energies_kev))
        or min(args.energies_kev) <= 0
        or args.zone_nodes < 2
        or args.zone_nodes % 2
        or args.refined_zone_nodes % 2
        or args.refined_zone_nodes <= args.zone_nodes
    ):
        parser.error("finite positive energies and even zone nodes, refined > coarse, required")
    profile = axial_profile("tungsten", (0, 0, 1), entry.TEMPERATURE_K)
    payload = {
        "meta": {
            "issue": 585,
            "material": "tungsten",
            "axis_uvw": [0, 0, 1],
            "temperature_K": entry.TEMPERATURE_K,
            "transition": "2p (bands 1, 2) -> 1s (band 0), vertical in k",
            "weights": "static entry population x dipole rate; no evolution",
            "cutoffs_over_well_momentum": [2.0, 3.0],
            "zone_midpoint_nodes_per_axis": [args.zone_nodes, args.refined_zone_nodes],
        },
        "cases": [
            screen_energy(
                profile,
                kinetic_keV,
                DEFAULT_DIVERGENCE_FRACTIONS,
                args.zone_nodes,
                args.refined_zone_nodes,
                _tungsten_delta,
            )
            for kinetic_keV in args.energies_kev
        ],
    }
    json.dump(payload, fp=sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
