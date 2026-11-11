"""Finite-beta audit of the sub-MeV W <001> 2p -> 1s line (#585).

Run from the repository root with::

    PYRITE_MC_BACKEND=cpu OMP_NUM_THREADS=4 uv run python checks/channeling_submev_w001_kinematics.py

For a potential ``V`` independent of the axial coordinate ``z``, the
spin-free Klein--Gordon equation ``[(E - V)^2 - c^2 p^2 - m^2 c^4] psi = 0``
separates exactly. With ``psi = exp(i p_z z / hbar) phi`` and
``c^2 p_z^2 = E^2 - m^2 c^4 - 2 E eps``, the transverse equation is

    [c^2 p_perp^2 / (2E) + V - V^2 / (2E)] phi = eps phi.

The transverse mass ``E / c^2 = gamma m`` is therefore exact, not paraxial.
The production Hamiltonian keeps ``V`` and drops ``-V^2/(2E)``. Writing
``V = Vbar + U`` with the cell mean ``Vbar`` (mean inner potential) and
``U[0, 0] = 0``, the dropped term is a constant plus ``-U Vbar / E - U^2/(2E)``.
This check solves the k = 0 problem with ``U (1 - Vbar/E) - U^2/(2E)``,
taking ``Vbar`` from ``mean_inner_potential_eV`` (``cr-mean-inner-potential``).

The production line energy ``hbar w = d_eps / (1 - beta)`` (forward) is first
order in ``hbar w / E``. The exact forward root conserves total energy and
``p_z`` with ``eps`` re-solved at the final energy::

    c p_z(E, eps_2p(E)) - c p_z(E - hbar w, eps_1s(E - hbar w)) = hbar w.

The photon is along the axis, so the Bloch k is unchanged. No spin, no
longitudinal lattice harmonics, no dielectric response, k = 0 only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

import numpy as np
from scipy.signal import convolve2d

from pyrite.materials.channeling.axial import axial_profile, axial_states
from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.channeling import longitudinal_kinematics
from pyrite.montecarlo.channeling.entry import mean_inner_potential_eV

TEMPERATURE_K = 293.15
DEFAULT_ENERGIES_KEV = (30.0, 50.0, 100.0, 200.0, 300.0)


def _orders(profile, gamma: float, depth_eV: float, cutoff: float) -> tuple[int, int]:
    """Same well-momentum cutoff as the axial screen and entry audit."""
    q = cutoff * np.sqrt(2.0 * gamma * M_E_EV * depth_eV) / HBARC_EV_ANG
    return tuple(
        min(int(np.ceil(q * length / (2.0 * np.pi))), cap)
        for length, cap in zip(profile.period_ang, profile.max_order, strict=True)
    )


def kg_profile(profile, total_energy_eV: float, mean_eV: float = 0.0):
    """Return the profile with ``U (1 - Vbar/E) - U^2/(2E)`` on doubled orders.

    ``U^2`` coefficients are the exact linear convolution of the stored
    (already truncated) coefficients; its constant is removed so that
    ``U[0, 0] = 0`` still holds.
    """
    u = np.asarray(profile.potential_eV)
    square = convolve2d(u, u, mode="full")
    padded = np.zeros_like(square)
    m, n = (s // 2 for s in u.shape)
    padded[m : m + u.shape[0], n : n + u.shape[1]] = u
    centre = tuple(s // 2 for s in square.shape)
    effective = padded * (1.0 - mean_eV / total_energy_eV) - square / (2.0 * total_energy_eV)
    effective[centre] = 0.0
    order = tuple(2 * v for v in profile.max_order)
    return dataclasses.replace(profile, potential_eV=effective, max_order=order)


def pair_energies(
    profile, total_energy_eV: float, orders, *, kg: bool, mean_eV: float = 0.0
) -> tuple[float, float]:
    """Return ``(eps_1s, eps_2p)`` [eV] at k = 0 for total energy ``E``."""
    solved = kg_profile(profile, total_energy_eV, mean_eV) if kg else profile
    states = axial_states(solved, total_energy_eV / M_E_EV, max_order=orders)
    e = states.energies_eV
    if abs(e[2] - e[1]) > 1e-6 * abs(e[1] - e[0]):
        raise ValueError("2p doublet is not degenerate at k = 0")
    return float(e[0]), float(e[1])


def _cpz(total_energy_eV: float, eps_eV: float) -> float:
    return float(np.sqrt(total_energy_eV**2 - M_E_EV**2 - 2.0 * total_energy_eV * eps_eV))


def exact_forward_line(
    profile, total_energy_eV, orders, *, kg: bool, mean_eV: float = 0.0, tol_eV=1e-9
) -> float:
    """Solve the forward energy and p_z conservation root by secant."""
    upper = pair_energies(profile, total_energy_eV, orders, kg=kg, mean_eV=mean_eV)[1]
    initial = _cpz(total_energy_eV, upper)

    def residual(w: float) -> float:
        final = pair_energies(profile, total_energy_eV - w, orders, kg=kg, mean_eV=mean_eV)[0]
        return initial - _cpz(total_energy_eV - w, final) - w

    gamma = total_energy_eV / M_E_EV
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    lower, high = pair_energies(profile, total_energy_eV, orders, kg=kg, mean_eV=mean_eV)
    w0 = (high - lower) / (1.0 - beta)
    w1 = w0 * (1.0 + 1e-4)
    f0, f1 = residual(w0), residual(w1)
    for _ in range(50):
        if f1 == f0:
            break
        w0, w1 = w1, w1 - f1 * (w1 - w0) / (f1 - f0)
        f0, f1 = f1, residual(w1)
        if abs(f1) < tol_eV:
            return float(w1)
    raise RuntimeError("forward kinematic root did not converge")


def audit_energy(profile, kinetic_keV: float, cutoff: float = 2.0, mean_eV: float = 0.0) -> dict:
    """Compare production gap/line with the KG term and exact kinematics."""
    gamma, beta = longitudinal_kinematics(1e3 * kinetic_keV)
    total = gamma * M_E_EV
    lo, hi = profile.extrema()
    orders = _orders(profile, gamma, hi - lo, cutoff)
    lower, upper = pair_energies(profile, total, orders, kg=False)
    kg_lower, kg_upper = pair_energies(profile, total, orders, kg=True, mean_eV=mean_eV)
    gap, kg_gap = upper - lower, kg_upper - kg_lower
    doppler = gap / (1.0 - beta)
    exact = exact_forward_line(profile, total, orders, kg=False)
    exact_kg = exact_forward_line(profile, total, orders, kg=True, mean_eV=mean_eV)
    return {
        "kinetic_keV": kinetic_keV,
        "gamma": gamma,
        "beta": beta,
        "orders": list(orders),
        "well_depth_eV": hi - lo,
        "gap_eV": gap,
        "kg_gap_eV": kg_gap,
        "kg_gap_rel": (kg_gap - gap) / gap,
        "kg_level_shifts_eV": [kg_lower - lower, kg_upper - upper],
        "doppler_line_eV": doppler,
        "exact_line_eV": exact,
        "exact_line_rel": (exact - doppler) / doppler,
        "exact_kg_line_eV": exact_kg,
        "exact_kg_line_rel": (exact_kg - doppler) / doppler,
        "recoil_scale_hw_over_E": doppler / total,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--energies-kev", type=float, nargs="+", default=DEFAULT_ENERGIES_KEV)
    parser.add_argument("--cutoff", type=float, nargs="+", default=(2.0, 3.0))
    args = parser.parse_args(argv)
    if not np.all(np.isfinite(args.energies_kev)) or min(args.energies_kev) <= 0:
        parser.error("finite positive energies required")
    profile = axial_profile("tungsten", (0, 0, 1), TEMPERATURE_K)
    mean = mean_inner_potential_eV("tungsten")
    payload = {
        "meta": {
            "issue": 585,
            "material": "tungsten",
            "axis_uvw": [0, 0, 1],
            "temperature_K": TEMPERATURE_K,
            "k_inv_ang": [0.0, 0.0],
            "assumption": "spin-free KG, z-independent continuum potential, forward photon",
            "mean_inner_potential_eV": mean,
            "omitted": ["spin", "longitudinal harmonics", "dielectric response"],
        },
        "cases": [
            {f"cutoff_{c:g}": audit_energy(profile, t, c, mean) for c in args.cutoff}
            for t in args.energies_kev
        ],
    }
    json.dump(payload, fp=sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
