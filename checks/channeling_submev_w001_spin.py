"""First-order Dirac spin correction to the sub-MeV W <001> 2p -> 1s line (#585).

Run from the repository root with::

    PYRITE_MC_BACKEND=cpu OMP_NUM_THREADS=4 uv run python checks/channeling_submev_w001_spin.py

The production transverse Hamiltonian is spin-free. This check bounds the
first-order Dirac correction to it. For a static potential energy ``V`` and
total energy ``E``, the positive-energy plane-wave spinor is
``u(p) = N (chi, c sigma.p chi / (E + m c^2))`` with
``N^2 = (E + m c^2) / (2 E)``. The first-order (Born) matrix element of ``V``
between two such spinors at the same ``E``, each on shell (``p_z`` set by
``|p|^2 = E^2/c^2 - m^2 c^2``), is

    V_q u(p')^dag u(p) = V_q N^2 [1 + c^2 (p'.p + i sigma.(p' x p)) / (E + m c^2)^2],

with ``q = (p' - p) / hbar``. Its spin-independent part, minus 1, is a
Darwin-type correction. Its spin part is the spin-orbit coupling. For
``p = p_z z + p_perp`` with ``V`` independent of ``z``, ``p' x p`` has a
longitudinal piece ``p_z (p'_perp - p_perp) x z`` and a transverse piece
``(p'_perp x p_perp)_z``. In position space the longitudinal piece is
``(hbar c beta / (2 (E + m c^2))) sigma.(grad V x z)``. That is the
Thomas--BMT precession energy of a spin moving at ``v`` in the field
``-grad V / e``, ``(hbar / 2 m c) (beta / (gamma + 1)) |grad V|``. The
non-relativistic limit is ``hbar / (4 m^2 c^2) sigma.(grad V x p)``.

The check solves the spin-free problem (``plain``), adds the Darwin-type
part (``darwin``), then adds the spin part (``spin``), at ``k = 0`` and
cutoffs 2 and 3 well momenta. ``plain`` reproduces ``axial_states``. At
``k = 0`` time reversal makes every level a Kramers doublet. The 1s doublet
and the four 2p spin states are identified by rank. The record gives level
shifts, the 2p fine-structure splitting and each 2p sub-level's gap change
against ``plain``. The on-shell form is the consistent first-order
correction beyond Klein--Gordon: eliminating the lower Dirac component gives
``c^2 [-i hbar grad V.p + hbar sigma.(grad V x p)] / (2 E (E + m c^2))``,
whose Hermitian part is this on-shell Darwin term and whose remainder is a
commutator with no first-order shift. A common ``p_z`` instead adds a spurious
``O(V^2 / E)`` term. The result is signed. It omits the ``-V^2/(2E)``
Klein--Gordon term (see ``channeling_submev_w001_kinematics.py``), relative
``O(V / (E + m c^2))`` corrections to the Darwin and spin terms, longitudinal
lattice harmonics and the dielectric response.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from pyrite.materials.channeling.axial import axial_profile
from pyrite.materials.crystal import HBAR2_2M_EV_ANG2, HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.channeling import line_energy_eV, longitudinal_kinematics
from pyrite.montecarlo.channeling.axial_bloch import axial_states

try:
    from checks.channeling_submev_w001_kinematics import _orders
except ModuleNotFoundError:  # direct ``python checks/...`` run
    from channeling_submev_w001_kinematics import _orders

TEMPERATURE_K = 293.15
DEFAULT_ENERGIES_KEV = (30.0, 50.0, 100.0, 200.0, 300.0, 500.0, 750.0, 1000.0)
CUTOFFS = (2.0, 3.0)
SIGMA = (
    np.array([[0, 1], [1, 0]], complex),
    np.array([[0, -1j], [1j, 0]], complex),
    np.array([[1, 0], [0, -1]], complex),
)


def basis(profile, gamma: float, orders):
    """Plane-wave orders, transverse wavevectors and the spin-free Hamiltonian."""
    mx, my = orders
    om, on = np.meshgrid(np.arange(-mx, mx + 1), np.arange(-my, my + 1), indexing="ij")
    ords = np.stack([om.ravel(), on.ravel()], 1)
    gx, gy = profile.g_inv_ang
    kvec = np.stack([ords[:, 0] * gx, ords[:, 1] * gy], 1)
    pm, pn = profile.max_order
    dm = ords[:, None, 0] - ords[None, :, 0]
    dn = ords[:, None, 1] - ords[None, :, 1]
    inside = (np.abs(dm) <= pm) & (np.abs(dn) <= pn)
    u = np.where(
        inside, profile.potential_eV[np.clip(dm + pm, 0, 2 * pm), np.clip(dn + pn, 0, 2 * pn)], 0.0
    )
    kinetic = HBAR2_2M_EV_ANG2 / gamma * np.sum(kvec**2, 1)
    return ords, kvec, u, u + np.diag(kinetic)


def dirac_terms(u, kvec, total_eV: float):
    """Darwin-type and spin-orbit matrices from ``V_q u(p')^dag u(p)``.

    Rows index ``p'``, columns ``p``. Returns ``(darwin, (sx, sy, sz))`` where
    the spin part is ``sum_a sigma_a * s_a``.

    Validation: cr-submev-dirac-spin-correction
    """
    rest = M_E_EV
    cp = HBARC_EV_ANG * kvec  # transverse c p [eV]
    # Each component is an on-shell spinor at E, so u(p)^dag u(p) = 1 exactly.
    cpz = np.sqrt(total_eV**2 - rest**2 - np.sum(cp**2, 1))
    n2 = (total_eV + rest) / (2.0 * total_eV)
    scale = n2 / (total_eV + rest) ** 2
    dot = cp @ cp.T + np.multiply.outer(cpz, cpz)
    darwin = u * (n2 * (1.0 + dot / (total_eV + rest) ** 2) - 1.0)
    # p' x p with p' = (cp'_x, cp'_y, cpz'), p = (cp_x, cp_y, cpz)
    px_r, py_r, pz_r = cp[:, None, 0], cp[:, None, 1], cpz[:, None]
    px_c, py_c, pz_c = cp[None, :, 0], cp[None, :, 1], cpz[None, :]
    cross_x = py_r * pz_c - pz_r * py_c
    cross_y = pz_r * px_c - px_r * pz_c
    cross_z = px_r * py_c - py_r * px_c
    spin = tuple(1j * u * scale * c for c in (cross_x, cross_y, cross_z))
    return darwin, spin


def solve(profile, gamma: float, total_eV: float, orders, variant: str) -> np.ndarray:
    _, kvec, u, h0 = basis(profile, gamma, orders)
    if variant == "plain":
        return np.linalg.eigvalsh(h0)
    darwin, spin = dirac_terms(u, kvec, total_eV)
    h = h0 + darwin
    if variant == "darwin":
        return np.linalg.eigvalsh(h)
    n = h.shape[0]
    full = np.kron(h, np.eye(2))
    for s, pauli in zip(spin, SIGMA, strict=True):
        full += np.kron(s, pauli)
    if not np.allclose(full, full.conj().T, atol=1e-10 * np.abs(full).max()):
        raise RuntimeError("spin Hamiltonian is not Hermitian")
    energies = np.linalg.eigvalsh(full)
    assert energies.size == 2 * n
    return energies


def levels(energies: np.ndarray, spinful: bool) -> dict:
    """1s and 2p energies; spinful spectra list every Kramers member."""
    if spinful:
        return {"1s": energies[:2].tolist(), "2p": energies[2:6].tolist()}
    return {"1s": [float(energies[0])], "2p": energies[1:3].tolist()}


def case(profile, kinetic_eV: float, cutoff: float) -> dict:
    lo, hi = profile.extrema()
    gamma, _ = longitudinal_kinematics(kinetic_eV)
    total = gamma * M_E_EV
    orders = _orders(profile, gamma, hi - lo, cutoff)
    t0 = time.perf_counter()
    plain = solve(profile, gamma, total, orders, "plain")
    reference = axial_states(profile, gamma, max_order=orders).energies_eV
    darwin = solve(profile, gamma, total, orders, "darwin")
    spin = solve(profile, gamma, total, orders, "spin")
    lp, ld, ls = levels(plain, False), levels(darwin, False), levels(spin, True)
    gap0 = np.mean(lp["2p"]) - lp["1s"][0]
    gap_d = np.mean(ld["2p"]) - ld["1s"][0]
    gaps_s = [e - np.mean(ls["1s"]) for e in ls["2p"]]
    return {
        "cutoff": cutoff,
        "orders": list(orders),
        "plain_vs_axial_states_eV": float(np.max(np.abs(plain[:6] - reference[:6]))),
        "plain_eV": lp,
        "darwin_shift_eV": {
            "1s": ld["1s"][0] - lp["1s"][0],
            "2p": float(np.mean(ld["2p"]) - np.mean(lp["2p"])),
        },
        "spin_eV": ls,
        "kramers_1s_split_eV": float(ls["1s"][1] - ls["1s"][0]),
        "fine_2p_split_eV": float(np.mean(ls["2p"][2:]) - np.mean(ls["2p"][:2])),
        "gap_plain_eV": float(gap0),
        "gap_change_darwin_rel": float(gap_d / gap0 - 1.0),
        "gap_change_spin_rel": [float(g / gap0 - 1.0) for g in gaps_s],
        "line_plain_eV": float(line_energy_eV(gap0, kinetic_eV)),
        "line_change_spin_eV": [
            float(line_energy_eV(g, kinetic_eV) - line_energy_eV(gap0, kinetic_eV)) for g in gaps_s
        ],
        "seconds": time.perf_counter() - t0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--energies-kev", nargs="+", type=float, default=list(DEFAULT_ENERGIES_KEV))
    parser.add_argument("--cutoffs", nargs="+", type=float, default=list(CUTOFFS))
    args = parser.parse_args(argv)
    profile = axial_profile("tungsten", (0, 0, 1), TEMPERATURE_K)
    rows = []
    for kev in args.energies_kev:
        rows.append(
            {
                "kinetic_keV": kev,
                "cases": [case(profile, 1e3 * kev, c) for c in args.cutoffs],
            }
        )
        print(f"{kev:g} keV done", file=sys.stderr, flush=True)
    meta = {"issue": 585, "axis": "W <001>", "temperature_K": TEMPERATURE_K, "k": [0.0, 0.0]}
    json.dump({"meta": meta, "energies": rows}, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
