"""Transverse decoherence budget Gamma(Delta x) against formation lengths (#501).

Tabulates the cell-averaged decoherence rate ``Gamma(Delta)`` of the #504
jump operators (``decoherence_rate``) for the cells of the #501 plan and
compares it with the lengths over which a quantum-coherent emission effect
must keep a transverse superposition alive:

- the channeling-radiation formation length ``L_CR = hbar v / Delta E_10``
  of the lowest bound-bound line at ``k = 0`` (when two bound states exist);
- the per-period formation length ``1 / g`` of a PXR/CBS line from the
  plane's first reciprocal vector, ``phase-matching omega (1 - beta cos theta)
  = v g_par``;
- the decoherence lengths ``1 / Gamma`` at ``Delta = d / 2`` and ``1``
  angstrom, and the momentum-diffusion coefficient ``2 Gamma / Delta^2`` at
  the smallest nonzero ``Delta`` (an effective, log-sensitive value; see the
  research note).

Cells: Si(110) at 30 keV, 300 keV, 3 MeV and 30 MeV (TDS and electronic);
diamond(110) at the same energies, using its sourced 34 eV valence oscillator
in the production electronic kernel; and the HOPG/vdW
catalog crystals at 30 keV, which have no channeling thermal data and are
reported as not computable.

    PYRITE_MC_BACKEND=cpu uv run python checks/quantum_decoherence_budget.py

Exit 0 when the limits hold: ``Gamma(0) = 0``; ``Gamma_TDS(Delta -> inf)``
saturates (relative change below 1e-3 from 100 to 1000 angstrom); and
``beta^2 Gamma_TDS`` is energy independent (``C`` scales as ``1 / v^2``) to
1e-6. A few seconds.

Validation: quantum-decoherence-budget
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from pyrite.materials.channeling import ChannelingDataError, planar_profile
from pyrite.materials.channeling.transverse import longitudinal_kinematics, transverse_states
from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.channeling import decoherence_rate

ENERGIES_EV = (3.0e4, 3.0e5, 3.0e6, 3.0e7)
VDW_CELLS = (("hopg", (0, 0, 1)), ("hbn", (0, 0, 1)), ("mos2", (0, 0, 1)), ("wse2", (0, 0, 1)))


def _deltas(period: float) -> np.ndarray:
    return np.array([0.0, 0.03, 0.1, 0.3, 1.0, 0.5 * period, period, 3.0, 10.0, 100.0, 1000.0])


def _cr_formation_ang(profile, energy_eV: float) -> tuple[int, float]:
    gamma, beta = longitudinal_kinematics(energy_eV)
    states = transverse_states(profile, gamma, 0.0)
    if states.n_bound < 2:
        return states.n_bound, float("nan")
    delta = float(states.energies_eV[1] - states.energies_eV[0])
    return states.n_bound, HBARC_EV_ANG * beta / delta


def _row(label: str, profile, energy_eV: float, oscillators=None) -> dict:
    deltas = _deltas(profile.period_ang)
    rate = decoherence_rate(profile, energy_eV, deltas, oscillators=oscillators)
    n_bound, l_cr = _cr_formation_ang(profile, energy_eV)
    _, beta = longitudinal_kinematics(energy_eV)
    half, one = 5, 4  # indices of Delta = d/2 and 1 angstrom in _deltas
    total = rate.total
    return {
        "label": label,
        "energy_eV": energy_eV,
        "beta": beta,
        "deltas": deltas,
        "tds": rate.tds,
        "electronic": rate.electronic,
        "n_bound": n_bound,
        "l_cr_ang": l_cr,
        "l_pxr_ang": 1.0 / profile.g_inv_ang,
        "ell_half_ang": 1.0 / total[half],
        "ell_one_ang": 1.0 / total[one],
        "diffusion_inv_ang3": 2.0 * total[1] / deltas[1] ** 2,
        "gamma_l_cr": total[half] * l_cr,
    }


def _print_row(row: dict, note: str = "") -> None:
    energy = row["energy_eV"]
    tag = f"{energy / 1e6:g} MeV" if energy >= 1e6 else f"{energy / 1e3:g} keV"
    print(f"\n{row['label']} {tag}{note}")
    print("  Delta[A]   " + " ".join(f"{d:9.3g}" for d in row["deltas"]))
    print("  G_tds[1/A] " + " ".join(f"{v:9.3g}" for v in row["tds"]))
    print("  G_el[1/A]  " + " ".join(f"{v:9.3g}" for v in row["electronic"]))
    print(
        f"  n_bound(k=0) {row['n_bound']}; L_CR {row['l_cr_ang']:.4g} A; "
        f"L_PXR(1/g) {row['l_pxr_ang']:.3g} A; 1/G(d/2) {row['ell_half_ang']:.4g} A; "
        f"1/G(1 A) {row['ell_one_ang']:.4g} A; 2G/D^2 {row['diffusion_inv_ang3']:.3g} 1/A^3; "
        f"G(d/2) L_CR {row['gamma_l_cr']:.3g}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--temperature-K", type=float, default=293.15)
    args = parser.parse_args(argv)
    temperature = args.temperature_K

    failures: list[str] = []
    rows = []
    silicon = planar_profile("silicon", (1, 1, 0), temperature)
    diamond = planar_profile("diamond", (1, 1, 0), temperature)
    for energy in ENERGIES_EV:
        rows.append(_row("Si(110)", silicon, energy))
        _print_row(rows[-1])
    for energy in ENERGIES_EV:
        rows.append(_row("diamond(110)", diamond, energy))
        _print_row(rows[-1])

    print("\nHOPG/vdW cells at 30 keV:")
    for crystal, hkl in VDW_CELLS:
        try:
            planar_profile(crystal, hkl, temperature)
        except ChannelingDataError as exc:
            print(f"  {crystal}{hkl}: not computable ({exc})")
        else:
            failures.append(f"{crystal} unexpectedly has channeling data; add it to the table")

    for row in rows:
        scale = max(float(row["tds"][-1] + row["electronic"][-1]), 1e-300)
        if abs(row["tds"][0]) + abs(row["electronic"][0]) > 1e-12 * scale:
            failures.append(f"{row['label']} {row['energy_eV']:g}: Gamma(0) != 0")
        if abs(row["tds"][-1] / row["tds"][-2] - 1.0) > 1e-3:
            failures.append(f"{row['label']} {row['energy_eV']:g}: TDS rate does not saturate")
    for label in ("Si(110)", "diamond(110)"):
        scaled = [r["tds"][1:] * r["beta"] ** 2 for r in rows if r["label"] == label]
        spread = max(float(np.max(np.abs(s / scaled[0] - 1.0))) for s in scaled)
        print(f"\n{label}: max relative spread of beta^2 Gamma_TDS over energy {spread:.2e}")
        if spread > 1e-6:
            failures.append(f"{label}: beta^2 Gamma_TDS is energy dependent ({spread:.2e})")

    for failure in failures:
        print(f"FAIL: {failure}")
    print("PASS" if not failures else f"{len(failures)} failure(s)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
