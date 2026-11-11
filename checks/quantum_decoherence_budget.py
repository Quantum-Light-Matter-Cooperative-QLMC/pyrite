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

Cells: Si(110), diamond(110) and MoS2(001) at 30 keV, 300 keV, 3 MeV and
30 MeV. Si and diamond carry TDS and electronic rates; diamond uses its
sourced 34 eV valence oscillator in the production electronic kernel. MoS2
carries TDS only, from its uniaxial room-temperature thermal rows (Schoenfeld
et al. 1983; basal planes need no beam direction). Its electronic rate is
unsupported: ``decoherence_rate`` has only the scalar radial response, and the
directional MoS2 valence model is provisional (#541). Because every channel
adds ``Gamma >= 0``, its TDS-only lengths are upper bounds on ``1/Gamma`` and
its ``Gamma L_CR`` and diffusion values are lower bounds. The MoS2 rows hold
only at 295 +- 10 K, so other temperatures report MoS2 as not computable
instead of extrapolating. HOPG, hBN and WSe2 (001) lack channeling thermal data
and are listed as not computable. A supported cell that loses its data, or an
unsupported cell that gains it, fails the gate until the table is reviewed.

    PYRITE_MC_BACKEND=cpu uv run python checks/quantum_decoherence_budget.py

Exit 0 when the limits hold in every computed cell: ``Gamma(0) = 0``;
``Gamma_TDS(Delta -> inf)`` saturates (relative change below 1e-3 from 100 to
1000 angstrom); and ``beta^2 Gamma_TDS`` is energy independent (``C`` scales
as ``1 / v^2``) to 1e-6. Nonfinite rates or limit statistics fail. Diagnostic
formation lengths remain unavailable when fewer than two bound states exist.
About half a minute on one CPU.

Validation: quantum-decoherence-budget
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from pyrite.materials.channeling import (
    ChannelingDataError,
    UniaxialThermalSite,
    load_thermal_sites,
    planar_profile,
)
from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.channeling import (
    decoherence_rate,
    longitudinal_kinematics,
    transverse_states,
)

ENERGIES_EV = (3.0e4, 3.0e5, 3.0e6, 3.0e7)
# Reviewed material coverage (#1071): computed cells, and cells whose
# channeling data are absent. Moving a crystal between them needs review.
# The flag marks a supported electronic rate; without it only TDS is reported.
CELLS = (
    ("Si(110)", "silicon", (1, 1, 0), True),
    ("diamond(110)", "diamond", (1, 1, 0), True),
    ("MoS2(001)", "mos2", (0, 0, 1), False),
)
UNAVAILABLE_CELLS = (("hopg", (0, 0, 1)), ("hbn", (0, 0, 1)), ("wse2", (0, 0, 1)))


def _deltas(period: float) -> np.ndarray:
    return np.array([0.0, 0.03, 0.1, 0.3, 1.0, 0.5 * period, period, 3.0, 10.0, 100.0, 1000.0])


def _cr_formation_ang(profile, energy_eV: float) -> tuple[int, float]:
    gamma, beta = longitudinal_kinematics(energy_eV)
    states = transverse_states(profile, gamma, 0.0)
    if states.n_bound < 2:
        return states.n_bound, float("nan")
    delta = float(states.energies_eV[1] - states.energies_eV[0])
    return states.n_bound, HBARC_EV_ANG * beta / delta


def _row(label: str, profile, energy_eV: float, electronic: bool = True) -> dict:
    deltas = _deltas(profile.period_ang)
    rate = decoherence_rate(profile, energy_eV, deltas)
    n_bound, l_cr = _cr_formation_ang(profile, energy_eV)
    _, beta = longitudinal_kinematics(energy_eV)
    half, one = 5, 4  # indices of Delta = d/2 and 1 angstrom in _deltas
    total = rate.total if electronic else rate.tds
    return {
        "label": label,
        "energy_eV": energy_eV,
        "beta": beta,
        "deltas": deltas,
        "tds": rate.tds,
        "electronic": rate.electronic if electronic else None,
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
    if row["electronic"] is None:
        print("  G_el[1/A]  unsupported (scalar radial response; directional model provisional)")
        upper, lower = "<= ", ">= "
    else:
        print("  G_el[1/A]  " + " ".join(f"{v:9.3g}" for v in row["electronic"]))
        upper = lower = ""
    print(
        f"  n_bound(k=0) {row['n_bound']}; L_CR {row['l_cr_ang']:.4g} A; "
        f"L_PXR(1/g) {row['l_pxr_ang']:.3g} A; 1/G(d/2) {upper}{row['ell_half_ang']:.4g} A; "
        f"1/G(1 A) {upper}{row['ell_one_ang']:.4g} A; "
        f"2G/D^2 {lower}{row['diffusion_inv_ang3']:.3g} 1/A^3; "
        f"G(d/2) L_CR {lower}{row['gamma_l_cr']:.3g}"
    )


def _outside_thermal_window(crystal: str, temperature_K: float) -> str:
    """Return why a single-temperature thermal row excludes ``temperature_K``, or ``""``."""
    for (name, element), site in load_thermal_sites().items():
        if (
            name == crystal
            and isinstance(site, UniaxialThermalSite)
            and abs(temperature_K - site.temperature_K) > site.tolerance_K
        ):
            return (
                f"{element} thermal row holds only at "
                f"{site.temperature_K:g} +- {site.tolerance_K:g} K"
            )
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--temperature-K", type=float, default=293.15)
    args = parser.parse_args(argv)
    temperature = args.temperature_K

    failures: list[str] = []
    rows = []
    labels = []
    for label, crystal, hkl, electronic in CELLS:
        outside = _outside_thermal_window(crystal, temperature)
        if outside:
            print(f"\n{label}: not computable at {temperature:g} K ({outside})")
            continue
        try:
            profile = planar_profile(crystal, hkl, temperature)
        except ChannelingDataError as exc:
            failures.append(f"{label} lost its channeling data ({exc}); review the table")
            continue
        labels.append(label)
        for energy in ENERGIES_EV:
            rows.append(_row(label, profile, energy, electronic))
            _print_row(rows[-1])

    print("\nCells without channeling data:")
    for crystal, hkl in UNAVAILABLE_CELLS:
        try:
            planar_profile(crystal, hkl, temperature)
        except ChannelingDataError as exc:
            print(f"  {crystal}{hkl}: not computable ({exc})")
        else:
            failures.append(f"{crystal} unexpectedly has channeling data; add it to the table")

    for row in rows:
        electronic = row["electronic"] if row["electronic"] is not None else 0.0 * row["tds"]
        if not (
            np.all(np.isfinite(row["tds"]))
            and np.all(np.isfinite(electronic))
            and np.isfinite(row["beta"])
        ):
            failures.append(f"{row['label']} {row['energy_eV']:g}: nonfinite decoherence rates")
            continue
        scale = max(float(row["tds"][-1] + electronic[-1]), 1e-300)
        if abs(row["tds"][0]) + abs(electronic[0]) > 1e-12 * scale:
            failures.append(f"{row['label']} {row['energy_eV']:g}: Gamma(0) != 0")
        saturation = abs(row["tds"][-1] / row["tds"][-2] - 1.0)
        if not np.isfinite(saturation) or saturation > 1e-3:
            failures.append(f"{row['label']} {row['energy_eV']:g}: TDS rate does not saturate")
    for label in labels:
        scaled = [r["tds"][1:] * r["beta"] ** 2 for r in rows if r["label"] == label]
        spread = max(float(np.max(np.abs(s / scaled[0] - 1.0))) for s in scaled)
        print(f"\n{label}: max relative spread of beta^2 Gamma_TDS over energy {spread:.2e}")
        if not np.isfinite(spread) or spread > 1e-6:
            failures.append(f"{label}: beta^2 Gamma_TDS is energy dependent ({spread:.2e})")

    for failure in failures:
        print(f"FAIL: {failure}")
    print("PASS" if not failures else f"{len(failures)} failure(s)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
