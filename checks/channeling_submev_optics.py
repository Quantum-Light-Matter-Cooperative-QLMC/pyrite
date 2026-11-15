"""Static optics bound for sub-MeV channeling-radiation candidate lines (#585).

Run with ``uv run python checks/channeling_submev_optics.py``. Prints JSON.

For each candidate bound->bound line recorded by the sub-MeV screens (planar
``channeling_submev_screen.py``, axial ``channeling_submev_axial_screen.py``
and ``channeling_submev_multistring.py``), this evaluates the medium at the
vacuum forward line energy, at both ends of the recorded range:

* Refractive index ``n = 1 - delta - i beta_idx`` from the Chantler
  isolated-atom factors through ``crystal.optical_constants``, and
  ``f1 / Z`` (1 in the free-electron limit).
* Absorption length ``L_abs = lambda / (4 pi beta_idx)``. The EPDL total
  narrow-beam attenuation length (``linear_attenuation_inv_mm``) is given as an
  independent cross-check of the absorptive part.
* The forward Doppler factor in the medium. The resonance condition
  ``omega (1 - beta_e n cos theta) = Delta_eps / hbar`` with ``n = 1 - delta``
  gives the line ``Delta_eps / (1 - beta_e (1 - delta))`` instead of the
  vacuum ``Delta_eps / (1 - beta_e)``. ``delta`` is evaluated at the shifted
  energy by fixed-point iteration. With ``x = beta_e delta / (1 - beta_e)``
  the relative shift is exactly ``-x / (1 + x)`` within first-order
  kinematics; the cap ``x <= 0.3`` limits reliance on the tables, not an
  expansion. No medium line is given below ``3 hbar omega_p``.
* Mean escape of a line emitted uniformly through a slab of thickness ``t``
  and leaving along the normal: ``T = (1 - exp(-t / L)) / (t / L)``.
* Flags: line below ``3 hbar omega_p``; within 3 % of an EPDL photoelectric
  edge (where isolated-atom tables miss solid-state edge structure); ``delta
  < 0`` (``n > 1``, Cherenkov-like phase matching possible); Chantler/EPDL
  ``L_abs`` disagreement above a factor 1.5.

The line energies are copied from those records (ranges over the zone,
cutoff 2 except where noted) so this check stays a fast optics record. It
uses isolated-atom tables in a homogeneous, isotropic medium. Not included:
crystal-field, plasmon or excitonic structure, surfaces, the transition
radiation background, angular apertures and detector response. No support
verdict. Ledger: ``cr-submev-refractive-doppler`` (research record).
"""

from __future__ import annotations

import json
import sys

import numpy as np

from pyrite.materials.atomic import Z_TABLE, henke_dispersion
from pyrite.materials.attenuation import (
    _resolve_composition,
    linear_attenuation_inv_mm,
    plasma_energy_eV,
)
from pyrite.materials.crystal import HC_EV_ANG, optical_constants
from pyrite.materials.photon_cross_sections import photoelectric_edges
from pyrite.montecarlo.channeling import longitudinal_kinematics

# (material, geometry, transition, kinetic keV, forward vacuum line range eV, source record)
LINES = (
    ("silicon", "(111)", "1->0", 30, (15.5, 19.0), "planar screen"),
    ("silicon", "(111)", "1->0", 100, (22.7, 29.6), "planar screen"),
    ("silicon", "(111)", "1->0", 200, (33.0, 45.0), "planar screen"),
    ("silicon", "(111)", "1->0", 500, (67.5, 78.3), "planar screen"),
    ("silicon", "(111)", "1->0", 1000, (133.2, 139.5), "planar screen"),
    ("diamond", "(111)", "1->0", 200, (45.8, 49.0), "planar screen"),
    ("diamond", "(111)", "1->0", 500, (99.2, 127.5), "planar screen"),
    ("diamond", "(111)", "1->0", 750, (154.4, 211.2), "planar screen"),
    ("diamond", "(111)", "1->0", 1000, (217.6, 271.8), "planar screen"),
    ("silicon", "<110>", "inter-string", 30, (6.9, 9.3), "multi-string, cutoff 1.5"),
    ("silicon", "<110>", "inter-string", 300, (10.0, 11.1), "multi-string, cutoff 1.5"),
    ("diamond", "<110>", "inter-string", 30, (12.2, 32.6), "multi-string, cutoff 1.5"),
    ("diamond", "<110>", "inter-string", 300, (33.8, 58.1), "multi-string, cutoff 1.5"),
    ("silicon", "<111>", "on-string", 1000, (440.1, 452.7), "multi-string, cutoff 1.5"),
    ("silicon", "<100>", "2->0", 1000, (342.0, 368.0), "axial screen"),
    ("tungsten", "<001>", "2p->1s", 30, (333.0, 337.0), "axial screen"),
    ("tungsten", "<001>", "2p->1s", 50, (387.0, 390.0), "axial screen"),
    ("tungsten", "<001>", "2p->1s", 100, (518.0, 520.0), "axial screen"),
    ("tungsten", "<001>", "2p->1s", 200, (800.0, 800.5), "axial screen"),
    ("tungsten", "<001>", "2p->1s", 300, (1119.0, 1120.0), "axial screen"),
    ("tungsten", "<001>", "2p->1s", 500, (1876.5, 1877.2), "axial screen"),
    ("tungsten", "<001>", "2p->1s", 750, (3035.0, 3036.0), "axial screen"),
    ("tungsten", "<001>", "2p->1s", 1000, (4412.0, 4413.0), "axial screen"),
    ("tungsten", "<111>", "2p->1s", 30, (404.5, 405.2), "multi-string, cutoff 1.5"),
    ("tungsten", "<111>", "2p->1s", 100, (620.4, 620.7), "multi-string, cutoff 1.5"),
    ("tungsten", "<111>", "2p->1s", 300, (1317.8, 1317.9), "multi-string, cutoff 1.5"),
)
SLAB_NM = (20.0, 100.0, 500.0, 2000.0)
PLASMA_MULTIPLE = 3.0
EDGE_FRACTION = 0.03
TABLE_RATIO = 1.5
ITERATIONS = 50
MAX_REFRACTION = 0.3


def medium(material: str, energy_eV: float) -> dict:
    """Index, ``f1 / Z`` and absorption lengths of a catalog medium at one energy."""
    comp = _resolve_composition(material)
    delta = beta = f1_sum = z_sum = 0.0
    for element, density in comp:
        d, b = optical_constants(element, energy_eV, density)
        delta += float(d)
        beta += float(b)
        f_prime, _ = henke_dispersion(element, energy_eV)
        f1_sum += (Z_TABLE[element] + float(f_prime)) * density
        z_sum += Z_TABLE[element] * density
    lam = HC_EV_ANG / energy_eV
    mu_epdl = float(linear_attenuation_inv_mm(material, np.array([energy_eV]))[0])
    return {
        "delta": delta,
        "beta_idx": beta,
        "f1_over_Z": f1_sum / z_sum,
        "L_abs_nm": lam / (4.0 * np.pi * beta) / 10.0,
        "L_abs_epdl_nm": 1e6 / mu_epdl,
    }


def medium_line(material: str, vacuum_line_eV: float, beta_e: float) -> dict | None:
    """Forward line in the medium with ``delta`` taken at the shifted energy.

    ``None`` where the weak-refraction premise fails: the refraction parameter
    exceeds ``MAX_REFRACTION`` at any iterate, or the iteration does not settle.

    Validation: cr-submev-refractive-doppler
    """
    gap = vacuum_line_eV * (1.0 - beta_e)
    line = vacuum_line_eV
    for _ in range(ITERATIONS):
        delta = medium(material, line)["delta"]
        if abs(beta_e * delta / (1.0 - beta_e)) > MAX_REFRACTION:
            return None
        new = gap / (1.0 - beta_e * (1.0 - delta))
        if abs(new - line) < 1e-9 * line:
            return {"line_eV": new, "shift_eV": new - vacuum_line_eV, "delta_at_line": delta}
        line = new
    return None


def escape(l_abs_nm: float) -> dict:
    """Mean normal-exit transmission of a line emitted uniformly in a slab.

    Validation: cr-submev-refractive-doppler
    """
    out = {}
    for t in SLAB_NM:
        x = t / l_abs_nm
        out[f"{t:g}nm"] = float(-np.expm1(-x) / x)
    return out


def nearest_edge(material: str, energy_eV: float) -> tuple[float, float]:
    edges = [e for el, _ in _resolve_composition(material) for e, _ in photoelectric_edges(el)]
    nearest = min(edges, key=lambda e: abs(e - energy_eV))
    return float(nearest), float(abs(energy_eV - nearest) / energy_eV)


def evaluate(row) -> dict:
    material, geometry, transition, kev, (lo, hi), source = row
    _, beta_e = longitudinal_kinematics(1e3 * kev)
    plasma = plasma_energy_eV(material)
    ends = []
    for vacuum in (lo, hi):
        opt = medium(material, vacuum)
        below_floor = vacuum < PLASMA_MULTIPLE * plasma
        # Below 3 hbar omega_p collective response dominates; no optics model.
        shifted = None if below_floor else medium_line(material, vacuum, beta_e)
        edge, distance = nearest_edge(material, vacuum)
        ratio = opt["L_abs_nm"] / opt["L_abs_epdl_nm"]
        ends.append(
            {
                "vacuum_line_eV": vacuum,
                **opt,
                "refraction_parameter": beta_e * opt["delta"] / (1.0 - beta_e),
                "medium_line": shifted,
                "escape": escape(opt["L_abs_nm"]),
                "nearest_edge_eV": edge,
                "flags": {
                    "below_3_plasma": below_floor,
                    "near_edge": distance < EDGE_FRACTION,
                    "n_above_1": opt["delta"] < 0.0,
                    "tables_disagree": not (1 / TABLE_RATIO < ratio < TABLE_RATIO),
                },
            }
        )
    return {
        "material": material,
        "geometry": geometry,
        "transition": transition,
        "kinetic_keV": kev,
        "beta_e": beta_e,
        "plasma_eV": plasma,
        "source": source,
        "ends": ends,
    }


def main() -> int:
    meta = {
        "issue": 585,
        "optics": "Chantler isolated-atom f1/f2 via crystal.optical_constants",
        "cross_check": "EPDL total narrow-beam attenuation",
        "slab_nm": list(SLAB_NM),
        "flags": {
            "plasma_multiple": PLASMA_MULTIPLE,
            "edge_fraction": EDGE_FRACTION,
            "table_ratio": TABLE_RATIO,
        },
    }
    json.dump({"meta": meta, "lines": [evaluate(r) for r in LINES]}, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
