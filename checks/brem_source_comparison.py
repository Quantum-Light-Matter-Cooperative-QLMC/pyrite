"""EEDL and BremsLib bremsstrahlung against the Seltzer--Berger tables.

For every BremsLib catalogue element and Seltzer--Berger incident-energy node
from 1 keV to 30 MeV, compares the scaled spectrum ``chi = (beta^2/Z^2) k
dsigma/dk`` pointwise on the table's ``kappa = k/T`` nodes, the hard-photon
cross section above ``kappa = 0.05``, and the radiative first moment. See
``pyrite.validation.brem_sources`` for the reduction.

Gate (claim ``brem-source-comparison``, BremsLib only) on the validated domain
``10 keV <= T <= 30 MeV``, ``0.05 <= kappa <= 0.95``:

* pointwise ``chi`` ratio within ``[1/(1 + 1/Z) - 0.05, 1.05]``. The lower
  bound allows the electron-electron share that Seltzer--Berger includes and
  BremsLib, an electron-atom library, does not. Below ``kappa = 0.05``
  screening raises that share towards ``xi/Z`` with ``xi > 1`` (B at 30 MeV
  reaches 0.70 at ``kappa -> 0``), and the tip ``kappa > 0.95`` is resolved
  differently by the two grids, so both edges are reported, not gated;
* first-moment ratio within ``[1/(1 + 1/Z) - 0.03, 1.03]``.

EEDL is reported, not gated: its fixed-photon-energy interpolation between
decade-spaced incident panels is a known defect (#174). Its values at the
EEDL panel nodes are listed separately as the data-versus-interpolation split.

Exit 0 pass, 1 fail, 2 skip (table or BremsLib release not installed).

Validation: brem-source-comparison

Run:
  PYRITE_MC_BACKEND=cpu uv run python checks/brem_source_comparison.py --download
  PYRITE_MC_BACKEND=cpu uv run python checks/brem_source_comparison.py --output REPORT.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pyrite.materials._transport_data import TRANSPORT_ELEMENTS
from pyrite.montecarlo.spectrum.brem import load_bremsstrahlung_cross_sections
from pyrite.validation.brem_sources import (
    SeltzerBergerUnavailableError,
    beta_squared,
    compare_sources,
    load_seltzer_berger,
)
from pyrite.xsgen._errors import TableNotFoundError
from pyrite.xsgen.bremslib import release
from pyrite.xsgen.bremslib import tables as bremslib_tables

T_MIN_MEV, T_MAX_MEV, KAPPA_MIN, KAPPA_MAX = 0.01, 30.0, 0.05, 0.95
POINTWISE_TOLERANCE, MOMENT_TOLERANCE = 0.05, 0.03


def _electron_electron_floor(atomic_number: int) -> float:
    return 1.0 / (1.0 + 1.0 / atomic_number)


def _eedl_panel_moments(table, elements):
    """EEDL first moment at its own panels: data quality without interpolation."""
    rows = []
    for element, Z in elements:
        eedl = load_bremsstrahlung_cross_sections(element)
        for index, energy_eV in enumerate(eedl.distribution_incident_energy_eV):
            energy = energy_eV / 1e6
            if not table.incident_energy_MeV[0] <= energy <= T_MAX_MEV:
                continue
            k = eedl.photon_energy_eV_by_incident[index]
            p = eedl.photon_probability_density_per_eV_by_incident[index]
            sigma = np.exp(
                np.interp(
                    np.log(energy_eV),
                    np.log(eedl.incident_energy_eV),
                    np.log(eedl.total_cross_section_cm2),
                )
            )
            moment = sigma * np.trapezoid(k * p, k) * 1e27  # eV mb
            reference = (
                np.trapezoid(table.chi(Z, energy), table.kappa)
                * Z**2
                / beta_squared(energy)
                * energy_eV
            )
            rows.append({"element": element, "T_MeV": energy, "moment_ratio": moment / reference})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true", help="fetch the pinned table")
    parser.add_argument("--table", type=Path, help="path to nist_brems.data")
    parser.add_argument("--output", type=Path, help="write a JSON report")
    args = parser.parse_args(argv)

    try:
        table = load_seltzer_berger(args.table, download=args.download)
    except SeltzerBergerUnavailableError as exc:
        print(f"SKIP: {exc}")
        return 2
    by_z = {data["Z"]: symbol for symbol, data in TRANSPORT_ELEMENTS.items()}
    elements = [(by_z[z], z) for z in release.catalogue_elements()]
    try:
        tables = bremslib_tables.load_bremsstrahlung_tables([e for e, _ in elements])
    except TableNotFoundError as exc:
        print(f"SKIP: released BremsLib tables are not installed: {exc}")
        return 2
    energies = [float(t) for t in table.incident_energy_MeV if 0.001 <= t <= T_MAX_MEV * (1 + 1e-9)]
    results = compare_sources(table, elements, energies, bremslib_tables=tables)

    failures = []
    gated_kappa = (table.kappa >= KAPPA_MIN) & (table.kappa <= KAPPA_MAX)
    for model in ("bremslib", "eedl"):
        print(f"\n== {model}: chi ratio envelope over {KAPPA_MIN} <= kappa <= {KAPPA_MAX}")
        print(f"{'el':>3} {'T[MeV]':>8} {'min':>7} {'max':>7} {'moment':>7} {'hard':>7}")
        for r in (r for r in results if r.model == model):
            ratio = r.chi_ratio[gated_kappa]
            print(
                f"{r.element:>3} {r.incident_energy_MeV:8.3f} {ratio.min():7.3f} "
                f"{ratio.max():7.3f} {r.first_moment_ratio:7.3f} "
                f"{r.hard_cross_section_ratio:7.3f}"
            )
            if model != "bremslib" or r.incident_energy_MeV < T_MIN_MEV * (1 - 1e-9):
                continue
            floor = _electron_electron_floor(r.atomic_number)
            if ratio.min() < floor - POINTWISE_TOLERANCE or ratio.max() > 1 + POINTWISE_TOLERANCE:
                failures.append((r.element, r.incident_energy_MeV, "chi", ratio.min(), ratio.max()))
            moment = r.first_moment_ratio
            if moment < floor - MOMENT_TOLERANCE or moment > 1 + MOMENT_TOLERANCE:
                failures.append((r.element, r.incident_energy_MeV, "moment", moment, moment))

    panel_rows = _eedl_panel_moments(table, elements)
    panel = np.array([row["moment_ratio"] for row in panel_rows])
    print(
        f"\nEEDL first moment at its own panels (no interpolation): "
        f"{panel.min():.3f}..{panel.max():.3f} over {panel.size} panels"
    )

    if args.output:
        report = {
            "reference": "Seltzer-Berger BREME.DAT via EGSnrc nist_brems.data",
            "gate": {
                "T_MeV": [T_MIN_MEV, T_MAX_MEV],
                "kappa": [KAPPA_MIN, KAPPA_MAX],
                "pointwise_tolerance": POINTWISE_TOLERANCE,
                "moment_tolerance": MOMENT_TOLERANCE,
            },
            "comparisons": [
                {
                    "model": r.model,
                    "element": r.element,
                    "T_MeV": r.incident_energy_MeV,
                    "kappa": r.kappa.tolist(),
                    "chi_ratio": r.chi_ratio.tolist(),
                    "hard_cross_section_ratio": r.hard_cross_section_ratio,
                    "first_moment_ratio": r.first_moment_ratio,
                }
                for r in results
            ],
            "eedl_panel_moments": panel_rows,
            "failures": [list(f) for f in failures],
        }
        args.output.write_text(json.dumps(report, indent=1))
    for failure in failures:
        print("FAIL", failure)
    print("PASS" if not failures else f"FAIL: {len(failures)} BremsLib gate violations")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
