"""BremsLib photon angular shape against the Schiff formula (Koch-Motz 2BS).

For every BremsLib catalogue element, every table incident energy in
``[T_MIN, T_MAX]`` and the reduced photon energies ``kappa = 0.1..0.8``, compares
the enclosed-flux angles ``theta_50`` and ``theta_90`` of the stored
double-differential cross section with Schiff's small-angle formula. See
``pyrite.validation.brem_angular``.

The gate is ``|ratio - 1| <= 0.06`` for 3 <= Z < 46 and 0.15 for Z >= 46, where
the Born-approximation Schiff formula omits the Coulomb distortion that the
partial-wave library includes. It is set from the measured envelope; it is a
shape-agreement gate on a Born-approximation reference, not a claim of
experimental accuracy. Z < 3 (hydrogen) is compared and reported, not gated:
Schiff's Thomas-Fermi screening is 4-5 % off for Z = 1, 2 (see
``pyrite.validation.brem_angular.gate_tolerance``). Above ``kappa = 0.8`` the
tip is reported, not gated. Every element's range is printed. Exit 0 pass,
1 fail, 2 skip (BremsLib tables not installed).

Validation: bremslib-angular-schiff

Run:
  PYRITE_MC_BACKEND=cpu uv run python checks/brem_angular_comparison.py [--output REPORT.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pyrite.materials._transport_data import TRANSPORT_ELEMENTS
from pyrite.validation.brem_angular import (
    HIGH_Z_MIN,
    KAPPA_MAX,
    KAPPA_MIN,
    QUANTILES,
    SCHIFF_MIN_Z,
    TOLERANCE_HIGH_Z,
    TOLERANCE_LOW_Z,
    compare_angular_shape,
    gate_tolerance,
)
from pyrite.xsgen._errors import TableNotFoundError
from pyrite.xsgen.bremslib import release
from pyrite.xsgen.bremslib import tables as bremslib_tables

T_MIN_KEV, T_MAX_KEV = 5000.0, 30000.0


def element_ranges(rows: list[dict]) -> dict[str, dict]:
    """Per-element ratio ranges: gated, reported below the Z floor, and the tip."""
    ranges = {}
    for element in dict.fromkeys(r["element"] for r in rows):
        mine = [r for r in rows if r["element"] == element]
        entry = {"Z": mine[0]["Z"]}
        groups = (
            ("gated", lambda r: r["gated"]),
            ("reported", lambda r: not r["gated"] and r["kappa"] <= KAPPA_MAX),
            ("tip", lambda r: r["kappa"] > KAPPA_MAX),
        )
        for label, keep in groups:
            v = [x for r in mine if keep(r) for x in r["ratio"].values()]
            if v:
                entry[label] = {"min": min(v), "max": max(v), "nodes": len(v) // len(QUANTILES)}
        ranges[element] = entry
    return ranges


def print_ranges(ranges: dict[str, dict]) -> None:
    """Print every element's ranges, the ungated low-Z block first."""
    ungated = [e for e, r in ranges.items() if r["Z"] < SCHIFF_MIN_Z]
    print(f"ungated (Z < {SCHIFF_MIN_Z}, reported only): {', '.join(ungated) or 'none'}")
    for element, entry in ranges.items():
        parts = [
            f"{label} {entry[label]['min']:.3f}..{entry[label]['max']:.3f} over {entry[label]['nodes']}"
            for label in ("gated", "reported", "tip")
            if label in entry
        ]
        print(f"{element:>3} Z={entry['Z']:<3} " + "; ".join(parts))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="write a JSON report")
    args = parser.parse_args(argv)

    by_z = {data["Z"]: symbol for symbol, data in TRANSPORT_ELEMENTS.items()}
    elements = [(by_z[z], z) for z in release.catalogue_elements()]
    try:
        tables = bremslib_tables.load_bremsstrahlung_tables([e for e, _ in elements])
    except TableNotFoundError as exc:
        print(f"SKIP: released BremsLib tables are not installed: {exc}")
        return 2

    rows, failures = [], []
    for element, Z in elements:
        table = tables.get(element)
        if table is None:
            continue
        for i, energy in enumerate(table.incident_energy_keV):
            # Staged energies carry round-off (4999.999..., 30000.00...4 keV).
            if not T_MIN_KEV * (1 - 1e-9) <= energy <= T_MAX_KEV * (1 + 1e-9):
                continue
            for j, kappa in enumerate(table.nominal_reduced_energy):
                if not 0.1 <= kappa <= 0.9:
                    continue
                ratios = compare_angular_shape(table, i, j)
                tolerance = gate_tolerance(Z, float(kappa))
                gated = tolerance is not None
                rows.append(
                    {
                        "element": element,
                        "Z": Z,
                        "T_keV": float(energy),
                        "kappa": float(kappa),
                        "gated": gated,
                        "ratio": {str(p): float(r) for p, r in ratios.items()},
                    }
                )
                if gated and any(abs(r - 1.0) > tolerance for r in ratios.values()):
                    failures.append((element, float(energy), float(kappa), ratios))
    ranges = element_ranges(rows)
    print_ranges(ranges)
    for group, label in ((True, "gated"), (False, f"reported (Z < {SCHIFF_MIN_Z} or kappa > 0.8)")):
        for p in QUANTILES:
            v = np.array([r["ratio"][str(p)] for r in rows if r["gated"] == group])
            if v.size:
                print(f"theta_{int(100 * p)} {label}: {v.min():.3f}..{v.max():.3f} over {v.size}")
    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "reference": "Koch-Motz 1959 Formula 2BS (Schiff)",
                    "gate": {
                        "T_keV": [T_MIN_KEV, T_MAX_KEV],
                        "kappa": [KAPPA_MIN, KAPPA_MAX],
                        "tolerance_low_z": TOLERANCE_LOW_Z,
                        "tolerance_high_z": TOLERANCE_HIGH_Z,
                        "high_z_min": HIGH_Z_MIN,
                        "min_z": SCHIFF_MIN_Z,
                    },
                    "ungated_elements": {e: r for e, r in ranges.items() if r["Z"] < SCHIFF_MIN_Z},
                    "element_ranges": ranges,
                    "comparisons": rows,
                    "failures": [[f[0], f[1], f[2]] for f in failures],
                },
                indent=1,
            )
        )
    for f in failures:
        print("FAIL", f)
    print("PASS" if not failures else f"FAIL: {len(failures)} nodes out of tolerance")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
