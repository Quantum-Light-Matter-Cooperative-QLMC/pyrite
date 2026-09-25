"""EEDL shell ionization and characteristic production against Bote--Salvat.

For every catalogue transport element, compares the packaged EEDL MF=23
subshell cross sections with the Bote--Salvat formulas for each K/L/M shell
both tabulate, at overvoltages 1.1--1000, at the same incident energy and with
Bote--Salvat evaluated at EEDL's binding energy; repeats the comparison at
EEDL's own tabulation nodes to separate data from interpolation; and compares
characteristic production per line family (K, L1, L2, L3, M) at 10--300 keV and
as approximate thick-target yields for 30, 100 and 262 keV beams. See
``pyrite.validation.shell_ionization``.

Gate (claim ``eedl-shell-ionization-comparison``): the Bote--Salvat
transcription reproduces the ``xion.f`` reference values that
BoteSalvatICX.jl tests against within 1 %, and every EEDL-edge ratio agrees
with the ratios at its adjacent native EEDL nodes to 8 % (the Bote--Salvat
denominator curves between EEDL's sparse nodes), i.e. the reported
disagreement is in the EEDL data, not PyRITE's interpolation. The EEDL/Bote--Salvat disagreement
itself is recorded, not gated.

Exit 0 pass, 1 fail, 2 skip (parameters not installed).

Validation: eedl-shell-ionization-comparison

Run:
  PYRITE_MC_BACKEND=cpu uv run python checks/shell_ionization_comparison.py --download
  PYRITE_MC_BACKEND=cpu uv run python checks/shell_ionization_comparison.py --output REPORT.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pyrite.materials._transport_data import TRANSPORT_ELEMENTS
from pyrite.validation.shell_ionization import (
    LINE_FAMILIES,
    BoteSalvatUnavailableError,
    compare_production,
    compare_shells,
    load_bote_salvat,
)

OVERVOLTAGES = (1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 30.0, 100.0, 1000.0)
PRODUCTION_ENERGIES_EV = (10e3, 30e3, 100e3, 300e3)
THICK_TARGET_BEAMS_EV = (30e3, 100e3, 262.4e3)
# (Z, shell index 1=K..9=M5, energy eV, xion.f cross section cm^2), from
# BoteSalvatICX.jl test/xione.jl at the pinned commit.
XION_ANCHORS = (
    (12, 4, 7.71792e01, 3.85315e-18),
    (12, 4, 1.29569e04, 1.24124e-18),
    (12, 4, 5.78762e06, 1.30393e-19),
    (23, 2, 2.90068e08, 5.64142e-21),
    (23, 2, 1.53993e04, 2.63509e-20),
    (45, 7, 6.68344e08, 1.62343e-20),
    (45, 7, 1.22321e03, 2.11571e-19),
    (67, 1, 1.63117e05, 1.36404e-23),
    (78, 9, 2.98538e03, 2.02808e-20),
    (99, 3, 1.00000e09, 1.39024e-22),
    (99, 3, 2.66073e04, 3.85171e-24),
    (1, 1, 6.87860e02, 1.57729e-17),
)


def _display_ratio(value: float, family: str, energy_eV: float, k_edge_eV: float | None) -> str:
    if np.isfinite(value):
        return f"{value:5.2f}"
    if family == "K" and k_edge_eV is not None and energy_eV <= k_edge_eV:
        return "below K edge"
    return "undefined"


def _json_ratios(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true", help="fetch the pinned parameters")
    parser.add_argument("--table", type=Path, help="path to BoteSalvatICX.jl src/xione.jl")
    parser.add_argument("--output", type=Path, help="write a JSON report")
    args = parser.parse_args(argv)
    try:
        parameters = load_bote_salvat(args.table, download=args.download)
    except BoteSalvatUnavailableError as exc:
        print(f"SKIP: {exc}")
        return 2

    failures = []
    shells = ("K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5")
    anchor_error = max(
        abs(parameters[z].cross_section_cm2(shells[s - 1], energy) / value - 1.0)
        for z, s, energy, value in XION_ANCHORS
    )
    print(f"Bote-Salvat transcription vs xion.f: max relative error {anchor_error:.2e}")
    if anchor_error > 0.01:
        failures.append(("xion.f anchors", anchor_error))

    report = {"shells": [], "production": []}
    elements = sorted(TRANSPORT_ELEMENTS.items(), key=lambda item: item[1]["Z"])
    print(f"\nEEDL/Bote-Salvat at the EEDL binding energy; U = {OVERVOLTAGES}")
    for element, data in elements:
        for result in compare_shells(parameters, element, data["Z"], OVERVOLTAGES):
            own, native = result.eedl_edge_ratio, result.native_node_ratio
            print(
                f"{element:>2} {result.shell:>2} Eb {result.eedl_binding_eV:9.1f} "
                f"(BS {result.bote_salvat_edge_eV:9.1f}) "
                + " ".join(f"{value:5.2f}" for value in own)
            )
            neighboring = result.neighboring_node_ratio
            defined = np.all(np.isfinite(neighboring), axis=1)
            low = np.min(neighboring, axis=1)
            high = np.max(neighboring, axis=1)
            outside = defined & ((own < low * 0.92) | (own > high * 1.08))
            if np.any(outside):
                failures.append(
                    (element, result.shell, "interpolation", result.overvoltage[outside].tolist())
                )
            report["shells"].append(
                {
                    "element": element,
                    "shell": result.shell,
                    "eedl_binding_eV": result.eedl_binding_eV,
                    "bote_salvat_edge_eV": result.bote_salvat_edge_eV,
                    "overvoltage": result.overvoltage.tolist(),
                    "same_energy_ratio": result.same_energy_ratio.tolist(),
                    "eedl_edge_ratio": own.tolist(),
                    "neighboring_node_ratio": [_json_ratios(row) for row in neighboring],
                    "native_node_ratio_range": [float(native.min()), float(native.max())]
                    if native.size
                    else None,
                }
            )

    print(
        f"\nProduction EEDL/Bote-Salvat per line family {LINE_FAMILIES}: at "
        f"{[e / 1e3 for e in PRODUCTION_ENERGIES_EV]} keV | thick target "
        f"{[e / 1e3 for e in THICK_TARGET_BEAMS_EV]} keV"
    )
    for element, data in elements:
        result = compare_production(
            parameters, element, data["Z"], PRODUCTION_ENERGIES_EV, THICK_TARGET_BEAMS_EV
        )
        for family in LINE_FAMILIES:
            thin, thick = result.ratio[family], result.thick_target_ratio[family]
            if np.all(np.isnan(thin)) and np.all(np.isnan(thick)):
                continue
            print(
                f"{element:>2} {family:>2} "
                + " ".join(
                    _display_ratio(v, family, e, result.k_edge_eV)
                    for v, e in zip(thin, result.incident_energy_eV, strict=True)
                )
                + " | "
                + " ".join(
                    _display_ratio(v, family, e, result.k_edge_eV)
                    for v, e in zip(thick, result.thick_target_energy_eV, strict=True)
                )
            )
        report["production"].append(
            {
                "element": element,
                "incident_energy_eV": result.incident_energy_eV.tolist(),
                "ratio": {k: _json_ratios(v) for k, v in result.ratio.items()},
                "thick_target_energy_eV": result.thick_target_energy_eV.tolist(),
                "thick_target_ratio": {
                    k: _json_ratios(v) for k, v in result.thick_target_ratio.items()
                },
                "k_edge_eV": result.k_edge_eV,
            }
        )

    if args.output:
        report["xion_anchor_max_relative_error"] = anchor_error
        report["failures"] = [list(map(str, f)) for f in failures]
        args.output.write_text(json.dumps(report, indent=1, allow_nan=False))
    for failure in failures:
        print("FAIL", failure)
    print("PASS" if not failures else f"FAIL: {len(failures)} gate violations")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
