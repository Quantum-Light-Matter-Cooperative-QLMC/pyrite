"""Thresholded ``Dans_Diffraction`` validation for crystallography.

The check builds P1 ``Dans_Diffraction`` crystals from the internal pyrite
structures, then compares lattice parameters, ``|g|``, and ``|F_hkl|^2`` for a
small material/reflection set. Non-resonant factors compare identical
Waasmaier--Kirfel tables tightly. Dispersive factors compare PyRITE's
Chantler/FFAST data with independent Henke/CXRO data over 1--8 keV.

Run with the pinned optional dependency:

    uv run --group oracle python checks/dans_diffraction_oracle.py

Missing dependencies, non-finite results, or threshold violations return
nonzero. This remains an implementation-side external anchor, not a human
physics sign-off.

Validation: dans-diffraction-oracle
"""

import os
import sys

from tabulate import tabulate

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pyrite.validation.validation_oracles import (  # noqa: E402
    DEFAULT_DANS_TOLERANCES,
    DansDiffractionUnavailableError,
    validate_dans_crystal,
)

EXAMPLES = {
    "silicon": [(1, 1, 1), (2, 2, 0), (4, 0, 0)],
    "sapphire": [(0, 0, 6), (1, 1, 0), (1, 0, 4)],
    "mote2_product": [(0, 0, 2), (1, 0, 0), (1, 0, 3)],
}
DISPERSIVE_ENERGIES_EV = (1000.0, 2000.0, 3000.0, 8000.0)


def main() -> int:
    try:
        reports = [
            validate_dans_crystal(
                crystal,
                hkls,
                DISPERSIVE_ENERGIES_EV,
                use_henke=True,
            )
            for crystal, hkls in EXAMPLES.items()
        ]
        reports.append(
            validate_dans_crystal(
                "sapphire",
                EXAMPLES["sapphire"],
                (8000.0,),
                use_henke=False,
            )
        )
    except DansDiffractionUnavailableError as exc:
        print(exc)
        print("RESULT: SKIP — rerun with `uv run --group oracle`.")
        return 2

    limits = DEFAULT_DANS_TOLERANCES
    rows = []
    for report in reports:
        mode = "dispersive" if report.use_henke else "non-resonant"
        lattice = report.lattice
        rows.extend(
            [
                _row(
                    report.crystal,
                    mode,
                    "max |Δ length| [Å]",
                    "-",
                    max(abs(delta) for delta in lattice.lattice_delta[:3]),
                    limits.lattice_length_abs_ang,
                ),
                _row(
                    report.crystal,
                    mode,
                    "max |Δ angle| [deg]",
                    "-",
                    max(abs(delta) for delta in lattice.lattice_delta[3:]),
                    limits.lattice_angle_abs_deg,
                ),
                _row(
                    report.crystal,
                    mode,
                    "|Δ volume| [Å³]",
                    "-",
                    abs(lattice.volume_delta_ang3),
                    limits.volume_abs_ang3,
                ),
                _row(
                    report.crystal,
                    mode,
                    "max relative Δ|g|",
                    "-",
                    max(item.relative_delta for item in report.geometry),
                    limits.geometry_relative,
                ),
            ]
        )
        sf_limit = (
            limits.dispersive_structure_factor_relative
            if report.use_henke
            else limits.nonresonant_structure_factor_relative
        )
        energies = sorted({item.photon_E_eV for item in report.structure_factors})
        for energy_eV in energies:
            observed = max(
                item.relative_delta
                for item in report.structure_factors
                if item.photon_E_eV == energy_eV
            )
            rows.append(
                _row(
                    report.crystal,
                    mode,
                    "max relative Δ|F|²",
                    f"{energy_eV:g}",
                    observed,
                    sf_limit,
                )
            )

    print(
        tabulate(
            rows,
            headers=["crystal", "mode", "check", "energy [eV]", "observed", "limit", "status"],
            floatfmt=".6e",
        )
    )
    failures = [failure for report in reports for failure in report.failures]
    if failures:
        print("\nThreshold failures:")
        for failure in failures:
            print(f"- {failure}")
        print("RESULT: FAIL")
        return 1
    print("\nRESULT: PASS")
    return 0


def _row(crystal: str, mode: str, check: str, energy: str, observed: float, limit: float):
    status = "PASS" if observed <= limit else "FAIL"
    return [crystal, mode, check, energy, observed, limit, status]


if __name__ == "__main__":
    raise SystemExit(main())
