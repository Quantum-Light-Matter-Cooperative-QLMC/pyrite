"""Optional ``Dans_Diffraction`` validation examples for crystallography.

The check builds P1 ``Dans_Diffraction`` crystals from the internal cxr_mc
structures, then compares lattice parameters, ``|g|``, and ``|F_hkl|^2`` for a
small material/reflection set. It is an independent-oracle smoke test, not a
production dependency and not a signed-off physics claim.

Run after installing the optional oracle package in the active environment:

    uv run python checks/dans_diffraction_oracle.py

Validation: dans-diffraction-oracle
"""

from __future__ import annotations

import os
import sys

from tabulate import tabulate

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from cxr_mc.validation_oracles import (  # noqa: E402
    DansDiffractionUnavailableError,
    build_dans_crystal_from_cxr,
    compare_lattice,
    compare_reflection_geometry,
    compare_structure_factor_magnitudes,
)

EXAMPLES = {
    "silicon": [(1, 1, 1), (2, 2, 0), (4, 0, 0)],
    "sapphire": [(0, 0, 6), (1, 1, 0), (1, 0, 4)],
    "mote2_product": [(0, 0, 2), (1, 0, 0), (1, 0, 3)],
}


def main() -> int:
    rows = []
    for crystal, hkls in EXAMPLES.items():
        try:
            oracle = build_dans_crystal_from_cxr(crystal)
        except DansDiffractionUnavailableError as exc:
            print(exc)
            print("Skipping optional Dans_Diffraction validation examples.")
            return 0

        lattice = compare_lattice(crystal, oracle)
        rows.append(
            [
                crystal,
                "cell",
                "-",
                abs(lattice.volume_delta_ang3),
                max(abs(delta) for delta in lattice.lattice_delta),
                "-",
            ]
        )

        for geometry in compare_reflection_geometry(crystal, oracle, hkls):
            rows.append(
                [
                    crystal,
                    "q",
                    geometry.hkl,
                    geometry.absolute_delta_inv_ang,
                    geometry.relative_delta,
                    "-",
                ]
            )

        for sf in compare_structure_factor_magnitudes(crystal, oracle, hkls, 8000.0):
            rows.append(
                [
                    crystal,
                    "|F|^2",
                    sf.hkl,
                    sf.absolute_delta,
                    sf.relative_delta,
                    f"{sf.cxr_abs_f_sq:.6g} / {sf.oracle_abs_f_sq:.6g}",
                ]
            )

    print(
        tabulate(
            rows,
            headers=["crystal", "check", "hkl", "abs delta", "rel delta", "cxr/oracle"],
            floatfmt=".6e",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
