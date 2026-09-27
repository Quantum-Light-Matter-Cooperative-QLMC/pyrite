"""Thresholded xraylib comparison for the EADL relaxation cascade (#91).

xraylib's ``CS_FluorShell_Kissel_Cascade`` and ``CS_FluorLine_Kissel_Cascade``
are an independent implementation of the full radiative and nonradiative
vacancy cascade through M5, with EADL97 nonradiative topology and Krause-based
fluorescence yields, Coster--Kronig probabilities, and radiative rates. Both
sides are fed the same primaries -- xraylib's own photoionization partials
``CS_Photo_Partial`` for K..M5 at one photon energy -- so the comparison
isolates relaxation from ionization.

Two quantities per case:

- vacancy enhancement ``n_i / P_i``, the expected vacancies that ever occupy
  subshell ``i`` over its primaries. This is the cascade topology test and is
  gated at 10% on K, L and Au M subshells.
- line cross section, PyRITE ``sum_i n_i F_il`` over xraylib's. This also
  carries the EADL-versus-Krause radiative-yield disagreement, so only K-alpha
  and Au L3 lines are gated, at 5%; the other lines are reported.

Run with the pinned optional dependency:

    uv run --group oracle python checks/xraylib_cascade_oracle.py

Missing xraylib returns 2; threshold violations return 1. This is
implementation-side external evidence, not a human physics sign-off.

Validation: characteristic-radiation
"""

import math

import numpy as np
from tabulate import tabulate

from pyrite.montecarlo.eadl_relaxation import load_eadl_relaxation, vacancy_cascade
from pyrite.montecarlo.spectrum.characteristic import _elam_fluorescence_yields

SHELLS = ("K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5")
LINES = (
    "KL2",
    "KL3",
    "KM2",
    "KM3",
    "L1M2",
    "L1M3",
    "L2M4",
    "L2N4",
    "L3M1",
    "L3M4",
    "L3M5",
    "L3N5",
    "M3N5",
    "M4N6",
    "M5N6",
    "M5N7",
)
# (element, photon energy keV, open primaries)
CASES = (
    ("Si", 3.0, "K+L"),
    ("Cu", 12.0, "K+L+M"),
    ("Cu", 5.0, "L+M"),
    ("Au", 100.0, "K+L+M"),
    ("Au", 15.0, "L+M"),
    ("Au", 12.5, "L3+M"),
    ("Au", 5.0, "M"),
)
CUTOFF_EV = 50.0
VACANCY_LIMIT = 0.10
LINE_LIMIT = 0.05
# Near-valence M subshells of Si and Cu are reported, not gated.
GATED_SHELLS = {"Si": SHELLS[:4], "Cu": SHELLS[:4], "Au": SHELLS}
GATED_LINES = {
    "Si": ("KL2", "KL3"),
    "Cu": ("KL2", "KL3"),
    "Au": ("KL2", "KL3", "L3M4", "L3M5", "L3N5"),
}


def _pyrite_cascade(element, primaries, fluorescence_yields):
    """Subshell vacancies and per-level-pair line cross sections from EADL."""
    relaxation = load_eadl_relaxation(element)
    yields = (
        _elam_fluorescence_yields(element, relaxation) if fluorescence_yields == "elam" else None
    )
    _daughters, visits = vacancy_cascade(relaxation, CUTOFF_EV, fluorescence_yields=yields)
    labels = relaxation.shell_labels
    label_of = dict(zip(relaxation.shell_designators, labels, strict=True))
    source = np.array([primaries.get(label, 0.0) for label in labels])
    vacancies = source @ visits
    lines: dict[str, float] = {}
    for row, shell in enumerate(relaxation.subshells):
        if shell.binding_energy_eV <= CUTOFF_EV:
            continue
        scale = 1.0
        if yields is not None and shell.shell_designator in yields:
            omega = shell.fluorescence_yield
            if 0.0 < omega and shell.auger_probability.sum() > 0.0:
                scale = yields[shell.shell_designator] / omega
        for final, probability in zip(
            shell.radiative_final, shell.radiative_probability, strict=True
        ):
            key = labels[row] + label_of[int(final)]
            lines[key] = lines.get(key, 0.0) + scale * probability * vacancies[row]
    return dict(zip(labels, vacancies, strict=True)), lines


def main() -> int:
    try:
        import xraylib as xrl
    except ImportError as exc:
        print(exc)
        print("RESULT: SKIP — rerun with `uv run --group oracle`.")
        return 2

    rows = []
    failures = []

    def record(case, quantity, reference, eadl, elam, gated, limit):
        deviation = eadl / reference - 1.0
        ok = math.isfinite(deviation) and abs(deviation) <= limit
        status = ("PASS" if ok else "FAIL") if gated else "report"
        rows.append([*case, quantity, reference, eadl, elam, eadl / reference, status])
        if gated and not ok:
            failures.append(f"{' '.join(case)} {quantity}: EADL/xraylib {eadl / reference:.4f}")

    for element, energy_keV, tag in CASES:
        Z = xrl.SymbolToAtomicNumber(element)
        case = (element, f"{energy_keV:g}", tag)
        primaries = {}
        for shell in SHELLS:
            try:
                primaries[shell] = xrl.CS_Photo_Partial(
                    Z, getattr(xrl, f"{shell}_SHELL"), energy_keV
                )
            except ValueError:
                primaries[shell] = 0.0
        eadl_n, eadl_lines = _pyrite_cascade(element, primaries, "eadl")
        elam_n, elam_lines = _pyrite_cascade(element, primaries, "elam")
        for shell in SHELLS:
            code = getattr(xrl, f"{shell}_SHELL")
            if primaries[shell] <= 0.0 or shell not in eadl_n:
                continue
            try:
                reference = xrl.CS_FluorShell_Kissel_Cascade(Z, code, energy_keV) / xrl.FluorYield(
                    Z, code
                )
            except ValueError:
                continue
            p = primaries[shell]
            record(
                case,
                f"n/P {shell}",
                reference / p,
                eadl_n[shell] / p,
                elam_n[shell] / p,
                shell in GATED_SHELLS[element],
                VACANCY_LIMIT,
            )
        for line in LINES:
            try:
                reference = xrl.CS_FluorLine_Kissel_Cascade(
                    Z, getattr(xrl, f"{line}_LINE"), energy_keV
                )
            except ValueError:
                continue
            if reference <= 0.0:
                continue
            record(
                case,
                f"line {line}",
                reference,
                eadl_lines.get(line, 0.0),
                elam_lines.get(line, 0.0),
                line in GATED_LINES[element],
                LINE_LIMIT,
            )

    print(
        tabulate(
            rows,
            headers=[
                "el",
                "E [keV]",
                "primaries",
                "quantity",
                "xraylib",
                "PyRITE eadl",
                "PyRITE elam",
                "eadl/xraylib",
                "status",
            ],
            floatfmt=".4g",
        )
    )
    print(f"\nLimits: vacancy enhancement {VACANCY_LIMIT:.0%}, gated lines {LINE_LIMIT:.0%}.")
    if failures:
        print("\nThreshold failures:")
        for failure in failures:
            print(f"- {failure}")
        print("RESULT: FAIL")
        return 1
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
