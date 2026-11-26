"""Write PyRITE's per-EADL-subshell Compton-profile table (issue #1231).

Maintainer-only; needs the ``oracle`` dependency group (``xraylib==4.3.0``).
For every element Z=1--100 and every occupied EADL2025 subshell (the shell set
of :func:`pyrite.montecarlo.eadl_relaxation.load_eadl_relaxation`) it records
the one-electron Hartree--Fock Compton profile at zero projected momentum,
``J_i(0)`` in inverse atomic units of momentum, as tabulated by Biggs,
Mendelsohn & Mann, At. Data Nucl. Data Tables 16, 201 (1975) and distributed
by xraylib 4.3.0 (``ComptonProfile_Partial(Z, shell, 0)``; BSD-3-Clause, see
``THIRD-PARTY-NOTICES.md``). xraylib normalizes each partial profile to one
electron, the convention of PENELOPE's analytical profiles.

The ``source`` column states how each value was obtained:

* ``xraylib`` -- the same x-ray subshell label in xraylib;
* ``partner:<label>`` -- Biggs tabulates one nonrelativistic ``n,l`` orbital
  for lighter atoms, which xraylib stores under one ``j`` subshell only; the
  spin--orbit partner with the same ``n,l`` supplies the value;
* ``neighbor:Z=<z>`` -- the subshell is occupied in EADL but in no Biggs
  configuration of this element (only Bk 6d); the same subshell of the
  preceding element is used.

Binding energies and occupation numbers are not stored: the sampler reads
both from EADL2025, so a Compton vacancy uses EADL designators.

Usage::

    pixi run \\
        python scripts/generate_compton_profiles.py [--write]

Without ``--write`` the script checks that the packaged table is reproduced
byte for byte. Pin the printed SHA-256 in
``pyrite.montecarlo.transport.compton_profiles.COMPTON_PROFILE_SHA256``.

Validation: photon-compton-scattering
"""

import argparse
import hashlib
import sys
from pathlib import Path

TABLE = (
    Path(__file__).resolve().parents[1]
    / "src/pyrite/data/photon_cross_sections/compton_profile_j0.csv"
)

HEADER = """\
# One-electron Hartree-Fock Compton profiles at p_z = 0, J_i(0) [1/(atomic unit
# of momentum)], per occupied EADL2025 subshell, Z = 1-100.
# Source: F. Biggs, L. B. Mendelsohn, J. B. Mann, At. Data Nucl. Data Tables 16,
# 201-309 (1975), via xraylib 4.3.0 ComptonProfile_Partial (BSD-3-Clause;
# THIRD-PARTY-NOTICES.md). Generator: scripts/generate_compton_profiles.py.
# designator: EADL/ENDF-6 subshell designator (1=K, 2=L1, ...).
z,designator,label,j0_au,source
"""

_PARTNERS = {
    f"{shell}{a}": f"{shell}{b}"
    for shell, pairs in {
        "L": ((2, 3),),
        "M": ((2, 3), (4, 5)),
        "N": ((2, 3), (4, 5), (6, 7)),
        "O": ((2, 3), (4, 5), (6, 7)),
        "P": ((2, 3), (4, 5)),
    }.items()
    for low, high in pairs
    for a, b in ((low, high), (high, low))
}


def _partial(xraylib, z: int, label: str) -> float | None:
    shell = getattr(xraylib, f"{label}_SHELL", None)
    if shell is None:
        return None
    try:
        value = float(xraylib.ComptonProfile_Partial(z, shell, 0.0))
    except ValueError:
        return None
    return value if value > 0 else None


def _value(xraylib, z: int, label: str) -> tuple[float, str]:
    value = _partial(xraylib, z, label)
    if value is not None:
        return value, "xraylib"
    partner = _PARTNERS.get(label)
    if partner is not None and (value := _partial(xraylib, z, partner)) is not None:
        return value, f"partner:{partner}"
    for candidate in (label, partner):
        if candidate is not None and (value := _partial(xraylib, z - 1, candidate)) is not None:
            return value, f"neighbor:Z={z - 1}"
    raise ValueError(f"no Biggs profile for Z={z} subshell {label}")


def build() -> bytes:
    import xraylib

    from pyrite.montecarlo.eadl_relaxation import load_eadl_relaxation
    from pyrite.montecarlo.eedl_ionization import EEDL_SUBSHELL_LABELS

    rows = [HEADER]
    for z in range(1, 101):
        relaxation = load_eadl_relaxation(xraylib.AtomicNumberToSymbol(z))
        for designator in sorted(relaxation.shell_designators):
            label = EEDL_SUBSHELL_LABELS[designator]
            value, source = _value(xraylib, z, label)
            rows.append(f"{z},{designator},{label},{value:.6g},{source}\n")
    return "".join(rows).encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="overwrite the packaged table")
    args = parser.parse_args(argv)
    data = build()
    print(hashlib.sha256(data).hexdigest())
    if args.write:
        TABLE.write_bytes(data)
        return 0
    if TABLE.read_bytes() != data:
        print(f"{TABLE} differs from the regenerated table", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
