"""Atomic shell occupations from the public SBETHE reference-data deposit."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..materials.atomic import Z_TABLE
from .eedl_ionization import EEDL_SUBSHELL_LABELS, load_eedl_shell_ionization

PDATCONF_SHA256 = "cd239554bb6e823692ea4611d443df8684b4cace06006fc271a4168cb78c62d2"


@dataclass(frozen=True, slots=True)
class AtomicShell:
    """One occupied free-atom subshell; energies are in eV."""

    atomic_number: int
    designator: int
    label: str
    orbital: str
    occupation: int
    ionization_energy_eV: float
    compton_profile_au: float
    eadl_width_eV: float
    campbell_papp_width_eV: float


@dataclass(frozen=True, slots=True)
class EEDLShellMatch:
    """EEDL subshells joined to one SBETHE shell, with signed binding differences.

    ``eedl_labels`` has more than one entry only when EEDL splits an ``n,l``
    shell into spin-orbit partners that SBETHE fills in one ``j`` subshell.
    ``binding_differences_eV`` is EEDL minus SBETHE for each EEDL label.
    """

    shell: AtomicShell
    eedl_labels: tuple[str, ...]
    eedl_binding_energies_eV: tuple[float, ...]
    binding_differences_eV: tuple[float, ...]


_PRINCIPAL = {"K": 1, "L": 2, "M": 3, "N": 4, "O": 5, "P": 6, "Q": 7}


def _nl(label: str) -> str:
    """Return ``n,l`` for an x-ray subshell label, e.g. ``M3`` -> ``3p``."""
    index = 1 if label == "K" else int(label[1:])
    return f"{_PRINCIPAL[label[0]]}{'spdfgh'[index // 2]}"


def _default_path() -> Path:
    from ..xsgen.sources import installed_data_dir

    return installed_data_dir("sbethe", "sdbase") / "pdatconf.p14"


def load_atomic_shells(path: str | Path | None = None) -> dict[int, tuple[AtomicShell, ...]]:
    """Read occupied shells from ``pdatconf.p14`` without bundling its data.

    The default fetched source is checksum-pinned. An explicit path permits
    inspection of alternate source revisions, including test fixtures. Each
    element must have unique shell designators and labels, labels consistent
    with their ``nlj`` orbitals, and occupations summing to Z.

    Source: SBETHE v2 deposit ``pdatconf.p14``; free-atom inputs, not a material
    conduction-band assignment. Limit: an empty or incomplete element fails.
    Validation: sbethe-atomic-shell-inputs
    """
    source = _default_path() if path is None else Path(path)
    if path is None and not source.is_file():
        from ..xsgen._errors import DataFetchError

        raise DataFetchError(
            f"SBETHE shell data is not installed ({source}); run `pyrite tables fetch sbethe`"
        )
    data = source.read_bytes()
    if path is None and hashlib.sha256(data).hexdigest() != PDATCONF_SHA256:
        raise ValueError(f"{source}: pdatconf.p14 checksum mismatch")
    grouped: dict[int, list[AtomicShell]] = {}
    for line_number, line in enumerate(data.decode("ascii").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(f"{source}:{line_number}: expected nine shell columns")
        try:
            z, designator, occupation = int(fields[0]), int(fields[1]), int(fields[4])
            energy, profile, eadl, campbell = (float(value) for value in fields[5:])
        except ValueError as exc:
            raise ValueError(f"{source}:{line_number}: invalid shell value") from exc
        if (
            not 1 <= z <= 99
            or designator < 1
            or occupation < 1
            or energy <= 0.0
            or profile < 0.0
            or eadl < 0.0
            or campbell < 0.0
            or not np.all(np.isfinite((energy, profile, eadl, campbell)))
        ):
            raise ValueError(f"{source}:{line_number}: invalid shell bounds")
        shell = AtomicShell(
            z, designator, fields[2], fields[3], occupation, energy, profile, eadl, campbell
        )
        grouped.setdefault(z, []).append(shell)
    if not grouped:
        raise ValueError(f"{source}: no atomic shell records")
    for z, shells in grouped.items():
        if len({shell.designator for shell in shells}) != len(shells):
            raise ValueError(f"{source}: duplicate shell designator for Z={z}")
        if len({shell.label for shell in shells}) != len(shells):
            raise ValueError(f"{source}: duplicate shell label for Z={z}")
        for shell in shells:
            try:
                consistent = _nl(shell.label) == shell.orbital[:2]
            except KeyError, ValueError, IndexError:
                consistent = False
            if not consistent:
                raise ValueError(
                    f"{source}: Z={z} shell {shell.label} disagrees with {shell.orbital}"
                )
        if sum(shell.occupation for shell in shells) != z:
            raise ValueError(f"{source}: shell occupations do not sum to Z={z}")
    return {z: tuple(shells) for z, shells in grouped.items()}


def match_eedl_shells(
    element: str, atomic_number: int, shells: tuple[AtomicShell, ...]
) -> tuple[EEDLShellMatch, ...]:
    """Join EEDL vacancy channels to SBETHE shells by x-ray label.

    EEDL designators follow ENDF-6 MF=23 (``MT - 533``, with O8/O9 slots),
    whereas ``pdatconf.p14`` numbers only occupied PENELOPE shells, so the
    join uses labels, never raw designators. An EEDL subshell without an
    equal label joins the unique SBETHE shell with the same ``n,l`` when
    SBETHE places that shell's electrons in one spin-orbit partner. Rates
    from both partners then describe one ``n,l`` vacancy; the partner label
    is retained. An EEDL shell whose ``n,l`` is absent from SBETHE raises.
    SBETHE shells without an EEDL channel receive no rate. The binding
    differences are diagnostic; this join does not define a transfer model.

    Validation: sbethe-atomic-shell-inputs
    """
    try:
        expected_number = Z_TABLE[element]
    except KeyError:
        raise ValueError(f"unknown element {element!r}") from None
    if expected_number != atomic_number:
        raise ValueError(f"{element}: atomic number does not match shell input")
    if any(shell.atomic_number != atomic_number for shell in shells):
        raise ValueError("atomic shell list contains another element")
    by_label = {shell.label: shell for shell in shells}
    joined: dict[str, list[tuple[str, float]]] = {}
    for eedl in load_eedl_shell_ionization(element):
        label = EEDL_SUBSHELL_LABELS[eedl.shell_designator]
        shell = by_label.get(label)
        if shell is None:
            partners = [s for s in shells if s.orbital[:2] == _nl(label)]
            if len(partners) != 1:
                raise ValueError(
                    f"{element}: EEDL shell {label} ({_nl(label)}) has no unique SBETHE n,l match"
                )
            shell = partners[0]
        joined.setdefault(shell.label, []).append((label, eedl.binding_energy_eV))
    matches = []
    for shell in shells:
        if shell.label not in joined:
            continue
        labels, energies = zip(*joined[shell.label], strict=True)
        matches.append(
            EEDLShellMatch(
                shell,
                labels,
                energies,
                tuple(energy - shell.ionization_energy_eV for energy in energies),
            )
        )
    return tuple(matches)
