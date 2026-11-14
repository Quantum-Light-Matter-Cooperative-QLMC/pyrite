"""Measured low-order structure factors: catalog record, decoder, and bonding-charge corrections.

The independent-atom model (IAM) writes the unit-cell structure factor as a sum
of spherical free-atom form factors, ``F_IAM(G) = sum_j f0_j(s) exp(i G.R_j)``.
Covalent bonding moves valence charge into the bonds, which the IAM misses: it
changes the low-order reflections of Si and diamond by up to a few percent and
gives the IAM-extinct (222) reflection a nonzero value. A crystal record may
therefore carry measured structure factors. The correction added to every
structure-factor sum is the static (thermally unsmeared) difference

    dF(G) = F_meas(G) exp(+B_src s^2) - F_IAM,static(G),     s = |G| / (4 pi),

in electrons per unit cell, in the module-wide ``exp(+i G.R)`` convention.
Callers multiply ``dF`` by their own Debye--Waller factor ``exp(-B s^2)``, so
the bonding charge is smeared like the atoms it belongs to and the catalog's
own ``B`` (and temperature) stay in one place. Anomalous ``f'``/``f''`` stay
in the per-atom form factors: the measured values are corrected for anomalous
dispersion and nuclear scattering by their sources.

Convention ``dynamic-per-atom`` (Cummings--Hart, Takama et al., Lu--Zunger--
Deutsch): each tabulated ``F`` is the measured thermally smeared value per atom.
With ``P(G) = sum_j exp(i G.R_j)`` the basis geometric sum,

* ``P(G) != 0``: ``F_meas,cell(G) = F P(G)``; the bonding charge keeps the
  IAM phase, and ``dF = (F exp(B_src s^2) - f0(s)) P(G)``;
* ``P(G) = 0`` (IAM-extinct, e.g. (222)): ``F_meas,cell(G) = N F exp(i G.o)``
  with ``N`` the basis size and ``o`` the source's centrosymmetric origin, in
  whose frame ``F`` is real (positive ``F``: in phase with charge at ``o``).

Each tabulated reflection stands for its whole ``m-3m`` family (all signed
permutations), so the record is accepted for cubic single-element crystals only.

Limiting cases: with no record, or with ``F = f0(s) exp(-B_src s^2)``, every
correction is zero and the IAM is recovered exactly.

Validation: bonding-charge-structure-factors
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np

from ._catalog_decode import _direction, _Errors, _number, _table

_CONVENTIONS = frozenset({"dynamic-per-atom"})
_TEXT_KEYS = ("validation_id", "citation", "doi", "temperature")
_KEYS = {*_TEXT_KEYS, "zotero_item", "convention", "B_ang2", "origin", "reflections"}
_EXTINCT_ATOL = 1e-6


@dataclass(frozen=True)
class MeasuredReflection:
    """One tabulated measured reflection.

    Parameters
    ----------
    hkl
        Representative Miller indices of the reflection family.
    F
        Measured dynamic structure factor per atom [electrons].
    sigma
        Its estimated standard deviation [electrons].
    """

    hkl: tuple[int, int, int]
    F: float
    sigma: float


@dataclass(frozen=True)
class MeasuredStructureFactors:
    """Measured structure factors attached to a catalog crystal.

    Parameters
    ----------
    validation_id
        Validation-ledger identifier of the bonding-charge claim.
    citation, doi
        Source of the tabulated values.
    zotero_item
        Optional Zotero item key of the primary measurement.
    convention
        Tabulation convention; only ``"dynamic-per-atom"``.
    B_ang2
        Debye--Waller ``B`` [angstrom^2] the source's dynamic values carry.
    temperature
        Measurement temperature as stated by the source.
    origin
        Fractional position, in the CIF frame, of the source's centrosymmetric
        origin; fixes the phase of IAM-extinct reflections.
    reflections
        Measured reflections.
    """

    validation_id: str
    citation: str
    doi: str
    zotero_item: str | None
    convention: str
    B_ang2: float
    temperature: str
    origin: tuple[float, float, float]
    reflections: tuple[MeasuredReflection, ...]


def _text(row: Mapping[str, object], key: str, path: str, errors: _Errors) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.add(f"{path}.{key}", "must be a nonempty string")
        return ""
    return value.strip()


def _reflection(item: object, path: str, errors: _Errors) -> MeasuredReflection | None:
    row = _table(item, path, errors)
    if row is None:
        return None
    errors.keys(row, path, {"hkl", "F", "sigma"})
    hkl = _direction(row.get("hkl"), f"{path}.hkl", errors)
    F = _number(row.get("F"))
    sigma = _number(row.get("sigma"))
    if F is None:
        errors.add(f"{path}.F", "must be finite")
    if sigma is None or sigma <= 0:
        errors.add(f"{path}.sigma", "must be finite and positive")
    if hkl is None or F is None or sigma is None or sigma <= 0:
        return None
    return MeasuredReflection(hkl=hkl, F=F, sigma=sigma)


def _is_cubic(lattice: Mapping[str, str | float]) -> bool:
    lengths = [float(lattice[key]) for key in ("a", "b", "c")]
    angles = [float(lattice.get(key, 90.0)) for key in ("alpha", "beta", "gamma")]
    return bool(np.allclose(lengths, lengths[0]) and np.allclose(angles, 90.0))


def _origin(raw: object, path: str, errors: _Errors) -> tuple[float, float, float] | None:
    if isinstance(raw, list) and len(raw) == 3 and all(_number(v) is not None for v in raw):
        return cast(tuple[float, float, float], tuple(float(v) for v in raw))
    errors.add(path, "must be three finite fractional coordinates")
    return None


def _reflections(raw: object, path: str, errors: _Errors) -> tuple[MeasuredReflection, ...]:
    if not isinstance(raw, list) or not raw:
        errors.add(path, "must be a nonempty array")
        return ()
    out: list[MeasuredReflection] = []
    families: set[tuple[int, ...]] = set()
    for index, item in enumerate(raw):
        reflection = _reflection(item, f"{path}[{index}]", errors)
        if reflection is None:
            continue
        family = tuple(sorted(abs(v) for v in reflection.hkl))
        if family in families:
            errors.add(f"{path}[{index}]", "duplicate reflection family")
            continue
        families.add(family)
        out.append(reflection)
    return tuple(out)


def _check_cell(info, path: str, errors: _Errors) -> None:
    if info is None:
        return
    if not _is_cubic(info.lattice):
        errors.add(path, "requires a cubic cell (reflections stand for m-3m families)")
    if len({el for el, _ in info.basis}) != 1:
        errors.add(path, "requires a single-element crystal")


def parse_measured_structure_factors(
    raw: object, path: str, errors: _Errors, info
) -> MeasuredStructureFactors | None:
    """Decode an optional ``[measured_structure_factors]`` crystal table.

    ``info`` is the crystal's parsed :class:`CrystalInfo` (or ``None`` when the
    CIF failed); the record is refused for a non-cubic or multi-element cell.

    Validation: bonding-charge-structure-factors
    """
    if raw is None:
        return None
    row = _table(raw, path, errors)
    if row is None:
        return None
    errors.keys(row, path, _KEYS)
    texts = {key: _text(row, key, path, errors) for key in _TEXT_KEYS}
    zotero = row.get("zotero_item")
    if zotero is not None and (not isinstance(zotero, str) or not zotero.strip()):
        errors.add(f"{path}.zotero_item", "must be a nonempty string")
    convention = row.get("convention")
    if convention not in _CONVENTIONS:
        errors.add(f"{path}.convention", f"must be one of {sorted(_CONVENTIONS)}")
    B = _number(row.get("B_ang2"))
    if B is None or B < 0:
        errors.add(f"{path}.B_ang2", "must be finite and nonnegative")
    origin = _origin(row.get("origin"), f"{path}.origin", errors)
    reflections = _reflections(row.get("reflections"), f"{path}.reflections", errors)
    _check_cell(info, path, errors)
    if any(item.startswith((f"{path}: ", f"{path}.")) for item in errors.items):
        return None
    return MeasuredStructureFactors(
        validation_id=texts["validation_id"],
        citation=texts["citation"],
        doi=texts["doi"],
        zotero_item=cast(str, zotero).strip() if zotero is not None else None,
        convention=str(convention),
        B_ang2=cast(float, B),
        temperature=texts["temperature"],
        origin=cast(tuple[float, float, float], origin),
        reflections=reflections,
    )


def _family(hkl: tuple[int, int, int]) -> set[tuple[int, int, int]]:
    """All ``m-3m`` equivalents (signed permutations) of ``hkl``."""
    members: set[tuple[int, int, int]] = set()
    for perm in itertools.permutations(hkl):
        for sh, sk, sl in itertools.product((1, -1), repeat=3):
            members.add((sh * perm[0], sk * perm[1], sl * perm[2]))
    return members


def bonding_corrections(
    record: MeasuredStructureFactors, lattice, basis, *, g_of, f0_of
) -> Mapping[tuple[int, int, int], complex]:
    """Return static ``dF(G)`` [electrons/cell] for every member of every tabulated family.

    ``g_of(hkl)`` returns ``|G|`` [1/angstrom] and ``f0_of(element, g)`` the
    free-atom form factor, so this module needs no crystal-physics import.

    Validation: bonding-charge-structure-factors
    """
    element = basis[0][0]
    positions = np.array([r for _, r in basis], dtype=float)
    origin = np.asarray(record.origin, dtype=float)
    out: dict[tuple[int, int, int], complex] = {}
    for reflection in record.reflections:
        for hkl in sorted(_family(reflection.hkl)):
            h = np.asarray(hkl, dtype=float)
            g = float(g_of(hkl))
            s = g / (4.0 * math.pi)
            undo = math.exp(record.B_ang2 * s * s)
            P = complex(np.exp(2j * math.pi * positions @ h).sum())
            if abs(P) > _EXTINCT_ATOL:
                out[hkl] = (reflection.F * undo - float(f0_of(element, g))) * P
            else:
                phase = complex(np.exp(2j * math.pi * float(h @ origin)))
                out[hkl] = len(basis) * reflection.F * undo * phase
    return out
