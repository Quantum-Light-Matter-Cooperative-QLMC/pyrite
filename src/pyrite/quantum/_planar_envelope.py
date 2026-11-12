"""Repository-owned planar admission table (#1105). Data and exact matching only.

Each entry is one measured planar case: its resolved physical fields, the
production zone numerics it runs at, and the paired-arm relative error ``epsilon``
measured for it (#1056, the #993 method on the carrier-fibred planar reference
``cr-planar-fiber-unraveling``). :func:`match_planar_case` looks an entry up
from resolved fields and returns it or refuses; nothing is interpolated or
extrapolated, and callers cannot extend the table.

``epsilon`` bounds the relative error of the band-weighted on-axis line source
(the forward-detector ``1->0``, ``2->1``, ``3->2`` lines) per incident electron
at crystal thickness, at ``joint_confidence``. It is measured on the fibre
reference with the production 300 eV margin; the production density-matrix
zone matched that reference to within 7% at 17 MeV (#1056). It bounds the band
integral only, not individual spectral bins.

Admission is a curated maintainer decision, not a threshold on ``epsilon``:
the measured Kephart configurations (epsilon 113-572%) are not admitted and run
only under the opt-in (#1105).

Validation: cr-planar-fiber-unraveling
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ._admission import PlanarAdmissionError, unadmitted_hint

_REL_TOL = 1e-9
#: Largest plane incidence [rad] treated as exact alignment (frame roundoff).
_ALIGNED_RAD = 1e-9


@dataclass(frozen=True)
class PlanarZoneNumerics:
    """Production zone numerics an entry was admitted at; part of the solver identity."""

    basis_size: int = 16
    bloch_cells: int = 16
    dz_ang: float = 250.0
    zone_max_ang: float | None = None


@dataclass(frozen=True)
class AdmittedPlanarCase:
    """One admitted planar case and the evidence that admits it.

    Energies are kinetic [eV]; ``thickness_ang`` is a closed range [angstrom];
    ``incidence_rad`` is the vacuum angle between beam and planes. Admitted
    cases are plane waves: no angular or energy spread.
    """

    key: str
    crystal: str
    plane_hkl: tuple[int, int, int]
    kinetic_energy_eV: float
    thickness_ang: tuple[float, float]
    incidence_rad: float
    temperature_K: float
    margin_eV: float
    numerics: PlanarZoneNumerics
    epsilon: float
    joint_confidence: float
    observable: str
    evidence_id: str
    evidence_record: str
    particle: str = "electron"


_RECORDS = "docs/validation/check-records/planar_fiber_runs/budget"

ADMITTED_PLANAR_CASES: Mapping[str, AdmittedPlanarCase] = MappingProxyType(
    {
        # The bundled channeling_demo profile: 17 MeV, Si (1-10), 2 um, beam in
        # the planes. Measured as the {110} family (identical planar potential).
        "si110-17MeV-2um": AdmittedPlanarCase(
            key="si110-17MeV-2um",
            crystal="silicon",
            plane_hkl=(1, -1, 0),
            kinetic_energy_eV=1.7e7,
            thickness_ang=(20000.0, 20000.0),
            incidence_rad=0.0,
            temperature_K=293.15,
            margin_eV=300.0,
            numerics=PlanarZoneNumerics(),
            epsilon=0.04862891554506509,
            joint_confidence=0.95,
            observable="line-source:((1, 0), (2, 1), (3, 2)):forward-detector",
            evidence_id="1056/demo_17@ce8ea207f",
            evidence_record=f"{_RECORDS}/report-demo_17.json",
        ),
    }
)


@dataclass(frozen=True)
class PlanarCaseQuery:
    """Resolved fields of a requested planar case, for exact lookup."""

    crystal: str
    plane_hkl: tuple[int, int, int]
    kinetic_energy_eV: float
    thickness_ang: float
    incidence_rad: float
    temperature_K: float
    margin_eV: float
    numerics: PlanarZoneNumerics
    plane_wave: bool
    particle: str = "electron"


def _same(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=_REL_TOL, abs_tol=0.0)


def _same_plane(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    return tuple(a) == tuple(b) or tuple(-v for v in a) == tuple(b)


def _mismatches(query: PlanarCaseQuery, case: AdmittedPlanarCase) -> list[str]:
    lo, hi = case.thickness_ang
    checks = {
        "crystal": query.crystal == case.crystal,
        "plane_hkl": _same_plane(query.plane_hkl, case.plane_hkl),
        "particle": query.particle == case.particle,
        "kinetic_energy_eV": _same(query.kinetic_energy_eV, case.kinetic_energy_eV),
        "thickness_ang": (lo * (1 - _REL_TOL) <= float(query.thickness_ang) <= hi * (1 + _REL_TOL)),
        "incidence_rad": abs(float(query.incidence_rad) - case.incidence_rad) <= _ALIGNED_RAD,
        "temperature_K": _same(query.temperature_K, case.temperature_K),
        "margin_eV": _same(query.margin_eV, case.margin_eV),
        "numerics": query.numerics == case.numerics,
        "beam spread": query.plane_wave,
    }
    return [name for name, ok in checks.items() if not ok]


def match_planar_case(query: PlanarCaseQuery) -> AdmittedPlanarCase:
    """Return the admitted case equal to ``query`` or refuse naming the opt-in.

    Floats match to a relative 1e-9 (unit round trips only); the refusal names
    the closest entry's mismatched fields.

    Validation: cr-planar-fiber-unraveling
    """
    if not isinstance(query, PlanarCaseQuery):
        raise TypeError("match_planar_case requires a PlanarCaseQuery")
    best: tuple[str, list[str]] | None = None
    for key, case in ADMITTED_PLANAR_CASES.items():
        missing = _mismatches(query, case)
        if not missing:
            return case
        if best is None or len(missing) < len(best[1]):
            best = (key, missing)
    detail = "" if best is None else f" (closest {best[0]} differs in {', '.join(best[1])})"
    raise PlanarAdmissionError(
        "planar channeling case is outside the measured admission table"
        + detail
        + unadmitted_hint()
    )
