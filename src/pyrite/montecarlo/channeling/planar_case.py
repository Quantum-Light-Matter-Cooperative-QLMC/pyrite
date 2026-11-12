"""Resolve a lowered planar case into its admission query and record (#1105).

The plane incidence is the vacuum angle between the beam and the channeling
planes in the same slab frame as native transport
(:func:`~pyrite.montecarlo.channeling.handoff.case_plane_incidence`). A case
is admitted only when every resolved field equals one entry of
:data:`~pyrite.quantum._planar_envelope.ADMITTED_PLANAR_CASES`; otherwise it
runs only under ``channeling_admission="accept-inaccurate"`` and its record is
``{"status": "unadmitted"}`` with no error budget.

Validation: cr-planar-fiber-unraveling
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...quantum._admission import UNADMITTED, PlanarAdmissionError
from ...quantum._planar_envelope import (
    ADMITTED_PLANAR_CASES,
    AdmittedPlanarCase,
    PlanarCaseQuery,
    PlanarZoneNumerics,
    match_planar_case,
)

Case = Mapping[str, Any]

#: Case fields that give the beam an angular or energy spread, or replace the
#: plane-wave entrance; any of them makes a case a non-plane-wave query.
_SPREAD_FIELDS = ("transverse_distribution", "energy_spread_frac", "channeling_source")

#: The record of a case run under the opt-in: no case key, no epsilon.
UNADMITTED_RECORD: Mapping[str, object] = {"status": UNADMITTED}


def _plane_wave(case: Case) -> bool:
    return all(not case.get(field) for field in _SPREAD_FIELDS)


def planar_case_query(case: Case, zone: Mapping[str, Any]) -> PlanarCaseQuery:
    """Admission query of ``case`` for the planar zone record ``zone``.

    ``zone`` holds ``plane_hkl``, ``temperature_K``, ``margin_eV`` and the zone
    numerics (the keys of ``channeling_case_record``); ``case`` need not carry it.
    """
    from .handoff import case_plane_incidence

    incidence = case_plane_incidence({**dict(case), "channeling": zone})
    cap = zone["zone_max_ang"]
    h, k, ell = (int(v) for v in zone["plane_hkl"])
    return PlanarCaseQuery(
        crystal=str(case["crystal"]),
        plane_hkl=(h, k, ell),
        kinetic_energy_eV=float(case["E0_keV"]) * 1e3,
        thickness_ang=float(case["thickness_ang"]),
        incidence_rad=abs(float(incidence.geometric_rad)),
        temperature_K=float(zone["temperature_K"]),
        margin_eV=float(zone["margin_eV"]),
        numerics=PlanarZoneNumerics(
            basis_size=int(zone["basis_size"]),
            bloch_cells=int(zone["bloch_cells"]),
            dz_ang=float(zone["dz_ang"]),
            zone_max_ang=None if cap is None else float(cap),
        ),
        plane_wave=_plane_wave(case),
        particle=str(case.get("particle", "electron")),
    )


def planar_admission_record(admitted: AdmittedPlanarCase) -> dict[str, object]:
    """Freeze an admitted entry's key, epsilon and evidence into the zone record."""
    if not isinstance(admitted, AdmittedPlanarCase):
        raise TypeError("planar_admission_record requires an admitted case")
    return {
        "status": "admitted",
        "case_key": admitted.key,
        "epsilon": admitted.epsilon,
        "joint_confidence": admitted.joint_confidence,
        "observable": admitted.observable,
        "evidence_id": admitted.evidence_id,
        "evidence_record": admitted.evidence_record,
    }


def check_admission_record(record: object) -> None:
    """Refuse an admission record that is neither unadmitted nor a table entry's.

    Structural only: a record's key selects the entry whose record it must
    equal in full, so no epsilon or evidence a caller writes is accepted.
    :func:`require_planar_admission` re-resolves the physical case.
    """
    if not isinstance(record, Mapping):
        raise PlanarAdmissionError("channeling admission must be a record (#1105)")
    if dict(record) == dict(UNADMITTED_RECORD):
        return
    key = record.get("case_key")
    if not isinstance(key, str) or key not in ADMITTED_PLANAR_CASES:
        raise PlanarAdmissionError("planar admission names no repository table entry (#1105)")
    if dict(record) != planar_admission_record(ADMITTED_PLANAR_CASES[key]):
        raise PlanarAdmissionError("planar admission record differs from the table (#1105)")


def require_planar_admission(case: Case) -> AdmittedPlanarCase | None:
    """Re-resolve ``case`` and return its entry, or ``None`` when unadmitted.

    An admitted record must equal the lookup for the case's own fields; an
    unadmitted record must not match any entry (an admitted case never runs
    under the unadmitted marker).
    """
    zone = case["channeling"]
    record = zone["admission"]
    check_admission_record(record)
    query = planar_case_query(case, zone)
    try:
        admitted = match_planar_case(query)
    except PlanarAdmissionError:
        if dict(record) == dict(UNADMITTED_RECORD):
            return None
        raise
    if dict(record) != planar_admission_record(admitted):
        raise PlanarAdmissionError(
            f"case {case.get('name')!r} is admitted as {admitted.key} but its record differs"
        )
    return admitted
