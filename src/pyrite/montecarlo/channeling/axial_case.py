"""Resolve a lowered case into the exact axial admission query (#994).

The beam direction is taken into crystal coordinates with the same orientation
convention as planar channeling (:func:`beam_in_crystal`), then
expressed in the axial profile frame (``AxialProfile.axes`` rows): the tilt is
the angle between beam and axis, and its azimuth is measured from the
profile's first transverse axis toward its second. Admission compares these
resolved values with the repository lookup; no other frame is accepted.

Validation: cr-axial-persisted-channel
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import numpy as np

from ...quantum._admission import AxialAdmissionError
from ...quantum._axial_envelope import (
    ADMITTED_AXIAL_CASES,
    AXIAL_FIBRE_MODEL_MARKER,
    AdmittedAxialCase,
    AxialCaseQuery,
    match_axial_case,
)

Case = Mapping[str, Any]

#: Tilts below this [deg] are frame roundoff (profile axes carry ~1e-16 noise).
ALIGNED_TILT_DEG = 1e-9


def _rotation(case: Case) -> np.ndarray:
    """Crystal (construction) to slab-frame rotation of ``case``."""
    from ...materials.crystal import CRYSTALS, _orientation_R

    rotation = _orientation_R(
        CRYSTALS[case["crystal"]]["lattice"],
        case.get("beam_uvw"),
        case.get("azimuth_rad", 0.0),
        case.get("recip_miscut_rad"),
        surface_hkl=case.get("surface_hkl"),
    )
    return np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)


def beam_in_crystal(case: Case) -> np.ndarray:
    """Unit beam direction in crystal (construction) coordinates."""
    from ..geometry import tilted_geometry

    beam, _ = tilted_geometry(
        0.0, np.deg2rad(case.get("tilt_deg", 0.0)), np.deg2rad(case.get("tilt_azim_deg", 0.0))
    )
    return _rotation(case).T @ np.asarray(beam, dtype=float)


def axial_beam_angles(case: Case, axis_uvw: tuple[int, int, int]) -> tuple[float, float]:
    """Beam tilt from ``axis_uvw`` and its profile-frame azimuth [deg].

    Tilts below :data:`ALIGNED_TILT_DEG` resolve to exact alignment with zero
    azimuth. Refuses a beam travelling against the axis direction.
    """
    from ...materials.channeling.axial import axial_profile

    axes = np.asarray(axial_profile(case["crystal"], axis_uvw).axes, dtype=float)
    x, y, z = axes @ beam_in_crystal(case)
    if z <= 0.0:
        raise ValueError(f"case {case['name']!r}: the beam travels against axis {list(axis_uvw)}")
    tilt = math.degrees(math.atan2(math.hypot(x, y), z))
    if tilt < ALIGNED_TILT_DEG:
        return 0.0, 0.0
    return tilt, math.degrees(math.atan2(y, x))


def axial_case_query(case: Case, axis_uvw: tuple[int, int, int], temperature_K: float):
    """Admission query of one lowered case for the axial fibre model."""
    tilt, azimuth = axial_beam_angles(case, axis_uvw)
    return AxialCaseQuery(
        crystal=str(case["crystal"]),
        axis_uvw=(int(axis_uvw[0]), int(axis_uvw[1]), int(axis_uvw[2])),
        kinetic_energy_eV=float(case["E0_keV"]) * 1e3,
        thickness_ang=float(case["thickness_ang"]),
        tilt_deg=tilt,
        tilt_azimuth_deg=azimuth,
        temperature_K=float(temperature_K),
        particle=str(case.get("particle", "electron")),
    )


def axial_channeling_record(admitted: AdmittedAxialCase) -> dict[str, object]:
    """Freeze the calibrated fibre model and its evidence into Case.channeling.

    This records an admission *lookup result*, never caller-provided settings.
    The digest covers resolved source settings. An independent model-generation
    marker separates it from planar and the unverified v1 axial recipe.
    """
    if not isinstance(admitted, AdmittedAxialCase):
        raise TypeError("axial_channeling_record requires an admitted case")
    settings = {
        "crystal": admitted.crystal,
        "axis_uvw": list(admitted.axis_uvw),
        "kinetic_energy_eV": admitted.kinetic_energy_eV,
        "thickness_ang": admitted.thickness_ang,
        "tilt_deg": admitted.tilt_deg,
        "tilt_azimuth_deg": admitted.tilt_azimuth_deg,
        "temperature_K": admitted.temperature_K,
        "numerics": asdict(admitted.numerics),
    }
    settings_digest = hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {
        "model": "axial",
        "model_marker": AXIAL_FIBRE_MODEL_MARKER,
        "case_key": admitted.key,
        "settings_digest": settings_digest,
        "axis_uvw": list(admitted.axis_uvw),
        "temperature_K": admitted.temperature_K,
        "fibre_numerics": settings["numerics"],
        "band_eV": list(admitted.band_eV),
        "depth_budget": admitted.depth_budget,
        "joint_confidence": admitted.joint_confidence,
        "free_free": admitted.free_free,
        "free_free_upper": admitted.free_free_upper,
        "free_free_in_cut": admitted.free_free_in_cut,
        "free_free_in_cut_sem": admitted.free_free_in_cut_sem,
        "evidence_id": admitted.evidence_id,
        "evidence_record": admitted.evidence_record,
        "free_free_record": admitted.free_free_record,
        "observable": admitted.observable,
    }


#: Largest angle [rad] between the axis and the slab normal treated as roundoff.
_NORMAL_AXIS_TOL_RAD = 1e-9


def axial_zone_frame(case: Case, axis_uvw: tuple[int, int, int]) -> np.ndarray:
    """Slab-frame unit vectors of the profile x, y and axis (columns).

    The fibre model's depth is measured along the axis and its calibration is
    for a slab whose thickness is that depth, so the axis must be the slab
    normal; any other cut refuses.
    """
    from ...materials.channeling.axial import axial_profile

    axes = np.asarray(axial_profile(case["crystal"], axis_uvw).axes, dtype=float)
    frame = _rotation(case) @ axes.T
    if np.hypot(frame[0, 2], frame[1, 2]) > _NORMAL_AXIS_TOL_RAD or frame[2, 2] <= 0:
        raise AxialAdmissionError(
            f"axial fibre requires axis {list(axis_uvw)} along the slab normal"
        )
    return frame


def admitted_axial_case(case: Case) -> AdmittedAxialCase:
    """Re-resolve ``case`` against the repository lookup and return its entry.

    The stored record must equal the lookup result for the case's own physical
    fields: a case key, budget or record alone grants nothing.
    """
    record = case.get("channeling")
    key = record.get("case_key") if isinstance(record, Mapping) else None
    if not isinstance(key, str) or key not in ADMITTED_AXIAL_CASES:
        raise AxialAdmissionError("axial record has no repository-owned admission (#994)")
    template = ADMITTED_AXIAL_CASES[key]
    admitted = match_axial_case(axial_case_query(case, template.axis_uvw, template.temperature_K))
    if not isinstance(record, Mapping) or dict(record) != axial_channeling_record(admitted):
        raise AxialAdmissionError("axial resolved record differs from repository lookup (#994)")
    axial_zone_frame(case, admitted.axis_uvw)
    return admitted
