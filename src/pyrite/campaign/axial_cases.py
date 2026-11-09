"""Resolve a lowered case into the exact axial admission query (#994).

The beam direction is taken into crystal coordinates with the same orientation
convention as planar channeling (``channeling_cases._beam_in_crystal``), then
expressed in the axial profile frame (``AxialProfile.axes`` rows): the tilt is
the angle between beam and axis, and its azimuth is measured from the
profile's first transverse axis toward its second. Admission compares these
resolved values with the repository lookup; no other frame is accepted.
"""

from __future__ import annotations

import math

import numpy as np

from ..montecarlo.case import Case
from ..quantum._axial_envelope import AxialCaseQuery

#: Tilts below this [deg] are frame roundoff (profile axes carry ~1e-16 noise).
ALIGNED_TILT_DEG = 1e-9


def axial_beam_angles(case: Case, axis_uvw: tuple[int, int, int]) -> tuple[float, float]:
    """Beam tilt from ``axis_uvw`` and its profile-frame azimuth [deg].

    Tilts below :data:`ALIGNED_TILT_DEG` resolve to exact alignment with zero
    azimuth. Refuses a beam travelling against the axis direction.
    """
    from ..materials.channeling.axial import axial_profile
    from .channeling_cases import _beam_in_crystal

    axes = np.asarray(axial_profile(case["crystal"], axis_uvw).axes, dtype=float)
    x, y, z = axes @ _beam_in_crystal(case)
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
