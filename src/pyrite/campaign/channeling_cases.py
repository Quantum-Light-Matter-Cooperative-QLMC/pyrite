"""Per-case selection and gates of the planar channeling zone (ADR-0018, #506).

``Physics(channeling_model="planar")`` adds one resolved ``channeling`` record
to each case; ``"off"`` adds nothing, so case payloads, content keys and RNG
streams stay bit-for-bit. The gates run here, after geometry and energy are
known and before any transport:

- crystals without channeling thermal data (placeholder ``B``) are refused;
- energies above 100 MeV are refused (quasiclassical regime, #440);
- energies above 30 MeV warn: post-handoff transport lacks radiative loss
  until #439, so physical-support claims stop there;
- fewer than one bound state at the beam energy (``n_b < 1``) is refused;
- a beam within a few axial critical angles of a low-index axis warns, since
  the planar model ignores axial channeling.

Geometries the one-way handoff does not support (stacks, grooves, mosaic,
GDF beams) are refused by :class:`~pyrite.montecarlo.case.Case` itself.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ..montecarlo.case import Case, channeling_case_record

if TYPE_CHECKING:
    from .._numerics import Numerics, Physics

#: Highest beam kinetic energy the planar quantum zone accepts [keV].
CHANNELING_MAX_ENERGY_KEV = 100e3
#: Above this energy [keV] a run warns: radiative loss after handoff is #439.
CHANNELING_RADIATIVE_WARN_KEV = 30e3
#: Crystal temperature of the zone's thermal smearing [K].
CHANNELING_TEMPERATURE_K = 293.15
#: Warn when the beam lies within this many axial critical angles of an axis.
AXIS_WARN_CRITICAL_ANGLES = 3.0


def _beam_in_crystal(case: Case) -> np.ndarray:
    """Unit beam direction in crystal (construction) coordinates."""
    from ..materials.crystal import CRYSTALS, _orientation_R
    from ..montecarlo.geometry import tilted_geometry

    beam, _ = tilted_geometry(0.0, np.deg2rad(case["tilt_deg"]), np.deg2rad(case["tilt_azim_deg"]))
    rotation = _orientation_R(
        CRYSTALS[case["crystal"]]["lattice"],
        case["beam_uvw"],
        case.get("azimuth_rad", 0.0),
        case.get("recip_miscut_rad"),
        surface_hkl=case["surface_hkl"],
    )
    return beam if rotation is None else np.asarray(rotation).T @ beam


def _axis_proximity(case: Case) -> tuple[tuple[int, int, int], float, float] | None:
    """Nearest low-index axis, its angle and Lindhard ``psi_1`` [rad].

    ``psi_1 = sqrt(4 Z e^2 / (p v d))`` with ``d`` the shortest lattice
    translation along the axis and ``Z`` the basis-mean atomic number; a
    warning heuristic, not an emission or transport input.
    """
    from ..materials.atomic import Z_TABLE
    from ..materials.channeling import longitudinal_kinematics
    from ..materials.crystal import CRYSTALS, E2_EV_ANG, M_E_EV, _direct_lattice_vectors

    info = CRYSTALS[case["crystal"]]
    beam = _beam_in_crystal(case)
    vectors = _direct_lattice_vectors(info["lattice"])
    z_mean = float(np.mean([Z_TABLE[element] for element, _ in info["basis"]]))
    kinetic_eV = float(case["E0_keV"]) * 1e3
    gamma, beta = longitudinal_kinematics(kinetic_eV)
    pv_eV = gamma * M_E_EV * beta**2
    best: tuple[tuple[int, int, int], float, float] | None = None
    for u, v, w in itertools.product((-1, 0, 1), repeat=3):
        uvw = (int(u), int(v), int(w))
        if uvw == (0, 0, 0) or uvw < (0, 0, 0):  # one sign per axis
            continue
        axis = sum(c * v for c, v in zip(uvw, vectors, strict=True))
        length = float(np.linalg.norm(axis))
        angle = float(np.arccos(min(1.0, abs(float(beam @ axis)) / length)))
        psi_1 = float(np.sqrt(4.0 * z_mean * E2_EV_ANG / (pv_eV * length)))
        if best is None or angle / psi_1 < best[1] / best[2]:
            best = (uvw, angle, psi_1)
    return best


def resolve_channeling(cases: list[Case], physics: Physics, numerics: Numerics) -> list[Case]:
    """Attach the channeling record to every case and apply the gates.

    Returns ``cases`` unchanged when ``physics.channeling_model == "off"``.

    Raises
    ------
    ValueError
        For a crystal without channeling data, an energy above 100 MeV, no
        bound state at the beam energy, or an unsupported case combination.
    """
    from ..materials.channeling import longitudinal_kinematics, planar_profile, transverse_states
    from ..montecarlo.channeling import DECHANNELING_MARGIN_EV

    if physics.channeling_model == "off":
        return cases
    assert physics.channeling_plane is not None  # Physics validated the pair
    record = channeling_case_record(
        physics.channeling_plane,
        temperature_K=CHANNELING_TEMPERATURE_K,
        margin_eV=DECHANNELING_MARGIN_EV,
        basis_size=numerics.channeling_basis_size,
        bloch_cells=numerics.channeling_bloch_cells,
        dz_ang=numerics.channeling_dz_ang,
        zone_max_ang=numerics.channeling_zone_max_ang,
    )
    out = []
    for case in cases:
        energy_keV = float(case["E0_keV"])
        if energy_keV > CHANNELING_MAX_ENERGY_KEV:
            raise ValueError(
                f"channeling_model='planar' supports beam energies up to 100 MeV; case "
                f"{case['name']!r} is {energy_keV / 1e3:g} MeV (quasiclassical regime, #440)"
            )
        if energy_keV > CHANNELING_RADIATIVE_WARN_KEV:
            warnings.warn(
                f"case {case['name']!r}: above 30 MeV, transport after dechanneling lacks "
                "radiative energy loss (#439); channeling results are not physically supported",
                UserWarning,
                stacklevel=3,
            )
        profile = planar_profile(case["crystal"], record["plane_hkl"], CHANNELING_TEMPERATURE_K)
        gamma, _ = longitudinal_kinematics(energy_keV * 1e3)
        n_bound = transverse_states(profile, gamma).n_bound
        if n_bound < 1:
            raise ValueError(
                f"case {case['name']!r}: plane {record['plane_hkl']} of {case['crystal']!r} "
                f"has no bound transverse state at {energy_keV:g} keV"
            )
        nearest = _axis_proximity(case)
        if nearest is not None and nearest[1] < AXIS_WARN_CRITICAL_ANGLES * nearest[2]:
            uvw, angle, psi_1 = nearest
            warnings.warn(
                f"case {case['name']!r}: the beam is {angle * 1e3:.3g} mrad from the "
                f"<{''.join(str(c) for c in uvw)}> axis (axial critical angle "
                f"{psi_1 * 1e3:.3g} mrad); the planar zone omits axial channeling",
                UserWarning,
                stacklevel=3,
            )
        out.append(replace(case, channeling=dict(record)))
    return out
