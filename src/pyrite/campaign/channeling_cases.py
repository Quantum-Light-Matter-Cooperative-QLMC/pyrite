"""Per-case selection and gates of the planar channeling zone (ADR-0018, #506).

``Channeling(model="planar")`` adds one resolved ``channeling`` record
to each case; ``channeling=None`` adds nothing, so case payloads, content keys and RNG
streams stay bit-for-bit. The gates run here, after geometry and energy are
known and before any transport:

- crystals without channeling thermal data (placeholder ``B``) are refused;
- energies above 100 MeV are refused (quasiclassical regime, #440);
- energies above 30 MeV warn: post-handoff transport lacks radiative loss
  until #439, so physical-support claims stop there;
- fewer than one bound state at the beam energy (``n_b < 1``) is refused;
- a beam within a few axial critical angles of a low-index axis warns, since
  the planar model ignores axial channeling;
- a case outside the measured admission table (planar
  ``ADMITTED_PLANAR_CASES``, axial ``ADMITTED_AXIAL_CASES``) is refused unless
  ``Channeling.admission="accept-inaccurate"``, which runs it with an
  ``unadmitted`` record and no error budget, and warns (#1105). The opt-in
  relaxes only the table match, never the gates above.

Geometries the one-way handoff does not support (stacks, grooves, mosaic,
GDF beams) are refused by :class:`~pyrite.montecarlo.case.Case` itself.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np

from ..montecarlo.case import Case, channeling_case_record

if TYPE_CHECKING:
    from ..quantum import Channeling

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
    from ..montecarlo.channeling.axial_case import beam_in_crystal

    return beam_in_crystal(case)


def _axis_proximity(case: Case) -> tuple[tuple[int, int, int], float, float] | None:
    """Nearest low-index axis, its angle and Lindhard ``psi_1`` [rad].

    ``psi_1 = sqrt(4 Z e^2 / (p v d))`` with ``d`` the shortest lattice
    translation along the axis and ``Z`` the basis-mean atomic number; a
    warning heuristic, not an emission or transport input.
    """
    from ..materials.atomic import Z_TABLE
    from ..materials.crystal import CRYSTALS, E2_EV_ANG, M_E_EV, _direct_lattice_vectors
    from ..montecarlo.channeling import longitudinal_kinematics

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


def resolve_channeling(cases: list[Case], channeling: Channeling | None) -> list[Case]:
    """Attach the channeling record to every case and apply the gates.

    Returns ``cases`` unchanged when ``channeling`` is ``None``.

    Raises
    ------
    ValueError
        For a crystal without channeling data, an energy above 100 MeV, no
        bound state at the beam energy, an unsupported case combination, or
        (by default) a case outside the admission table.
    """
    if channeling is None:
        return cases
    if channeling.model == "axial":
        return _resolve_axial(cases, channeling)
    return _resolve_planar(cases, channeling)


def _accepts_unadmitted(channeling: Channeling) -> bool:
    from ..quantum._admission import CHANNELING_ADMISSION_OPT_IN

    return channeling.admission == CHANNELING_ADMISSION_OPT_IN


def _warn_unadmitted(case: Case, model: str) -> None:
    warnings.warn(
        f"case {case['name']!r}: {model} channeling case is outside the measured admission "
        "table and runs under Channeling(admission='accept-inaccurate'); its results may be "
        "very inaccurate and are marked unadmitted, with no error budget (#1105)",
        UserWarning,
        stacklevel=4,
    )


def _resolve_axial(cases: list[Case], channeling: Channeling) -> list[Case]:
    from ..quantum._admission import AxialAdmissionError
    from ..quantum._axial_envelope import match_axial_case, unadmitted_axial_case
    from .axial_cases import axial_case_query, axial_channeling_record
    from .axial_spec import refuse_unverified_axial_run

    if channeling.axial is None or channeling.axial["version"] != 2:
        refuse_unverified_axial_run()  # v1 remains parseable, never executable
    u, v, w = cast(list[int], channeling.axial["axis_uvw"])
    axis = (u, v, w)
    out = []
    for case in cases:
        # Match only resolved physical cases; do not infer or extrapolate
        # calibration settings from an API/profile recipe.
        query = axial_case_query(case, axis, CHANNELING_TEMPERATURE_K)
        try:
            resolved = match_axial_case(query)
        except AxialAdmissionError:
            if not _accepts_unadmitted(channeling):
                raise
            resolved = unadmitted_axial_case(query)
            _warn_unadmitted(case, "axial")
        out.append(replace(case, channeling=axial_channeling_record(resolved)))
    return out


def _planar_admission(case: Case, zone: dict[str, object], channeling: Channeling):
    """Admission record of one planar case, or refuse (#1105)."""
    from ..montecarlo.channeling.planar_case import (
        UNADMITTED_RECORD,
        planar_admission_record,
        planar_case_query,
    )
    from ..quantum._admission import PlanarAdmissionError
    from ..quantum._planar_envelope import match_planar_case

    try:
        return planar_admission_record(match_planar_case(planar_case_query(case, zone)))
    except PlanarAdmissionError as error:
        if not _accepts_unadmitted(channeling):
            raise PlanarAdmissionError(f"case {case['name']!r}: {error}") from None
    _warn_unadmitted(case, "planar")
    return dict(UNADMITTED_RECORD)


def _resolve_planar(cases: list[Case], channeling: Channeling) -> list[Case]:
    from ..materials.channeling import planar_profile
    from ..montecarlo.channeling import (
        DECHANNELING_MARGIN_EV,
        longitudinal_kinematics,
        transverse_states,
    )

    assert channeling.plane is not None  # Channeling validated the pair
    numerics = channeling.numerics
    zone = channeling_case_record(
        channeling.plane,
        temperature_K=CHANNELING_TEMPERATURE_K,
        margin_eV=DECHANNELING_MARGIN_EV,
        basis_size=numerics.basis_size,
        bloch_cells=numerics.bloch_cells,
        dz_ang=numerics.step_ang,
        zone_max_ang=numerics.zone_max_ang,
        admission={},
    )
    out = []
    for case in cases:
        energy_keV = float(case["E0_keV"])
        if energy_keV > CHANNELING_MAX_ENERGY_KEV:
            raise ValueError(
                f"Channeling(model='planar') supports beam energies up to 100 MeV; case "
                f"{case['name']!r} is {energy_keV / 1e3:g} MeV (quasiclassical regime, #440)"
            )
        if energy_keV > CHANNELING_RADIATIVE_WARN_KEV:
            warnings.warn(
                f"case {case['name']!r}: above 30 MeV, transport after dechanneling lacks "
                "radiative energy loss (#439); channeling results are not physically supported",
                UserWarning,
                stacklevel=4,
            )
        profile = planar_profile(case["crystal"], zone["plane_hkl"], CHANNELING_TEMPERATURE_K)
        gamma, _ = longitudinal_kinematics(energy_keV * 1e3)
        n_bound = transverse_states(profile, gamma).n_bound
        if n_bound < 1:
            raise ValueError(
                f"case {case['name']!r}: plane {zone['plane_hkl']} of {case['crystal']!r} "
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
                stacklevel=4,
            )
        admission = _planar_admission(case, zone, channeling)
        out.append(replace(case, channeling={**zone, "admission": admission}))
    return out
