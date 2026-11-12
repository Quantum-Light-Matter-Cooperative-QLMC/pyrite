"""Repository-owned axial admission envelope (#994). Data and exact matching only.

Each entry is one resolved fibre-source case (#992) whose photon-source depth
budget was measured in #993 and is held here, keyed by the exact physical case,
fibre numerics and observable. Callers never supply or override a calibration:
:func:`match_axial_case` looks an entry up from resolved physical fields and
returns the stored one or refuses. A caller-built ``AxialPhotonDepthEvidence``,
research constructor or ownership tag is not an input here.

``depth_budget`` is the #993 bound on the relative error of the on-axis
photons/sr per incident electron integrated over ``band_eV``, for yield at
crystal thickness, at ``joint_confidence``. It bounds that band integral only,
not individual spectral bins, and is declared with results rather than gated
by a tolerance (maintainer decision, 2026-11-09).

``free_free`` names the owner of the in-cut free→free block that the fibre
source with ``boundary="max"`` does not score (#992). It is always ``"bound"``:
native PXR/CBS line scoring is off on axial zone rows
(``cr-axial-zone-row-native-lines``), so the block is omitted and carried as
the declared one-sided ``free_free_upper`` (relative to the band yield;
measured mean + 3 SE). ``free_free_in_cut`` is the measured block
(``boundary="none"`` minus ``"max"``, chunk-paired with the calibration
control). Free→free above the retained cut is scored by neither estimator on
zone rows and is not bounded here.

Validation: cr-axial-depth-calibration
Validation: cr-axial-zone-row-native-lines
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from ._admission import UNADMITTED, AxialAdmissionError, unadmitted_hint

_REL_TOL = 1e-9
#: Model generation of the calibrated fibre source; distinct from planar and
#: from the non-executable v1 axial recipe.
AXIAL_FIBRE_MODEL_MARKER = "axial-fibre-v2"


@dataclass(frozen=True)
class AxialFibreNumerics:
    """Fibre-source numerics a calibration was measured at; never caller-tunable."""

    max_order: int = 9
    outgoing_order_max: int = 20
    step_ang: float = 15.0
    harmonic_max: int = 4
    grid_points: int = 32
    cut_multiple: float = 1.0
    boundary: Literal["max"] = "max"
    electronic: bool = True
    tds: Literal["strings"] = "strings"
    absorption: bool = True


@dataclass(frozen=True)
class AdmittedAxialCase:
    """One admitted axial case and the evidence that admits it.

    ``tilt_azimuth_deg`` is measured from the axial profile's first reciprocal
    axis; no crystal symmetry reduction is applied. Energies are kinetic, in eV.
    """

    key: str
    crystal: str
    axis_uvw: tuple[int, int, int]
    kinetic_energy_eV: float
    thickness_ang: float
    tilt_deg: float
    tilt_azimuth_deg: float
    temperature_K: float
    numerics: AxialFibreNumerics
    band_eV: tuple[float, float]
    depth_budget: float
    joint_confidence: float
    free_free: Literal["bound"]
    free_free_upper: float
    free_free_in_cut: float
    free_free_in_cut_sem: float
    evidence_id: str
    evidence_record: str
    free_free_record: str
    observable: str = "on-axis"
    particle: str = "electron"

    def __post_init__(self) -> None:
        if self.free_free != "bound" or not self.free_free_upper >= self.free_free_in_cut:
            raise ValueError("free_free must be 'bound' with an upper bound covering the block")


_RECORDS = "docs/validation/check-records/axial-source"

ADMITTED_AXIAL_CASES: Mapping[str, AdmittedAxialCase] = MappingProxyType(
    {
        "si-fibre/A": AdmittedAxialCase(
            key="si-fibre/A",
            crystal="silicon",
            axis_uvw=(0, 0, 1),
            kinetic_energy_eV=3.5e6,
            thickness_ang=3000.0,
            tilt_deg=0.088,
            tilt_azimuth_deg=0.0,
            temperature_K=293.15,
            numerics=AxialFibreNumerics(),
            band_eV=(800.0, 10000.0),
            depth_budget=0.039720652973856484,
            joint_confidence=0.95,
            free_free="bound",
            free_free_upper=0.08057945671257792,
            free_free_in_cut=0.07832616270160697,
            free_free_in_cut_sem=0.0007510980036569821,
            evidence_id="993/A@07c0f5f3e",
            evidence_record=f"{_RECORDS}/depth-calibration-993-A.json",
            free_free_record=f"{_RECORDS}/free-free-994.json",
        ),
        "si-fibre/B": AdmittedAxialCase(
            key="si-fibre/B",
            crystal="silicon",
            axis_uvw=(0, 0, 1),
            kinetic_energy_eV=3.5e6,
            thickness_ang=3000.0,
            tilt_deg=0.0,
            tilt_azimuth_deg=0.0,
            temperature_K=293.15,
            numerics=AxialFibreNumerics(),
            band_eV=(800.0, 10000.0),
            depth_budget=0.09705397354242776,
            joint_confidence=0.95,
            free_free="bound",
            free_free_upper=0.18874404816708856,
            free_free_in_cut=0.18193684747509553,
            free_free_in_cut_sem=0.0022690668973310135,
            evidence_id="993/B@07c0f5f3e",
            evidence_record=f"{_RECORDS}/depth-calibration-993-B.json",
            free_free_record=f"{_RECORDS}/free-free-994.json",
        ),
    }
)


#: Keys of the lookup entries the standard runner may execute. An entry is
#: calibrated when it is in :data:`ADMITTED_AXIAL_CASES`; it becomes runnable
#: only when every claim its run uses (:data:`AXIAL_RUN_CLAIMS`) is
#: ``anchored``. Repository-owned; callers cannot extend it.
RUNNABLE_AXIAL_CASES: frozenset[str] = frozenset({"si-fibre/A", "si-fibre/B"})

#: Ledger claims every admitted fibre run relies on (#994).
AXIAL_RUN_CLAIMS: tuple[str, ...] = (
    "cr-axial-fiber-regression",
    "cr-axial-depth-calibration",
    "cr-axial-zone-handoff-physical",
    "cr-axial-source-partition",
    "cr-axial-zone-row-native-lines",
    "cr-axial-persisted-channel",
)


def require_runnable_axial_case(key: object) -> None:
    """Refuse a calibrated case whose production claims are not yet anchored.

    An unadmitted case (``Channeling(admission="accept-inaccurate")``) runs the
    same adapter outside every calibrated entry and is never gated here.
    """
    if key != UNADMITTED and key not in RUNNABLE_AXIAL_CASES:
        raise AxialAdmissionError(
            f"axial case {key} is calibrated but not yet runnable: the production "
            "adapter claim cr-axial-persisted-channel is not anchored for it (#994)"
        )


@dataclass(frozen=True)
class AxialCaseQuery:
    """Resolved physical fields of a requested axial case, for exact lookup."""

    crystal: str
    axis_uvw: tuple[int, int, int]
    kinetic_energy_eV: float
    thickness_ang: float
    tilt_deg: float
    tilt_azimuth_deg: float
    temperature_K: float
    particle: str = "electron"
    observable: str = "on-axis"


def _same(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=_REL_TOL, abs_tol=0.0)


def _azimuth_matches(requested: float, admitted: float, tilt_deg: float) -> bool:
    if tilt_deg == 0.0:
        return True  # azimuth is undefined at exact alignment
    turn = (float(requested) - float(admitted)) % 360.0
    return min(turn, 360.0 - turn) <= _REL_TOL * 360.0


def _mismatches(query: AxialCaseQuery, case: AdmittedAxialCase) -> list[str]:
    checks = {
        "crystal": query.crystal == case.crystal,
        "axis_uvw": tuple(query.axis_uvw) == case.axis_uvw,
        "particle": query.particle == case.particle,
        "observable": query.observable == case.observable,
        "kinetic_energy_eV": _same(query.kinetic_energy_eV, case.kinetic_energy_eV),
        "thickness_ang": _same(query.thickness_ang, case.thickness_ang),
        "temperature_K": _same(query.temperature_K, case.temperature_K),
        "tilt_deg": (
            query.tilt_deg == case.tilt_deg == 0.0 or _same(query.tilt_deg, case.tilt_deg)
        ),
        "tilt_azimuth_deg": _azimuth_matches(
            query.tilt_azimuth_deg, case.tilt_azimuth_deg, case.tilt_deg
        ),
    }
    return [name for name, ok in checks.items() if not ok]


def match_axial_case(query: AxialCaseQuery) -> AdmittedAxialCase:
    """Return the admitted case equal to ``query`` or refuse.

    Every physical field must match one stored entry exactly (floats to a
    relative 1e-9, for unit round trips only). The refusal names the closest
    entry's mismatched fields; nothing is interpolated or extrapolated.

    Validation: cr-axial-depth-calibration
    """
    if not isinstance(query, AxialCaseQuery):
        raise TypeError("match_axial_case requires an AxialCaseQuery")
    best: tuple[str, list[str]] | None = None
    for key, case in ADMITTED_AXIAL_CASES.items():
        missing = _mismatches(query, case)
        if not missing:
            return case
        if best is None or len(missing) < len(best[1]):
            best = (key, missing)
    detail = "" if best is None else f"; closest {best[0]} differs in {', '.join(best[1])}"
    raise AxialAdmissionError(
        "axial case is outside the admitted envelope (#994; "
        "docs/research/physics/axial-first-envelope-886.md)" + detail + unadmitted_hint()
    )


@dataclass(frozen=True)
class UnadmittedAxialCase:
    """An axial case run under ``Channeling(admission="accept-inaccurate")`` (#1105).

    The resolved physical fields of a case outside the lookup, run with the
    default fibre numerics. It carries no calibrated band, depth budget or
    free->free bound: its results may be very inaccurate. Solver-validity
    refusals (slab normal, plane wave, entrance order window, observable,
    unsupported options) still apply.
    """

    crystal: str
    axis_uvw: tuple[int, int, int]
    kinetic_energy_eV: float
    thickness_ang: float
    tilt_deg: float
    tilt_azimuth_deg: float
    temperature_K: float
    numerics: AxialFibreNumerics = field(default_factory=AxialFibreNumerics)
    observable: str = "on-axis"
    particle: str = "electron"
    key: str = UNADMITTED


#: A resolved axial fibre case: a lookup entry or an opted-in unadmitted case.
AxialFibreCase = AdmittedAxialCase | UnadmittedAxialCase


def unadmitted_axial_case(query: AxialCaseQuery) -> UnadmittedAxialCase:
    """Unadmitted case for ``query``; refuses a query that matches an entry."""
    try:
        admitted = match_axial_case(query)
    except AxialAdmissionError:
        pass
    else:
        raise AxialAdmissionError(f"axial case is admitted as {admitted.key}; it is not unadmitted")
    if query.observable != "on-axis" or query.particle != "electron":
        raise AxialAdmissionError("axial fibre supports on-axis electron cases only")
    return UnadmittedAxialCase(
        crystal=query.crystal,
        axis_uvw=query.axis_uvw,
        kinetic_energy_eV=float(query.kinetic_energy_eV),
        thickness_ang=float(query.thickness_ang),
        tilt_deg=float(query.tilt_deg),
        tilt_azimuth_deg=float(query.tilt_azimuth_deg),
        temperature_K=float(query.temperature_K),
    )


def check_axial_line_band(case: AxialFibreCase, energy_eV) -> None:
    """Refuse a line grid reaching outside the calibrated band of ``case``.

    An unadmitted case has no calibrated band; only an empty grid refuses.

    Validation: cr-axial-depth-calibration
    """
    values = [float(e) for e in energy_eV]
    if not values:
        raise AxialAdmissionError("axial line grid is empty")
    if isinstance(case, UnadmittedAxialCase):
        return
    lo, hi = case.band_eV
    if min(values) < lo or max(values) > hi:
        raise AxialAdmissionError(
            f"axial line grid {min(values):g}-{max(values):g} eV leaves the calibrated "
            f"band {lo:g}-{hi:g} eV of {case.key}"
        )
