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
source with ``boundary="max"`` does not score (#992): ``"native"`` when native
CBS scores it on tilted zone rows, ``"bound"`` when it is omitted and carried as
the declared one-sided ``free_free_upper`` (relative to the band yield).

Validation: cr-axial-depth-calibration
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from ._admission import AxialAdmissionError

_REL_TOL = 1e-9


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
    free_free: Literal["native", "bound"]
    free_free_upper: float | None
    evidence_id: str
    evidence_record: str
    observable: str = "on-axis"
    particle: str = "electron"

    def __post_init__(self) -> None:
        if (self.free_free == "bound") != (self.free_free_upper is not None):
            raise ValueError("free_free_upper is required exactly when free_free='bound'")


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
            free_free="native",
            free_free_upper=None,
            evidence_id="993/A@07c0f5f3e",
            evidence_record=f"{_RECORDS}/depth-calibration-993-A.json",
        ),
    }
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
        "docs/research/physics/axial-first-envelope-886.md)" + detail
    )


def check_axial_line_band(case: AdmittedAxialCase, energy_eV) -> None:
    """Refuse a line grid reaching outside the calibrated band of ``case``.

    Validation: cr-axial-depth-calibration
    """
    values = [float(e) for e in energy_eV]
    if not values:
        raise AxialAdmissionError("axial line grid is empty")
    lo, hi = case.band_eV
    if min(values) < lo or max(values) > hi:
        raise AxialAdmissionError(
            f"axial line grid {min(values):g}-{max(values):g} eV leaves the calibrated "
            f"band {lo:g}-{hi:g} eV of {case.key}"
        )
