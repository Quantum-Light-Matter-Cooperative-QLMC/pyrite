"""Repository-owned axial admission lookup (#994).

Each entry must equal its retained #993 calibration record, and only an exact
resolved case matches. Admission itself stays refused until the production
adapters land; see ``test_channeling_axial_envelope.py``.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from pyrite.quantum._admission import AxialAdmissionError
from pyrite.quantum._axial_envelope import (
    ADMITTED_AXIAL_CASES,
    AdmittedAxialCase,
    AxialCaseQuery,
    check_axial_line_band,
    match_axial_case,
)

ROOT = Path(__file__).resolve().parents[3]


def _query(case: AdmittedAxialCase, **updates) -> AxialCaseQuery:
    fields = {f.name for f in dataclasses.fields(AxialCaseQuery)}
    base = {name: getattr(case, name) for name in fields}
    return AxialCaseQuery(**{**base, **updates})


@pytest.mark.parametrize("key", sorted(ADMITTED_AXIAL_CASES))
def test_entry_equals_its_retained_calibration_record(key):
    case = ADMITTED_AXIAL_CASES[key]
    record = json.loads((ROOT / case.evidence_record).read_text())
    spec, control = record["spec"], record["settings"]["control"]
    band = record["bands"]["whole"]
    assert spec["case_key"] == key
    assert spec["evidence_id"] == case.evidence_id
    assert band["observable_key"] == f"on-axis/whole/{case.band_eV[0]:g}-{case.band_eV[1]:g}eV"
    assert band["simultaneous_confidence"] == case.joint_confidence
    last = band["intervals"][-1]
    assert last["stop_ang"] == case.thickness_ang == control["thickness_ang"]
    assert last["relative_error_upper"] == case.depth_budget
    n = case.numerics
    expected = {
        "axis": list(case.axis_uvw),
        "energy_eV": case.kinetic_energy_eV,
        "tilt_deg": case.tilt_deg,
        "tilt_azimuth_deg": case.tilt_azimuth_deg,
        "e_min_eV": case.band_eV[0],
        "e_max_eV": case.band_eV[1],
        "max_order": n.max_order,
        "outgoing_order_max": n.outgoing_order_max,
        "step_ang": n.step_ang,
        "harmonic_max": n.harmonic_max,
        "grid_points": n.grid_points,
        "cut_multiple": n.cut_multiple,
        "boundary": n.boundary,
        "electronic": n.electronic,
        "absorption": n.absorption,
        "reference": False,
    }
    assert {k: control[k] for k in expected} == expected


@pytest.mark.parametrize("key", sorted(ADMITTED_AXIAL_CASES))
def test_exact_case_matches_and_unit_roundoff_is_tolerated(key):
    case = ADMITTED_AXIAL_CASES[key]
    assert match_axial_case(_query(case)) is case
    assert match_axial_case(_query(case, thickness_ang=case.thickness_ang * (1 + 1e-13))) is case
    assert match_axial_case(_query(case, tilt_azimuth_deg=case.tilt_azimuth_deg + 360.0)) is case


CHANGES = {
    "crystal": {"crystal": "diamond"},
    "axis": {"axis_uvw": (1, 1, 1)},
    "axis-sign": {"axis_uvw": (0, 0, -1)},
    "energy": {"kinetic_energy_eV": 3.6e6},
    "thickness-thinner": {"thickness_ang": 1500.0},
    "thickness-thicker": {"thickness_ang": 3015.0},
    "tilt": {"tilt_deg": 0.0},
    "tilt-azimuth": {"tilt_azimuth_deg": 90.0},
    "temperature": {"temperature_K": 100.0},
    "positron": {"particle": "positron"},
    "off-axis": {"observable": "off-axis"},
}


@pytest.mark.parametrize("change", CHANGES.values(), ids=CHANGES.keys())
def test_any_other_case_refuses_and_names_the_mismatch(change):
    case = ADMITTED_AXIAL_CASES["si-fibre/A"]
    with pytest.raises(AxialAdmissionError, match="differs in") as info:
        match_axial_case(_query(case, **change))
    assert "#994" in str(info.value)


def test_no_symmetry_reduction_of_the_tilt_azimuth():
    case = ADMITTED_AXIAL_CASES["si-fibre/A"]
    for azimuth in (90.0, 180.0, -0.088):
        with pytest.raises(AxialAdmissionError, match="tilt_azimuth_deg"):
            match_axial_case(_query(case, tilt_azimuth_deg=azimuth))


def test_lookup_takes_no_caller_evidence():
    with pytest.raises(TypeError):
        match_axial_case({"crystal": "silicon", "depth_budget": 0.0})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AxialCaseQuery(**{**vars(_query(ADMITTED_AXIAL_CASES["si-fibre/A"])), "depth_budget": 0})
    with pytest.raises(TypeError):
        ADMITTED_AXIAL_CASES["si-fibre/B"] = ADMITTED_AXIAL_CASES["si-fibre/A"]  # type: ignore[index]


def test_line_grid_must_stay_inside_the_calibrated_band():
    case = ADMITTED_AXIAL_CASES["si-fibre/A"]
    check_axial_line_band(case, [800.0, 5000.0, 10000.0])
    for grid in ([700.0, 5000.0], [5000.0, 10001.0], []):
        with pytest.raises(AxialAdmissionError):
            check_axial_line_band(case, grid)


def test_free_free_bound_is_declared_exactly_when_the_block_is_omitted():
    case = ADMITTED_AXIAL_CASES["si-fibre/A"]
    assert case.free_free == "native" and case.free_free_upper is None
    with pytest.raises(ValueError):
        dataclasses.replace(case, free_free="bound")
    with pytest.raises(ValueError):
        dataclasses.replace(case, free_free_upper=0.1)
