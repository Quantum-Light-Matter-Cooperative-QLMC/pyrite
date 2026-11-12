"""#1105: channeling cases outside the admission tables refuse unless opted in."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

import pyrite as pr
from pyrite import Numerics, Physics, api
from pyrite.campaign.axial_cases import axial_channeling_record
from pyrite.campaign.channeling_cases import resolve_channeling
from pyrite.campaign.profiles import case_content_key
from pyrite.montecarlo.channeling.planar_case import (
    planar_admission_record,
    require_planar_admission,
)
from pyrite.quantum._admission import AxialAdmissionError, PlanarAdmissionError
from pyrite.quantum._axial_envelope import ADMITTED_AXIAL_CASES, UnadmittedAxialCase
from pyrite.quantum._planar_envelope import ADMITTED_PLANAR_CASES
from tests.campaign.test_axial_fibre_milestone3 import case as axial_case
from tests.campaign.test_axial_fibre_milestone3 import physics as axial_physics

ROOT = Path(__file__).resolve().parents[2]
DEMO = ADMITTED_PLANAR_CASES["si110-17MeV-2um"]
PLANAR = dict(channeling_model="planar", channeling_plane=(1, -1, 0), radiative_model="uncoupled")
OPT_IN = dict(channeling_admission="accept-inaccurate")


def _planar_case(physics, *, energy_keV=17000.0, thickness_ang=20000.0, numerics=None):
    scene = pr.Scene(
        pr.Beam(
            energy_keV=energy_keV, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()
        ),
        pr.geometry.Slab(
            "silicon",
            footprint=pr.geometry.Footprint(5.0, 5.0),
            thickness_ang=thickness_ang,
            tilt_deg=3.0,
            tilt_azim_deg=45.0,
        ),
        pr.Detector(),
        physics=physics,
    )
    return api.build_case(scene, numerics or pr.Numerics())


# -- planar ---------------------------------------------------------------------


def test_demo_case_is_admitted_and_carries_its_epsilon():
    case = _planar_case(pr.Physics(**PLANAR))
    admission = case["channeling"]["admission"]
    assert admission == planar_admission_record(DEMO)
    assert admission["epsilon"] == pytest.approx(0.0486, abs=5e-5)
    assert require_planar_admission(case) is DEMO


def test_demo_epsilon_equals_its_evidence_record():
    record = json.loads((ROOT / DEMO.evidence_record).read_text())
    assert record["epsilon"] == [DEMO.epsilon]
    assert record["simultaneous_confidence"] == DEMO.joint_confidence
    assert record["observable_key"] == DEMO.observable
    assert record["budget_um"] == [DEMO.thickness_ang[1] / 1e4]


@pytest.mark.parametrize(
    "updates, field",
    [({"energy_keV": 5000.0}, "kinetic_energy_eV"), ({"thickness_ang": 50000.0}, "thickness_ang")],
)
def test_unadmitted_planar_refuses_by_default_naming_the_opt_in(updates, field):
    with pytest.raises(PlanarAdmissionError, match=f"differs in {field}.*accept-inaccurate"):
        _planar_case(pr.Physics(**PLANAR), **updates)


def test_numerics_off_the_admitted_solver_refuse():
    with pytest.raises(PlanarAdmissionError, match="numerics"):
        _planar_case(pr.Physics(**PLANAR), numerics=Numerics(channeling_basis_size=12))


def test_opt_in_marks_unadmitted_without_epsilon_and_warns():
    with pytest.warns(UserWarning, match="very inaccurate"):
        case = _planar_case(pr.Physics(**PLANAR, **OPT_IN), energy_keV=5000.0)
    assert case["channeling"]["admission"] == {"status": "unadmitted"}
    assert require_planar_admission(case) is None


def test_opt_in_leaves_admitted_cases_admitted():
    case = _planar_case(pr.Physics(**PLANAR, **OPT_IN))
    assert case["channeling"]["admission"]["case_key"] == DEMO.key


def test_unadmitted_and_admitted_never_share_a_content_key():
    admitted = _planar_case(pr.Physics(**PLANAR))
    forged = {**dict(admitted["channeling"]), "admission": {"status": "unadmitted"}}
    unadmitted = replace(admitted, channeling=forged)
    assert case_content_key(unadmitted) != case_content_key(admitted)
    # An admitted case cannot run under the unadmitted marker either.
    with pytest.raises(PlanarAdmissionError, match=f"admitted as {DEMO.key}"):
        require_planar_admission(unadmitted)


def test_physical_refusals_hold_under_the_opt_in():
    with pytest.raises(ValueError, match="up to 100 MeV"):
        _planar_case(pr.Physics(**PLANAR, **OPT_IN), energy_keV=150e3)


def test_tampered_admission_records_refuse():
    case = _planar_case(pr.Physics(**PLANAR))
    zone = dict(case["channeling"])
    with pytest.raises(PlanarAdmissionError, match="differs from the table"):
        replace(case, channeling={**zone, "admission": {**zone["admission"], "epsilon": 0.0}})
    with pytest.raises(PlanarAdmissionError, match="no repository table entry"):
        replace(case, channeling={**zone, "admission": {"status": "admitted"}})
    other = _planar_case(pr.Physics(**PLANAR, **OPT_IN), energy_keV=5000.0)
    stolen = {**dict(other["channeling"]), "admission": zone["admission"]}
    with pytest.raises(PlanarAdmissionError, match="outside"):
        require_planar_admission(replace(other, channeling=stolen))


def test_physics_refuses_opt_in_without_a_model_and_unknown_values():
    with pytest.raises(ValueError, match="requires a channeling model"):
        Physics(**OPT_IN)
    with pytest.raises(ValueError, match="channeling_admission must be one of"):
        Physics(**PLANAR, channeling_admission="yes")


# -- axial ----------------------------------------------------------------------


def test_admitted_axial_record_is_unchanged_by_the_opt_in():
    opted = replace(axial_physics(), **OPT_IN)
    out = resolve_channeling([axial_case()], opted, Numerics())
    assert out[0]["channeling"] == axial_channeling_record(ADMITTED_AXIAL_CASES["si-fibre/A"])


def test_unadmitted_axial_refuses_by_default_naming_the_opt_in():
    thin = replace(axial_case(), thickness_ang=2000.0)
    with pytest.raises(AxialAdmissionError, match="thickness_ang.*accept-inaccurate"):
        resolve_channeling([thin], axial_physics(), Numerics())


def test_unadmitted_axial_runs_marked_without_budget():
    thin = replace(axial_case(), thickness_ang=2000.0)
    opted = replace(axial_physics(), **OPT_IN)
    with pytest.warns(UserWarning, match="very inaccurate"):
        (out,) = resolve_channeling([thin], opted, Numerics())
    record = out["channeling"]
    assert record["admission"] == "unadmitted"
    assert record["case_key"] == "unadmitted"
    assert not {"depth_budget", "free_free_upper", "evidence_id", "band_eV"} & set(record)
    admitted = resolve_channeling([axial_case()], axial_physics(), Numerics())[0]
    assert case_content_key(out) != case_content_key(admitted)
    from pyrite.montecarlo.channeling.axial_case import admitted_axial_case

    assert isinstance(admitted_axial_case(out), UnadmittedAxialCase)


def test_admitted_axial_case_cannot_run_as_unadmitted():
    thin = replace(axial_case(), thickness_ang=2000.0)
    opted = replace(axial_physics(), **OPT_IN)
    with pytest.warns(UserWarning):
        (out,) = resolve_channeling([thin], opted, Numerics())
    with pytest.raises(AxialAdmissionError, match="admitted as si-fibre/A"):
        replace(out, thickness_ang=3000.0)


def test_axial_solver_refusals_hold_under_the_opt_in():
    opted = replace(axial_physics(), **OPT_IN)
    with pytest.raises(AxialAdmissionError, match="on-axis"):
        resolve_channeling(
            [replace(axial_case(), thickness_ang=2000.0, theta_obs_rad=1e-3)], opted, Numerics()
        )
