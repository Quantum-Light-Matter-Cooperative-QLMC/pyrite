"""The first axial envelope is empty, so even its candidate cases must refuse (#886).

A failure here means axial admission changed. Update
``docs/research/physics/axial-first-envelope-886.md`` and supply anchored source
and depth evidence for the admitted case; do not loosen these parameters.
"""

import pytest

from pyrite._channeling_admission import (
    AxialAdmissionError,
    refuse_axial_channeling,
    refuse_axial_channeling_cases,
)
from pyrite.montecarlo.case import refuse_axial_channeling as case_module_refusal

_RECORD = {"model": "axial", "axis_uvw": [0, 0, 1], "zone_max_ang": 1000.0}


def _case(**updates):
    return {"channeling": dict(_RECORD), "E0_keV": 5000.0, "crystal": "si", **updates}


CANDIDATES = {
    "candidate-si-5mev": {},
    "candidate-diamond-30mev": {"crystal": "diamond", "E0_keV": 30000.0},
    "energy-below-candidate": {"E0_keV": 500.0},
    "energy-above-candidate": {"E0_keV": 54500.0},
    "material-outside-candidate": {"crystal": "w"},
    "layered-material": {"crystal": "mos2"},
    "positron": {"particle": "positron"},
    "stack": {"layers": [{"thickness_ang": 100.0}, {"thickness_ang": 200.0}]},
    "groove": {"groove": {"pitch_nm": 50.0}},
    "mosaic": {"mosaic_fwhm_rad": 1e-3},
    "off-axis-detector": {"dtheta_obs_rad": 0.01},
    "cuda-backend": {"transport_backend": "cuda"},
    "adaptive-consumer": {"adaptive": True},
    "capture-consumer": {"trajectory_capture": True},
}


@pytest.mark.parametrize("updates", CANDIDATES.values(), ids=CANDIDATES.keys())
def test_every_case_refuses_while_the_envelope_is_empty(updates):
    with pytest.raises(AxialAdmissionError, match="#994"):
        refuse_axial_channeling(_case(**updates))


@pytest.mark.parametrize(
    "case",
    [
        {"channeling_model": "axial"},
        {"channeling_axial": {"axis_uvw": [0, 0, 1]}},
        {"channeling": {"model": "axial"}},
    ],
    ids=["unlowered-selector", "unlowered-spec", "lowered-record"],
)
def test_every_axial_selector_shape_refuses(case):
    with pytest.raises(AxialAdmissionError):
        refuse_axial_channeling(case)


def test_caller_supplied_evidence_cannot_grant_admission():
    case = _case(axial_depth_evidence=object(), channeling_axial_evidence=object())
    with pytest.raises(AxialAdmissionError):
        refuse_axial_channeling(case)


def test_batch_refuses_for_a_later_axial_case_before_any_is_consumed():
    consumed = []

    def cases():
        for case in ({"channeling_model": "off"}, _case()):
            consumed.append(case)
            yield case

    with pytest.raises(AxialAdmissionError):
        refuse_axial_channeling_cases(cases())
    assert len(consumed) == 2


def test_off_and_planar_cases_stay_admitted():
    refuse_axial_channeling({"channeling_model": "off"})
    refuse_axial_channeling({"channeling": {"model": "planar"}})
    refuse_axial_channeling({})


def test_case_module_reexports_the_single_admission_owner():
    assert case_module_refusal is refuse_axial_channeling
