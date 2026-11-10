"""Lowered cases resolve to the exact axial admission query (#994)."""

import pytest

from pyrite.campaign.axial_cases import axial_beam_angles, axial_case_query
from pyrite.quantum._admission import AxialAdmissionError
from pyrite.quantum._axial_envelope import ADMITTED_AXIAL_CASES, match_axial_case


def _case(**updates):
    base = dict(
        name="si-axial",
        crystal="silicon",
        beam_uvw=None,
        surface_hkl=(0, 0, 1),
        tilt_deg=0.088,
        tilt_azim_deg=225.0,
        E0_keV=3500.0,
        thickness_ang=3000.0,
    )
    return {**base, **updates}


def test_profile_azimuth_follows_the_lab_azimuth_one_to_one():
    angles = [axial_beam_angles(_case(tilt_azim_deg=a), (0, 0, 1)) for a in (225.0, 270.0)]
    assert angles[0][0] == pytest.approx(0.088, rel=1e-12)
    assert angles[0][1] == pytest.approx(0.0, abs=1e-9)
    assert angles[1][1] == pytest.approx(45.0, abs=1e-9)


def test_case_a_on_a_default_si001_slab_matches_its_lookup_entry():
    query = axial_case_query(_case(), (0, 0, 1), 293.15)
    assert match_axial_case(query) is ADMITTED_AXIAL_CASES["si-fibre/A"]


@pytest.mark.parametrize(
    "updates",
    [{"tilt_azim_deg": 0.0}, {"tilt_deg": 0.1}, {"E0_keV": 3000.0}, {"thickness_ang": 6000.0}],
    ids=["azimuth", "tilt", "energy", "thickness"],
)
def test_nearby_lab_setups_refuse(updates):
    with pytest.raises(AxialAdmissionError):
        match_axial_case(axial_case_query(_case(**updates), (0, 0, 1), 293.15))


def test_beam_against_the_axis_refuses():
    with pytest.raises(ValueError, match="against axis"):
        axial_beam_angles(_case(), (0, 0, -1))


def test_exact_alignment_has_zero_azimuth():
    assert axial_beam_angles(_case(tilt_deg=0.0), (0, 0, 1)) == (0.0, 0.0)
