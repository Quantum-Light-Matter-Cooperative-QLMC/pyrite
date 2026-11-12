"""#994 milestone 3: public fibre selector, closed admission and cache identity."""

from dataclasses import replace

import pytest

from pyrite import Physics
from pyrite.campaign.axial_cases import axial_channeling_record
from pyrite.campaign.channeling_cases import resolve_channeling
from pyrite.montecarlo.case import Case
from pyrite.quantum import Channeling
from pyrite.quantum._admission import AxialAdmissionError
from pyrite.quantum._axial_envelope import ADMITTED_AXIAL_CASES
from pyrite.quantum._axial_spec import (
    AXIAL_FIBRE_MODEL_MARKER,
    axial_spec_identity,
    resolve_axial_spec,
)
from pyrite.quantum._channeling import validate_channeling_run


def spec(axis=(0, 0, 1)):
    return {"version": 2, "axis_uvw": list(axis)}


def channeling():
    return Channeling("axial", axial=spec())


def case(*, tilt=0.088, azimuth=225.0, coherent=False):
    return Case(
        name="si-axial",
        crystal="silicon",
        composition=[("Si", 0.05)],
        hkl_list=[(1, 1, 1)],
        B_ang2=0.0,
        E0_keV=3500.0,
        thickness_ang=3000.0,
        crystal_width_mm=None,
        crystal_height_mm=None,
        beam_fwhm_mm=None,
        E_grid=(800.0, 10000.0, 50.0),
        E_grid_line=(800.0, 10000.0, 50.0),
        E_grid_brem=(800.0, 10000.0, 50.0),
        theta_obs_rad=0.0,
        tilt_deg=tilt,
        tilt_azim_deg=azimuth,
        beam_uvw=None,
        surface_hkl=(0, 0, 1),
        mosaic_fwhm_rad=None,
        mosaic_mc_fwhm_rad=None,
        mosaic_mc_nodes=1,
        abs_layers=None,
        layer_radiators=None,
        brem_file=None,
        Ne=8,
        Ne_brem=8,
        seed=0,
        spec_chunk=None,
        brem_chunk=None,
        dtheta_obs_rad=0.0,
        domega_sr=0.0,
        energy_model="midpoint",
        **({"coherent_emission": True} if coherent else {}),
    )


def test_public_v2_is_minimal_roundtripping_and_distinct_from_v1():
    normalized = resolve_axial_spec(spec((0, 0, 2)))
    assert normalized == {"version": 2, "model": "axial", "axis_uvw": [0, 0, 1]}
    assert resolve_axial_spec(normalized) == normalized
    assert channeling().axial == normalized
    assert axial_spec_identity(normalized) == axial_spec_identity(spec())
    v1 = resolve_axial_spec(
        {
            "axis_uvw": [0, 0, 1],
            "zone_max_ang": 1500.0,
            "entry": {"version": 1, "kind": "gaussian", "waist_ang": 0.5},
        }
    )
    assert axial_spec_identity(normalized) != axial_spec_identity(v1)
    assert AXIAL_FIBRE_MODEL_MARKER == "axial-fibre-v2"


@pytest.mark.parametrize(
    "additional",
    [
        {"temperature_K": 293.15},
        {"seed": 0},
        {"entry": {}},
        {"numerics": {}},
        {"zone_max_ang": 3000},
        {"boundary": "max"},
        {"source": "fibre"},
        {"free_free": "native"},
    ],
)
def test_public_v2_never_accepts_caller_physics(additional):
    with pytest.raises(ValueError, match="unknown"):
        resolve_axial_spec({**spec(), **additional})


def test_repository_record_is_detached_and_includes_provenance():
    entry = ADMITTED_AXIAL_CASES["si-fibre/A"]
    rec = axial_channeling_record(entry)
    assert rec["model_marker"] == "axial-fibre-v2"
    assert rec["free_free"] == "bound"
    assert rec["free_free_upper"] == entry.free_free_upper
    assert rec["fibre_numerics"]["boundary"] == "max"
    assert rec["depth_budget"] == entry.depth_budget
    assert rec["evidence_id"] == entry.evidence_id
    rec["fibre_numerics"]["max_order"] = 3
    assert axial_channeling_record(entry)["fibre_numerics"]["max_order"] == 9


def test_case_a_resolves_with_its_declared_bound():
    out = resolve_channeling([case()], channeling())
    assert out[0]["channeling"] == axial_channeling_record(ADMITTED_AXIAL_CASES["si-fibre/A"])
    assert out[0]["channeling"]["free_free"] == "bound"


def test_case_b_resolves_with_its_declared_bound():
    entry = ADMITTED_AXIAL_CASES["si-fibre/B"]
    out = resolve_channeling([case(tilt=0.0)], channeling())
    assert out[0]["channeling"]["case_key"] == entry.key
    assert out[0]["channeling"]["free_free_upper"] == entry.free_free_upper


@pytest.mark.parametrize("tilt", [0.088, 0.0])
def test_coherent_emission_is_refused_for_every_axial_case(tilt):
    # Native CBS is scored per flight regardless of the flag; it adds only
    # bunch-phased scoring, which has no axial contract (#994).
    with pytest.raises(ValueError, match="incoherent emission"):
        validate_channeling_run(
            channeling(),
            Physics(emission=("incoherent", "coherent"), radiative_model="uncoupled"),
            "midpoint",
        )
    with pytest.raises(AxialAdmissionError, match="does not support coherent emission"):
        resolve_channeling([case(tilt=tilt, coherent=True)], channeling())


def test_v2_refuses_other_energies_observables_and_tampered_records():
    with pytest.raises(AxialAdmissionError, match="kinetic_energy"):
        resolve_channeling([replace(case(), E0_keV=3000)], channeling())
    with pytest.raises(AxialAdmissionError, match="on-axis"):
        resolve_channeling([replace(case(), theta_obs_rad=0.001)], channeling())
    resolved = resolve_channeling([case()], channeling())[0]
    fake = dict(resolved["channeling"])
    fake["depth_budget"] = 0.0
    with pytest.raises(AxialAdmissionError, match="differs from repository"):
        replace(resolved, channeling=fake)
