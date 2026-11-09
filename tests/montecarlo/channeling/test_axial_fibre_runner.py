"""#994 milestone 4: calibrated axial fibre cases through the standard runner.

No lookup entry is runnable in the repository yet (the adapter claim is not
anchored), so the runner tests mark si-fibre/B runnable for themselves.
"""

from dataclasses import replace

import numpy as np
import pytest

import pyrite as pr
from pyrite.campaign.lowering import build_case
from pyrite.checkpoints import _checkpoint_store
from pyrite.config import FixedSampling
from pyrite.detectors import EnergyBins, PointDetector
from pyrite.montecarlo import runner
from pyrite.montecarlo.case import (
    AxialAdmissionError,
    admit_axial_channeling,
    refuse_axial_channeling,
    refuse_channeling,
)
from pyrite.montecarlo.channeling.axial_case import axial_zone_frame
from pyrite.quantum import _axial_envelope
from pyrite.quantum._axial_envelope import ADMITTED_AXIAL_CASES
from pyrite.results.store import store_result

LINES, BREM = 4, 6
GRID = np.linspace(800.0, 10000.0, 47)


def fibre_case(key="B", *, lines=LINES, brem=BREM):
    """A calibrated case lowered from public objects alone."""
    tilt = dict(tilt_deg=0.088, tilt_azim_deg=225.0) if key == "A" else dict(tilt_deg=0.0)
    beam = pr.Beam(
        energy_keV=3500.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()
    )
    target = pr.geometry.Slab(
        "silicon",
        thickness_ang=3000.0,
        width=None,
        height=None,
        cut=pr.geometry.CrystalCut(surface_hkl=(0, 0, 1)),
        allow_normal_incidence=True,
        **tilt,
    )
    numerics = pr.Numerics(
        sampling=FixedSampling(cxr_histories=lines, brem_histories=brem),
        energy_bins=EnergyBins(cxr=GRID, brem=GRID),
    )
    physics = pr.Physics(
        channeling_model="axial",
        channeling_axial={"version": 2, "axis_uvw": [0, 0, 1]},
        emission="incoherent",
        radiative_model="uncoupled",
    )
    return build_case(pr.Scene(beam, target, PointDetector()), numerics, physics=physics)


@pytest.fixture(scope="module")
def case_b():
    return fibre_case("B")


@pytest.fixture(scope="module")
def runnable_b():
    """Module-wide stand-in for the repository marking si-fibre/B runnable."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_axial_envelope, "RUNNABLE_AXIAL_CASES", frozenset({"si-fibre/B"}))
        yield


def test_no_calibrated_case_is_runnable_in_the_repository(case_b):
    assert _axial_envelope.RUNNABLE_AXIAL_CASES == frozenset()
    for case in (case_b, case_b.to_dict(), fibre_case("A")):
        with pytest.raises(AxialAdmissionError, match="not yet runnable"):
            admit_axial_channeling(case)
    with pytest.raises(AxialAdmissionError, match="not yet runnable"):
        runner.run_cases([case_b], engine="cpu", progress=False)
    with pytest.raises(AxialAdmissionError, match="not yet runnable"):
        runner.run_case(case_b)
    with pytest.raises(AxialAdmissionError, match="not yet runnable"):
        store_result({}, case_b, {})


def test_runner_admits_a_runnable_case_and_no_caller_record(case_b, runnable_b):
    admit_axial_channeling(case_b)
    admit_axial_channeling(case_b.to_dict())
    assert case_b["channeling"]["case_key"] == "si-fibre/B"
    forged = case_b.to_dict()
    forged["channeling"] = {**forged["channeling"], "depth_budget": 0.0}
    with pytest.raises(AxialAdmissionError, match="differs from repository"):
        admit_axial_channeling(forged)
    with pytest.raises(AxialAdmissionError, match="#623"):
        admit_axial_channeling({"channeling_model": "axial", "channeling_axial": {"version": 1}})
    with pytest.raises(AxialAdmissionError, match="#623"):
        admit_axial_channeling({"channeling": {"model": "axial", "axis_uvw": [0, 0, 1]}})
    admit_axial_channeling({"channeling": None})


def test_marking_one_case_runnable_admits_no_other(runnable_b):
    case = fibre_case("A")
    assert case["channeling"]["free_free"] == "native"
    with pytest.raises(AxialAdmissionError, match="not yet runnable"):
        admit_axial_channeling(case)


def test_other_consumers_and_backends_refuse_before_work(case_b, runnable_b):
    with pytest.raises(AxialAdmissionError, match="recompute does not support axial fibre"):
        refuse_axial_channeling(case_b, "recompute")
    with pytest.raises(AxialAdmissionError, match="trajectory capture does not support"):
        refuse_channeling(case_b, "trajectory capture")
    with pytest.raises(AxialAdmissionError, match="CPU only"):
        runner.run_cases([case_b], engine="gpu", progress=False)
    with pytest.raises(AxialAdmissionError, match="directional scoring"):
        runner.run_case(case_b, observation_directions=np.array([[0.0, 0.0, 1.0]]))
    with pytest.raises(AxialAdmissionError, match="electron-block transport does not support"):
        runner._transport_case(case_b, block_electrons=2)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (dict(crystal_width_mm=1.0, crystal_height_mm=1.0), "laterally unbounded"),
        (dict(energy_spread_frac=1e-3), "energy spread"),
        (dict(dtheta_obs_rad=1e-3), "zero-aperture"),
        (dict(theta_obs_rad=1e-3), "zero-aperture"),
        (dict(E_grid_line=np.linspace(500.0, 10000.0, 20)), "calibrated band"),
        (dict(E0_keV=3000.0), "kinetic_energy_eV"),
        (dict(thickness_ang=6000.0), "thickness_ang"),
        (dict(tilt_deg=0.05), "tilt_deg"),
        (dict(coherent_emission=True), "coherent emission"),
        (dict(energy_model="exact"), "midpoint"),
    ],
)
def test_cases_outside_the_envelope_refuse_at_construction(case_b, change, message):
    with pytest.raises(ValueError, match=message):
        replace(case_b, **change)


def test_a_finite_aperture_detector_refuses_at_lowering():
    beam = pr.Beam(
        energy_keV=3500.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()
    )
    target = pr.geometry.Slab(
        "silicon",
        thickness_ang=3000.0,
        tilt_deg=0.0,
        width=None,
        height=None,
        cut=pr.geometry.CrystalCut(surface_hkl=(0, 0, 1)),
        allow_normal_incidence=True,
    )
    physics = pr.Physics(
        channeling_model="axial",
        channeling_axial={"version": 2, "axis_uvw": [0, 0, 1]},
        radiative_model="uncoupled",
    )
    numerics = pr.Numerics(
        sampling=FixedSampling(cxr_histories=2, brem_histories=2),
        energy_bins=EnergyBins(cxr=GRID, brem=GRID),
    )
    for detector, message in (
        (pr.Detector(observation_angle_deg=0.0), "zero-aperture"),
        (PointDetector(observation_angle_deg=1.0), "zero-aperture"),
    ):
        with pytest.raises(AxialAdmissionError, match=message):
            build_case(pr.Scene(beam, target, detector), numerics, physics=physics)


def test_axis_must_be_the_slab_normal(case_b):
    assert np.allclose(axial_zone_frame(case_b, (0, 0, 1))[:, 2], [0.0, 0.0, 1.0])
    cut = {**case_b.to_dict(), "surface_hkl": (1, 1, 1)}
    with pytest.raises(AxialAdmissionError, match="slab normal"):
        axial_zone_frame(cut, (0, 0, 1))


@pytest.fixture(scope="module")
def produced(case_b, runnable_b):
    transport = runner._transport_case(case_b)
    return transport, runner._spectrum_case(case_b, transport)


@pytest.mark.slow
def test_one_fibre_history_gives_the_source_and_the_fates(case_b, produced):
    from pyrite.materials.attenuation import linear_attenuation_inv_mm
    from pyrite.montecarlo.channeling.axial_fiber_case import axial_fiber_inputs
    from pyrite.montecarlo.channeling.axial_fiber_regression import fiber_on_axis_source

    transport, out = produced
    entry = ADMITTED_AXIAL_CASES["si-fibre/B"]
    inputs = axial_fiber_inputs(entry)
    direct = fiber_on_axis_source(
        inputs.channels,
        inputs.entry_orders,
        kinetic_energy_eV=entry.kinetic_energy_eV,
        hbar_v_eV_ang=inputs.hbar_v_eV_ang,
        depths_ang=inputs.depths_ang,
        photon_energy_eV=GRID,
        incident_ids=range(LINES),
        seed=int(case_b["seed"]),
        carrier_inv_ang=inputs.carrier_inv_ang,
        boundary_eV=inputs.boundary_eV,  # boundary="max": no free->free scored
        absorption_inv_ang=linear_attenuation_inv_mm("silicon", GRID) * 1e-7,
        exit_path_ang=inputs.depths_ang[-1] - inputs.depths_ang,
        search_step_ang=inputs.search_step_ang,
    )
    np.testing.assert_array_equal(out["E_grid"], GRID)
    np.testing.assert_array_equal(out["spec_channeling"], direct.spectrum)
    assert np.isfinite(out["spec_channeling"]).all()
    info = transport["segs"]["channeling"]
    kinds = [event.kind for event in direct.events]
    assert info["n_fibre_primaries"] == LINES
    assert info["n_fibre_retained_exit"] == kinds.count("retained")
    assert info["n_fibre_outgoing_native"] == kinds.count("outgoing") + kinds.count("tail")
    assert info["n_fibre_unresolved"] == kinds.count("unresolved")
    assert info["depth_budget"] == entry.depth_budget
    assert info["free_free_upper"] == entry.free_free_upper


@pytest.mark.slow
def test_zone_rows_launches_and_brem_only_electrons(case_b, produced):
    transport, _ = produced
    segs = transport["segs"]
    info = segs["channeling"]
    ids = np.asarray(segs["electron_id"])
    zone = np.asarray(segs["channeling_zone"])
    assert sorted(ids[zone]) == list(range(LINES))  # one zone row per line electron
    assert not zone[ids >= LINES].any()
    assert set(ids[ids >= LINES]) == set(range(LINES, BREM))  # native from the face
    first = np.asarray(segs["flight_id"]) == 0
    np.testing.assert_allclose(np.asarray(segs["v_hat"])[zone], [[0.0, 0.0, 1.0]] * LINES)
    assert np.all(np.asarray(segs["L_ang"])[zone] <= 3000.0)
    assert np.all(np.asarray(segs["E_end_keV"])[zone] < 3500.0)
    assert np.all(first[zone])
    launched = np.unique(ids[~zone & (ids < LINES)])
    assert launched.size == info["n_fibre_outgoing_native"]
    assert segs["n_transmitted"] >= info["n_fibre_retained_exit"]
    assert segs["Ne"] == BREM


@pytest.mark.slow
def test_source_and_budgets_round_trip_through_results_and_checkpoints(case_b, produced, tmp_path):
    _, out = produced
    records = {}
    store_result(records, case_b, dict(out))
    _checkpoint_store.save("silicon", tmp_path, records)
    loaded = _checkpoint_store.load("silicon", tmp_path)
    record = next(iter(next(iter(loaded.values())).values()))
    np.testing.assert_array_equal(record["spec_channeling"], out["spec_channeling"])
    stored = record["case"]["channeling"]
    entry = ADMITTED_AXIAL_CASES["si-fibre/B"]
    assert stored["model_marker"] == "axial-fibre-v1"
    assert stored["depth_budget"] == entry.depth_budget
    assert stored["free_free"] == "bound"
    assert stored["free_free_upper"] == entry.free_free_upper
    assert stored["evidence_record"] == entry.evidence_record


@pytest.mark.slow
def test_public_simulate_runs_a_runnable_case_from_public_inputs(runnable_b, produced):
    _, out = produced
    beam = pr.Beam(
        energy_keV=3500.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()
    )
    target = pr.geometry.Slab(
        "silicon",
        thickness_ang=3000.0,
        tilt_deg=0.0,
        width=None,
        height=None,
        cut=pr.geometry.CrystalCut(surface_hkl=(0, 0, 1)),
        allow_normal_incidence=True,
    )
    result = pr.simulate(
        beam,
        target,
        PointDetector(),
        physics=pr.Physics(
            channeling_model="axial",
            channeling_axial={"version": 2, "axis_uvw": [0, 0, 1]},
            radiative_model="uncoupled",
        ),
        numerics=pr.Numerics(
            sampling=FixedSampling(cxr_histories=LINES, brem_histories=BREM),
            energy_bins=EnergyBins(cxr=GRID, brem=GRID),
        ),
    )
    np.testing.assert_array_equal(result.channeling.photon_energy_eV, GRID)
    np.testing.assert_array_equal(result.channeling.density, out["spec_channeling"])
    assert result.case["channeling"]["case_key"] == "si-fibre/B"
