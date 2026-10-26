"""Channeling direction scoring and physical observations share the source (#521)."""

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite.detectors import EnergyBins
from pyrite.montecarlo import runner


@pytest.fixture(scope="module")
def case():
    scene = pr.Scene(
        pr.Beam(5000.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab(
            "silicon",
            5000.0,
            tilt_deg=3.0,
            tilt_azim_deg=45.0,
            footprint=pr.geometry.Footprint(5.0, 5.0),
        ),
        pr.Detector(),
        physics=pr.Physics(
            channeling_model="planar", channeling_plane=(1, -1, 0), radiative_model="uncoupled"
        ),
    )
    return api.build_case(
        scene,
        pr.Numerics(
            cxr_trials=4,
            brem_trials=4,
            channeling_basis_size=8,
            channeling_bloch_cells=4,
            channeling_dz_ang=500.0,
            energy_bins=EnergyBins(
                cxr=np.linspace(1500.0, 3000.0, 31), brem=np.array([5000.0, 10000.0])
            ),
        ),
    )


@pytest.mark.slow
def test_directions_rescore_channeling_from_one_transport(case, monkeypatch):
    transport = runner._transport_case(case, transport_core="per-electron")
    calls = []

    def reuse(*args, **kwargs):
        calls.append(args[0])
        return transport

    monkeypatch.setattr(runner, "_transport_case", reuse)
    directions = np.array([[0.0, 0.0, 1.0], [0.01, 0.0, np.sqrt(1 - 0.01**2)]])
    actual = runner.run_case_directions(case, directions)
    assert len(calls) == 1
    for index, direction in enumerate(directions):
        expected = runner._spectrum_case(case, {**transport, "n_hat": direction})
        np.testing.assert_array_equal(
            actual["spec_channeling_by_direction"][index], expected["spec_channeling"]
        )
    assert np.any(actual["spec_channeling_by_direction"] > 0)
    assert not np.array_equal(*actual["spec_channeling_by_direction"])
    scalar = runner.run_case(case, observation_directions=directions)
    np.testing.assert_array_equal(
        scalar["directional"]["spec_channeling_by_direction"],
        actual["spec_channeling_by_direction"],
    )


@pytest.mark.slow
@pytest.mark.parametrize("acquisition", [False, True])
def test_planar_observation_carries_channeling_once(case, monkeypatch, acquisition):
    detector = pr.detectors.PlanarDetector(
        pr.detectors.PlanarPose((0.0, 0.0, 50.0)),
        pixels=pr.detectors.PixelGrid((1, 1), (0.1, 0.1)),
        response=pr.detectors.IdealPhotonCounter(),
        scorer=pr.config.PixelScorer((1, 1)),
    )
    scene = pr.Scene(
        pr.Beam(5000.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab("silicon", 5000.0, footprint=pr.geometry.Footprint(5.0, 5.0)),
        detector,
        physics=pr.Physics(
            channeling_model="planar", channeling_plane=(1, -1, 0), radiative_model="uncoupled"
        ),
        acquisition=(
            pr.observations.Acquisition(exposure_s=1.0, measured_edges_eV=(0, 1500, 3000))
            if acquisition
            else None
        ),
    )
    monkeypatch.setattr(api, "_source_case", lambda *_: (case, {}))

    def directional(source, directions, **kwargs):
        assert source is case
        shape = (len(directions), 2)
        return dict(
            E_grid=np.array([1500.0, 3000.0]),
            E_grid_brem=np.array([1500.0, 3000.0]),
            spec_by_direction=np.ones(shape),
            spec_characteristic_by_direction=np.full(shape, 2.0),
            spec_channeling_by_direction=np.full(shape, 4.0),
            brem_wide_by_direction=np.zeros(shape),
        )

    monkeypatch.setattr(api, "run_case_directions", directional)
    result = api.simulate(
        scene.beam, scene.target, detector, physics=scene.physics, acquisition=scene.acquisition
    )
    assert result.channeling is not None
    np.testing.assert_allclose(result.channeling.density, 4 * result.cxr.incoherent.density)
    np.testing.assert_allclose(result.combined_cxr().density, 7 * result.cxr.incoherent.density)
    if acquisition:
        plan = api.observation_plan(scene, pr.Numerics())
        assert plan.source_identity_digest == result.provenance["identity_digest"]
