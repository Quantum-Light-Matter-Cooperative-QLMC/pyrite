"""Expected CR source, slab coordinates, censoring and real checkpoint round trip."""

from types import SimpleNamespace

import numpy as np
import pytest

import pyrite as pr
from pyrite._spectral_components import cxr_spectrum
from pyrite.checkpoints import _checkpoint_store
from pyrite.detectors import EnergyBins
from pyrite.montecarlo.channeling.emission import _History
from pyrite.montecarlo.runner.channeling_emission import source_frame
from pyrite.results.store import store_result


def test_geometric_survival_integrates_each_path_once():
    # Independently: (int_0^0 + int_0^2 + int_0^7 + int_0^10) (2+3z) dz / 4.
    history = _History(np.array([0.0, 5.0, 10.0]), np.array([[2.0], [17.0], [32.0]]), np.ones(3))
    limits = np.array([0.0, 2.0, 7.0, 10.0])
    actual = history.censored(limits)
    expected = np.mean(2 * limits + 1.5 * limits**2)
    np.testing.assert_allclose(actual.integrate(actual.rates), [expected], rtol=1e-14)
    assert actual.censored(np.zeros(4)).integrate(actual.rates).item() == 0


def test_source_frame_is_right_handed_and_rotates_with_the_slab():
    beam = np.array([0.2, -0.1, 1.0])
    beam /= np.linalg.norm(beam)
    normal = np.array([0.7, 0.7, 0.0])
    normal /= np.linalg.norm(normal)
    frame = source_frame(beam, normal)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-14)
    assert np.linalg.det(frame) == pytest.approx(1.0)
    np.testing.assert_allclose(frame[:, 2], beam)
    np.testing.assert_allclose(normal @ frame[:, 1], 0.0, atol=1e-14)
    angle = 0.6
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    np.testing.assert_allclose(source_frame(rotation @ beam, rotation @ normal), rotation @ frame)


def test_producer_censors_misses_sides_and_cutoffs_using_only_the_line_population(monkeypatch):
    from pyrite.montecarlo.runner import channeling_emission as producer

    case = dict(
        crystal="silicon",
        composition=[("Si", 0.05)],
        E0_keV=100.0,
        thickness_ang=10.0,
        crystal_width_mm=4e-7,
        crystal_height_mm=4e-7,
        E_cut_lines_keV=1.0,
        channeling=dict(
            plane_hkl=(1, 0, 0),
            temperature_K=293.15,
            basis_size=8,
            bloch_cells=4,
            margin_eV=300.0,
            dz_ang=1.0,
        ),
    )
    monkeypatch.setattr(producer, "zone_solver", lambda *_args: (None, None))
    monkeypatch.setattr(
        producer, "case_plane_incidence", lambda _case: SimpleNamespace(plane_normal=[1, 0, 0])
    )

    def constant_source(_solver, path, grid, *_angles, **options):
        # A constant unit rate gives the mean physical source path directly.
        history = _History(np.array([0.0, path]), np.ones((2, 1)), np.ones(2))
        history = history.censored(options["path_limits_ang"])
        return SimpleNamespace(spectrum=np.full_like(grid, history.integrate(history.rates).item()))

    monkeypatch.setattr(producer, "channeling_spectrum", constant_source)
    # First primary hits its cutoff at 4 A; second misses; third hits a side
    # after 2.5 A. The fourth is a brem-only primary and must not normalize CR.
    transport = dict(
        Ne_lines=3,
        E_grid=np.array([1000.0, 2000.0]),
        n_hat=np.array([0, 0, 1]),
        channeling_beam_dir=np.array([0, 0, 1]),
        segs=dict(
            initial_r_ang=np.array([[0, 0, 0], [3, 0, 0], [0, 0, 0], [0, 0, 0]]),
            initial_v_hat=np.array([[0, 0, 1], [0, 0, 1], [0.8, 0, 0.6], [0, 0, 1]]),
            initial_E_keV=np.array([3.0, 100.0, 100.0, 100.0]),
            channeling=dict(
                path_ang=10.0, stopping_keV_per_ang=0.5, incidence_rad=0.0, divergence_rad=0.0
            ),
        ),
    )
    np.testing.assert_allclose(producer.case_channeling_spectrum(case, transport), (4 + 2.5) / 3)


@pytest.fixture(scope="module")
def produced():
    beam = pr.Beam(
        energy_keV=5000.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()
    )
    target = pr.geometry.Slab(
        "silicon",
        thickness_ang=5000.0,
        tilt_deg=3.0,
        tilt_azim_deg=45.0,
        footprint=pr.geometry.Footprint(5.0, 5.0),
    )
    numerics = pr.Numerics(
        cxr_trials=4,
        brem_trials=4,
        energy_bins=EnergyBins(
            cxr=np.linspace(1500.0, 3000.0, 31), brem=np.array([5000.0, 10000.0])
        ),
        channeling_basis_size=8,
        channeling_bloch_cells=4,
        channeling_dz_ang=500.0,
    )
    physics = pr.Physics(
        channeling_model="planar", channeling_plane=(1, -1, 0), radiative_model="uncoupled"
    )
    return pr.simulate(beam, target, pr.Detector(), physics=physics, numerics=numerics)


def test_public_result_produces_a_separate_finite_channel(produced):
    assert produced.channeling is not None
    np.testing.assert_array_equal(
        produced.channeling.photon_energy_eV, produced.cxr.incoherent.photon_energy_eV
    )
    assert np.isfinite(produced.channeling.density).all()
    assert np.any(produced.channeling.density > 0)
    np.testing.assert_allclose(
        produced.combined_cxr().density,
        produced.combined_cxr(include_channeling=False).density + produced.channeling.density,
    )


def test_produced_channel_round_trips_without_double_counting(produced, tmp_path):
    # Store the real producer output through the normal dataset/checkpoint owners.
    output = {
        "E_grid": produced.cxr.incoherent.photon_energy_eV,
        "spec": produced.cxr.incoherent.density,
        "spec_channeling": produced.channeling.density,
        "spec_characteristic": produced.characteristic.density,
        "E_grid_brem": produced.brem.photon_energy_eV,
        "brem_wide": produced.brem.density,
        "brem": np.interp(
            produced.cxr.incoherent.photon_energy_eV,
            produced.brem.photon_energy_eV,
            produced.brem.density,
        ),
        "eta": 0.0,
    }
    records = {}
    store_result(records, produced.case, output)
    _checkpoint_store.save("silicon", tmp_path, records)
    loaded = _checkpoint_store.load("silicon", tmp_path)
    record = next(iter(next(iter(loaded.values())).values()))
    np.testing.assert_array_equal(record["spec_channeling"], produced.channeling.density)
    np.testing.assert_allclose(cxr_spectrum(record), produced.combined_cxr().density)
