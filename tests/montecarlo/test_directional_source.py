"""Detector-free source geometry, shared scoring and rate contracts (#590)."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite import Beam, Footprint, Numerics, PointBunch, PointSpot, Slab
from pyrite.directional import observation_directions
from pyrite.radiation import directional_source
from pyrite.results.directional import DirectionalSpectrum


def source_result():
    angles = np.array([-180.0, -90.0, 0.0, 90.0])
    vectors = observation_directions(angles)
    return DirectionalSpectrum(
        angles,
        np.array([100.0, 200.0]),
        vectors,
        vectors,
        np.ones((4, 2)),
        np.ones(4, dtype=bool),
        {"bunch_charge_pc": 2.0, "rep_rate_hz": 3000.0},
        {},
    )


def test_rate_normalization_zero_linear_default_and_no_solid_angle():
    """Validation: directional-source-rate-normalization (BIPM exact e)."""
    result = source_result()
    np.testing.assert_array_equal(result.rate(0), 0)
    np.testing.assert_allclose(result.rate(1), 6.241509074460763e9, rtol=2e-15)
    np.testing.assert_allclose(result.rate(2), 2 * result.rate(1), rtol=2e-15)
    assert result.current_na == 6
    np.testing.assert_allclose(result.rate(), 3.744905444676458e10, rtol=2e-15)
    other = replace(result, case={**result.case, "domega_sr": 5.0, "dtheta_obs_rad": 0.3})
    np.testing.assert_array_equal(other.rate(), result.rate())
    # Changing angular sample density is not an angular-bin/acceptance factor.
    fine = replace(result, angle_deg=np.arange(8), density=np.ones((8, 2)))
    np.testing.assert_array_equal(fine.rate(1)[0], result.rate(1)[0])


@pytest.mark.parametrize("current", [-1, np.nan, np.inf, -np.inf])
def test_invalid_mean_current_rejected(current):
    with pytest.raises(ValueError, match="finite nonnegative"):
        source_result().rate(current)


def test_full_circle_basis_and_azimuth():
    np.testing.assert_allclose(
        observation_directions([-180, -90, 0, 90, 180]),
        [[0, 0, -1], [-1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, -1]],
        atol=2e-16,
    )
    np.testing.assert_allclose(
        observation_directions([-90, 90], 90), [[0, -1, 0], [0, 1, 0]], atol=2e-16
    )


@pytest.fixture
def inputs():
    return (
        Beam(30.0, transverse=PointSpot(), longitudinal=PointBunch()),
        Slab("silicon", 20.0, tilt_deg=30.0, tilt_azim_deg=45.0, footprint=None),
    )


def test_source_lowering_rotation_and_no_detector(inputs, monkeypatch):
    import pyrite.directional as owner
    from pyrite import detectors
    from pyrite.montecarlo.geometry import sample_to_lab_R

    def forbidden(*args, **kwargs):
        raise AssertionError("detector must not be constructed")

    def score(case, directions, **kwargs):
        assert case["domega_sr"] == 0
        from pyrite._energy_grid_encoding import decode_energy_grid

        np.testing.assert_array_equal(decode_energy_grid(case["E_grid"]), [100.0, 200.0])
        assert case["Ne"] == case["Ne_brem"] == 3
        assert kwargs["components"] == ("pxr-cbs",)
        values = {"pxr-cbs": np.ones((len(directions), 2))}
        return values, np.ones(len(directions), dtype=bool)

    monkeypatch.setattr(detectors.Detector, "__post_init__", forbidden)
    monkeypatch.setattr(owner, "source_outputs", score)
    monkeypatch.setattr(owner, "case_table_markers", lambda case: {})
    monkeypatch.setattr(owner, "run_provenance", lambda *args: {})
    result = directional_source(
        *inputs,
        energy_eV=[100.0, 200.0],
        angle_deg=[-180.0, -90.0, 0.0, 90.0],
        numerics=Numerics(cxr_trials=3, brem_trials=50),
        seed=42,
    )
    rotation = sample_to_lab_R(np.deg2rad(30), np.deg2rad(45))
    expected = observation_directions([-180.0, -90.0, 0.0, 90.0]) @ rotation
    np.testing.assert_allclose(result.direction_sample, expected, atol=1e-15)
    np.testing.assert_allclose(
        result.direction_sample @ rotation.T, result.direction_lab, atol=1e-15
    )
    assert result.case["seed"] == 42
    with pytest.raises(ValueError, match="domega_sr"):
        replace(result.case, domega_sr=-1.0)


def test_tr_lowering_sets_history_model_and_preserves_oriented_frame(inputs, monkeypatch):
    import pyrite.directional as owner

    def score(case, directions, **kwargs):
        assert case["transition_radiation"] == {"model": "planar-stack", "coherence": "history"}
        assert case["Ne"] == case["Ne_brem"] == 3
        return {"transition": np.ones((len(directions), 2))}, np.ones(len(directions), dtype=bool)

    monkeypatch.setattr(owner, "source_outputs", score)
    monkeypatch.setattr(owner, "case_table_markers", lambda case: {})
    monkeypatch.setattr(owner, "run_provenance", lambda *args: {})
    result = owner.directional_sources(
        *inputs,
        energy_eV=[100, 200],
        angle_deg=[-120, 60],
        numerics=Numerics(cxr_trials=3),
        components=("transition",),
    )["transition"]
    assert result.component == "transition-radiation-history"
    assert result.case["tilt_deg"] == 30 and result.case["tilt_azim_deg"] == 45
    with pytest.raises(ValueError, match="footprint"):
        owner.directional_sources(
            inputs[0],
            replace(inputs[1], footprint=Footprint(5, 5)),
            energy_eV=[100, 200],
            components=("transition",),
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"angle_deg": [-180.0, 180.0]}, "periodic"),
        ({"energy_eV": [0.0, 200.0]}, "positive"),
        ({"direction_batch": 0}, "positive"),
        ({"seed": -1}, "nonnegative"),
    ],
)
def test_invalid_requests(inputs, kwargs, match):
    options = {"energy_eV": [100.0, 200.0], **kwargs}
    with pytest.raises(ValueError, match=match):
        directional_source(*inputs, **options)


@pytest.mark.slow
def test_shared_transport_matches_fixed_direction_cxr_only(inputs, monkeypatch):
    import pyrite.directional as owner
    from pyrite import EnergyBins, Scene
    from pyrite.api import build_case
    from pyrite.montecarlo import runner

    controls = Numerics(
        cxr_trials=2, brem_trials=2, energy_bins=EnergyBins(cxr=np.linspace(100, 3000, 33))
    )
    case = build_case(Scene(*inputs, detector=None, numerics=controls))
    transport = runner._transport_case(case)
    calls = []

    def reuse(*args, **kwargs):
        calls.append(1)
        return transport

    monkeypatch.setattr(runner, "_transport_case", reuse)
    monkeypatch.setattr(owner, "case_table_markers", lambda case: {})
    monkeypatch.setattr(owner, "run_provenance", lambda *args: {})

    def forbidden(*args, **kwargs):
        raise AssertionError("excluded radiation channel scored")

    monkeypatch.setattr(runner, "_characteristic_from_segments", forbidden)
    monkeypatch.setattr(runner, "_brem_wide_from_segments", forbidden)
    results = []
    for batch in (1, 4):
        result = directional_source(
            *inputs,
            energy_eV=transport["E_grid"],
            angle_deg=[-180.0, -90.0, 0.0, 90.0],
            numerics=controls,
            direction_batch=batch,
        )
        results.append(result)
        for i, direction in enumerate(result.direction_sample):
            expected = runner._cxr_for_segments(
                transport["segs"],
                transport["E_grid"],
                result.case,
                direction,
                result.case.get("abs_layers"),
                None,
                Ne=2,
            )
            np.testing.assert_allclose(result.density[i], expected, rtol=1e-13, atol=0)
    assert len(calls) == 2  # exactly one transport request per map, never per angle
    np.testing.assert_array_equal(results[0].density, results[1].density)


@pytest.mark.slow
def test_shared_transport_components_match_fixed_direction_scorers(inputs, monkeypatch):
    import pyrite.directional as owner
    from pyrite import EnergyBins, Scene
    from pyrite.api import build_case
    from pyrite.directional import directional_sources
    from pyrite.montecarlo import runner

    controls = Numerics(
        cxr_trials=2, brem_trials=2, energy_bins=EnergyBins(cxr=np.linspace(500, 3000, 26))
    )
    case = build_case(Scene(*inputs, detector=None, numerics=controls))
    transport = runner._transport_case(case)
    monkeypatch.setattr(runner, "_transport_case", lambda *args, **kwargs: transport)
    monkeypatch.setattr(owner, "case_table_markers", lambda case: {})
    monkeypatch.setattr(owner, "run_provenance", lambda *args: {})
    results = directional_sources(
        *inputs,
        energy_eV=transport["E_grid"],
        angle_deg=[-120.0, 0.0, 60.0],
        numerics=controls,
        components=("pxr-cbs", "characteristic", "bremsstrahlung"),
    )
    assert [r.component for r in results.values()] == [
        "incoherent-pxr-cbs",
        "characteristic",
        "bremsstrahlung",
    ]
    for i, direction in enumerate(results["pxr-cbs"].direction_sample):
        args = (transport["segs"], transport["E_grid"], results["pxr-cbs"].case, direction)
        layers = results["pxr-cbs"].case.get("abs_layers")
        np.testing.assert_allclose(
            results["characteristic"].density[i],
            runner._characteristic_from_segments(*args, layers, Ne=transport["Ne_brem"]),
            rtol=1e-13,
        )
        np.testing.assert_allclose(
            results["bremsstrahlung"].density[i],
            runner._brem_wide_from_segments(*args, layers, Ne=transport["Ne_brem"]),
            rtol=1e-13,
        )


def test_grazing_explicitly_masked_without_scoring(monkeypatch):
    from types import SimpleNamespace

    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import directional_source as owner

    monkeypatch.setattr(
        runner,
        "_transport_case",
        lambda *args, **kw: {"segs": {}, "Ne_lines": 7, "E_grid": np.array([100.0, 200.0])},
    )
    monkeypatch.setattr(runner, "_segments_on_device", lambda segs: segs)
    seen = []

    def score(batch, slot):
        seen.append(batch.n_hat)
        assert batch.electron_limit == 7
        return SimpleNamespace(values=np.array([1.0, 2.0]))

    monkeypatch.setattr(
        owner, "select_emission", lambda case: SimpleNamespace(cxr=SimpleNamespace(evaluate=score))
    )
    values, supported = owner.source_outputs(
        {"E_grid": np.array([100.0, 200.0])}, [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    )
    np.testing.assert_array_equal(supported, [True, False, True])
    assert np.isnan(values["pxr-cbs"][1]).all()
    assert len(seen) == 2


def test_detector_acceptance_remains_strictly_positive():
    from pyrite import AngularAcceptance, Detector

    with pytest.raises(ValueError):
        Detector(acceptance=AngularAcceptance(polar_deg=1.0, solid_angle_sr=0.0))


def test_finite_footprint_scores_face_parallel_directions(monkeypatch):
    """Side faces give every direction an escape path; nothing is masked."""
    from types import SimpleNamespace

    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import directional_source as owner

    monkeypatch.setattr(
        runner,
        "_transport_case",
        lambda *args, **kw: {"segs": {}, "Ne_lines": 7, "E_grid": np.array([100.0, 200.0])},
    )
    monkeypatch.setattr(runner, "_segments_on_device", lambda segs: segs)
    monkeypatch.setattr(
        owner,
        "select_emission",
        lambda case: SimpleNamespace(
            cxr=SimpleNamespace(evaluate=lambda b, s: SimpleNamespace(values=np.ones(2)))
        ),
    )
    values, supported = owner.source_outputs(
        {"E_grid": np.array([100.0, 200.0]), "crystal_width_mm": 5.0, "crystal_height_mm": 5.0},
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    )
    assert supported.all() and np.isfinite(values["pxr-cbs"]).all()


def test_components_use_their_live_populations_and_slots(monkeypatch):
    """Characteristic/continuum use the brem population; unrequested never run."""
    from types import SimpleNamespace

    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import directional_source as owner

    monkeypatch.setattr(
        runner,
        "_transport_case",
        lambda *args, **kw: {
            "segs": {},
            "Ne_lines": 3,
            "Ne_brem": 5,
            "E_grid": np.array([100.0, 200.0]),
        },
    )
    monkeypatch.setattr(runner, "_segments_on_device", lambda segs: segs)
    seen = []

    def adapter(slot, scale):
        def evaluate(batch, requested):
            assert requested == slot
            seen.append((slot, batch.electron_limit))
            return SimpleNamespace(values=scale * np.ones(2))

        return SimpleNamespace(evaluate=evaluate)

    def forbidden(batch, slot):
        raise AssertionError("unrequested component scored")

    monkeypatch.setattr(
        owner,
        "select_emission",
        lambda case: SimpleNamespace(
            cxr=SimpleNamespace(evaluate=forbidden),
            characteristic=adapter("spec_characteristic", 2.0),
            continuum=adapter("brem_wide", 3.0),
        ),
    )
    values, _ = owner.source_outputs(
        {"E_grid": np.array([100.0, 200.0]), "crystal_width_mm": 5.0},
        [[0.0, 0.0, 1.0]],
        components=("characteristic", "bremsstrahlung"),
    )
    assert set(values) == {"characteristic", "bremsstrahlung"}
    np.testing.assert_array_equal(values["bremsstrahlung"], 3.0)
    assert seen == [("spec_characteristic", 5), ("brem_wide", 5)]
    with pytest.raises(ValueError, match="subset"):
        owner.source_outputs({"E_grid": np.array([1.0])}, [[0, 0, 1.0]], components=("tr",))


def test_line_window_ceiling_bound_and_kinetic_cap():
    """Validation: line-grid-kinematic-bandwidth; line-grid-emission-ceiling."""
    from pyrite.directional import line_window_ceiling_eV
    from pyrite.materials import CATALOG
    from pyrite.materials.crystal import reciprocal_g_vector

    g = reciprocal_g_vector((2, 2, 0), CATALOG.crystal("silicon").lattice)[1]
    gamma = 1 + 1000e3 / 510998.95
    beta = np.sqrt(1 - gamma**-2)
    case = {"crystal": "silicon", "hkl_list": [(1, 1, 1), (2, 2, 0)], "E0_keV": 1000.0}
    expected = 1973.269804 * beta * g / (1 - beta)
    assert expected <= line_window_ceiling_eV(case) < expected + 100
    # Doppler bound grows ~2 gamma^2: by 100 MeV energy conservation binds.
    assert line_window_ceiling_eV({**case, "E0_keV": 1e5}) == 1e8


def _escape_scan(footprint, keV, energy, angles):
    from pyrite._numerics import CxrReflections
    from pyrite.campaign.geometry import Footprint

    return directional_source(
        Beam(keV, transverse=PointSpot(), longitudinal=PointBunch()),
        Slab(
            "silicon",
            2000.0,
            tilt_deg=45.0,
            footprint=None if footprint is None else Footprint(footprint, footprint),
        ),
        energy_eV=energy,
        angle_deg=angles,
        numerics=Numerics(cxr_trials=2, cxr_reflections=CxrReflections(max_reflections=4)),
    )


def test_finite_target_escape_limits():
    """Infinite-footprint limit, deep physical face-parallel minimum, side exit.

    Validation: finite-transverse-crystal; segment-escape-average
    """
    angles = np.array([-60.0, -45.0, 0.0, 90.0, 135.0])
    energy = np.linspace(100.0, 2000.0, 65)
    infinite, wide, narrow = (_escape_scan(f, 30.0, energy, angles) for f in (None, 5.0, 0.01))
    face = [1, 4]  # tilt -/+ 90 deg: parallel to the slab faces
    np.testing.assert_array_equal(infinite.supported, [True, False, True, True, False])
    assert wide.supported.all() and narrow.supported.all()
    keep = infinite.supported
    np.testing.assert_allclose(wide.density[keep], infinite.density[keep], rtol=1e-12)
    yields = [np.trapezoid(r.density, energy, axis=1) for r in (wide, narrow)]
    # 2.5 mm of silicon is opaque at these energies; a 10 um footprint is not.
    assert np.all(yields[0][face] < 1e-6 * yields[0][2])
    assert np.all(yields[1][face] > 1e3 * yields[0][face])


def test_relativistic_forward_band_is_window_truncation():
    """#590 near-forward 1 MeV 'null': lines blueshift above a 5 keV stop."""
    angles = np.array([0.0, 30.0])
    result = _escape_scan(5.0, 1000.0, np.geomspace(100.0, 63_200.0, 400), angles)
    assert result.provenance["line_window_ceiling_eV"] > 5000
    energy, density = result.energy_eV, result.density
    total = np.trapezoid(density, energy, axis=1)
    below = np.trapezoid(density[:, energy <= 5000], energy[energy <= 5000], axis=1)
    assert total[0] > 0.5 * total[1]  # forward emission is not suppressed
    assert below[0] < 0.05 * total[0] and below[1] > 0.5 * total[1]
