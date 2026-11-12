"""Independent run arguments retain legacy values, identities and refusals (#695)."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite import Scene, Sweep, api
from pyrite.beams import Beam, PointBunch, Spot
from pyrite.config import FixedSampling, Numerics, Physics
from pyrite.detectors import Detector, EnergyBins
from pyrite.geometry import Footprint, Slab
from pyrite.quantum import Channeling, ChannelingNumerics
from tests.helpers.case_compare import assert_same_case


@pytest.fixture
def scene():
    return Scene(
        Beam(5_000.0, transverse=Spot(), longitudinal=PointBunch()),
        Slab("silicon", 1_000.0, tilt_deg=30.0, footprint=Footprint(5.0, 5.0)),
        Detector(),
        physics=Physics(bremsstrahlung_model="eedl", radiative_model="uncoupled"),
    )


@pytest.fixture
def numerics():
    return Numerics(
        sampling=FixedSampling(cxr_histories=1, brem_histories=1),
        energy_bins=EnergyBins(cxr=(1_000.0, 1_500.0), brem=(1_000.0, 1_500.0)),
    )


@pytest.fixture
def channeling():
    return Channeling(
        "planar",
        plane=(1, -1, 0),
        numerics=ChannelingNumerics(basis_size=12, bloch_cells=8, step_ang=100.0),
        admission="accept-inaccurate",
    )


@pytest.fixture
def transport(monkeypatch):
    calls = []

    def run(case, *, transport_core):
        calls.append((case, transport_core))
        return {
            "E_grid": np.array([1_000.0, 1_500.0]),
            "spec": np.array([2.0, 3.0]),
            "brem": np.array([4.0, 5.0]),
        }

    monkeypatch.setattr(api, "run_case", run)
    return calls


def test_one_scene_runs_under_independent_configurations(scene, numerics, transport):
    original = replace(scene)
    first = api.simulate(scene, numerics=numerics)
    second = api.simulate(
        scene,
        physics=replace(scene.physics, brem_source="none"),
        numerics=replace(
            numerics,
            sampling=replace(numerics.sampling, cxr_histories=2),
            transport_core="per-electron",
        ),
    )

    np.testing.assert_array_equal(first.brem.density, [4.0, 5.0])
    np.testing.assert_array_equal(second.brem.density, [0.0, 0.0])
    assert first.case["Ne"] == 1
    assert second.case["Ne"] == 2
    assert first.provenance.identity_digest != second.provenance.identity_digest
    assert [core for _, core in transport] == ["auto", "per-electron"]
    assert scene == original


def test_channeling_component_scene_and_embedded_calls_are_equivalent(
    scene, numerics, channeling, transport
):
    independent = api.simulate(scene, numerics=numerics, channeling=channeling)
    components = api.simulate(
        scene.beam,
        scene.target,
        scene.detector,
        physics=scene.physics,
        numerics=numerics,
        channeling=channeling,
    )
    embedded = api.simulate(replace(scene, numerics=numerics, channeling=channeling))

    record = independent.case["channeling"]
    assert (record["plane_hkl"], record["basis_size"], record["bloch_cells"]) == (
        (1, -1, 0),
        12,
        8,
    )
    assert record["dz_ang"] == 100.0
    for result in (components, embedded):
        assert result.case.to_dict() == independent.case.to_dict()
        assert result.provenance.identity_digest == independent.provenance.identity_digest
        np.testing.assert_array_equal(
            result.cxr.incoherent.density, independent.cxr.incoherent.density
        )
        assert result.provenance.scene.channeling is channeling
    assert independent.provenance["channeling"] is channeling
    assert scene.channeling is None


def test_channeling_omission_override_and_explicit_disable(scene, numerics, channeling):
    embedded = replace(scene, channeling=channeling)
    inherited = api.build_case(embedded, numerics)
    disabled = api.build_case(embedded, numerics, channeling=None)
    changed = api.build_case(
        embedded,
        numerics,
        channeling=replace(channeling, numerics=replace(channeling.numerics, step_ang=125.0)),
    )

    assert inherited["channeling"]["dz_ang"] == 100.0
    assert changed["channeling"]["dz_ang"] == 125.0
    assert disabled.to_dict() == api.build_case(scene, numerics).to_dict()
    assert embedded.channeling is channeling


@pytest.mark.parametrize("mode", ["unsupported-physics", "invalid-type"])
def test_channeling_errors_precede_material_preparation(
    scene, numerics, channeling, monkeypatch, mode
):
    from pyrite.campaign import lowering

    def forbidden(*args, **kwargs):
        pytest.fail("invalid composition reached case/material preparation")

    monkeypatch.setattr(lowering, "build_cases", forbidden)
    physics = scene.physics
    if mode == "unsupported-physics":
        physics = replace(physics, emission=("incoherent", "coherent"))
        error, message = ValueError, "incoherent emission only"
    else:
        channeling = object()
        error, message = TypeError, "channeling must be a Channeling or None"
    with pytest.raises(error, match=message):
        api.simulate(scene, numerics=numerics, physics=physics, channeling=channeling)
    with pytest.raises(error, match=message):
        api.build_case(scene, numerics, physics=physics, channeling=channeling)


def test_sweep_keeps_channeling_controls_and_seed_order(scene, numerics, channeling):
    sweep = Sweep(replace(scene, numerics=numerics), {"beam.energy_keV": (5000, 6000)})
    cases = sweep.cases(channeling=channeling)
    assert [case["seed"] for case in cases] == [1, 2]
    assert [case["E0_keV"] for case in cases] == [5000, 6000]
    assert all(case["channeling"]["dz_ang"] == 100.0 for case in cases)
    assert sweep.base.channeling is None


def test_configured_channeling_observation_matches_scene_lowering(scene, numerics, channeling):
    from pyrite.campaign.legacy import adapt_configured_sweep
    from pyrite.campaign.sweep import Sweep as ConfiguredSweep
    from pyrite.campaign.sweep import build_cases
    from pyrite.observations import Observation
    from pyrite.results import Settings

    old = ConfiguredSweep(
        material="silicon",
        beam=scene.beam.to_spec(),
        target=scene.target,
        detector=scene.detector,
    )
    sweep = adapt_configured_sweep(old, Settings(n_electrons=1, n_electrons_brem=1))
    observation = Observation(detector=Detector(45.0))
    actual = sweep.cases(
        numerics, physics=scene.physics, channeling=channeling, observation=observation
    )[0]
    expected = api.build_case(scene, numerics, channeling=channeling, observation=observation)
    assert actual.to_dict() == expected.to_dict()
    # Resolving another request leaves the adapter's original physics untouched.
    np.testing.assert_equal(sweep.cases()[0].to_dict(), build_cases(old, 1, 1)[0].to_dict())


def test_configured_observation_changes_direction_and_source_identity():
    from pyrite.campaign.legacy import adapt_configured_sweep
    from pyrite.campaign.profiles import case_content_key
    from pyrite.campaign.sweep import Sweep as ConfiguredSweep
    from pyrite.observations import Observation
    from pyrite.results import Settings

    old = ConfiguredSweep(
        material="hopg",
        beam=Beam(30.0, transverse=Spot(), longitudinal=PointBunch()).to_spec(),
        thickness_ang=1000.0,
        tilt_deg=30.0,
    )
    sweep = adapt_configured_sweep(old, Settings(n_electrons=1, n_electrons_brem=1))
    original = sweep.cases()[0]
    changed = sweep.cases(observation=Observation(detector=Detector(45.0)))[0]
    assert changed["theta_obs_rad"] == pytest.approx(np.pi / 4)
    assert case_content_key(changed) != case_content_key(original)
    assert_same_case(sweep.cases()[0].to_dict(), original.to_dict())


def test_result_records_independent_effective_configuration(scene, numerics, transport):
    physics = replace(scene.physics, brem_source="none")
    result = api.simulate(scene, physics=physics, numerics=numerics)
    assert result.provenance["input_scene"] is scene
    assert result.provenance["physics"] == physics
    assert result.provenance["channeling"] is None
    assert result.provenance["acquisition"] is None
    assert scene.physics.brem_source != "none"
