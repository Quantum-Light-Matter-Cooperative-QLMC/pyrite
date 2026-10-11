"""Public policy relocation preserves source and observation content identity."""

import inspect
import pickle
import warnings
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite._api_deprecations import API_DEPRECATIONS
from pyrite.campaign.profiles import case_content_key


def scene(**kwargs):
    return pr.Scene(
        pr.Beam(30.0, transverse=pr.PointSpot(), longitudinal=pr.PointBunch()),
        pr.Slab("hopg", thickness_ang=1000.0, tilt_deg=30.0, footprint=None),
        **kwargs,
    )


def digest(case):
    # No transport or table resolution: both payloads use the same table markers.
    return case_content_key(case, xsgen_tables={})


@pytest.mark.parametrize("name", [f.name for f in fields(pr.Physics)])
def test_every_moved_default_preserves_case_and_digest(name):
    value = getattr(pr.Physics(), name)
    with pytest.warns(DeprecationWarning, match=f"Numerics\\({name}=") as caught:
        old = pr.Numerics(**{name: value})
    assert len(caught) == 1
    assert "0.14.0" in str(caught[0].message)
    assert "Physics" in str(caught[0].message)
    before = api.build_case(scene(), old)
    after = api.build_case(scene(physics=pr.Physics(**{name: value})))
    assert before == after
    assert digest(before) == digest(after)


@pytest.mark.parametrize(
    "choices",
    [
        {"straggling": True},
        {"inelastic_model": "continuous", "elastic_model": "mott"},
        {
            "inelastic_model": "shell-soft-hard",
            "inelastic_cutoff_eV": 80.0,
            "secondary_threshold_eV": 100.0,
        },
        {
            "bremsstrahlung_model": "eedl",
            "radiative_model": "uncoupled",
            "atomic_electron_deflection": "none",
        },
        {"emission": "both", "brem_source": "none"},
        {
            "bremsstrahlung_model": "bremslib",
            "radiative_model": "bremslib-soft-hard",
            "radiative_cutoff_eV": 80.0,
            "inelastic_model": "shell-soft-hard",
            "inelastic_cutoff_eV": 50.0,
            "secondary_threshold_eV": 100.0,
            "pair_production_model": "penelope-2024",
            "positron_transport": True,
        },
    ],
)
def test_nondefault_model_families_preserve_case_and_digest(choices):
    with pytest.warns(DeprecationWarning):
        old = pr.Numerics(**choices)
    before = api.build_case(scene(), old)
    after = api.build_case(scene(physics=pr.Physics(**choices)))
    assert before == after
    assert digest(before) == digest(after)


@pytest.mark.parametrize("planar", [False, True])
def test_grid_relocation_preserves_case_and_digest(planar):
    bins = pr.EnergyBins(cxr=np.array([1000.0, 1050.0]), brem=np.array([1000.0, 2000.0]))
    factory = (
        (
            lambda **kw: pr.PlanarDetector(
                pr.PlanarPose.from_observation(200.0, 90.0),
                pixels=pr.PixelGrid((2, 2), (0.1, 0.1)),
                **kw,
            )
        )
        if planar
        else pr.Detector
    )
    with pytest.warns(DeprecationWarning, match="energy_bins") as caught:
        legacy_detector = factory(energy_bins=bins)
    assert len(caught) == 1
    old = api.build_case(scene(detector=legacy_detector))
    new_scene = scene(detector=factory(), numerics=pr.Numerics(energy_bins=bins))
    new = api.build_case(new_scene)
    assert old == new
    assert digest(old) == digest(new)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        copied = replace(legacy_detector)
        assert api.build_case(scene(detector=copied), pr.Numerics(energy_bins=bins)) == new
    assert not caught
    with pytest.raises(ValueError, match="conflicting.*energy_bins"):
        api.build_case(scene(detector=legacy_detector), pr.Numerics(energy_bins=pr.EnergyBins()))


def test_pixel_observation_identity_is_unchanged_by_grid_relocation():
    bins = pr.EnergyBins(cxr=np.array([1000.0, 1050.0]), brem=np.array([1000.0, 2000.0]))
    detector = pr.PlanarDetector(
        pr.PlanarPose.from_observation(200.0, 90.0),
        pixels=pr.PixelGrid((2, 2), (0.1, 0.1)),
        response=pr.IdealPhotonCounter(),
    )
    acquisition = pr.Acquisition(exposure_s=1.0, measured_edges_eV=(0.0, 1000.0, 2000.0))
    with pytest.warns(DeprecationWarning):
        old_detector = pr.PlanarDetector(
            detector.pose, pixels=detector.pixels, response=detector.response, energy_bins=bins
        )
    old = api.observation_plan(scene(detector=old_detector, acquisition=acquisition), pr.Numerics())
    new = api.observation_plan(
        scene(detector=detector, acquisition=acquisition), pr.Numerics(energy_bins=bins)
    )
    output = {
        "E_grid": bins.cxr,
        "E_grid_brem": bins.brem,
        "spec_by_direction": np.ones((1, 2)),
        "spec_characteristic_by_direction": np.zeros((1, 2)),
        "brem_wide_by_direction": np.ones((1, 2)),
    }
    assert old.assemble(output, {}).identity == new.assemble(output, {}).identity


def test_models_are_frozen_and_integration_constraints_remain():
    with pytest.raises(FrozenInstanceError):
        pr.Physics().straggling = True
    with pytest.raises(ValueError, match="midpoint"):
        api.build_case(
            scene(physics=pr.Physics(inelastic_model="shell-soft-hard", inelastic_cutoff_eV=50.0)),
            pr.Numerics(energy_model="frozen"),
        )
    with pytest.raises(ValueError, match="pair_production_model"):
        pr.Physics(positron_transport=True)
    with pytest.raises(ValueError, match="secondary_threshold_eV"):
        pr.Physics(pair_production_model="penelope-2024")


def test_deprecated_numerics_choices_survive_dataclass_copies_and_conflicts():
    with pytest.warns(DeprecationWarning):
        old = pr.Numerics(straggling=True)
    copied = replace(old, cxr_trials=10)
    assert api.build_case(scene(), copied)["straggling"] is True
    assert api.build_case(scene(physics=pr.Physics(straggling=True)), copied)["straggling"] is True
    with pytest.raises(ValueError, match="conflicting.*straggling"):
        api.build_case(scene(physics=pr.Physics()), copied)
    changed = replace(scene(), physics=pr.Physics(straggling=True))
    assert api.build_case(changed)["straggling"] is True


def test_reflection_alias_and_paths_preserve_cases():
    with pytest.warns(DeprecationWarning, match="Convergence"):
        old_reflections = pr.Convergence(n_families=2, mosaic_nodes=3)
    with pytest.warns(DeprecationWarning, match="convergence"):
        old_numerics = pr.Numerics(convergence=old_reflections)
    new_numerics = pr.Numerics(cxr_reflections=pr.CxrReflections(n_families=2, mosaic_nodes=3))
    assert api.build_case(scene(), old_numerics) == api.build_case(scene(), new_numerics)
    with pytest.warns(DeprecationWarning, match="convergence"):
        old_sweep = pr.Sweep(scene(), {"numerics.convergence.n_families": [2, 3]})
    new_sweep = pr.Sweep(scene(), {"numerics.cxr_reflections.n_families": [2, 3]})
    assert old_sweep.cases() == new_sweep.cases()


def test_physics_and_sampling_sweep_paths_use_each_expanded_policy():
    with pytest.warns(DeprecationWarning, match="straggling"):
        old = pr.Sweep(
            scene(), {"numerics.straggling": [False, True], "numerics.cxr_trials": [10, 20]}
        )
    new = pr.Sweep(scene(), {"physics.straggling": [False, True], "numerics.cxr_trials": [10, 20]})
    cases = new.cases(pr.Numerics(cxr_trials=1, brem_trials=2))
    assert cases == old.cases(pr.Numerics(cxr_trials=1, brem_trials=2))
    assert [c["Ne"] for c in cases] == [10, 20, 10, 20]
    assert [c.get("straggling", False) for c in cases] == [False, False, True, True]


def test_deprecated_grid_paths_preserve_other_grids_and_warn_once():
    bins = pr.EnergyBins(cxr=np.array([1000.0, 1050.0]), brem=np.array([1000.0, 2000.0]))
    with pytest.warns(DeprecationWarning):
        detector = pr.Detector(energy_bins=bins)
    grids = [np.array([1100.0, 1150.0]), np.array([1200.0, 1250.0])]
    with pytest.warns(DeprecationWarning, match="energy_bins") as caught:
        old = pr.Sweep(scene(detector=detector), {"detector.energy_bins.cxr": grids})
    assert len(caught) == 1
    new = pr.Sweep(
        scene(numerics=pr.Numerics(energy_bins=bins)), {"numerics.energy_bins.cxr": grids}
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        assert old.cases() == new.cases()
    assert not caught
    for _, expanded in old.expand():
        np.testing.assert_array_equal(expanded.numerics.energy_bins.brem, bins.brem)


def test_alias_conflicts_and_equal_inputs_are_explicit():
    with pytest.warns(DeprecationWarning):
        old = pr.Convergence(n_families=2)
    with pytest.warns(DeprecationWarning):
        same = pr.Numerics(convergence=old, cxr_reflections=pr.CxrReflections(n_families=2))
    assert same.cxr_reflections.n_families == 2
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="conflicting"):
            pr.Numerics(convergence=old, cxr_reflections=pr.CxrReflections(n_families=3))
    with pytest.warns(DeprecationWarning):
        same_scene = scene(physics=pr.Physics(emission="both"), emission="both")
    assert same_scene.physics.emission == "both"
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="conflicting"):
            scene(physics=pr.Physics(), emission="both")
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="conflicting.*axis"):
            pr.Sweep(scene(), {"numerics.straggling": [False], "physics.straggling": [True]})


def test_simulate_forwards_legacy_policy_once_and_preserves_identity(monkeypatch):
    def run(case, **kwargs):
        return {
            "E_grid": np.array([1.0, 2.0]),
            "spec": np.array([3.0, 4.0]),
            "brem": np.array([5.0, 6.0]),
            "spec_coherent": np.array([7.0, 8.0]),
        }

    monkeypatch.setattr(api, "run_case", run)
    s = scene()
    with pytest.warns(DeprecationWarning) as caught:
        old = pr.simulate(s.beam, s.target, s.detector, emission="both", brem_source="none")
    assert len(caught) == 2
    new = pr.simulate(
        s.beam, s.target, s.detector, physics=pr.Physics(emission="both", brem_source="none")
    )
    assert old.case == new.case
    assert old.provenance["identity_digest"] == new.provenance["identity_digest"]
    np.testing.assert_array_equal(old.brem.density, new.brem.density)
    assert new.scene.physics.emission == "both"
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="conflicting"):
            pr.simulate(s.beam, s.target, s.detector, physics=pr.Physics(), emission="both")


def test_ownership_deprecations_are_registered_with_two_minor_window():
    for name in (
        "Convergence",
        "Numerics(convergence=)",
        "simulate(emission=)",
        "simulate(brem_source=)",
        "Detector(energy_bins=)",
        "PlanarDetector(energy_bins=)",
    ):
        entry = API_DEPRECATIONS[name]
        assert entry.deprecated_in == "0.12.0"
        assert entry.remove_in == "0.14.0"


def test_legacy_positionals_and_pickle_paths_stay_usable():
    with pytest.warns(DeprecationWarning, match="straggling") as caught:
        numerics = pr.Numerics(450, 100, None, None, "auto", "auto", True)
    assert len(caught) == 1
    assert api.build_case(scene(), numerics)["straggling"] is True
    base = scene()
    with pytest.warns(DeprecationWarning, match="Scene\\(emission="):
        old = pr.Scene(base.beam, base.target, base.detector, (), None, None, "both")
    assert old.physics.emission == "both"
    bins = pr.EnergyBins(cxr=np.array([1000.0, 1050.0]))
    assert pickle.loads(pickle.dumps(bins)) == bins
    assert pr.EnergyBins.__module__ == "pyrite.detectors.spec"
    for cls in (pr.Scene, pr.Numerics, pr.Detector, pr.PlanarDetector):
        assert not any(name.startswith("_") for name in inspect.signature(cls).parameters)
