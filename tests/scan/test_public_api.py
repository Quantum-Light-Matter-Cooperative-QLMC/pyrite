import json

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite.campaign.sweep import Sweep as LegacySweep
from pyrite.campaign.sweep import build_cases
from pyrite.detectors import EnergyBins
from pyrite.montecarlo import run_case
from pyrite.montecarlo.spectrum import (
    BREM_ENDF_PARSERPY_VERSION,
    BREMSSTRAHLUNG_MODEL,
    CHARACTERISTIC_MODEL,
    CHARACTERISTIC_XRAYDB_VERSION,
)
from pyrite.montecarlo.transport import STOPPING_MODEL


def _inputs():
    beam = pr.Beam(energy_keV=30.0, transverse=pr.Spot(), longitudinal=pr.PointBunch())
    target = pr.Slab("hopg", thickness_ang=1_000.0, tilt_deg=30.0, footprint=pr.Footprint(5.0, 5.0))
    detector = pr.Detector()
    numerics = pr.Numerics(
        cxr_trials=1,
        brem_trials=1,
        energy_bins=EnergyBins(cxr=np.array([1000.0, 1050.0]), brem=np.array([1000.0, 1050.0])),
    )
    return beam, target, detector, numerics


def test_build_case_is_the_existing_case_builder_boundary() -> None:
    beam, target, detector, numerics = _inputs()
    scene = pr.Scene(beam, target, detector)

    canonical = api.build_case(scene, numerics)
    legacy = build_cases(
        LegacySweep(
            material="hopg",
            beam=beam,
            target=target,
            detector=detector,
            energy_bins=_inputs()[3].energy_bins,
        ),
        n_electrons=1,
        n_electrons_brem=1,
    )[0]

    assert canonical == legacy


@pytest.mark.parametrize("use_scene", [False, True])
def test_simulate_returns_intrinsic_result_and_provenance_without_store(
    monkeypatch, use_scene
) -> None:
    beam, target, detector, numerics = _inputs()
    seen = {}

    def fake_run_case(case, *, transport_core):
        seen["case"] = case
        seen["transport_core"] = transport_core
        return {
            "E_grid": np.array([1.0, 2.0]),
            "spec": np.array([3.0, 4.0]),
            "spec_characteristic": np.array([0.25, 0.5]),
            "E_grid_brem": np.array([1.0, 2.0, 3.0]),
            "brem_wide": np.array([5.0, 6.0, 7.0]),
            "brem": np.array([5.0, 6.0]),
            "line_data_coverage": [{"schema": "line-data-coverage.v1", "counts": {"pairs": 2}}],
        }

    monkeypatch.setattr(api, "run_case", fake_run_case)

    physics = pr.Physics(bremsstrahlung_model="eedl")
    if use_scene:
        result = pr.simulate(pr.Scene(beam, target, detector, numerics=numerics, physics=physics))
    else:
        result = pr.simulate(beam, target, detector, numerics=numerics, physics=physics)

    np.testing.assert_array_equal(result.cxr.incoherent.density, [3.0, 4.0])
    np.testing.assert_array_equal(result.characteristic.density, [0.25, 0.5])
    np.testing.assert_array_equal(result.brem.density, [5.0, 6.0, 7.0])
    assert result.case is seen["case"]
    assert seen["transport_core"] == "auto"
    assert result.provenance["scene"].target == target
    assert len(result.provenance["identity_digest"]) == 64
    assert result.provenance["kinematic_validity"]["reflections"]
    assert "kinematic_validity" not in result.case
    assert result.provenance.line_data_coverage[0]["counts"]["pairs"] == 2
    assert "line_data_coverage" not in result.case
    assert result.provenance["xsgen_tables"]
    from pyrite._physics_contract import ResolvedPhysicsView
    from pyrite.campaign.profiles import _jsonable

    view = ResolvedPhysicsView.from_json(json.dumps(result.provenance.resolved_physics))
    assert dict(view.table_markers) == result.provenance.xsgen_tables
    assert view.parameters == _jsonable(dict(result.case))
    assert result.provenance["identity_digest"] == api.case_content_key(
        result.case,
        xsgen_tables=result.provenance["xsgen_tables"],
    )
    # Which collision-stopping model produced the numbers is part of the record,
    # not just of the digest that separates the two models' checkpoints.
    assert result.provenance["stopping_model"] == STOPPING_MODEL
    assert result.provenance["characteristic_model"] == CHARACTERISTIC_MODEL
    assert result.provenance["bremsstrahlung_model"] == BREMSSTRAHLUNG_MODEL
    assert result.provenance["versions"]["xraydb"] == CHARACTERISTIC_XRAYDB_VERSION
    assert result.provenance["versions"]["endf-parserpy"] == BREM_ENDF_PARSERPY_VERSION


def test_simulate_is_bit_for_bit_the_existing_single_case_path() -> None:
    beam, target, detector, numerics = _inputs()
    expected = run_case(
        api.build_case(
            pr.Scene(beam, target, detector, physics=pr.Physics(bremsstrahlung_model="eedl")),
            numerics,
        )
    )

    result = pr.simulate(
        beam, target, detector, numerics=numerics, physics=pr.Physics(bremsstrahlung_model="eedl")
    )

    np.testing.assert_array_equal(result.cxr.incoherent.photon_energy_eV, expected["E_grid"])
    np.testing.assert_array_equal(result.cxr.incoherent.density, expected["spec"])
    np.testing.assert_array_equal(
        result.characteristic.density,
        expected["spec_characteristic"],
    )
    np.testing.assert_array_equal(result.brem.density, expected["brem_wide"])


@pytest.mark.parametrize("source", [None, "none"])
def test_scene_model_switches_select_returned_arrays(monkeypatch, source) -> None:
    beam, target, detector, numerics = _inputs()

    monkeypatch.setattr(
        api,
        "run_case",
        lambda case, *, transport_core: {
            "E_grid": np.array([1.0, 2.0]),
            "spec": np.array([3.0, 4.0]),
            "spec_coherent": np.array([7.0, 8.0]),
            "E_grid_brem": np.array([1.0, 2.0]),
            "brem_wide": np.array([5.0, 6.0]),
            "brem": np.array([5.0, 6.0]),
        },
    )

    result = pr.simulate(
        beam,
        target,
        detector,
        numerics=numerics,
        physics=pr.Physics(emission="coherent", brem_source=source),
    )

    np.testing.assert_array_equal(result.cxr.coherent.density, [7.0, 8.0])
    np.testing.assert_array_equal(result.brem.density, [0.0, 0.0])


def test_build_sweep_cases_remains_public_from_api():
    from pyrite.api import build_sweep_cases

    assert callable(build_sweep_cases)


def test_run_provenance_records_git_state_in_repo_and_null_outside(monkeypatch, tmp_path) -> None:
    from pyrite import api
    from tests.helpers.physics_baseline import resolved_baseline

    case = resolved_baseline("legacy")["case"]
    monkeypatch.setattr(api, "case_kinematic_validity", lambda _case: {})
    monkeypatch.setattr(api, "case_bremsstrahlung_marker", lambda _case: "x")

    api.source_git_state.cache_clear()
    sha, dirty = api.source_git_state()
    assert sha is not None and len(sha) == 40
    assert isinstance(dirty, bool)
    provenance = api.run_provenance(case, None)
    assert (provenance["git_sha"], provenance["git_dirty"]) == (sha, dirty)

    # Not a git checkout: the package root moved to an empty directory.
    fake = tmp_path / "src" / "pyrite"
    fake.mkdir(parents=True)
    monkeypatch.setattr(api, "__file__", str(fake / "api.py"))
    api.source_git_state.cache_clear()
    try:
        provenance = api.run_provenance(case, None)
        assert provenance["git_sha"] is None
        assert provenance["git_dirty"] is None
    finally:
        monkeypatch.undo()
        api.source_git_state.cache_clear()


@pytest.mark.parametrize(
    "overrides",
    [{"target": object()}, {"detector": pr.Detector()}, {"filters": ()}, {"acquisition": object()}],
)
def test_simulate_rejects_mixed_scene_and_component_arguments(overrides):
    beam, target, detector, numerics = _inputs()
    with pytest.raises(TypeError, match="with a Scene, omit"):
        pr.simulate(pr.Scene(beam, target, detector, numerics=numerics), **overrides)


def test_simulate_scene_overrides_preserve_original_and_geometry(monkeypatch):
    from dataclasses import replace

    beam, target, detector, numerics = _inputs()
    scene = pr.Scene(
        beam, replace(target, tilt_deg=30.123456789, tilt_azim_deg=-75.23456789), detector
    )
    physics = pr.Physics(brem_source="none", bremsstrahlung_model="eedl")
    seen = {}

    def run(case, *, transport_core):
        seen["case"] = case
        return {"E_grid": np.array([1.0, 2.0]), "spec": np.ones(2), "brem": np.ones(2)}

    monkeypatch.setattr(api, "run_case", run)
    result = pr.simulate(scene, physics=physics, numerics=numerics)
    assert result.case["tilt_deg"] == 30.123456789
    assert result.case["tilt_azim_deg"] == -75.23456789
    assert result.case["Ne"] == numerics.cxr_trials
    assert result.provenance["scene"].physics == physics
    assert result.provenance["scene"].numerics is numerics
    assert scene.physics != physics and scene.numerics is not numerics
    np.testing.assert_array_equal(result.brem.density, [0.0, 0.0])


def test_simulate_scene_requires_detector():
    beam, target, _, _ = _inputs()
    with pytest.raises(TypeError, match="requires a detector"):
        pr.simulate(pr.Scene(beam, target, detector=None))
