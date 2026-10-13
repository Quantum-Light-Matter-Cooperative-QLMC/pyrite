"""Shared rectangular descriptions preserve source and observation contracts."""

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite.campaign import geometry
from pyrite.campaign.profiles import case_content_key
from pyrite.instrument import model, observation_identity
from pyrite.instrument.scene import scene_payload
from pyrite.materials import CATALOG, MediumSpec


def _body(*, key="silicon", kind="crystal", thickness_mm=0.0001):
    return pr.RectangularBody(pr.SubstanceRef(kind, key), (5.0, 5.0), thickness_mm)


def _scene(target=None, *, filters=()):
    return pr.Scene(
        beam=pr.Beam(energy_keV=30.0, transverse=pr.Spot(), longitudinal=pr.PointBunch()),
        target=pr.Target(_body()) if target is None else target,
        detector=pr.PlanarDetector(
            pr.PlanarPose((0, 0, 100)),
            pixels=pr.PixelGrid((4, 4), (0.5, 0.5)),
            response=pr.IdealPhotonCounter(),
            scorer=pr.PixelScorer((2, 2)),
        ),
        filters=filters,
        acquisition=pr.Acquisition(1.0, (0.0, 1000.0, 2000.0)),
        physics=pr.Physics(brem_source="none"),
        numerics=pr.Numerics(cxr_trials=1, brem_trials=1),
    )


def _identity(scene):
    resolved = pr.ResolvedObservation(
        scene.detector, scene.detector.scorer, scene.filter_plates, scene.acquisition
    )
    return observation_identity(
        "a" * 64,
        resolved,
        rep_rate_hz=scene.beam.rep_rate_hz,
        bunch_charge_pc=scene.beam.bunch_charge_pc,
        representative_directions_lab=np.array([[0.0, 0.0, 1.0]]),
        attenuation_arrays=(np.array([[1.0, 2.0]]),),
    )


def test_same_silicon_description_has_independent_target_and_filter_roles():
    body = _body(thickness_mm=0.1)
    target = pr.Target(body, tilt_deg=30.2, tilt_azim_deg=10.0, mosaic=True)
    window = pr.Filter(body, pr.PlanarPose((0, 0, 50)), name="window")
    other = replace(window, pose=pr.PlanarPose((4, 0, 60)), name="other")

    assert target.body is window.body is other.body
    assert target.to_slab() == pr.Slab(
        "silicon",
        thickness_ang=1_000_000.0,
        tilt_deg=30.2,
        tilt_azim_deg=10.0,
        footprint=pr.Footprint(5, 5),
        mosaic=True,
    )
    assert window.to_plate() == pr.FilterPlate("silicon", 0.1, (5, 5), window.pose, "window")
    assert other.to_plate().pose != window.to_plate().pose
    assert window.pose.center_mm == (0.0, 0.0, 50.0)
    with pytest.raises(FrozenInstanceError):
        body.thickness_mm = 2.0


def test_inline_body_freezes_composition_and_accepts_filters_only():
    composition = [["Al", 0.06]]
    body = pr.RectangularBody(MediumSpec("foil", composition), [5, 5], 0.1)
    composition[0][1] = 99.0
    assert body.substance.composition == (("Al", 0.06),)
    assert body.size_mm == (5.0, 5.0)
    assert pr.Filter(body, pr.PlanarPose((0, 0, 50))).to_plate().material == body.substance
    with pytest.raises(ValueError, match="crystalline SubstanceRef"):
        pr.Target(body)


def test_inline_body_rejects_mutable_non_scalar_density_values():
    with pytest.raises(TypeError, match="scalar densities"):
        pr.RectangularBody(MediumSpec("foil", (("Al", np.array([0.06])),)), (5, 5), 0.1)


def test_adapted_scene_retains_wrappers_and_lowers_to_identical_case_and_snapshot():
    target = pr.Target(_body(), tilt_deg=30.2)
    window = pr.Filter(_body(), pr.PlanarPose((0, 0, 50)), name="window")
    scene = _scene(target, filters=(window,))
    legacy = replace(scene, target=target.to_slab(), filters=(window.to_plate(),))

    assert scene.target is target and scene.filters == (window,)
    assert api.build_case(scene) == api.build_case(legacy)
    assert api.build_case(scene)["tilt_deg"] == 30.0  # existing angle quantization
    assert _identity(scene) == _identity(legacy)
    assert scene_payload(scene.filter_plates, scene.detector) == scene_payload(
        legacy.filter_plates, legacy.detector
    )
    assert scene.physics is legacy.physics and scene.numerics is legacy.numerics
    assert scene.detector.scorer is legacy.detector.scorer


def test_body_sweep_expands_before_legacy_lowering_and_preserves_order_and_seeds():
    scene = _scene()
    sweep = pr.Sweep(
        scene, {"beam.energy_keV": [30, 45], "target.body.thickness_mm": [0.0001, 0.0002]}
    )
    legacy = pr.Sweep(
        replace(scene, target=scene.target_spec),
        {"beam.energy_keV": [30, 45], "target.thickness_ang": [1000, 2000]},
    )
    assert [s.target.body.thickness_mm for _, s in sweep.expand()] == [
        0.0001,
        0.0002,
        0.0001,
        0.0002,
    ]
    new_cases, old_cases = sweep.cases(), legacy.cases()
    assert [c.seed for c in new_cases] == [1, 2, 3, 4]
    for new, old in zip(new_cases, old_cases, strict=True):
        assert replace(new, name=old.name) == old
        assert case_content_key(new) == case_content_key(old)


def test_filter_only_body_edits_preserve_source_and_change_observation_identity():
    window = pr.Filter(_body(), pr.PlanarPose((0, 0, 50)), name="window")
    scene = _scene(filters=(window,))
    moved = replace(scene, filters=(replace(window, pose=pr.PlanarPose((1, 0, 50))),))
    thicker = replace(
        scene, filters=(replace(window, body=replace(window.body, thickness_mm=0.1)),)
    )
    for changed in (moved, thicker):
        assert api.build_case(changed) == api.build_case(scene)
        assert _identity(changed).true_spatial_digest != _identity(scene).true_spatial_digest
    renamed = replace(scene, filters=(replace(window, name="new label"),))
    assert _identity(renamed) == _identity(scene)


def test_filter_sweep_paths_remain_on_shared_body_and_validate_each_expanded_pose():
    window = pr.Filter(_body(), pr.PlanarPose((0, 0, 50)))
    sweep = pr.Sweep(_scene(filters=(window,)), {"filters[0].body.thickness_mm": [0.0001, 0.1]})
    assert [s.filter_plates[0].thickness_mm for _, s in sweep.expand()] == [0.0001, 0.1]
    with pytest.raises(ValueError, match="full FilterPlate volume"):
        pr.Sweep(_scene(filters=(window,)), {"filters[0].pose.center_mm": [(0, 0, 101)]})


def test_explicit_filter_namespace_and_catalog_cannot_be_reinterpreted():
    custom = replace(
        CATALOG, media=MappingProxyType({"silicon": MediumSpec("silicon", (("Al", 0.06),))})
    )
    crystal_filter = pr.Filter(_body(), pr.PlanarPose((0, 0, 50)))
    medium_filter = pr.Filter(_body(kind="medium"), crystal_filter.pose)
    assert (
        crystal_filter.to_plate(catalog=custom).material.composition
        == CATALOG.crystals["silicon"].composition
    )
    assert medium_filter.to_plate(catalog=custom).material.composition == (("Al", 0.06),)
    empty = replace(custom, crystals=MappingProxyType({}))
    with pytest.raises(ValueError, match="unknown crystal substance"):
        crystal_filter.to_plate(catalog=empty)


def test_selected_catalog_collision_preserves_explicit_crystal_composition(monkeypatch):
    custom = replace(
        CATALOG, media=MappingProxyType({"silicon": MediumSpec("silicon", (("Al", 0.06),))})
    )
    monkeypatch.setattr(model, "CATALOG", custom)
    crystal_filter = pr.Filter(_body(), pr.PlanarPose((0, 0, 50)))
    medium_filter = pr.Filter(_body(kind="medium"), crystal_filter.pose)
    assert isinstance(crystal_filter.to_plate().material, MediumSpec)
    assert crystal_filter.to_plate().material.composition == CATALOG.crystals["silicon"].composition
    assert medium_filter.to_plate().material == "silicon"


@pytest.mark.parametrize("with_acquisition", [False, True])
def test_simulate_and_observation_planning_have_legacy_parity(monkeypatch, with_acquisition):
    def directional(case, directions, *, transport_core):
        return {
            "E_grid": np.array([5000.0, 6000.0]),
            "E_grid_brem": np.array([5000.0, 6000.0]),
            "spec_by_direction": np.ones((len(directions), 2)),
            "spec_characteristic_by_direction": np.full((len(directions), 2), 0.25),
            "brem_wide_by_direction": np.full((len(directions), 2), 0.5),
        }

    monkeypatch.setattr(api, "run_case_directions", directional)
    window = pr.Filter(_body(), pr.PlanarPose((0, 0, 50)))
    scene = _scene(filters=(window,))
    if not with_acquisition:
        scene = replace(scene, acquisition=None)
    kwargs = dict(physics=scene.physics, numerics=scene.numerics, acquisition=scene.acquisition)
    wrapped = pr.simulate(scene.beam, scene.target, scene.detector, filters=scene.filters, **kwargs)
    legacy = pr.simulate(
        scene.beam, scene.target_spec, scene.detector, filters=scene.filter_plates, **kwargs
    )
    assert wrapped.case == legacy.case
    np.testing.assert_array_equal(wrapped.cxr.incoherent.density, legacy.cxr.incoherent.density)
    np.testing.assert_array_equal(wrapped.brem.density, legacy.brem.density)
    assert wrapped.provenance["identity_digest"] == legacy.provenance["identity_digest"]
    assert (
        wrapped.provenance["observation_identity_digest"]
        == legacy.provenance["observation_identity_digest"]
    )
    if with_acquisition:
        plan = api.observation_plan(scene, scene.numerics)
        assert plan.observation.filters == scene.filter_plates


def test_target_rejects_recipe_namespace_redirection(monkeypatch):
    custom = replace(CATALOG, materials=MappingProxyType({"silicon": CATALOG.materials["hopg"]}))
    monkeypatch.setattr(geometry, "CATALOG", custom)
    with pytest.raises(ValueError, match="conflicts with a runnable recipe"):
        pr.Target(_body())


def test_bunch_front_matching_accepts_wrapped_targets_without_changing_the_result():
    scene = _scene(pr.Target(_body(key="hopg")))
    legacy = replace(scene, target=scene.target_spec)
    assert scene.phase_matched_bunch_front_tilt(
        (0, 0, 2), observation_direction_lab=(0, 0, 1)
    ) == legacy.phase_matched_bunch_front_tilt((0, 0, 2), observation_direction_lab=(0, 0, 1))


@pytest.mark.parametrize("thickness", [0, -1, float("nan"), float("inf"), [0.1, 0.2], True])
def test_body_rejects_non_scalar_or_nonphysical_thickness(thickness):
    with pytest.raises((TypeError, ValueError), match="thickness_mm"):
        _body(thickness_mm=thickness)


@pytest.mark.parametrize(
    "policy", ["electron-transport", "fluorescence", "diffraction", "secondary-production"]
)
def test_filter_rejects_unsupported_interaction_policies(policy):
    with pytest.raises(ValueError, match="primary-attenuation-only"):
        pr.Filter(_body(), pr.PlanarPose((0, 0, 50)), interaction=policy)


def test_unsupported_target_material_cut_pose_and_shapes_are_rejected():
    with pytest.raises(ValueError, match="crystalline SubstanceRef"):
        pr.Target(_body(kind="medium", key="sio2"))
    with pytest.raises(ValueError, match="lattice_cut"):
        pr.Target(_body(), lattice_cut="custom")
    with pytest.raises(TypeError, match="pose"):
        pr.Target(_body(), pose=pr.PlanarPose((0, 0, 50)))
    with pytest.raises(TypeError, match="RectangularBody"):
        pr.Filter(pr.Slab("silicon", footprint=None), pr.PlanarPose((0, 0, 50)))
    with pytest.raises(TypeError, match="scalar real angle"):
        pr.Target(_body(), tilt_deg=[20, 30])
