"""Named pixel CLI profiles compose the same scientific case as legacy inputs (#695)."""

from dataclasses import asdict, replace

import pytest

from pyrite import api
from pyrite.campaign.model import Scene
from pyrite.campaign.single_scene import (
    ProfileSimulation,
    compose_pixel_scene,
    resolve_pixel_scene,
)
from pyrite.cli.commands import _simulation
from pyrite.materials import load_material_catalog
from pyrite.observations import Observation


@pytest.fixture
def profile_parts():
    catalog = load_material_catalog(profile="pixelated_detector_demo")
    return resolve_pixel_scene(catalog, "hopg", "pixelated_detector_demo")


def test_profile_case_and_identity_survive_independent_composition(profile_parts):
    (
        beam,
        target,
        detector,
        filters,
        _,
        acquisition,
        numerics,
        physics,
        detector_id,
    ) = profile_parts
    legacy = Scene(
        beam,
        target,
        detector,
        filters,
        acquisition=acquisition,
        physics=physics,
        numerics=numerics,
    )
    configured = compose_pixel_scene(profile_parts)
    assert isinstance(configured, ProfileSimulation)
    assert configured.detector_id == detector_id
    assert configured.scene.detector.response is None
    assert set(asdict(configured.scene)) == {"beam", "target", "detector", "filters"}
    assert set(asdict(configured.scene)["detector"]) == {"pose", "size_mm", "pixels"}
    assert configured.observation.detector == "detector"
    assert configured.observation.response == detector.response
    assert configured.observation.acquisition == acquisition
    assert configured.observation.sampling == detector.scorer
    assert isinstance(configured.observation, Observation)

    from pyrite.campaign.profiles import case_content_key

    expected = api.build_case(legacy)
    actual = api.build_case(
        configured.scene,
        numerics=configured.numerics,
        physics=configured.physics,
        observation=configured.observation,
    )
    assert actual.to_dict() == expected.to_dict()
    assert case_content_key(actual) == case_content_key(expected)


def test_profile_observation_label_is_not_scientific_identity(profile_parts):
    original = compose_pixel_scene(profile_parts)
    changed = replace(
        original,
        observation=replace(original.observation, name="other-observation"),
    )
    assert (
        api.build_case(
            original.scene,
            numerics=original.numerics,
            physics=original.physics,
            observation=original.observation,
        ).to_dict()
        == api.build_case(
            changed.scene,
            numerics=changed.numerics,
            physics=changed.physics,
            observation=changed.observation,
        ).to_dict()
    )


def test_legacy_resolver_tuple_dispatches_independent_configuration(profile_parts, monkeypatch):
    class StopDispatch(Exception):
        pass

    received = []

    def fake_simulate(*args, **kwargs):
        received.append((args, kwargs))
        raise StopDispatch

    monkeypatch.setattr(_simulation._catalog_io, "catalog_text", lambda: ("", {}))
    monkeypatch.setattr(api, "simulate", fake_simulate)
    with pytest.raises(StopDispatch):
        _simulation.execute_simulation(
            "hopg",
            "pixelated_detector_demo",
            None,
            "table",
            None,
            resolver=lambda *_: profile_parts,
        )
    [(args, kwargs)] = received
    assert len(args) == 1 and isinstance(args[0], Scene)
    assert kwargs["physics"] == profile_parts[7]
    assert kwargs["numerics"] == profile_parts[6]
    assert kwargs["observation"].detector == "detector"
    assert kwargs["observation"].sampling == profile_parts[4]


def test_pixel_composition_normalizes_unit_values_before_case_lowering(profile_parts):
    pytest.importorskip("pint")
    from pyrite.units import ureg

    parts = list(profile_parts)
    parts[0] = replace(parts[0], energy_keV=parts[0].energy_keV * ureg.keV)
    composed = compose_pixel_scene(tuple(parts))
    original = compose_pixel_scene(profile_parts)
    assert composed.scene.beam.energy_keV == original.scene.beam.energy_keV
    assert (
        api.build_case(
            composed.scene,
            numerics=composed.numerics,
            physics=composed.physics,
            observation=composed.observation,
        ).to_dict()
        == api.build_case(
            original.scene,
            numerics=original.numerics,
            physics=original.physics,
            observation=original.observation,
        ).to_dict()
    )
