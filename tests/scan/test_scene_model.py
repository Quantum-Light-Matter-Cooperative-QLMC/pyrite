import pickle

import numpy as np
import pytest

from pyrite.api import build_sweep_cases
from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.geometry import Layer, Slab, Stack
from pyrite.campaign.legacy import analysis_from_legacy, numerics_from_legacy
from pyrite.campaign.model import Analysis, Convergence, Numerics, Scene, Sweep
from pyrite.campaign.sweep import BeamSpec
from pyrite.campaign.sweep import build_cases as build_legacy_cases
from pyrite.detectors import Detector
from pyrite.materials import CATALOG
from pyrite.results import Settings


def _scene() -> Scene:
    return Scene(
        beam=BeamSpec(energy_keV=30.0),
        target=Stack(
            layers=(Layer("mos2", 2_000.0), Layer("sio2", 2_850.0)),
            tilt_deg=30.0,
        ),
        detector=Detector(),
    )


def _case_bytes(cases) -> list[bytes]:
    return [pickle.dumps(case.to_dict(), protocol=5) for case in cases]


def test_scene_rejects_implicit_sweep_values() -> None:
    with pytest.raises(ValueError, match="put multiple values in Sweep.axes"):
        Scene(beam=BeamSpec(energy_keV=[30.0, 60.0]), target=Slab("hopg"))


def test_path_axes_expand_nested_and_indexed_fields_mechanically() -> None:
    sweep = Sweep(
        base=_scene(),
        axes={
            "beam.energy_keV": np.array([30.0, 45.0]),
            "target.layers[1].thickness_ang": [2_850.0, 5_000.0],
        },
    )

    expanded = sweep.expand()

    assert [scene.beam.energy_keV for _, scene in expanded] == [30.0, 30.0, 45.0, 45.0]
    assert [scene.target.layers[1].thickness_ang for _, scene in expanded] == [
        2_850.0,
        5_000.0,
        2_850.0,
        5_000.0,
    ]
    assert expanded[-1][0] == ("beam.energy_keV=45 target.layers[1].thickness_ang=5000")


@pytest.mark.parametrize(
    "path, message",
    [
        ("target.tlit_deg", "has no field"),
        ("target.layers[9].thickness_ang", "out of range"),
        ("target.layers.one", "has no field"),
    ],
)
def test_axis_paths_fail_at_construction(path: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Sweep(base=_scene(), axes={path: [1.0]})


def test_object_defaults_are_separate_by_lifetime() -> None:
    numerics = Numerics(convergence=Convergence(max_reflections=4))
    analysis = Analysis()

    assert numerics.n_electrons == 450
    assert numerics.convergence.max_reflections == 4
    assert analysis.convolve_with_det is False


def test_legacy_pair_distributes_settings_and_swept_scene_fields() -> None:
    from pyrite.campaign.sweep import Sweep as LegacySweep

    old = LegacySweep(
        material="hopg",
        beam=BeamSpec(energy_keV=[30.0, 60.0]),
        thickness_ang=[1_000.0, 2_000.0],
        tilt_deg=30.0,
    )
    settings = Settings(n_electrons=12, n_electrons_brem=6, emission="both")

    with pytest.warns(DeprecationWarning):
        sweep = Sweep.from_legacy(old, settings)

    assert len(sweep.expand()) == 4
    assert sweep.base.emission == "both"
    assert numerics_from_legacy(old, settings).n_electrons == 12
    assert analysis_from_legacy(settings).beam_current_na == settings.beam_current_na


def test_legacy_bridge_preserves_expanded_cases_exactly() -> None:
    from pyrite.campaign.sweep import Sweep as LegacySweep

    old = LegacySweep(
        material="hopg",
        beam=BeamSpec(energy_keV=[30.0, 60.0]),
        thickness_ang=[1_000.0, 2_000.0],
        tilt_deg=[20.0, 30.0],
        n_electrons=[2, 3],
    )
    settings = Settings(n_electrons=12, n_electrons_brem=6, emission="both")

    with pytest.warns(DeprecationWarning):
        converted = Sweep.from_legacy(old, settings)

    expected = build_legacy_cases(
        old,
        settings.n_electrons,
        settings.n_electrons_brem,
        coherent_emission=settings.coherent_emission,
    )
    assert _case_bytes(build_sweep_cases(converted)) == _case_bytes(expected)


def test_every_catalog_profile_round_trips_a_resolved_case_list() -> None:
    settings = default_settings("survey")
    for profile in CATALOG.profile_names:
        members = CATALOG.profile_materials(profile)
        material = members[0] if members else "hopg"
        old = material_sweep(material, fidelity="survey", catalog_profile=profile)
        with pytest.warns(DeprecationWarning):
            converted = Sweep.from_legacy(old, settings)
        expected = build_legacy_cases(
            old,
            settings.n_electrons,
            settings.n_electrons_brem,
            coherent_emission=settings.coherent_emission,
        )
        assert _case_bytes(build_sweep_cases(converted)) == _case_bytes(expected), profile
