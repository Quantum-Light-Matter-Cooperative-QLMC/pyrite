import json

import marimo as mo

from pyrite.apps.analysis_ui.pixels import (
    discover_observations,
    load_selected_observation,
    make_observation_selector,
    make_pixel_image_controls,
    make_pixel_selection_controls,
    resolve_pixel_image,
    spectrum_components,
)
from pyrite.apps.analysis_ui.views import render_pixel_detector
from pyrite.observations import ObservationStore, observation_from_result, observation_inventory
from tests.observations.test_store import _acquisition, _simulate, fake_directional  # noqa: F401


class _Values:
    def __init__(self, **values):
        self.value = values


def _inventory(tmp_path, acquisition=None):
    result = _simulate(acquisition=acquisition)
    store = ObservationStore("hopg", root=tmp_path / "observations")
    digest = store.put(observation_from_result(result))
    directory = tmp_path / "checkpoints" / "hopg"
    directory.mkdir(parents=True)
    (directory / "cases.json").write_text(
        json.dumps(
            {
                "schema": "cxr.case-manifest.v1",
                "cases": [
                    {
                        "name": "hopg_t1000",
                        "E0_keV": 30.0,
                        "content_key": result.provenance["identity_digest"],
                    }
                ],
            }
        )
    )
    return observation_inventory("hopg", checkpoint_dir=tmp_path / "checkpoints"), digest


def test_missing_checkpoint_or_observation_renders_an_explanation(tmp_path) -> None:
    assert discover_observations(None) is None
    assert "No checkpoint" in load_selected_observation(None, None).message

    empty = observation_inventory("hopg", checkpoint_dir=tmp_path / "checkpoints")
    assert make_observation_selector(mo, empty) is None
    state = load_selected_observation(empty, None)
    assert state.observation is None
    assert "pyrite profile physical-detector" in state.message

    view = render_pixel_detector(
        mo,
        state=state,
        selector=None,
        image_controls=None,
        selection_controls=None,
        resolved=None,
        theme="light",
    )
    assert "No stored pixel observation" in view.text


def test_selected_observation_renders_images_spectra_and_histogram(tmp_path) -> None:
    inventory, digest = _inventory(tmp_path)
    selector = make_observation_selector(mo, inventory)
    assert selector is not None and selector.value == digest
    state = load_selected_observation(inventory, selector)
    observation = state.observation
    assert observation is not None

    image_controls = make_pixel_image_controls(mo, observation)
    selection_controls = make_pixel_selection_controls(mo, observation)
    assert selection_controls.value == {"row": 2, "column": 3}
    # A Poisson view is offered only for a Poisson acquisition.
    assert image_controls.value["counts"] == "expected"
    resolved = resolve_pixel_image(observation, image_controls.value)

    view = render_pixel_detector(
        mo,
        state=state,
        selector=selector,
        image_controls=image_controls,
        selection_controls=selection_controls,
        resolved=resolved,
        theme="light",
    )

    assert "unit-efficiency, energy-preserving reference response" in view.text
    assert spectrum_components(observation) == ("line", "characteristic", "background")


def test_inverted_window_falls_back_to_the_full_range_with_a_warning(tmp_path) -> None:
    inventory, digest = _inventory(tmp_path)
    observation = inventory.load(digest)
    edges = observation.acquisition.measured_edges_eV

    scored, image, warnings = resolve_pixel_image(
        observation,
        _Values(kind="window", counts="expected", low=edges[2], high=edges[1], scale="log").value,
    )

    assert "full reporting range" in warnings[0]
    assert image.values.sum() == observation.acquisition_image().sum()


def test_poisson_observation_defaults_to_its_labelled_realization(tmp_path) -> None:
    inventory, digest = _inventory(
        tmp_path, acquisition=_acquisition(exposure_s=1.0e-9, mode="poisson", seed=1)
    )
    observation = inventory.load(digest)
    controls = make_pixel_image_controls(mo, observation)

    scored, image, warnings = resolve_pixel_image(observation, controls.value)

    assert controls.value["counts"] == "realized"
    assert image.value_title == "realized counts"
    assert "zero counts" in warnings[-1]


def test_image_resolution_is_memoized_per_observation_and_settings(tmp_path, monkeypatch) -> None:
    from pyrite.apps.analysis_ui import pixels

    inventory, digest = _inventory(tmp_path)
    observation = inventory.load(digest)
    values = make_pixel_image_controls(mo, observation).value
    calls = []
    original = pixels.observation_image

    def counting(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(pixels, "observation_image", counting)
    monkeypatch.setattr(pixels, "_IMAGE_CACHE", type(pixels._IMAGE_CACHE)())

    first = resolve_pixel_image(observation, values)
    again = resolve_pixel_image(inventory.load(digest), dict(values))
    other = resolve_pixel_image(observation, {**values, "kind": "coverage"})

    assert again is first
    assert other is not first
    assert calls == ["total", "coverage"]
