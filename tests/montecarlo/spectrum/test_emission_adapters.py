"""Existing-slot batch selection without new source physics (issue #485)."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from pyrite._physics_contract import ModelDescriptor
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import emission_adapters as ea


def _batch(case=None):
    return ea.EmissionBatch(
        segments={"Ne": 3, "n_layers": 1, "L_ang": np.ones(3)},
        energy_eV=np.array([1000.0, 2000.0, 3000.0]),
        case={"composition": [("Si", 0.05)]} if case is None else case,
        n_hat=np.array([1.0, 0.0, 0.0]),
        electron_limit=3,
    )


@pytest.mark.parametrize(
    "case,model,route",
    [
        ({}, "eedl", "uncoupled"),
        ({"bremsstrahlung_model": "bremslib"}, "bremslib", "uncoupled"),
        (
            {"bremsstrahlung_model": "bremslib", "radiative_model": "bremslib-soft-hard"},
            "bremslib",
            "coupled",
        ),
    ],
)
def test_continuum_selection_is_resolved_and_route_specific(case, model, route):
    adapter = ea.select_continuum(case)
    assert adapter.descriptor.id == f"emission.continuum/{model}"
    assert adapter.route == route
    with pytest.raises(ValueError, match="unknown physics descriptor"):
        ea.select_continuum({"bremsstrahlung_model": "auto"})


def test_catalog_registration_is_local_immutable_and_checks_slots():
    descriptor = ModelDescriptor("emission.cxr", "constant", "test", output_slots=("spec",))
    adapter = ea.EmissionAdapter(
        descriptor, lambda batch: ea.EmissionOutput("spec", np.ones(batch.energy_eV.shape))
    )
    entries = [adapter]
    local = ea.EmissionCatalog(entries)
    entries.clear()
    assert local.get(descriptor.id) is adapter
    assert all(entry.descriptor.id != descriptor.id for entry in ea.BUILTIN_EMISSION.adapters)
    extended = ea.BUILTIN_EMISSION.with_adapter(adapter)
    assert extended.get(descriptor.id) is adapter
    with pytest.raises(ValueError, match="duplicate"):
        extended.with_adapter(adapter)
    with pytest.raises(FrozenInstanceError):
        extended.adapters = ()
    with pytest.raises(ValueError, match="existing output slots"):
        ea.EmissionAdapter(replace(descriptor, output_slots=("new-channel",)), adapter.score)
    with pytest.raises(ValueError, match="existing output slots"):
        ea.EmissionAdapter(replace(descriptor, output_slots=("brem_wide",)), adapter.score)


def test_test_only_registered_producer_reaches_live_existing_slot(monkeypatch):
    from tests.montecarlo.coherent.test_coherent_emission import _runner_case, _runner_tp, _segments

    segments = _segments(3)
    staged = {**segments}
    uploaded = []
    seen = []
    grid = _runner_tp(segments)["E_grid"]
    descriptor = ModelDescriptor("emission.cxr", "constant", "test", output_slots=("spec",))

    def producer(batch):
        seen.append(batch)
        return ea.EmissionOutput("spec", np.full(batch.energy_eV.shape, 7.0))

    custom = ea.EmissionAdapter(descriptor, producer)
    local = ea.BUILTIN_EMISSION.with_adapter(custom)
    ordinary = ea.select_emission({})
    selection = replace(ordinary, cxr=local.get(descriptor.id))
    monkeypatch.setattr(ea, "select_emission", lambda case: selection)
    monkeypatch.setattr(
        runner,
        "_cxr_for_segments",
        lambda *args, **kwargs: pytest.fail("registered producer bypasses old CXR scorer"),
    )
    monkeypatch.setattr(
        runner.case_driver, "_segments_on_device", lambda value: uploaded.append(value) or staged
    )
    other = []

    def characteristic(segs, E, case, n, layers, **kwargs):
        other.append(("characteristic", segs, E, kwargs["Ne"]))
        return np.full(E.shape, 2.0)

    def continuum(segs, E, case, n, layers, **kwargs):
        other.append(("continuum", segs, E, kwargs["Ne"]))
        return np.full(E.shape, 3.0)

    monkeypatch.setattr(runner, "_characteristic_from_segments", characteristic)
    monkeypatch.setattr(runner, "_brem_wide_from_segments", continuum)
    output = runner._spectrum_case_impl(_runner_case(), _runner_tp(segments, ne_lines=2, ne_brem=3))
    np.testing.assert_array_equal(output["spec"], 7.0)
    np.testing.assert_array_equal(output["spec_characteristic"], 2.0)
    np.testing.assert_array_equal(output["brem"], 3.0)
    assert "spec_coherent" not in output
    assert uploaded == [segments]
    assert len(seen) == 1 and seen[0].segments is staged and seen[0].energy_eV is grid
    assert seen[0].electron_limit == 2
    assert [(name, ne) for name, _, _, ne in other] == [("characteristic", 3), ("continuum", 3)]
    assert all(segs is staged for _, segs, _, _ in other)


@pytest.mark.parametrize("model", ["eedl", "bremslib"])
def test_native_continuum_preserves_layer_order_tables_cutoff_and_population(monkeypatch, model):
    composition = [("Si", 0.05)]
    layers = [(0.0, 1.0, composition), (1.0, 2.0, [("O", 0.03)])]
    case = {"composition": composition, "abs_layers": layers, "E_cut_brem_keV": 2.0}
    if model == "bremslib":
        case["bremsstrahlung_model"] = model
    tables = {"Si": object(), "O": object()}
    loaded = []
    monkeypatch.setattr(runner, "_case_bremslib_tables", lambda case: loaded.append(case) or tables)
    monkeypatch.setattr(runner, "_admit_chunk", lambda *args, **kwargs: 19)
    segments = {"Ne": 3, "n_layers": 2, "L_ang": np.ones(3)}
    views = [{"L_ang": np.ones(1), "layer": 0}, {"L_ang": np.ones(2), "layer": 1}]
    monkeypatch.setattr(runner, "_segments_in_layer", lambda segs, index: views[index])
    calls = []

    def score(segs, E, **kwargs):
        calls.append((segs, E, kwargs))
        return np.full(E.shape, 1.0 if segs is views[0] else 2.0)

    monkeypatch.setattr(runner, "mc_brem_spectrum", score)
    batch = replace(_batch(case), segments=segments, abs_layers=layers)
    output = runner._brem_wide_from_segments(
        segments, batch.energy_eV, case, batch.n_hat, layers, Ne=3
    )
    np.testing.assert_array_equal(output, 3.0)
    assert [kwargs["composition"] for _, _, kwargs in calls] == [layers[0][2], layers[1][2]]
    for _, E, kwargs in calls:
        assert E is batch.energy_eV and kwargs["layers"] is layers
        assert kwargs["E_cut_keV"] == 2.0 and kwargs["electron_limit"] == 3
        assert kwargs["chunk"] == 19 and kwargs["cross_section_model"] == model
        if model == "bremslib":
            assert kwargs["bremslib_tables"] is tables
    assert len(loaded) == (1 if model == "bremslib" else 0)


def test_coupled_route_cannot_score_uncoupled_continuum(monkeypatch):
    case = {"bremsstrahlung_model": "bremslib", "radiative_model": "bremslib-soft-hard"}
    batch = _batch(case)
    called = []
    ordinary = ea.BUILTIN_EMISSION.get("emission.continuum/bremslib", "uncoupled")
    monkeypatch.setattr(
        runner, "_brem_wide_from_segments", lambda *args, **kwargs: called.append(True)
    )
    with pytest.raises(ValueError, match="coupled tracks"):
        ordinary.evaluate(batch, "brem_wide")
    assert not called
    with pytest.raises(ValueError, match="requires bremsstrahlung_model"):
        ea.select_continuum({"radiative_model": "bremslib-soft-hard"})


@pytest.mark.parametrize("slot,shape", [("spec_characteristic", (3,)), ("spec", (1,))])
def test_producer_must_return_declared_slot_on_native_grid(slot, shape):
    descriptor = ModelDescriptor("emission.cxr", "test", "test", output_slots=("spec",))
    adapter = ea.EmissionAdapter(descriptor, lambda batch: ea.EmissionOutput(slot, np.zeros(shape)))
    with pytest.raises(ValueError, match="component|native grid"):
        adapter.evaluate(_batch(), "spec")


def test_cxr_batch_forwards_shared_cache_and_policy_accumulators(monkeypatch):
    cache, truncation, temporal, capture = {}, {}, object(), object()
    batch = replace(_batch(), table_cache=cache, truncation_audit=truncation, temporal=temporal)
    seen = []

    def score(segs, E, case, direction, layers, groove, **kwargs):
        seen.append(kwargs)
        return np.ones(E.shape)

    monkeypatch.setattr(runner, "_cxr_for_segments", score)
    adapter = ea.BUILTIN_EMISSION.get("emission.cxr/cxr")
    adapter.evaluate(batch, "spec")
    adapter.evaluate(replace(batch, coherent=True, coefficient_capture=capture), "spec_coherent")
    assert all(kwargs["table_cache"] is cache and kwargs["temporal"] is temporal for kwargs in seen)
    assert seen[0]["truncation_audit"] is truncation and "coefficient_capture" not in seen[0]
    assert seen[1]["coefficient_capture"] is capture and "truncation_audit" not in seen[1]
    assert [kwargs["coherent"] for kwargs in seen] == [False, True]


def test_per_electron_wrapper_keeps_native_observable_shape(monkeypatch):
    grid = np.array([1000.0, 2000.0, 3000.0])
    weights = np.array([0.5, 1.0, 0.5])
    seen = []

    def score(segs, E, **kwargs):
        seen.append(kwargs)
        return np.array([11.0, 12.0])

    monkeypatch.setattr(runner, "mc_brem_spectrum", score)
    batch = replace(_batch(), energy_eV=grid, electron_limit=2, electron_band_weights=weights)
    output = ea.select_continuum(batch.case).evaluate(batch, "brem_wide")
    np.testing.assert_array_equal(output.values, [11.0, 12.0])
    assert seen[0]["electron_band_weights"] is weights and seen[0]["electron_limit"] == 2


def _custom_selection(monkeypatch, *, streaming=True, adaptive=True):
    seen = []

    def producer(batch):
        seen.append(batch)
        if batch.truncation_audit is not None:
            batch.truncation_audit["electron_mass"] = np.full(batch.electron_limit, 7.0)
        return ea.EmissionOutput(
            "spec_coherent" if batch.coherent else "spec", np.full(batch.energy_eV.shape, 7.0)
        )

    descriptor = ModelDescriptor(
        "emission.cxr",
        "test",
        "test",
        output_slots=("spec", "spec_coherent"),
        streaming_requirements=("whole-electron blocks",) if streaming else (),
        adaptive_observables=("cxr",) if adaptive else (),
    )
    selection = replace(ea.select_emission({}), cxr=ea.EmissionAdapter(descriptor, producer))
    monkeypatch.setattr(ea, "select_emission", lambda case: selection)
    return seen


def test_streaming_uses_selected_producer_and_one_upload_per_block(monkeypatch):
    from types import SimpleNamespace

    from pyrite.montecarlo.runner import artifacts, cxr_grid

    batch = _batch()
    case = {"crystal": "test", "E0_keV": 30.0}
    inputs = dict(
        E_grid=batch.energy_eV,
        E_brem=batch.energy_eV,
        n_hat=batch.n_hat,
        groove=None,
        Ne_lines=2,
        Ne_brem=3,
    )
    artifact = SimpleNamespace(path="test.h5", attrs={"segment_count": 2})
    monkeypatch.setattr(artifacts, "read_trajectory_artifact", lambda *a, **kw: artifact)
    monkeypatch.setattr(artifacts, "_checked_case_and_inputs", lambda *a: (case, inputs))
    blocks = [dict(Ne=3, L_ang=np.ones(1), n_backscattered=0, n_missed=0) for _ in range(2)]
    monkeypatch.setattr(artifacts, "iter_electron_row_blocks", lambda *a: iter(blocks))
    monkeypatch.setattr(cxr_grid, "cxr_truncation_audit", lambda *a, **kw: None)
    uploaded = []
    monkeypatch.setattr(runner, "_segments_on_device", lambda s: uploaded.append(s) or s)
    for name in ("_characteristic_from_segments", "_brem_wide_from_segments"):
        monkeypatch.setattr(runner, name, lambda s, E, *a, **kw: np.zeros(E.shape))
    seen = _custom_selection(monkeypatch)
    output = artifacts.stream_spectrum_from_artifact("test.h5", max_segments=1)
    np.testing.assert_array_equal(output["spec"], 14.0)
    assert uploaded == blocks
    assert [b.electron_limit for b in seen] == [2, 2]
    assert seen[0].table_cache is seen[1].table_cache

    _custom_selection(monkeypatch, streaming=False)
    uploaded.clear()
    with pytest.raises(artifacts.TrajectoryArtifactError, match="does not support streaming"):
        artifacts.stream_spectrum_from_artifact("test.h5")
    assert uploaded == []


def test_recompute_pair_uses_selected_producer_with_shared_segments(monkeypatch):
    batch = _batch()
    seen = _custom_selection(monkeypatch)
    monkeypatch.setattr(
        runner.case_driver,
        "_transport_cxr_for_case",
        lambda *a: (batch.segments, None, batch.n_hat, None, None),
    )
    uploads = []
    monkeypatch.setattr(runner.case_driver, "_segments_on_device", lambda s: uploads.append(s) or s)
    monkeypatch.setattr(
        runner, "_characteristic_from_segments", lambda s, E, *a, **kw: np.zeros(E.shape)
    )
    case = {"Ne": 2, "Ne_brem": 3}
    output = runner._cxr_pair_for_case(
        case, batch.energy_eV, want_coherent=True, return_characteristic=True
    )
    np.testing.assert_array_equal(output[0], 7.0)
    np.testing.assert_array_equal(output[1], 7.0)
    assert uploads == [batch.segments]
    assert [b.coherent for b in seen] == [False, True]
    assert seen[0].table_cache is seen[1].table_cache


def test_adaptive_observable_uses_selected_producer_and_refuses_unsupported(monkeypatch):
    from pyrite._numerics import Precision
    from pyrite.montecarlo.runner import adaptive

    batch = _batch()
    monkeypatch.setattr(
        adaptive,
        "_case_geometry",
        lambda case: (batch.energy_eV, batch.energy_eV, None, batch.n_hat, None),
    )
    settings = Precision(
        target_rse=0.1,
        max_electrons=6,
        block_electrons=3,
        observables=("cxr",),
        band_eV=(1000.0, 3000.0),
    )
    seen = _custom_selection(monkeypatch)
    measure = adaptive.case_measure({}, settings)
    np.testing.assert_array_equal(measure(0, 3, batch.segments)["cxr"], [7.0] * 3)
    assert len(seen) == 1
    _custom_selection(monkeypatch, adaptive=False)
    with pytest.raises(ValueError, match="adaptive observable cxr"):
        adaptive.case_measure({}, settings)


def test_invalid_continuum_is_refused_before_transport_or_capture(monkeypatch):
    monkeypatch.setattr(
        runner.case_driver, "_case_geometry", lambda case: pytest.fail("must preflight first")
    )
    with pytest.raises(ValueError, match="coupled radiative scoring requires"):
        runner._transport_case({"radiative_model": "bremslib-soft-hard"})


def test_documented_existing_slot_example_is_executable_and_reaches_live_scoring(monkeypatch):
    from pathlib import Path

    from tests.montecarlo.coherent.test_coherent_emission import _runner_case, _runner_tp, _segments

    page = Path(__file__).parents[3] / "docs/repo-design/compute/extending-physics.md"
    text = page.read_text(encoding="utf-8")
    example = text.split("<!-- executable-example: existing-slot -->", 1)[1]
    code = example.split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {}
    exec(compile(code, str(page), "exec"), namespace)
    selection = namespace["selection"]
    monkeypatch.setattr(ea, "select_emission", lambda case: selection)
    monkeypatch.setattr(runner, "_cxr_for_segments", lambda s, E, *a, **kw: np.ones(E.shape))
    monkeypatch.setattr(runner, "_brem_wide_from_segments", lambda s, E, *a, **kw: np.ones(E.shape))
    monkeypatch.setattr(
        runner,
        "_characteristic_from_segments",
        lambda *a, **kw: pytest.fail("documented producer must supply the existing slot"),
    )
    output = runner._spectrum_case_impl(_runner_case(), _runner_tp(_segments(3)))
    np.testing.assert_array_equal(output["spec_characteristic"], 0.0)
    np.testing.assert_array_equal(output["spec"], 1.0)
