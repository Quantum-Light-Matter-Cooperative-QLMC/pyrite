"""Independent prefix counts, guards, and fixed-count equivalence (#366).

Validation: adaptive-sample-size-stopping
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite._precision import Precision
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.adaptive import run_stopping_rule, without_wall_clock
from pyrite.montecarlo.runner.independent import IndependentStoppingMonitor
from tests.montecarlo.runner.test_adaptive_stopping import _RULE_ONLY, _case


def _policy(**kw):
    return Precision(
        **dict(
            target_rse=0.1,
            min_histories=40,
            max_histories=200,
            observables=("cxr", "brem"),
            brem_target_rse=0.2,
            brem_min_histories=40,
            brem_max_histories=200,
            **_RULE_ONLY,
        )
        | kw
    )


def test_brem_freezes_and_line_is_rechecked_until_brem_finishes():
    settings = _policy(brem_min_histories=60)
    measured = []

    def measure(start, stop, _):
        measured.append(start)
        return {"cxr": np.ones(stop - start), "brem": np.ones(stop - start)}

    record = run_stopping_rule(settings, measure)
    assert record["realized_electrons"] == record["realized_electrons_brem"] == 60
    assert record["stop_reason"] == "converged"
    assert record["statistics"]["cxr"]["n_electrons"] == 60
    # One call per population per chunk, no full-population reductions.
    assert measured[::2] == measured[1::2]
    assert measured[::2] == sorted(set(measured))


def test_frozen_brem_does_not_observe_later_blocks_and_resets_on_retry():
    settings = _policy(min_histories=80, brem_min_histories=40)
    cxr_calls, brem_calls = [], []

    def measure(calls, name):
        def block(start, stop, _):
            calls.append(start)
            return {name: np.ones(stop - start)}

        return block

    monitor = IndependentStoppingMonitor(
        settings, measure(cxr_calls, "cxr"), measure(brem_calls, "brem")
    )
    for _ in range(2):
        for start in range(0, 80, 20):
            monitor.on_block(start, start + 20, None)
            done = monitor.should_stop(start + 20)
        assert done
        record = monitor.record()
        assert record["realized_electrons_brem"] == 40
        assert record["statistics"]["brem"]["n_electrons"] == 40
        assert record["statistics"]["cxr"]["n_electrons"] == 80
    assert brem_calls == [0, 20, 0, 20]
    assert cxr_calls == [0, 20, 40, 60] * 2


def test_brem_maximum_records_its_own_failure_while_line_continues():
    settings = _policy(min_histories=80, brem_min_histories=20, brem_max_histories=40)
    record = run_stopping_rule(
        settings,
        lambda a, b, _: {
            "cxr": np.ones(b - a),
            "brem": np.zeros(b - a),
        },
    )
    assert record["statistics_limited"]
    assert record["realized_electrons"] == 80
    assert record["realized_electrons_brem"] == 40
    assert record["statistics"]["cxr"]["stop_reason"] == "converged"
    assert record["statistics"]["brem"]["stop_reason"] == "max_electrons"


def test_cxr_early_convergence_is_not_frozen_while_waiting_for_brem():
    settings = _policy(brem_min_histories=60)
    values = np.ones(200)
    values[40] = 1000
    record = run_stopping_rule(
        settings,
        lambda a, b, _: {
            "cxr": values[a:b],
            "brem": np.ones(b - a),
        },
    )
    assert record["realized_electrons"] == 200
    assert record["realized_electrons_brem"] == 60
    assert not record["statistics"]["cxr"]["converged"]
    assert record["statistics"]["cxr"]["stop_reason"] == "max_electrons"


def test_independent_pilot_is_clipped_to_brem_bounds():
    settings = _policy(pilot_histories=80, brem_min_histories=20, brem_max_histories=60)
    record = run_stopping_rule(
        settings,
        lambda a, b, _: {
            "cxr": np.ones(b - a),
            "brem": np.ones(b - a),
        },
    )
    assert record["realized_electrons"] == 80
    assert record["realized_electrons_brem"] == 60
    assert record["statistics"]["cxr"]["pilot"]["n_electrons"] == 80
    assert settings.brem_policy().pilot_histories == 60
    # The cap always checks directly, so no pilot projection is needed there.
    assert record["statistics"]["brem"]["pilot"] is None


@pytest.mark.slow
def test_light_case_brem_converges_first_with_equal_minima():
    from pyrite.montecarlo.runner.adaptive import run_case_adaptive

    settings = _policy(target_rse=0.035, brem_target_rse=0.1)
    _, record = run_case_adaptive(_case(), settings, transport_core="per-electron")
    assert record["stop_reason"] == "converged"
    assert record["realized_electrons_brem"] < record["realized_electrons"]
    for name, target in (("cxr", 0.035), ("brem", 0.1)):
        assert record["statistics"][name]["relative_se"] <= target


@pytest.mark.parametrize(
    "kw",
    [
        {"brem_target_rse": True},
        {"brem_target_rse": float("nan")},
        {"brem_target_rse": 0},
        {"brem_max_histories": 220},
        {"brem_min_histories": True},
        {"brem_min_histories": 80, "brem_max_histories": 60},
        {"brem_max_histories": 1},
        {"observables": ("brem",)},  # brem keys need the cxr observable (#1398)
    ],
)
def test_invalid_independent_policies(kw):
    with pytest.raises(ValueError):
        _policy(**kw)


@pytest.mark.parametrize("spread", [None, 0.001])
@pytest.mark.parametrize("core", ["per-electron", pytest.param("cuda", marks=pytest.mark.slow)])
@pytest.mark.slow
def test_realized_pair_matches_fixed_transport_and_spectra(spread, core):
    if core == "cuda":
        from pyrite.montecarlo.transport.batching import _cuda_transport_available

        if not _cuda_transport_available():
            pytest.skip("no CUDA transport backend")
    # Different cutoffs expose accidental brem transport beyond the prefix.
    case = {
        **_case(energy_spread_frac=spread, bunch_length_fs=5.0),
        # Seed 8 puts the maximum initial energy at electron 106, outside
        # the realized 80-electron population, forcing table-range replay.
        "seed": 8 if spread is not None else 1,
        "E_cut_lines_keV": 29.0,
        "E_cut_brem_keV": 1.0,
    }
    settings = _policy(
        target_rse=1.0,
        min_histories=80,
        max_histories=120,
        brem_target_rse=1.0,
        brem_min_histories=40,
        batch_means_band_eV=(1000.0, 10000.0),
    )
    requested = {
        **case,
        "Ne": settings.max_histories,
        "Ne_brem": settings.max_histories,
        "adaptive_precision": settings.to_dict(),
    }
    transport = runner._transport_case(requested, transport_core=core)
    output = runner._spectrum_case(requested, transport)
    record = output["adaptive_sampling"]
    n, nb = record["realized_electrons"], record["realized_electrons_brem"]
    assert nb < n
    assert record["stop_reason"] == "converged"
    assert record["replayed_transport"] is (spread is not None)
    for name, count in (("cxr", n), ("brem", nb)):
        status = record["statistics"][name]
        assert status["n_electrons"] == count
        assert status["relative_se"] <= status["target_rse"]
    fixed = {**case, "Ne": n, "Ne_brem": nb}
    reference_transport = runner._transport_case(fixed, transport_core=core)
    reference = runner._spectrum_case(fixed, reference_transport)
    for key, value in reference_transport["segs"].items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(transport["segs"][key], value, err_msg=key)
    for key in ("spec", "spec_characteristic", "brem", "brem_wide", "E_grid"):
        np.testing.assert_array_equal(output[key], reference[key], err_msg=key)
    batches = record["batch_means"]
    assert batches["n_batches_brem"] == min(20, nb)
    for key, axis in (("spec", "E_grid"), ("brem_wide", "E_grid_brem")):
        energies = np.asarray(output[axis])
        mask = (energies >= 1000) & (energies <= 10000)
        # Batch and whole-population reductions associate sums differently;
        # CUDA reductions use float32, including brem arrays widened on return.
        # The windowed node reduction's analytic far tail (#1191) carries that
        # rounding at the peak's scale into low tail bins.
        reduction_dtype = np.float32 if core == "cuda" else output[key].dtype
        tolerance = max(1e-12, 8 * np.finfo(reduction_dtype).eps)
        expected = output[key][mask]
        np.testing.assert_allclose(
            batches[key]["mean"], expected, rtol=tolerance, atol=tolerance * np.abs(expected).max()
        )


def _explicit(**kw):
    return dict(target_rse=0.1, min_histories=40, max_histories=200, **kw)


def test_cxr_only_policy_keeps_its_serialization_and_ignores_brem():
    """Brem never gates the shared stop (#1398): CXR alone decides, brem is not monitored."""
    shared = Precision(**_explicit(), **_RULE_ONLY)
    payload = shared.to_dict()
    assert not any(key.startswith("brem_") for key in payload)
    assert payload["observables"] == ("line",)
    assert Precision.from_dict(payload) == shared
    values = np.random.default_rng(366).lognormal(0, 0.5, 200)
    calls = []

    def measure(a, b, _):
        calls.append(a)
        return {"cxr": values[a:b], "brem": np.random.default_rng(a).pareto(1.1, b - a)}

    record = run_stopping_rule(shared, measure)
    assert record["observables"] == ["cxr"]
    assert set(record["statistics"]) == {"cxr"}
    # The CXR stop is the one CXR alone gives, whatever brem would have done.
    alone = run_stopping_rule(shared, lambda a, b, _: {"cxr": values[a:b]})
    assert record["realized_electrons"] == alone["realized_electrons"]
    assert record["stop_reason"] == alone["stop_reason"]
    assert record["sampling_model"] == "decision-points-v4"


@pytest.mark.parametrize("chunk", [7, 50])
def test_independent_record_does_not_depend_on_the_transport_chunk(chunk):
    """A brem freeze inside a chunk truncates and replays; the record is unchanged."""
    settings = _policy(brem_min_histories=60)
    values = np.random.default_rng(3).lognormal(0, 0.3, 200)

    def measure(a, b, _):
        return {"cxr": values[a:b], "brem": values[a:b]}

    whole = run_stopping_rule(settings, measure)
    chunked = run_stopping_rule(settings, measure, chunk=chunk)
    assert whole["realized_electrons_brem"] is not None
    assert without_wall_clock(chunked) == without_wall_clock(whole)


def test_independent_brem_stop_with_explicit_keys_round_trips():
    independent = _policy()
    assert Precision.from_dict(independent.to_dict()) == independent
    payload = independent.to_dict()
    assert payload["observables"] == ("line", "brem")
    assert (payload["brem_target_rse"], payload["brem_min_electrons"]) == (0.2, 40)
    assert payload["brem_max_electrons"] == 200
    # Equal bounds and targets recover the joint rule.
    values = np.random.default_rng(366).lognormal(0, 0.5, 200)

    def measure(a, b, _):
        return {"cxr": values[a:b], "brem": values[a:b]}

    equal = run_stopping_rule(replace(independent, brem_target_rse=independent.target_rse), measure)
    assert equal["realized_electrons"] == equal["realized_electrons_brem"]
    assert equal["sampling_model"] == "decision-points-v5"


@pytest.mark.parametrize(
    "kw, missing",
    [
        ({"observables": ("cxr", "brem")}, "brem_target_rse and brem_max_histories"),
        ({"brem_target_rse": 0.2}, "brem_max_histories"),
        ({"brem_max_histories": 100}, "brem_target_rse"),
        ({"brem_min_histories": 60}, "brem_target_rse and brem_max_histories"),
    ],
)
def test_incomplete_brem_request_warns_and_inherits_the_cxr_values(kw, missing):
    """#1398 deprecation: requested brem stopping inherits CXR's required keys."""
    with pytest.warns(DeprecationWarning, match="brem stopping without") as caught:
        legacy = Precision(**_explicit(), **_RULE_ONLY, **kw)
    assert "will be an error in 0.39.0" in str(caught[0].message)
    for name in missing.split(" and "):
        assert name in str(caught[0].message)
    assert legacy.observables == ("cxr", "brem")
    explicit = Precision(
        **_explicit(),
        **_RULE_ONLY,
        observables=("cxr", "brem"),
        brem_target_rse=kw.get("brem_target_rse", 0.1),
        # Inheritance carries the minimum too: an old config behaves as written.
        brem_min_histories=kw.get("brem_min_histories", 40),
        brem_max_histories=kw.get("brem_max_histories", 200),
    )
    assert legacy.brem_min_histories == kw.get("brem_min_histories", legacy.min_histories)
    if "brem_min_histories" not in kw:
        assert "brem_min_histories=40 (from min_histories)" in str(caught[0].message)
    # The same policy, hence the same identity, as the explicit spelling.
    assert legacy == explicit
    assert legacy.to_dict() == explicit.to_dict()


def test_inheritance_carries_the_cxr_minimum_capped_at_the_brem_maximum():
    """cleo_sub-like legacy config: min_histories=3000, observables cxr+brem."""
    kw = dict(target_rse=0.001, min_histories=3000, max_histories=60_000, **_RULE_ONLY)
    with pytest.warns(DeprecationWarning, match="brem_min_histories=3000 .from min_histories"):
        legacy = Precision(**kw, observables=("cxr", "brem"))
    assert (legacy.brem_target_rse, legacy.brem_min_histories, legacy.brem_max_histories) == (
        0.001,
        3000,
        60_000,
    )
    explicit = Precision(
        **kw,
        observables=("cxr", "brem"),
        brem_target_rse=0.001,
        brem_min_histories=3000,
        brem_max_histories=60_000,
    )
    assert legacy.to_dict() == explicit.to_dict()
    # A smaller explicit brem maximum caps the inherited minimum.
    with pytest.warns(DeprecationWarning):
        capped = Precision(**kw, brem_max_histories=2000)
    assert capped.brem_min_histories == 2000


def test_explicit_required_keys_keep_the_mirrored_minimum_default(recwarn):
    """With target and maximum given, an omitted minimum defaults like min_histories."""
    kw = dict(target_rse=0.001, min_histories=3000, max_histories=60_000, **_RULE_ONLY)
    policy = Precision(**kw, brem_target_rse=0.01, brem_max_histories=60_000)
    assert policy.brem_min_histories == 100
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_complete_brem_request_does_not_warn(recwarn):
    _policy()
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_incomplete_brem_request_is_refused_at_the_scheduled_removal(monkeypatch):
    import pyrite
    from pyrite._util.deprecations import API_DEPRECATIONS, BREM_INHERITANCE

    entry = API_DEPRECATIONS[BREM_INHERITANCE]
    assert entry.remove_in == "0.39.0"
    monkeypatch.setattr(pyrite, "__version__", entry.remove_in)
    with pytest.raises(ValueError, match="requires brem_target_rse and brem_max_histories"):
        Precision(**_explicit(), observables=("cxr", "brem"))
    with pytest.raises(ValueError, match="requires brem_max_histories"):
        Precision(**_explicit(), brem_target_rse=0.2)
    # Complete requests and CXR-only policies are unaffected by the removal.
    _policy()
    Precision(**_explicit())


def test_brem_only_observable_keeps_the_shared_rule_on_brem():
    brem_only = Precision(**_explicit(), observables=("brem",))
    assert brem_only.brem_target_rse is None
    assert "brem_target_rse" not in brem_only.to_dict()
    assert brem_only.to_dict()["observables"] == ("brem",)


def test_brem_cannot_exceed_the_cxr_maximum():
    with pytest.raises(ValueError, match="must not exceed max_histories"):
        _policy(brem_max_histories=201)
