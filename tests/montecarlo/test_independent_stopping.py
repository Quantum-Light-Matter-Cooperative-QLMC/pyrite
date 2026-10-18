"""Independent prefix counts, guards, and fixed-count equivalence (#366).

Validation: adaptive-sample-size-stopping
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite._precision import Precision
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.adaptive import run_stopping_rule
from pyrite.montecarlo.runner.independent import IndependentStoppingMonitor
from tests.montecarlo.test_adaptive_stopping import _RULE_ONLY, _case


def _policy(**kw):
    return Precision(
        **dict(
            target_rse=0.1,
            min_electrons=40,
            max_electrons=200,
            block_electrons=20,
            observables=("cxr", "brem"),
            brem_target_rse=0.2,
            **_RULE_ONLY,
        )
        | kw
    )


def test_brem_freezes_and_line_is_rechecked_until_brem_finishes():
    settings = _policy(brem_min_electrons=60)
    measured = []

    def measure(start, stop, _):
        measured.append(start)
        return {"cxr": np.ones(stop - start), "brem": np.ones(stop - start)}

    record = run_stopping_rule(settings, measure)
    assert record["realized_electrons"] == record["realized_electrons_brem"] == 60
    assert record["stop_reason"] == "converged"
    assert record["statistics"]["cxr"]["n_electrons"] == 60
    # Six calls: one per population per block, no full-population reductions.
    assert measured == [0, 0, 20, 20, 40, 40]


def test_frozen_brem_does_not_observe_later_blocks_and_resets_on_retry():
    settings = _policy(min_electrons=80, brem_min_electrons=40)
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
    settings = _policy(min_electrons=80, brem_min_electrons=20, brem_max_electrons=40)
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
    settings = _policy(brem_min_electrons=60)
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
    settings = _policy(pilot_electrons=80, brem_min_electrons=20, brem_max_electrons=60)
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
    assert settings.brem_policy().pilot_electrons == 60
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
        {"brem_max_electrons": 220},
        {"brem_min_electrons": 30},
        {"brem_min_electrons": True},
        {"brem_min_electrons": 80, "brem_max_electrons": 60},
        {"observables": ("cxr",)},
        {"brem_target_rse": None, "brem_min_electrons": 40},
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
        min_electrons=80,
        max_electrons=120,
        brem_target_rse=1.0,
        brem_min_electrons=40,
        batch_means_band_eV=(1000.0, 10000.0),
    )
    requested = {
        **case,
        "Ne": settings.max_electrons,
        "Ne_brem": settings.max_electrons,
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
    assert batches["n_batches_brem"] == nb // settings.block_electrons
    for key, axis in (("spec", "E_grid"), ("brem_wide", "E_grid_brem")):
        energies = np.asarray(output[axis])
        mask = (energies >= 1000) & (energies <= 10000)
        # Batch and whole-population reductions associate sums differently;
        # CUDA reductions use float32, including brem arrays widened on return.
        reduction_dtype = np.float32 if core == "cuda" else output[key].dtype
        tolerance = max(1e-12, 8 * np.finfo(reduction_dtype).eps)
        np.testing.assert_allclose(batches[key]["mean"], output[key][mask], rtol=tolerance)


def test_shared_policy_serialization_and_rule_are_unchanged():
    independent = _policy()
    shared = replace(independent, brem_target_rse=None)
    payload = shared.to_dict()
    assert not any(key.startswith("brem_") for key in payload)
    assert Precision.from_dict(payload) == shared
    assert Precision.from_dict(independent.to_dict()) == independent
    values = np.random.default_rng(366).lognormal(0, 0.5, 200)

    def measure(a, b, _):
        return {"cxr": values[a:b], "brem": values[a:b]}

    expected = run_stopping_rule(shared, measure)
    equal = run_stopping_rule(replace(independent, brem_target_rse=shared.target_rse), measure)
    assert (
        equal["realized_electrons"]
        == equal["realized_electrons_brem"]
        == expected["realized_electrons"]
    )
