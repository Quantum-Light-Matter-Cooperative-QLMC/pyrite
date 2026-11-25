"""Live streamed scoring of adaptive cases (#1280).

The monitored adaptive pass scores its kept chunks when the grid is explicit
(one transport, #1397); a policy grid keeps the #1280 measure-only route
(monitored pass, grid pass, fixed pass). For one seed the result must equal
the retained adaptive route and the #1280 measure-only route: the same
realized counts and sampling record, and spectra within streamed-vs-joined
summation order.

Validation: adaptive-sample-size-stopping
"""

import numpy as np
import pytest

from pyrite._precision import Precision
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import block_transport, streamed_adaptive
from pyrite.montecarlo.runner.adaptive import without_wall_clock
from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed, streaming_refusal
from tests.montecarlo.runner.test_adaptive_stopping import _RULE_ONLY, _case

SPECTRA = ("spec", "spec_characteristic", "brem_wide", "brem")


def _requested(spread=None, *, brem=True, **policy):
    case = {
        # Fixed families: streaming, not measured selection, is under test (#1301).
        **_case(n_families=2, energy_bins=None, energy_spread_frac=spread),
        # Seed 8 puts the maximum initial energy past the realized population,
        # forcing the energy-range replay (see test_independent_stopping).
        "seed": 8 if spread is not None else 1,
        "E_cut_lines_keV": 29.0,
        "E_cut_brem_keV": 1.0,
    }
    # ``brem=False``: CXR-only shared stop; brem is scored on every history (#1398).
    brem_keys = (
        dict(brem_target_rse=1.0, brem_min_histories=40, brem_max_histories=120) if brem else {}
    )
    settings = Precision(
        **_RULE_ONLY,
        target_rse=1.0,
        min_histories=80,
        max_histories=120,
        observables=("cxr", "brem") if brem else ("cxr",),
        **brem_keys,
        **policy,
    )
    return {
        **case,
        "Ne": settings.max_histories,
        "Ne_brem": settings.max_histories,
        "adaptive_precision": settings.to_dict(),
    }


def _retained(case):
    transport = runner._transport_case(case, transport_core="per-electron")
    return runner._spectrum_case(case, transport)


@pytest.mark.slow
@pytest.mark.parametrize("brem", [True, False], ids=["independent-brem", "cxr-only"])
@pytest.mark.parametrize("spread", [None, 0.001])
def test_streamed_adaptive_matches_retained(monkeypatch, spread, brem):
    case = _requested(spread, brem=brem)
    reference = _retained(case)

    def no_join(*args, **kwargs):
        raise AssertionError("streamed adaptive joined kept blocks")

    monkeypatch.setattr(block_transport, "merge_electron_blocks", no_join)
    streamed = run_case_streamed(case, 13, transport_core="per-electron")

    record, expected = streamed["adaptive_sampling"], reference["adaptive_sampling"]
    if brem:
        assert record["realized_electrons_brem"] < record["realized_electrons"]
    else:
        assert set(record["statistics"]) == {"cxr"}
    assert record["replayed_transport"] is (spread is not None)
    assert without_wall_clock(record) == without_wall_clock(expected)
    np.testing.assert_array_equal(streamed["E_grid"], reference["E_grid"])
    np.testing.assert_array_equal(streamed["E_grid_brem"], reference["E_grid_brem"])
    for key in SPECTRA:
        scale = np.abs(reference[key]).max()
        np.testing.assert_allclose(
            streamed[key], reference[key], rtol=1e-12, atol=1e-14 * scale, err_msg=key
        )
    for key in ("eta", "hit_frac", "n_segments"):
        assert streamed[key] == reference[key], key


@pytest.mark.parametrize(
    ("extra", "policy", "match"),
    [
        ({}, {"batch_means_band_eV": (1000.0, 10000.0)}, "batch means"),
    ],
)
def test_population_wide_adaptive_outputs_are_refused(monkeypatch, extra, policy, match):
    """Batch means keep the retained route; automatic selection streams (#1281)."""
    case = {**_requested(**policy), **extra}
    assert streaming_refusal(case, live=True) is not None

    def no_transport(*args, **kwargs):
        raise AssertionError("transport ran")

    monkeypatch.setattr(runner, "_transport_case", no_transport)
    with pytest.raises(ValueError, match=match):
        run_case_streamed(case, 10)


@pytest.mark.parametrize("brem", [True, False], ids=["independent-brem", "cxr-only"])
def test_plain_adaptive_case_is_live_streamable(brem):
    case = _requested(brem=brem)
    # The default sinc-oversampled line grid streams since #1280.
    assert case["line_grid_policy"]["resolution"]["policy"] == "sinc-oversampled"
    assert streaming_refusal(case, live=True) is None


def _no_headroom(monkeypatch):
    """No headroom for adaptive kept rows only; transport batches still fit."""
    from types import SimpleNamespace

    from pyrite.montecarlo.runner import host_admission
    from pyrite.montecarlo.transport.host_memory import HostMemoryResourceError

    def refuse(required, phase):
        if required > 0:
            raise HostMemoryResourceError(phase, required, 0)

    fake = SimpleNamespace(host_headroom_bytes=lambda: 0, admit_host_allocation=refuse)
    monkeypatch.setattr(host_admission, "host_memory", fake)


@pytest.mark.slow
def test_run_case_switches_to_streamed_when_retained_rows_exceed_headroom(monkeypatch):
    """Scans (``run_case``/``run_cases``) stream adaptive cases that would not fit (#1280)."""
    case = _requested()
    reference = _retained(case)
    _no_headroom(monkeypatch)
    with pytest.warns(RuntimeWarning, match="streamed adaptive"):
        out = runner.run_case(case, transport_core="per-electron")
    assert out["_adaptive_route"]["route"] == "streamed"
    assert without_wall_clock(out["adaptive_sampling"]) == without_wall_clock(
        reference["adaptive_sampling"]
    )
    for key in SPECTRA:
        scale = np.abs(reference[key]).max()
        np.testing.assert_allclose(
            out[key], reference[key], rtol=1e-12, atol=1e-14 * scale, err_msg=key
        )


@pytest.mark.parametrize("policy_grid", [False, True], ids=["single", "measure"])
def test_streamed_route_bounds_the_monitored_transport(monkeypatch, policy_grid):
    """#1417: the adaptive chunk floor must not override the streamed row budget."""
    from dataclasses import replace

    from pyrite.montecarlo.runner import case_driver, host_admission, recovery

    case = _requested(brem=False)
    settings = Precision.from_dict(case["adaptive_precision"])
    case["adaptive_precision"] = replace(settings, min_histories=8, max_histories=8).to_dict()
    case.update(Ne=8, Ne_brem=8)
    if not policy_grid:
        case = _explicit(case)
    _no_headroom(monkeypatch)
    # Every observed history exceeds this byte budget: the route must use one.
    monkeypatch.setattr(host_admission, "STREAMED_BLOCK_BYTES", 1)
    streamed = False
    spans = []
    original_route = recovery._streamed_adaptive
    simulate = case_driver.simulate_trajectories

    def switch(case, route, core):
        nonlocal streamed
        assert route.block_electrons == 1
        streamed = True
        return original_route(case, route, core)

    def bounded(energy, count, *args, **kwargs):
        if streamed:
            spans.append(count)
            assert count <= 1, "streamed route launched oversized retained transport"
        return simulate(energy, count, *args, **kwargs)

    monkeypatch.setattr(recovery, "_streamed_adaptive", switch)
    monkeypatch.setattr(case_driver, "simulate_trajectories", bounded)
    with pytest.warns(RuntimeWarning, match="streamed adaptive"):
        out = runner.run_case(case, transport_core="per-electron")
    assert spans
    assert out["_adaptive_route"]["scoring"] == ("measure" if policy_grid else "single")
    assert out["adaptive_sampling"]["realized_electrons"] == 8
    assert np.all(np.isfinite(out["spec"]))


@pytest.mark.parametrize(
    ("env", "policy"),
    [("off", {}), ("auto", {"batch_means_band_eV": (1000.0, 10000.0)})],
)
def test_route_switch_off_or_refused_keeps_the_admission_refusal(monkeypatch, env, policy):
    from pyrite.montecarlo.transport.host_memory import HostMemoryResourceError

    monkeypatch.setenv("PYRITE_STREAMED_ADAPTIVE", env)
    _no_headroom(monkeypatch)
    with pytest.raises(HostMemoryResourceError):
        runner.run_case(_requested(**policy), transport_core="per-electron")


# --- #1397: scoring in, and grid evidence from, the monitored pass ---------------


def _passes(monkeypatch):
    """Count full-population passes: monitored (adaptive) and fixed streamed."""
    counts = {"monitored": 0, "fixed": 0}
    monitored = block_transport.transport_monitored_blocks
    fixed = block_transport._stream_electron_blocks

    def count_monitored(*args, **kwargs):
        counts["monitored"] += 1
        return monitored(*args, **kwargs)

    def count_fixed(*args, **kwargs):
        counts["fixed"] += 1
        return fixed(*args, **kwargs)

    monkeypatch.setattr(block_transport, "transport_monitored_blocks", count_monitored)
    monkeypatch.setattr(block_transport, "_stream_electron_blocks", count_fixed)
    return counts


def _measure_route(monkeypatch, case, block):
    """The #1280 route (measure-only pass, then the fixed streamed case)."""
    with monkeypatch.context() as patch:
        patch.setattr(streamed_adaptive, "MONITORED_PASS", False)
        return run_case_streamed(case, block, transport_core="per-electron")


def _assert_spectra(streamed, reference, rtol=1e-12):
    for key in SPECTRA:
        scale = np.abs(reference[key]).max()
        np.testing.assert_allclose(
            streamed[key], reference[key], rtol=rtol, atol=1e-14 * scale, err_msg=key
        )
    for key in ("eta", "hit_frac", "n_segments"):
        assert streamed[key] == reference[key], key


def _explicit(case):
    return {key: value for key, value in case.items() if key != "line_grid_policy"}


@pytest.mark.slow
@pytest.mark.parametrize("spread", [None, 0.001])
def test_explicit_grid_scores_in_the_monitored_pass(monkeypatch, spread):
    """A stop inside a chunk, a brem freeze and (spread) the energy-range replay."""
    case = _explicit(_requested(spread))
    assert streamed_adaptive.adaptive_route(case) == "single"
    reference = _retained(case)
    two_pass = _measure_route(monkeypatch, case, 13)
    counts = _passes(monkeypatch)
    streamed = run_case_streamed(case, 13, transport_core="per-electron")

    replays = int(spread is not None)
    assert counts == {"monitored": 1 + replays, "fixed": 0}
    assert streamed["_adaptive_route"] == {
        "route": "streamed",
        "scoring": "single",
        "transports": 1 + replays,
    }
    assert two_pass["_adaptive_route"]["transports"] == 2 + replays
    record = streamed["adaptive_sampling"]
    assert record["realized_electrons_brem"] < record["realized_electrons"]
    assert record["replayed_transport"] is bool(replays)
    assert without_wall_clock(record) == without_wall_clock(two_pass["adaptive_sampling"])
    assert without_wall_clock(record) == without_wall_clock(reference["adaptive_sampling"])
    np.testing.assert_array_equal(streamed["E_grid"], reference["E_grid"])
    _assert_spectra(streamed, reference)
    _assert_spectra(streamed, two_pass)


def test_explicit_grid_coalesces_kept_chunks_at_the_brem_freeze(monkeypatch):
    """#1397: kept chunks join into blocks of ``block_electrons``; none straddles the freeze."""
    from pyrite.montecarlo.runner import stream_sizing, streamed_scoring

    case = _explicit(_requested())
    reference = _retained(case)
    # Chunks of 8 electrons (to the decision points) instead of one per side.
    monkeypatch.setattr(stream_sizing, "HOST_MIN_CHUNK", 8)
    real_add = streamed_scoring.BlockSpectrumScorer.add
    added = []

    def spy(self, segs):
        ids = np.asarray(segs["elec_id"])
        added.append((int(ids.min()), int(ids.max())))
        return real_add(self, segs)

    monkeypatch.setattr(streamed_scoring.BlockSpectrumScorer, "add", spy)
    chunks = []
    real_merge = block_transport.merge_electron_blocks

    def merge(blocks):
        chunks.append(len(blocks))
        return real_merge(blocks)

    monkeypatch.setattr(block_transport, "merge_electron_blocks", merge)
    streamed = run_case_streamed(case, 10**9, transport_core="per-electron")

    record = streamed["adaptive_sampling"]
    frozen = record["realized_electrons_brem"]
    assert frozen < record["realized_electrons"]
    # One block before the freeze, one after: the whole run buffered at this span.
    assert len(added) == 2 and max(chunks) > 1
    assert added[0][1] < frozen <= added[1][0]
    assert without_wall_clock(record) == without_wall_clock(reference["adaptive_sampling"])
    _assert_spectra(streamed, reference)


def test_provisional_count_is_the_next_power_of_two():
    assert [streamed_adaptive.provisional_count(n) for n in (1, 2, 3, 120, 128, 129)] == [
        1,
        2,
        4,
        128,
        128,
        256,
    ]


class _FakeOOM(RuntimeError):
    pass


def test_a_scoring_oom_rolls_back_and_commits_the_chunk_once(monkeypatch):
    """A failure after the scorer mutated its state rescores host rows once."""
    from pyrite import _backend
    from pyrite.montecarlo.runner import pool, streamed_scoring

    case = _explicit(_requested())
    clean = run_case_streamed(case, 13, transport_core="per-electron")

    real_add = streamed_scoring.BlockSpectrumScorer.add
    failures = []

    def failing_add(self, segs):
        real_add(self, segs)  # commits, then fails: the sink must roll it back
        if not failures:
            failures.append(int(segs["Ne"]))
            raise _FakeOOM("out of memory")

    monkeypatch.setattr(streamed_scoring.BlockSpectrumScorer, "add", failing_add)
    monkeypatch.setattr(runner, "_is_gpu_oom", lambda error: isinstance(error, _FakeOOM))
    monkeypatch.setattr(_backend, "is_device_array", lambda value: True)
    monkeypatch.setattr(pool, "_admit_host_segments", lambda: None)
    retried = run_case_streamed(case, 13, transport_core="per-electron")

    assert failures
    for key in (*SPECTRA, "n_segments"):
        np.testing.assert_array_equal(retried[key], clean[key], err_msg=key)


def _automatic(case):
    from tests.montecarlo.runner.test_reflection_selection_spectrum import SELECTION

    return {**case, "reflection_selection": {**SELECTION, "automatic": True}}


@pytest.mark.slow
@pytest.mark.parametrize("histories", [None, 2, 70])
def test_explicit_grid_selection_in_the_monitored_pass(monkeypatch, histories):
    """``None``: the whole population is the prefix (``Ne <= 4096``), selected at
    the stop. ``2``: the prefix completes mid-pass, before the brem freeze;
    ``70``: after it, so the selection reads the frozen ``Ne_brem``."""
    from pyrite.montecarlo.runner import reflection_cutoff

    from .test_streamed_selection import assert_same_selection

    if histories is not None:
        monkeypatch.setattr(reflection_cutoff, "SELECTION_HISTORIES", histories)
    case = _automatic(_explicit(_requested()))
    assert streaming_refusal(case, live=True) is None
    reference = _retained(case)
    two_pass = _measure_route(monkeypatch, case, 13)
    streamed = run_case_streamed(case, 13, transport_core="per-electron")

    assert streamed["_adaptive_route"]["transports"] == 1
    if histories == 70:
        frozen = streamed["adaptive_sampling"]["realized_electrons_brem"]
        assert frozen < histories < streamed["adaptive_sampling"]["realized_electrons"]
        assert streamed["reflection_selection"]["selection_histories"]["continuum"] == frozen
    assert without_wall_clock(streamed["adaptive_sampling"]) == without_wall_clock(
        reference["adaptive_sampling"]
    )
    assert_same_selection(streamed["reflection_selection"], reference["reflection_selection"])
    assert_same_selection(streamed["reflection_selection"], two_pass["reflection_selection"])
    _assert_spectra(streamed, reference)
    _assert_spectra(streamed, two_pass)


@pytest.mark.slow
@pytest.mark.parametrize("fraction", [1, 2])
def test_row_budget_selection_in_the_monitored_pass(monkeypatch, fraction):
    """#1368 under provisional scoring (#1397). ``1``: the budget is the realized
    rows, first met by the last kept electron, so neither route records a
    prefix. ``2``: half of them, so the prefix ends mid-pass."""
    from pyrite.montecarlo.runner import reflection_cutoff

    from .test_streamed_selection import assert_same_selection

    case = _automatic(_explicit(_requested()))
    rows = int(_retained(case)["n_segments"]) // fraction
    monkeypatch.setattr(reflection_cutoff, "SELECTION_ROWS", rows)
    reference = _retained(case)
    streamed = run_case_streamed(case, 13, transport_core="per-electron")

    prefix = reference["reflection_selection"].get("selection_histories")
    assert (prefix is None) == (fraction == 1)
    if prefix is not None:
        assert prefix["row_budget"] == rows <= prefix["rows"]
    assert_same_selection(streamed["reflection_selection"], reference["reflection_selection"])
    _assert_spectra(streamed, reference)


def _measured(case):
    from tests.montecarlo.transport.test_streamed_line_grid import _measured_policy

    return {**case, "line_grid_policy": _measured_policy(case)}


def _clear_grid_cache():
    from tests.montecarlo.transport.test_streamed_line_grid import _clear_grid_cache as clear

    clear()


@pytest.mark.slow
def test_policy_grid_keeps_the_measure_route(monkeypatch):
    """A policy grid needs the whole population: monitored, grid and fixed passes (#1397)."""
    case = _measured(_requested())
    assert streamed_adaptive.adaptive_route(case) == "measure"
    _clear_grid_cache()
    reference = _retained(case)
    _clear_grid_cache()
    counts = _passes(monkeypatch)
    missed = run_case_streamed(case, 13, transport_core="per-electron")
    assert counts == {"monitored": 1, "fixed": 2}
    assert missed["_adaptive_route"] == {"route": "streamed", "scoring": "measure", "transports": 3}
    assert missed["_streamed_line_grid_pass"]["transported"]
    assert without_wall_clock(missed["adaptive_sampling"]) == without_wall_clock(
        reference["adaptive_sampling"]
    )
    np.testing.assert_array_equal(missed["E_grid"], reference["E_grid"])
    _assert_spectra(missed, reference, rtol=1e-11)

    counts.update(monitored=0, fixed=0)
    hit = run_case_streamed(case, 13, transport_core="per-electron")
    assert counts == {"monitored": 1, "fixed": 1}
    assert hit["_adaptive_route"]["transports"] == 2
    assert hit["line_grid_resolved"]["cache"] == "hit"
    np.testing.assert_array_equal(hit["E_grid"], missed["E_grid"])
    _assert_spectra(hit, missed)


@pytest.mark.slow
def test_measure_route_names_each_transport_pass():
    """Live progress names every full-population pass, energy-range replay included."""
    from pyrite.montecarlo.runner import scheduling

    class PassLog(scheduling._TransportProgress):
        def __init__(self):
            super().__init__(False, None)
            self.labels = []

        def transport_pass(self, label):
            super().transport_pass(label)
            self.labels.append(label)

    case = _measured(_requested(0.001))
    _clear_grid_cache()
    for expected in (
        ["stopping", "energy-range replay", "grid", "scoring"],
        ["stopping", "energy-range replay", "scoring"],  # grid cache hit
    ):
        log = PassLog()
        token = runner._TRANSPORT_PROGRESS.set(log)
        try:
            out = run_case_streamed(case, 13, transport_core="per-electron")
        finally:
            runner._TRANSPORT_PROGRESS.reset(token)
        assert log.labels == expected
        assert out["_adaptive_route"]["transports"] == len(expected)
