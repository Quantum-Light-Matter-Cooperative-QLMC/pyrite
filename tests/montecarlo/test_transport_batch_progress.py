"""Within-case transport electron-batch progress (#272).

The progress hook is host-side bookkeeping on the per-electron batch driver: it
must report each completed batch exactly once (capacity replays included), and
it must never change what the transport computes.
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.campaign.config import material_sweep
from pyrite.campaign.sweep import build_cases
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import scheduling
from pyrite.montecarlo.transport import PerElectronTransportConfig, simulate_trajectories

BASE_CASE = dict(
    E0_keV=30.0,
    Ne=120,
    thickness_ang=5.0e4,
    element="Si",
    n_atoms_per_ang3=0.04996,
    seed=11,
    transport_core="per-electron",
)
SEGMENT_KEYS = ("r_mid", "v_hat", "L_ang", "E_keV", "t_ang", "t0_ang", "elec_id", "layer")
COUNT_KEYS = ("n_backscattered", "n_transmitted", "n_side_exited", "n_cutoff_stopped")
# seg_capacity=8 is far below what a 30 keV Si electron needs, so the probe batch
# overflows and replays; the small budget then splits the rest into many batches.
REPLAY_CONFIG = PerElectronTransportConfig(
    seg_capacity=8, scratch_budget_bytes=1 << 16, probe_electrons=4
)


def _identical(a, b):
    return all(np.array_equal(a[k], b[k]) for k in SEGMENT_KEYS) and all(
        a[k] == b[k] for k in COUNT_KEYS
    )


def _core_launches(monkeypatch):
    from pyrite.montecarlo.transport import batching

    launches = []
    real = batching._alloc_scratch

    def counting(xp, m, cap, midpoint=False):
        launches.append(m)
        return real(xp, m, cap, midpoint)

    monkeypatch.setattr(batching, "_alloc_scratch", counting)
    return launches


@pytest.mark.parametrize("elastic_model", ["elsepa", "sr"])
def test_progress_reports_each_completed_batch_once(monkeypatch, elastic_model):
    # "sr" takes the per-electron LUT driver, "elsepa" the exact one.
    launches = _core_launches(monkeypatch)
    reports = []
    simulate_trajectories(
        **BASE_CASE,
        elastic_model=elastic_model,
        per_electron_config=REPLAY_CONFIG,
        transport_progress=lambda done, total: reports.append((done, total)),
    )
    Ne = BASE_CASE["Ne"]
    done = [d for d, _ in reports]
    assert {t for _, t in reports} == {Ne}
    assert done[-1] == Ne
    assert all(b > a for a, b in zip(done, done[1:], strict=False))
    assert len(reports) >= 3
    # The probe replayed: more launches than batches, and the replay was not
    # reported -- electrons launched exceed electrons reported.
    assert len(launches) > len(reports)
    assert sum(launches) > Ne
    # Replaying at a larger capacity can shrink the batch to fit the budget.
    assert 0 < done[0] <= REPLAY_CONFIG.probe_electrons


@pytest.mark.parametrize("elastic_model", ["elsepa", "sr"])
def test_progress_does_not_change_transport(elastic_model):
    kwargs = dict(BASE_CASE, elastic_model=elastic_model, per_electron_config=REPLAY_CONFIG)
    off = simulate_trajectories(**kwargs)
    on = simulate_trajectories(**kwargs, transport_progress=lambda done, total: None)
    assert _identical(off, on)


def test_lockstep_core_reports_nothing():
    reports = []
    simulate_trajectories(
        **dict(BASE_CASE, transport_core="lockstep", Ne=8),
        transport_progress=lambda *a: reports.append(a),
    )
    assert reports == []


@pytest.fixture(scope="module")
def case():
    sweep = material_sweep("hbn")
    sweep = replace(sweep, beam=replace(sweep.beam, energy_keV=100.0))
    return build_cases(sweep, n_electrons=12, n_electrons_brem=12)[0]


def test_run_level_progress_is_bitwise_neutral(monkeypatch, case):
    real = runner.simulate_trajectories

    def batched(*args, **kwargs):
        return real(*args, per_electron_config=REPLAY_CONFIG, **kwargs)

    monkeypatch.setattr(runner, "simulate_trajectories", batched)
    off = runner._transport_case(case, transport_core="per-electron")
    reports = []
    token = runner._TRANSPORT_PROGRESS.set(lambda done, total: reports.append((done, total)))
    try:
        on = runner._transport_case(case, transport_core="per-electron")
    finally:
        runner._TRANSPORT_PROGRESS.reset(token)
    assert len(reports) >= 2
    assert reports[-1][0] == reports[-1][1]
    for key in ("r_mid", "v_hat", "L_ang", "E_keV", "t_ang", "elec_id"):
        np.testing.assert_array_equal(on["segs"][key], off["segs"][key])


def test_real_single_case_transport_advances_terminal_and_activity(monkeypatch, capsys, case):
    real = runner.simulate_trajectories

    def batched(*args, **kwargs):
        kwargs.update(transport_core="per-electron", per_electron_config=REPLAY_CONFIG)
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "simulate_trajectories", batched)
    monkeypatch.delenv("PYRITE_LOCAL_DASHBOARD", raising=False)
    activity = []
    runner.run_cases([case], max_workers=0, transport_only=True, on_activity=activity.append)
    counts = [e["transport_electrons_done"] for e in activity if "transport_electrons_done" in e]
    assert len(counts) >= 2
    assert all(b > a for a, b in zip(counts, counts[1:], strict=False))
    assert counts[-1] == case["Ne"]
    output = capsys.readouterr()
    assert "electrons" in output.err
    assert output.out == ""


# ---- run_cases wiring --------------------------------------------------------


def _fake_transport(batches):
    """A transport that reports ``batches`` through the installed sink."""

    def transport(case, record_timing=False, **_kwargs):
        sink = runner._TRANSPORT_PROGRESS.get()
        for done, total in batches:
            if sink is not None:
                sink(done, total)
        return None

    return transport


def _run(monkeypatch, batches, **kwargs):
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", False)
    monkeypatch.setattr(scheduling, "_transport_case", _fake_transport(batches))
    return runner.run_cases(
        [{"name": "c0", "Ne": 6}], max_workers=0, transport_only=True, **kwargs
    )


def test_multi_batch_case_shows_a_nested_electron_bar(monkeypatch, capsys):
    monkeypatch.delenv("PYRITE_LOCAL_DASHBOARD", raising=False)
    _run(monkeypatch, [(2, 6), (4, 6), (6, 6)])
    assert "electrons" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("batches", "progress", "dashboard"),
    [
        ([(6, 6)], True, None),  # single batch
        ([(2, 6), (6, 6)], False, None),  # progress off
        ([(2, 6), (6, 6)], True, "1"),  # local dashboard owns the terminal
    ],
)
def test_no_electron_bar_when_not_wanted(monkeypatch, capsys, batches, progress, dashboard):
    if dashboard is None:
        monkeypatch.delenv("PYRITE_LOCAL_DASHBOARD", raising=False)
    else:
        monkeypatch.setenv("PYRITE_LOCAL_DASHBOARD", dashboard)
    _run(monkeypatch, batches, progress=progress)
    assert "electrons" not in capsys.readouterr().err


def test_progress_false_without_listeners_installs_no_sink(monkeypatch):
    seen = []

    def transport(case, record_timing=False, **_kwargs):
        seen.append(runner._TRANSPORT_PROGRESS.get())

    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", False)
    monkeypatch.setattr(scheduling, "_transport_case", transport)
    runner.run_cases([{"name": "c0"}], max_workers=0, transport_only=True, progress=False)
    assert seen == [None]
    assert runner._TRANSPORT_PROGRESS.get() is None


def test_activity_carries_throttled_electron_counts(monkeypatch):
    # A frozen clock: only a pass's first and final batches pass the throttle.
    monkeypatch.setattr(scheduling, "perf_counter", lambda: 0.0)
    activity = []
    batches = [(2, 6), (4, 6), (6, 6), (1, 3), (3, 3)]  # two passes (cascade)
    _run(monkeypatch, batches, progress=False, on_activity=activity.append)
    electron = [
        (e["transport_electrons_done"], e["transport_electrons_total"])
        for e in activity
        if "transport_electrons_done" in e
    ]
    assert electron == [(2, 6), (6, 6), (1, 3), (3, 3)]
    assert all(e["phase"] == "serial_case" and e["case_index"] == 0 for e in activity[1:-1])
    assert "transport_electrons_done" not in activity[-1]  # idle clears it


def test_electron_activity_updates_after_throttle_interval(monkeypatch):
    clock = iter([0.0, 0.5, 1.0, 1.2])
    monkeypatch.setattr(scheduling, "perf_counter", lambda: next(clock))
    activity = []
    sink = scheduling._TransportProgress(False, lambda done, total: activity.append((done, total)))
    for done in (2, 4, 6, 8):
        sink(done, 8)
    assert activity == [(2, 8), (6, 8), (8, 8)]


def test_transport_error_closes_bar_and_restores_sink(monkeypatch):
    closed = []

    class Bar:
        n = 0

        def update(self, delta):
            self.n += delta

        def close(self):
            closed.append(True)

    def fail(case, **kwargs):
        runner._TRANSPORT_PROGRESS.get()(2, 6)
        raise RuntimeError("transport failed")

    monkeypatch.delenv("PYRITE_LOCAL_DASHBOARD", raising=False)
    monkeypatch.setattr(scheduling, "_progress_bar", lambda **kwargs: kwargs.get("iterable", Bar()))
    monkeypatch.setattr(scheduling, "_transport_case", fail)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", False)
    with pytest.raises(RuntimeError, match="transport failed"):
        runner.run_cases([{"name": "c0"}], max_workers=0, transport_only=True)
    assert closed == [True]
    assert runner._TRANSPORT_PROGRESS.get() is None


def test_each_multibatch_pass_gets_a_bar_but_single_batch_passes_do_not(monkeypatch):
    bars = []

    class Bar:
        n = 0
        closed = False

        def __init__(self, total):
            self.total = total
            bars.append(self)

        def update(self, delta):
            self.n += delta

        def close(self):
            self.closed = True

    monkeypatch.setattr(scheduling, "_progress_bar", lambda **kwargs: Bar(kwargs["total"]))
    sink = scheduling._TransportProgress(True, None)
    for done, total in [(2, 6), (6, 6), (1, 1), (1, 3), (3, 3)]:
        sink(done, total)
    assert [(bar.total, bar.n, bar.closed) for bar in bars] == [(6, 6, True), (3, 3, True)]
