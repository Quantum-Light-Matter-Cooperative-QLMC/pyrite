from contextlib import contextmanager

import numpy as np

import cxr_mc.montecarlo.runner as runner


class DummyOOM(Exception):
    pass


def _tp():
    return {"E_grid": np.zeros(2000), "E_brem": np.zeros(2000)}


def test_retry_succeeds_after_halving(monkeypatch):
    monkeypatch.setattr(runner, "_GPU_OOM", (DummyOOM,))
    calls = []

    def fake_spectrum(case, tp, record_timing=False):
        calls.append((case.get("spec_chunk"), case.get("brem_chunk")))
        if len(calls) <= 3:
            raise DummyOOM
        return {"ok": True}

    monkeypatch.setattr(runner, "_spectrum_case", fake_spectrum)
    case = {"spec_chunk": 30_000, "brem_chunk": 100_000}
    out = runner._spectrum_case_retry(case, _tp(), max_retries=3)

    assert out["ok"]
    assert case == {"spec_chunk": 30_000, "brem_chunk": 100_000}
    assert calls == [
        (30_000, 100_000),
        (15_000, 50_000),
        (7_500, 25_000),
        (3_750, 12_500),
    ]


def test_retry_reraises_on_exhaustion(monkeypatch):
    monkeypatch.setattr(runner, "_GPU_OOM", (DummyOOM,))
    monkeypatch.setattr(
        runner,
        "_spectrum_case",
        lambda c, tp, record_timing=False: (_ for _ in ()).throw(DummyOOM()),
    )
    import pytest

    with pytest.raises(DummyOOM):
        runner._spectrum_case_retry({}, _tp(), max_retries=2)


def test_spectrum_case_emits_nsys_phase_ranges(monkeypatch):
    entered = []

    @contextmanager
    def record_range(message):
        entered.append(message)
        yield

    monkeypatch.setattr(runner, "_nsys_range", record_range)
    monkeypatch.setattr(
        runner,
        "_lines_for_segments",
        lambda *_args, **_kwargs: np.array([1.0, 2.0]),
    )
    monkeypatch.setattr(
        runner,
        "_brem_wide_from_segments",
        lambda *_args, **_kwargs: np.array([3.0, 4.0]),
    )
    segments = {
        "n_backscattered": 0,
        "n_missed": 0,
        "Ne": 1,
        "L_ang": np.array([1.0]),
    }
    case = {"name": "heavy case", "crystal": "mos2", "E0_keV": 300.0}
    tp = {
        "E_grid": np.array([100.0, 200.0]),
        "E_brem": np.array([100.0, 200.0]),
        "n_hat": np.array([0.0, 0.0, 1.0]),
        "segs": segments,
        "segs_b": segments,
        "groove": None,
    }

    runner._spectrum_case(case, tp)

    assert entered == [
        "cxr.spectrum_case:heavy case",
        "cxr.lines",
        "cxr.brem",
        "cxr.interpolate",
    ]
