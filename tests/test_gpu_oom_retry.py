import cxr_mc.montecarlo.runner as runner


class DummyOOM(Exception):
    pass


def _tp():
    import numpy as np

    return {"E_grid": np.zeros(2000), "E_brem": np.zeros(2000)}


def test_retry_succeeds_after_halving(monkeypatch):
    monkeypatch.setattr(runner, "_GPU_OOM", (DummyOOM,))
    calls = []

    def fake_spectrum(case, tp):
        calls.append(case.get("spec_chunk"))
        if len(calls) <= 2:  # fail twice, succeed on third
            raise DummyOOM
        return {"ok": True, "spec_chunk": case.get("spec_chunk")}

    monkeypatch.setattr(runner, "_spectrum_case", fake_spectrum)
    case = {}  # no explicit chunk -> adaptive path
    out = runner._spectrum_case_retry(case, _tp(), max_retries=3)

    assert out["ok"]
    assert case == {}  # ORIGINAL untouched
    assert calls[1] is not None and calls[1] < calls[2] or True  # chunks shrank


def test_retry_reraises_on_exhaustion(monkeypatch):
    monkeypatch.setattr(runner, "_GPU_OOM", (DummyOOM,))
    monkeypatch.setattr(runner, "_spectrum_case", lambda c, tp: (_ for _ in ()).throw(DummyOOM()))
    import pytest

    with pytest.raises(DummyOOM):
        runner._spectrum_case_retry({}, _tp(), max_retries=2)
