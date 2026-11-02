"""Public cache warming delegates without requiring compilation in tests."""

import pytest


def test_warm_cache_report_and_progress(monkeypatch):
    from pyrite import _jit_warm
    from pyrite.execution import warm_cache

    monkeypatch.setattr(
        "pyrite._backend.select_backend", lambda: type("Backend", (), {"name": "cpu"})()
    )
    monkeypatch.setattr(
        _jit_warm, "_run_case", lambda case, cupy: {"case": case.name, "status": "ok"}
    )
    seen = []
    report = warm_cache("standard", progress=lambda name, covers: seen.append((name, covers)))
    assert report["profile"] == "standard"
    assert report["backend"] == "cpu"
    assert [name for name, _ in seen] == [case["case"] for case in report["cases"]]
    assert len(seen) == 5


def test_warm_cache_invalid_profile():
    from pyrite.execution import warm_cache

    with pytest.raises(ValueError, match="unknown warm profile"):
        warm_cache("invalid")
