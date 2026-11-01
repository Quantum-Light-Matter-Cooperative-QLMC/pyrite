"""Patch seams and identities of the ``runner`` facade over ``case_driver`` (#852).

Moving the per-case code into ``case_driver`` re-homed the globals its calls
resolve against. These tests pin which module a replacement must target to take
effect, and that the facade keeps the historical import and pickle identities.
"""

import pickle

import pytest

from pyrite.campaign.config import material_sweep
from pyrite.campaign.sweep import build_cases
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import case_driver

MOVED = (
    "run_case",
    "run_case_directions",
    "_transport_case",
    "_spectrum_case",
    "_spectrum_case_impl",
    "_cxr_for_case",
    "_cxr_for_segments",
    "_cxr_for_segments_once",
    "_cxr_pair_for_case",
    "_transport_cxr_for_case",
    "_directional_outputs",
    "_spectrum_stage_done",
)


@pytest.mark.parametrize("name", MOVED)
def test_facade_reexports_the_case_driver_object(name):
    assert getattr(runner, name) is getattr(case_driver, name)
    assert getattr(runner, name).__module__ == "pyrite.montecarlo.runner"


@pytest.mark.parametrize("name", MOVED)
def test_pickle_identity_keeps_the_package_qualname(name):
    payload = pickle.dumps(getattr(runner, name))
    assert b"case_driver" not in payload
    assert pickle.loads(payload) is getattr(runner, name)


def _stop(marker):
    def stop(*_args, **_kwargs):
        raise RuntimeError(marker)

    return stop


_CASE = build_cases(material_sweep("hbn"), n_electrons=2, n_electrons_brem=2)[0]


def test_simulate_trajectories_is_read_from_case_driver(monkeypatch):
    monkeypatch.setattr(runner, "simulate_trajectories", _stop("package"))
    monkeypatch.setattr(case_driver, "simulate_trajectories", _stop("case_driver"))
    with pytest.raises(RuntimeError, match="case_driver"):
        runner._transport_case(_CASE, transport_core="lockstep")


def test_package_transport_case_patch_reaches_run_case_and_directions(monkeypatch):
    monkeypatch.setattr(runner, "_transport_case", _stop("package"))
    with pytest.raises(RuntimeError, match="package"):
        runner.run_case(_CASE)
    with pytest.raises(RuntimeError, match="package"):
        runner.run_case_directions(_CASE, [[0.0, 0.0, 1.0]])
    with pytest.raises(RuntimeError, match="package"):
        case_driver._transport_cxr_for_case(_CASE)


def test_spectrum_case_impl_is_read_from_case_driver(monkeypatch):
    monkeypatch.setattr(case_driver, "_spectrum_case_impl", _stop("case_driver"))
    with pytest.raises(RuntimeError, match="case_driver"):
        runner._spectrum_case(_CASE, {"E_grid": None})


def test_timing_flag_is_read_late_from_the_package(monkeypatch):
    cleared = []
    monkeypatch.setattr(
        case_driver._cxr_setup,
        "SETUP_STATS",
        type("Stats", (), {"clear": lambda self: cleared.append(1)})(),
    )
    monkeypatch.setattr(case_driver, "_spectrum_case_impl", _stop("impl"))
    monkeypatch.setattr(runner, "_TIMING", False)
    with pytest.raises(RuntimeError, match="impl"):
        runner._spectrum_case(_CASE, {})
    assert cleared == []
    monkeypatch.setattr(runner, "_TIMING", True)
    with pytest.raises(RuntimeError, match="impl"):
        runner._spectrum_case(_CASE, {})
    assert cleared == [1]
