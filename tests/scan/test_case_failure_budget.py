"""``run_sweep(max_case_failures=N)``: finished cases persist; failures are reported."""

import pytest

from pyrite.runs import run
from pyrite.runs._case_failures import CaseFailureBudget, CaseFailuresError
from pyrite.runs.run import run_sweep
from tests.helpers.runner import fake_out
from tests.scan.test_run import _fake_case


def _failing_runner(bad, executed, clock=None):
    def run_cases(todo, *, callback, should_stop=None, on_case_error=None, **_kwargs):
        for index, case in enumerate(todo):
            if should_stop is not None and should_stop():
                break
            executed.append(case["name"])
            if clock is not None:
                clock[0] += 2
            if case["name"] in bad:
                error = ValueError(f"{case['name']} refused")
                if on_case_error is None or not on_case_error(index, case, error):
                    raise error
                continue
            callback(index, case, fake_out(case))

    return run_cases


def _cases():
    return [_fake_case(name, 300.0, tilt_deg=tilt) for tilt, name in enumerate("abcd")]


def test_budget_runs_the_rest_then_reports_failures_and_resume_retries_them(
    tmp_path, monkeypatch, capsys
):
    executed = []
    monkeypatch.setattr(run, "run_cases", _failing_runner({"b"}, executed))
    path = str(tmp_path / "hopg")

    with pytest.raises(CaseFailuresError) as caught:
        run_sweep(_cases(), {}, checkpoint_path=path, progress=False, max_case_failures=5)

    assert executed == ["a", "b", "c", "d"]
    assert [case["name"] for case, _ in caught.value.failures] == ["b"]
    assert "b at 300.0 keV: ValueError: b refused" in str(caught.value)
    assert "case b at 300.0 keV failed (1/5 tolerated)" in capsys.readouterr().err

    executed.clear()
    monkeypatch.setattr(run, "run_cases", _failing_runner(set(), executed))
    assert run_sweep(_cases(), {}, checkpoint_path=path, progress=False, max_case_failures=5)
    assert executed == ["b"]


def test_failure_past_the_budget_propagates_itself(tmp_path, monkeypatch):
    executed = []
    monkeypatch.setattr(run, "run_cases", _failing_runner({"a", "b", "c"}, executed))

    with pytest.raises(ValueError, match="c refused"):
        run_sweep(
            _cases(),
            {},
            checkpoint_path=str(tmp_path / "hopg"),
            progress=False,
            max_case_failures=2,
        )
    assert executed == ["a", "b", "c"]


def test_default_budget_is_zero_and_keeps_abort_on_first_failure(tmp_path, monkeypatch):
    executed = []
    monkeypatch.setattr(run, "run_cases", _failing_runner({"a"}, executed))

    with pytest.raises(ValueError, match="a refused"):
        run_sweep(_cases(), {}, checkpoint_path=str(tmp_path / "hopg"), progress=False)
    assert executed == ["a"]


def test_time_budget_pause_wins_over_failure_report(tmp_path, monkeypatch):
    executed, clock = [], [0]
    monkeypatch.setattr(run, "run_cases", _failing_runner({"a"}, executed, clock))

    complete = run_sweep(
        _cases(),
        {},
        checkpoint_path=str(tmp_path / "hopg"),
        progress=False,
        max_case_failures=5,
        max_seconds=3,
        time_fn=lambda: clock[0],
    )

    assert complete is False
    assert executed == ["a", "b"]


def test_budget_requires_local_execution(tmp_path):
    with pytest.raises(ValueError, match="requires local execution"):
        run_sweep(
            _cases(),
            {},
            checkpoint_path=str(tmp_path / "hopg"),
            progress=False,
            max_case_failures=1,
            executor=object(),
        )


def test_negative_budget_is_rejected():
    with pytest.raises(ValueError, match=">= 0"):
        CaseFailureBudget(-1)


def test_scan_run_continues_past_a_material_with_failed_cases(monkeypatch, capsys):
    from types import SimpleNamespace

    from pyrite.runs import scan

    ran = []

    def run_material(_args, material, **_kwargs):
        ran.append(material)
        if material == "hopg":
            raise CaseFailuresError([(_fake_case("cfg", 300.0), ValueError("refused"))])
        return True

    monkeypatch.setattr(scan, "_selected", lambda _args: ["hopg", "hbn"])
    monkeypatch.setattr(scan, "_run_material", run_material)
    args = SimpleNamespace(catalog_profile="standard", no_progress=True, max_minutes=None)

    with pytest.raises(SystemExit) as caught:
        scan.run(args)

    assert ran == ["hopg", "hbn"]
    assert "hopg [default]: 1 case(s) failed" in str(caught.value.code)
    assert "cfg at 300.0 keV: ValueError: refused" in str(caught.value.code)
