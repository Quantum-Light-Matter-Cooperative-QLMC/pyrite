"""Process/cache policy and stream contract of the startup benchmark."""

import json
import subprocess
from argparse import Namespace
from types import SimpleNamespace

import pytest

from pyrite.devtools import dev_cli, startup


@pytest.mark.parametrize("cache, expected_calls", [("warm", 4), ("cold", 3)])
def test_startup_uses_fresh_processes_and_isolates_cold_caches(
    monkeypatch, cache, expected_calls, tmp_path
):
    calls = []

    def child(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return SimpleNamespace(
            stderr="",
            stdout=json.dumps(
                {
                    "beam_s": 1.0,
                    "setup_s": 0.0,
                    "first_s": 3.0,
                    "repeat_s": 0.5,
                    "total_first_s": 4.0,
                    "identity_digest": "same",
                    "array_sha256": {"spectrum": "same"},
                }
            ),
        )

    monkeypatch.setattr(startup.subprocess, "run", child)
    monkeypatch.setattr(startup.platform, "platform", lambda: "test-platform")
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "user-cache"))
    report = startup.benchmark(repeats=3, cache=cache, profile=tmp_path / "first.prof")
    assert len(calls) == expected_calls + 1  # Profile outside measured samples.
    assert all(env["PYRITE_MC_BACKEND"] == "cpu" for _, env in calls)
    assert all("NUMBA_DISABLE_JIT" not in env for _, env in calls)
    assert len(report["samples"]) == 3
    assert report["timings_s"]["first_s"] == {"median": 3.0, "min": 3.0, "max": 3.0}
    assert all("PYRITE_STARTUP_PROFILE" not in env for _, env in calls[:-1])
    assert calls[-1][1]["PYRITE_STARTUP_PROFILE"] == str(tmp_path / "first.prof")
    caches = [env["NUMBA_CACHE_DIR"] for _, env in calls]
    if cache == "cold":
        assert len(set(caches)) == len(caches)
        assert str(tmp_path / "user-cache") not in caches
    else:
        assert set(caches) == {str(tmp_path / "user-cache")}


def test_startup_json_and_child_failure_streams(monkeypatch, capsys):
    args = Namespace(repeats=1, cache="warm", json=True, profile=None)
    monkeypatch.setattr(startup, "benchmark", lambda **kwargs: {"schema": "test"})
    startup.run(args)
    assert json.loads(capsys.readouterr().out) == {"schema": "test"}

    def fail(**kwargs):
        raise subprocess.CalledProcessError(1, "worker", stderr="missing table\n")

    monkeypatch.setattr(startup, "benchmark", fail)
    with pytest.raises(SystemExit) as error:
        startup.run(args)
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "missing table\n"


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_startup_rejects_invalid_repeats(value, capsys):
    with pytest.raises(SystemExit) as error:
        dev_cli.main(["startup", "--repeats", value])
    assert error.value.code == 2
    assert "--repeats" in capsys.readouterr().err


def test_startup_help_keeps_execution_unloaded(capsys):
    with pytest.raises(SystemExit) as error:
        dev_cli.main(["startup", "--help"])
    assert error.value.code == 0
    assert "--cache {warm,cold}" in capsys.readouterr().out
