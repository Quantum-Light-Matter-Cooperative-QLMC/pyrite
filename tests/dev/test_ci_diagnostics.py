"""Interrupted workers retain test identities and live Python stacks (#1367)."""

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.helpers.ci_diagnostics import WorkerDiagnostics, configure_test_diagnostics


def test_worker_records_test_phases_before_session_finishes(tmp_path):
    worker = WorkerDiagnostics(tmp_path, "gw1")
    try:
        worker.pytest_runtest_logstart("test_probe.py::test_probe", None)
        worker.pytest_runtest_logreport(
            SimpleNamespace(
                when="call", nodeid="test_probe.py::test_probe", outcome="failed", duration=0.25
            )
        )
        worker.pytest_runtest_logfinish("test_probe.py::test_probe", None)
        records = [json.loads(line) for line in Path(worker.events.name).read_text().splitlines()]
        assert [record["event"] for record in records] == ["start", "call", "finish"]
        assert records[1]["outcome"] == "failed"
        assert records[1]["duration"] == 0.25
        assert all(record["nodeid"] == "test_probe.py::test_probe" for record in records)
    finally:
        worker.pytest_unconfigure()


def test_killed_worker_retains_active_test_and_periodic_stack(tmp_path):
    script = """
import sys, time
from pathlib import Path
from tests.helpers.ci_diagnostics import WorkerDiagnostics
worker = WorkerDiagnostics(Path(sys.argv[1]), "gw0", stack_interval=0.05)
worker.pytest_runtest_logstart("test_probe.py::test_blocked", None)
worker.pytest_sessionstart()
time.sleep(60)
"""
    with subprocess.Popen([sys.executable, "-c", script, str(tmp_path)]) as process:
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                stacks = list(tmp_path.glob("*-stacks.log"))
                if stacks and "Timeout (" in stacks[0].read_text():
                    break
                assert process.poll() is None, "diagnostic worker exited before its stack dump"
                time.sleep(0.01)
            else:
                pytest.fail("diagnostic worker produced no stack dump within 10 seconds")
        finally:
            process.kill()
            process.wait(timeout=5)
    (events,) = list(tmp_path.glob("*.jsonl"))
    (stacks,) = list(tmp_path.glob("*-stacks.log"))
    assert json.loads(events.read_text())["nodeid"] == "test_probe.py::test_blocked"
    assert "Timeout (" in stacks.read_text()
    assert 'File "<string>"' in stacks.read_text()


@pytest.mark.parametrize("enabled, parallel", [(False, False), (True, True)])
def test_disabled_diagnostics_and_xdist_controller_write_nothing(
    tmp_path, monkeypatch, enabled, parallel
):
    monkeypatch.delenv("PYRITE_TEST_DIAGNOSTICS", raising=False)
    if enabled:
        monkeypatch.setenv("PYRITE_TEST_DIAGNOSTICS", str(tmp_path / "diagnostics"))
    config = SimpleNamespace(
        getoption=lambda *a, **kw: 2 if parallel else 0,
        pluginmanager=SimpleNamespace(register=lambda plugin: pytest.fail("registered")),
    )
    configure_test_diagnostics(config)
    assert not (tmp_path / "diagnostics").exists()
