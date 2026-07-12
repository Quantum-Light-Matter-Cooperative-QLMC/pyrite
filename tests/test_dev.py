"""Tests for the repository developer command runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def dev_module():
    path = Path(__file__).parents[1] / "scripts" / "dev.py"
    spec = importlib.util.spec_from_file_location("cxr_mc_dev", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_acp_up_stops_bridges_when_interrupted(dev_module, monkeypatch) -> None:
    class Process:
        def poll(self):
            return None

    stopped = []
    monkeypatch.setattr(dev_module, "start_acp_servers", lambda: [Process(), Process()])
    monkeypatch.setattr(dev_module, "stop_acp_servers", lambda: stopped.append(True))
    monkeypatch.setattr(
        dev_module.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt)
    )

    dev_module.cmd_acp_up(None)

    assert stopped == [True]
