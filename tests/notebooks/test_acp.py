"""Lifecycle tests for the optional local ACP bridges."""

from __future__ import annotations

import json

import pytest

from pyrite.apps import _acp


def test_running_acp_stops_bridges_after_notebook_exit(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(_acp, "start_acp_servers", lambda: events.append("start"))
    monkeypatch.setattr(_acp, "stop_acp_servers", lambda: events.append("stop"))

    with _acp.running_acp():
        events.append("notebook")

    assert events == ["start", "notebook", "stop"]


def test_running_acp_stops_bridges_after_notebook_error(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(_acp, "start_acp_servers", lambda: events.append("start"))
    monkeypatch.setattr(_acp, "stop_acp_servers", lambda: events.append("stop"))

    with pytest.raises(RuntimeError, match="notebook failed"):
        with _acp.running_acp():
            raise RuntimeError("notebook failed")

    assert events == ["start", "stop"]


def test_acp_start_records_each_bridge_pid(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(_acp, "ACP_STATE_PATH", tmp_path / "acp.json")
    calls = []

    class Process:
        def __init__(self, pid: int) -> None:
            self.pid = pid

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process(100 + len(calls))

    monkeypatch.setattr(_acp.subprocess, "Popen", fake_popen)

    processes = _acp.start_acp_servers()

    assert [process.pid for process in processes] == [101, 102]
    assert [call[0][3] for call in calls] == [
        "npx -y @agentclientprotocol/claude-agent-acp",
        "npx -y @agentclientprotocol/codex-acp",
    ]
    assert [call[0][-1] for call in calls] == ["3017", "3021"]
    assert json.loads(_acp.ACP_STATE_PATH.read_text()) == {
        "claude": {"pid": 101, "port": 3017},
        "codex": {"pid": 102, "port": 3021},
    }


def test_acp_down_stops_recorded_processes_and_clears_state(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(_acp, "ACP_STATE_PATH", tmp_path / "acp.json")
    _acp.ACP_STATE_PATH.write_text(
        json.dumps(
            {
                "claude": {"pid": 101, "port": 3017},
                "codex": {"pid": 102, "port": 3021},
            }
        )
    )
    stopped = []
    monkeypatch.setattr(_acp, "terminate_process_tree", stopped.append)

    _acp.stop_acp_servers()

    assert stopped == [101, 102]
    assert not _acp.ACP_STATE_PATH.exists()
