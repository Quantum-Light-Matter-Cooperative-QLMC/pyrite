"""Additional remote logs tests."""

import subprocess

import pytest

from pyrite import remote
from pyrite.console import dashboard as _dashboard
from pyrite.console.dashboard import poll as dashboard_poll
from pyrite.remote import transport, viewer
from tests.remote._remote_test_helpers import _remote_main


def test_follow_logs_use_stdin_closed_ssh(monkeypatch):
    runs = []
    monkeypatch.setattr(
        transport.subprocess,
        "run",
        lambda cmd: runs.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    assert remote.tail_logs("20260101-000000", follow=True) == 0

    assert len(runs) == 1
    assert runs[0][:3] == ["ssh", "-n", remote.HOST]
    assert len(runs[0]) == 4


def test_follow_logs_propagates_ssh_failure(monkeypatch):
    monkeypatch.setattr(
        transport.subprocess,
        "run",
        lambda cmd: subprocess.CompletedProcess(cmd, 255),
    )

    assert remote.tail_logs("20260101-000000", follow=True) == 1


def test_follow_logs_maps_interrupt_to_130_and_stderr(monkeypatch, capsys):
    def interrupt(_cmd):
        raise KeyboardInterrupt

    monkeypatch.setattr(transport.subprocess, "run", interrupt)

    assert remote.tail_logs("20260101-000000", follow=True) == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "job is unaffected" in captured.err


@pytest.mark.parametrize("status", [1, 130])
def test_follow_logs_status_propagates_through_remote_cli(monkeypatch, status):
    monkeypatch.setattr(viewer, "tail_logs", lambda _jobid, _follow: status)

    assert _remote_main(["logs", "--follow"]) == status


def test_stack_dump_signals_only_handler_registered_scans():
    from pyrite.remote import viewer  # noqa: F811

    command = viewer._stack_dump_command("j", 2)
    assert 'srun --jobid="$SID" --overlap --ntasks=1' in command
    assert "SigCgt" in command and ">> 9" in command
    assert "-m pyrite\\\\._entry\\\\.scan" in command  # not the uv wrapper or workers
    assert 'tail -c +"$((start + 1))" "$D/log"' in command


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected", "message"),
    [
        (0, "STACK j\n\nThread 0x1 (most recent call first):\n", 0, "signalled 1"),
        (3, "", 1, "predates the SIGUSR1 handler"),
    ],
)
def test_stack_dump_reports_signalled_and_legacy_jobs(
    monkeypatch, capsys, returncode, stdout, expected, message
):
    from pyrite.remote import viewer  # noqa: F811

    stderr = f"CXR_STACK_SIGNALLED {1 if returncode == 0 else 0} {0 if returncode == 0 else 1}\n"
    monkeypatch.setattr(
        viewer.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, returncode, stdout, stderr),
    )
    assert viewer.stack_dump("j", 0) == expected
    captured = capsys.readouterr()
    assert captured.out == stdout
    assert message in captured.err


def test_logs_identify_resolved_job_and_host(monkeypatch, capsys):
    commands = []
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: commands.append(command) or "LOG j · remote-host\n\nline\n",
    )

    remote.tail_logs("j")

    assert capsys.readouterr().out == "LOG j · remote-host\n\nline\n"
    assert 'printf "LOG %s · remote-host' in commands[0]


def test_key_listener_inactive_when_stdin_is_not_a_tty(monkeypatch):
    monkeypatch.setattr(dashboard_poll.sys.stdin, "isatty", lambda: False)

    listener = _dashboard.KeyListener()

    assert listener.active is False
    assert listener.poll() == []
    listener.stop()  # must be a harmless no-op
    assert viewer.KeyListener is _dashboard.KeyListener
