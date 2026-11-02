"""Remote attach tests."""

import pytest

from pyrite import remote
from pyrite.cli.commands import remote as cli  # noqa: F401
from pyrite.console.dashboard import state as dashboard_state
from pyrite.remote import (  # noqa: F401
    _queue_scripts,
    config,
    lifecycle,
    scripts,
    state,
    transport,
    viewer,
)
from tests.remote._remote_test_helpers import (
    _FakeKeyListener,
    _status_output,
)


def test_attach_redraws_the_status_report_until_terminal(monkeypatch, capsys):
    running = (
        '{"material":"hopg","total_cases":4,"cached_cases":1,'
        '"completed_new_cases":2,"state":"running"}'
    )
    done = (
        '{"material":"hopg","total_cases":4,"cached_cases":1,'
        '"completed_new_cases":3,"state":"done"}'
    )
    outputs = iter(
        [
            _status_output("running hopg [1/1] since now", progress=running),
            _status_output("done [1/1] now", squeue_state="NOT_QUEUED", progress=done),
        ]
    )
    monkeypatch.setattr(viewer, "_status_stream", lambda _cmd: outputs)

    assert remote.attach("20260101-000000") is True

    out = capsys.readouterr().out
    assert "ATTACHED" in out  # live banner with refresh counter
    assert "Progress" in out  # overall aggregate bar in the header
    assert "CASE PROGRESS" in out  # per-material bars at base verbosity
    assert "3/4" in out
    assert "JOB 20260101-000000 · FINISHED" in out
    assert "done [1/1] now" in out


def test_attach_sanitizes_hostile_state(monkeypatch, capsys):
    monkeypatch.setattr(
        viewer,
        "_status_stream",
        lambda _command: iter(
            [_status_output("done\x1b[2J hopg [1/1] now", squeue_state="NOT_QUEUED")]
        ),
    )

    assert remote.attach("20260101-000000") is True

    output = capsys.readouterr().out
    assert "\x1b" not in output
    assert "done?[2J hopg [1/1] now" in output


def test_attach_omits_log_tail_at_base_verbosity_but_still_fetches_progress(monkeypatch):
    commands = []
    monkeypatch.setattr(
        viewer,
        "_status_stream",
        lambda cmd: (
            commands.append(cmd) or iter([_status_output("done now", squeue_state="NOT_QUEUED")])
        ),
    )

    remote.attach("20260101-000000")

    assert '"$D"/progress/*.json' in commands[0]  # bars at every level need it
    assert "tail -c 32768" not in commands[0]  # log tail is still -vv only


def test_attach_forwards_double_verbose_to_the_status_command(monkeypatch):
    commands = []
    monkeypatch.setattr(
        viewer,
        "_status_stream",
        lambda cmd: (
            commands.append(cmd) or iter([_status_output("done now", squeue_state="NOT_QUEUED")])
        ),
    )

    remote.attach("20260101-000000", detail=2)

    assert 'tail -c 32768 "$D/log"' in commands[0]
    assert "squeue" in commands[0]
    assert "kill -0" not in commands[0]
    assert "/pid" not in commands[0]


def test_attach_watchdog_exits_on_a_stalled_chain(monkeypatch, capsys):
    output = _status_output("queued slice 2 now", squeue_state="NOT_QUEUED")
    monkeypatch.setattr(
        viewer,
        "_status_stream",
        lambda _cmd: iter([output] * viewer._POLL_GRACE_POLLS),
    )

    assert remote.attach("20260101-000000") is False
    assert "CHAIN STALLED" in capsys.readouterr().out


def test_attach_returns_false_when_the_viewer_is_interrupted(monkeypatch, capsys):
    monkeypatch.setattr(
        viewer,
        "_status_stream",
        lambda _cmd: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    assert remote.attach("20260101-000000") is False
    assert "VIEWER DISCONNECTED" in capsys.readouterr().out


def test_attach_header_shows_cancel_hint_only_when_keys_are_active():
    assert f"{viewer._CANCEL_ARM_KEY} cancels job" in viewer._attach_header(1, cancel_hint=True)
    assert f"{viewer._PULL_ARM_KEY} pulls progress" in viewer._attach_header(1, cancel_hint=True)
    assert "v cycles detail" in viewer._attach_header(1, cancel_hint=True)
    assert "cancels job" not in viewer._attach_header(1, cancel_hint=False)
    assert "cycles detail" not in viewer._attach_header(1, cancel_hint=False)


def test_attach_verbosity_key_cycles_restarts_stream_and_wraps(monkeypatch):
    running = _status_output("running hopg [1/1] since now", squeue_state="RUNNING")
    detailed = dashboard_state._encode_sections(
        {
            "JOB": "j",
            "META": "job: j\nmaterials: hopg\nslurm_job_id: 48291",
            "STATE": "running hopg [1/1] since now",
            "SQUEUE": "job_id=48291|state=RUNNING",
            "RESOURCES": "cpu_percent=25.0",
            "LOG": "detail-two log",
        }
    )
    done = _status_output("done now", squeue_state="NOT_QUEUED")
    streams = iter([[running], [running], [detailed, detailed], [done]])
    commands = []

    def status_stream(command):
        commands.append(command)
        return iter(next(streams))

    monkeypatch.setattr(viewer, "_status_stream", status_stream)
    fake_keys = _FakeKeyListener([["v"], ["v"], [], ["v"], []])
    monkeypatch.setattr(viewer, "KeyListener", lambda: fake_keys)
    frames = []
    monkeypatch.setattr(viewer, "render_frame", lambda frame, *, tty: frames.append(frame))

    assert remote.attach("20260101-000000") is True

    assert len(commands) == 4
    assert "tail -c 32768" not in commands[0]
    assert "tail -c 32768" not in commands[1]
    assert 'tail -c 32768 "$D/log"' in commands[2]
    assert "tail -c 32768" not in commands[3]
    assert "COMPUTE USAGE" in frames[2]
    assert "detail-two log" in frames[2]
    assert "COMPUTE USAGE" not in frames[3]
    assert "detail-two log" not in frames[3]
    assert fake_keys.stopped is True


def test_attach_header_shows_armed_confirm_banner():
    header = viewer._attach_header(1, armed=True)
    assert "CANCEL ARMED" in header
    assert f"press {viewer._CANCEL_CONFIRM_KEY} to confirm" in header


def test_attach_pull_keybinding_confirms_without_stopping_job(monkeypatch, capsys):
    running = _status_output(
        "running hopg [1/1] since now",
        squeue_state="RUNNING",
        progress=(
            '{"material":"hopg","total_cases":4,"cached_cases":1,'
            '"completed_new_cases":1,"state":"running"}'
        ),
    )
    done = _status_output("done now", squeue_state="NOT_QUEUED")
    monkeypatch.setattr(viewer, "_status_stream", lambda _cmd: iter([running, running, done]))
    fake_keys = _FakeKeyListener([["p"], ["y"], []])
    monkeypatch.setattr(viewer, "KeyListener", lambda: fake_keys)
    pulled = []
    monkeypatch.setattr(
        viewer,
        "_pull_attached_progress",
        lambda jobid, sections: pulled.append((jobid, sections)),
    )
    monkeypatch.setattr(
        lifecycle, "_stop_jobid", lambda _jobid: pytest.fail("pull must not cancel job")
    )

    assert remote.attach("20260101-000000") is True

    assert len(pulled) == 1
    assert pulled[0][0] == "20260101-000000"
    assert "PULL ARMED" in capsys.readouterr().out


def test_attach_pull_failure_reports_and_attachment_continues(monkeypatch, capsys):
    running = _status_output("running hopg [1/1] since now", squeue_state="RUNNING")
    done = _status_output("done now", squeue_state="NOT_QUEUED")
    monkeypatch.setattr(viewer, "_status_stream", lambda _cmd: iter([running, running, done]))
    monkeypatch.setattr(viewer, "KeyListener", lambda: _FakeKeyListener([["p"], ["y"], []]))

    def boom(jobid, sections):
        raise SystemExit("remote pull failed for 1 of 1 requested checkpoint(s): hbn")

    monkeypatch.setattr(viewer, "_pull_attached_progress", boom)
    monkeypatch.setattr(lifecycle, "_stop_jobid", lambda _j: pytest.fail("must not cancel"))

    assert remote.attach("20260101-000000") is True
    assert "PARTIAL PULL FAILED" in capsys.readouterr().out


def test_attach_cancel_keybinding_confirms_and_scancels_the_job(monkeypatch, capsys):
    """item 5: 'x' arms, 'y' on a LATER poll confirms -- two different keys, not
    a bare single keystroke -- and cancels via lifecycle._stop_jobid."""
    running = _status_output("running hopg [1/1] since now", squeue_state="RUNNING")
    outputs = iter([running, running])
    monkeypatch.setattr(viewer, "_status_stream", lambda _cmd: outputs)
    fake_keys = _FakeKeyListener([["x"], ["y"]])
    monkeypatch.setattr(viewer, "KeyListener", lambda: fake_keys)
    stopped = []
    monkeypatch.setattr(lifecycle, "_stop_jobid", lambda jobid: stopped.append(jobid))

    assert remote.attach("20260101-000000") is False

    assert stopped == ["20260101-000000"]
    assert fake_keys.stopped is True
    out = capsys.readouterr().out
    assert "CANCEL ARMED" in out
    assert "CANCELLED BY USER" in out


def test_attach_cancel_keybinding_disarms_on_any_other_key(monkeypatch, capsys):
    running = _status_output("running hopg [1/1] since now", squeue_state="RUNNING")
    done = _status_output("done now", squeue_state="NOT_QUEUED")
    outputs = iter([running, done])
    monkeypatch.setattr(viewer, "_status_stream", lambda _cmd: outputs)
    fake_keys = _FakeKeyListener([["x"], ["z"]])
    monkeypatch.setattr(viewer, "KeyListener", lambda: fake_keys)
    stopped = []
    monkeypatch.setattr(lifecycle, "_stop_jobid", lambda jobid: stopped.append(jobid))

    assert remote.attach("20260101-000000") is True

    assert stopped == []  # wrong confirm key disarms instead of cancelling
    assert "FINISHED" in capsys.readouterr().out


def test_attach_cancel_arm_expires_without_a_confirm_key(monkeypatch, capsys):
    running = _status_output("running hopg [1/1] since now", squeue_state="RUNNING")
    done = _status_output("done now", squeue_state="NOT_QUEUED")
    outputs = iter([running, done])
    monkeypatch.setattr(viewer, "_status_stream", lambda _cmd: outputs)
    fake_keys = _FakeKeyListener([["x"], []])
    monkeypatch.setattr(viewer, "KeyListener", lambda: fake_keys)
    # First frame arms at t=0 (armed_until = 0 + _CANCEL_ARM_SECONDS); every
    # later monotonic() call reads t=100, well past the window, so the second
    # frame's expiry check must disarm before it ever looks at (the empty)
    # keys.poll() for that frame.
    clock = iter([0.0])
    monkeypatch.setattr(viewer.time, "monotonic", lambda: next(clock, 100.0))
    stopped = []
    monkeypatch.setattr(lifecycle, "_stop_jobid", lambda jobid: stopped.append(jobid))

    assert remote.attach("20260101-000000") is True

    assert stopped == []
    out = capsys.readouterr().out
    assert out.count("CANCEL ARMED") == 1  # armed on frame 1, disarmed by frame 2
    assert "FINISHED" in out


def test_attach_cancel_request_failure_is_reported_without_a_traceback(monkeypatch, capsys):
    running = _status_output("running hopg [1/1] since now", squeue_state="RUNNING")
    outputs = iter([running, running])
    monkeypatch.setattr(viewer, "_status_stream", lambda _cmd: outputs)
    fake_keys = _FakeKeyListener([["x"], ["y"]])
    monkeypatch.setattr(viewer, "KeyListener", lambda: fake_keys)

    def _raise(jobid):
        raise SystemExit(f"job {jobid} is not an active SLURM job")

    monkeypatch.setattr(lifecycle, "_stop_jobid", _raise)

    assert remote.attach("20260101-000000") is False

    assert "CANCEL REQUEST FAILED" in capsys.readouterr().out


def test_attach_warns_when_defaulting_to_terminal_job(monkeypatch, capsys):
    """No job id + resolved job already terminal -> stderr warning, still attaches."""
    monkeypatch.setattr(state, "_latest_jobid", lambda: "sub_100keV")
    monkeypatch.setattr(
        state, "_job_state", lambda jobid: "cancelled [932] 2026-07-28T18:15:56-07:00"
    )
    seen = []
    monkeypatch.setattr(
        viewer, "_live_status", lambda jobid, detail: seen.append((jobid, detail)) or True
    )

    viewer.attach()

    captured = capsys.readouterr()
    assert "defaulting to sub_100keV" in captured.err
    assert "not running" in captured.err
    assert "defaulting" not in captured.out  # warning belongs on stderr
    assert seen == [("sub_100keV", 0)]


def test_attach_does_not_warn_for_explicit_jobid(monkeypatch, capsys):
    """An explicit job id is an informed choice: never read state, never warn."""

    def forbidden(jobid):
        raise AssertionError("state must not be read for an explicit job id")

    monkeypatch.setattr(state, "_job_state", forbidden)
    monkeypatch.setattr(viewer, "_live_status", lambda jobid, detail: True)

    viewer.attach("sub_100keV-3")

    assert "defaulting" not in capsys.readouterr().err


def test_attach_does_not_warn_when_default_job_is_live(monkeypatch, capsys):
    monkeypatch.setattr(state, "_latest_jobid", lambda: "sub_100keV-7")
    monkeypatch.setattr(state, "_job_state", lambda jobid: "running mos2 [4/21] since now")
    monkeypatch.setattr(viewer, "_live_status", lambda jobid, detail: True)

    viewer.attach()

    assert "defaulting" not in capsys.readouterr().err
