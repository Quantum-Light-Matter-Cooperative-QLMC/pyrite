"""Remote rebrem tests."""

import pytest

from pyrite import remote
from pyrite.cli.commands import remote as cli  # noqa: F401
from pyrite.console import dashboard as presentation
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
    _remote_main,
)


def test_rebrem_queue_script_flags_progress_and_markers():
    s = remote._rebrem_queue_script(
        "20260101-000000", ["hopg", "hbn"], ne_brem=1000, brem_step_eV=25.0, redo_all=True
    )
    assert "pyrite checkpoint recompute brem" in s
    assert "--brem-histories 1000" in s and "--step 25" in s and "--redo-all" in s
    # per-material progress record feeds the shared plain/attached status dashboard
    assert '--progress-file "$JOBDIR/progress/$m.json"' in s
    # completion markers match the scan queue's so _completed_materials works
    assert 'echo "completed: $m"' in s and 'echo "failed: $m"' in s
    assert "mats=(hopg hbn)" in s


def test_rebrem_queue_script_omits_unset_flags():
    s = remote._rebrem_queue_script("j", ["hopg"], ne_brem=None, brem_step_eV=None, redo_all=False)
    assert "--brem-histories" not in s and "--step" not in s and "--redo-all" not in s


def test_rebrem_metadata_keys_mode_line_and_live_job_plumbing():
    meta = remote._rebrem_queue_metadata("j", ["hopg"], 1000, 25.0, False)
    assert "kind: rebrem" in meta
    assert "materials: hopg" in meta and "quick: False" in meta  # _live_jobs fields
    summary = presentation.mode_summary(meta)
    assert "brem-only recompute" in summary
    assert "Ne_brem=1000" in summary and "step 25.0 eV" in summary
    # zhai detection must not shadow rebrem despite the ne_brem field
    assert "Zhai" not in summary


def test_rebrem_start_refuses_when_material_checkpoint_is_busy(monkeypatch):
    monkeypatch.setattr(state, "_live_jobs", lambda: [("job1", False, ["hopg"])])
    monkeypatch.setattr(transport, "sync_code", lambda: pytest.fail("must refuse before syncing"))
    with pytest.raises(SystemExit, match="refusing to start"):
        remote.start_rebrem_queue(["hopg"], ne_brem=1000)


def test_rebrem_start_dry_run_prints_without_ssh_or_sync(monkeypatch, capsys):
    monkeypatch.setattr(state, "_live_jobs", lambda: pytest.fail("dry-run must not check busy"))
    monkeypatch.setattr(transport, "sync_code", lambda: pytest.fail("dry-run must not sync"))
    monkeypatch.setattr(
        transport.subprocess, "run", lambda *a, **kw: pytest.fail("dry-run must not ssh")
    )

    jobid = remote.start_rebrem_queue(["hopg"], ne_brem=1000, dry_run=True)

    out = capsys.readouterr().out
    assert jobid in out and "pyrite checkpoint recompute brem" in out


def test_rebrem_cli_submits_attaches_and_pulls_completed(monkeypatch):
    events = []
    monkeypatch.setattr(
        lifecycle,
        "start_rebrem_queue",
        lambda mats, **kw: events.append(("start", mats, kw)) or "j",
    )
    monkeypatch.setattr(viewer, "attach", lambda jobid: events.append(("attach", jobid)) or True)
    monkeypatch.setattr(state, "_completed_materials", lambda jobid, mats: ["hopg"])
    monkeypatch.setattr(lifecycle, "pull", lambda stems, **kw: events.append(("pull", stems)))

    _remote_main(["rebrem", "hopg", "hbn", "--brem-histories", "1000", "--step", "25"])

    assert events[0][0] == "start" and events[0][1] == ["hopg", "hbn"]
    assert events[0][2]["ne_brem"] == 1000 and events[0][2]["brem_step_eV"] == 25.0
    assert events[1] == ("attach", "j")
    assert events[2] == ("pull", ["hopg"])


def test_rebrem_cli_skips_pull_when_viewer_disconnects(monkeypatch):
    monkeypatch.setattr(lifecycle, "start_rebrem_queue", lambda mats, **kw: "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: False)
    monkeypatch.setattr(
        lifecycle, "pull", lambda *a, **kw: pytest.fail("must not pull after disconnect")
    )

    _remote_main(["rebrem", "hopg"])


def test_rebrem_chunked_queue_script_self_resubmits():
    s = scripts._rebrem_chunked_queue_script(
        "J1", ["mos2", "w"], ne_brem=1000, brem_step_eV=None, redo_all=False, chunk_minutes=10.0
    )
    assert "pyrite checkpoint recompute brem" in s
    assert "--brem-histories 1000" in s
    assert "--max-minutes" in s
    assert "--nice=10000" in s
    assert '--progress-file "$JOBDIR/progress/$m.json"' in s
    # metadata unchanged by chunking
    meta = scripts._rebrem_queue_metadata("J1", ["mos2", "w"], 1000, None, False)
    assert "kind: rebrem" in meta
