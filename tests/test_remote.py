"""cxr_mc.remote (``cxr remote``): material-name validation and the
detached-queue runner generation.

These are pure-string/logic checks (no ssh), so they run anywhere. The one
exception is the clear-listing regression test, which executes the box-side
shell snippet under a local bash (skipped when bash is unavailable)."""

import shutil
import subprocess

import pytest

from cxr_mc import remote


def test_check_materials_accepts_crystal_keys():
    remote._check_materials(["mose2", "hopg", "mote2", "silicon"])  # no raise


@pytest.mark.parametrize("bad", ["rm -rf /", "a;b", "../etc", "a b", "", "m&n"])
def test_check_materials_rejects_injection(bad):
    with pytest.raises(SystemExit):
        remote._check_materials([bad])


def test_queue_script_has_per_material_scan_calls():
    s = remote._queue_script("20260101-000000", ["mose2", "wse2"], quick=True, workers=8)
    assert "scan.py" in s
    assert "--quick" in s and "--workers 8" in s
    assert "mose2" in s and "wse2" in s
    assert "20260101-000000" in s  # job id is embedded
    assert "mats=(mose2 wse2)" in s  # bash array drives the loop


def test_queue_script_no_flags_when_unset():
    s = remote._queue_script("j", ["mos2"], quick=False, workers=None)
    assert "--quick" not in s and "--workers" not in s


def test_launch_queue_command_backgrounds_only_runner():
    launch = remote._launch_queue_command("20260101-000000")

    assert "&& (nohup setsid bash" in launch
    assert "</dev/null &) && echo 'launched 20260101-000000'" in launch
    assert "</dev/null & echo" not in launch


def test_attach_uses_stdin_closed_ssh_for_live_view(monkeypatch):
    runs = []
    monkeypatch.setattr(remote.subprocess, "run", lambda cmd: runs.append(cmd))

    remote.attach("20260101-000000")

    assert len(runs) == 1
    assert runs[0][:3] == ["ssh", "-n", remote.HOST]
    assert len(runs[0]) == 4


def test_follow_logs_use_stdin_closed_ssh(monkeypatch):
    runs = []
    monkeypatch.setattr(remote.subprocess, "run", lambda cmd: runs.append(cmd))

    remote.tail_logs("20260101-000000", follow=True)

    assert len(runs) == 1
    assert runs[0][:3] == ["ssh", "-n", remote.HOST]
    assert len(runs[0]) == 4


def test_stems_quick_suffix():
    assert remote._stems(["mose2", "wse2"], True) == ["mose2_quick", "wse2_quick"]
    assert remote._stems(["mose2"], False) == ["mose2"]


def test_stop_materials_resolve_unique_live_jobs(monkeypatch):
    stopped = []
    monkeypatch.setattr(
        remote,
        "_live_jobs",
        lambda: [("job1", False, ["hopg"]), ("job2", False, ["mose2", "wse2"])],
    )
    monkeypatch.setattr(remote, "_stop_jobid", stopped.append)

    remote.stop_jobs(["wse2", "mose2"])

    assert stopped == ["job2"]


def test_stop_all_stops_every_live_job(monkeypatch):
    stopped = []
    monkeypatch.setattr(
        remote,
        "_live_jobs",
        lambda: [("job1", False, ["hopg"]), ("job2", True, ["mose2"])],
    )
    monkeypatch.setattr(remote, "_stop_jobid", stopped.append)

    remote.stop_jobs(all_jobs=True)

    assert stopped == ["job1", "job2"]


def test_stop_rejects_bad_material_before_live_job_lookup(monkeypatch):
    monkeypatch.setattr(
        remote,
        "_live_jobs",
        lambda: pytest.fail("must validate before checking live jobs"),
    )

    with pytest.raises(SystemExit):
        remote.stop_jobs(["bad;material"])


def test_stop_cli_accepts_materials(monkeypatch):
    calls = []
    monkeypatch.setattr(
        remote, "stop_jobs", lambda materials, all_jobs: calls.append((materials, all_jobs))
    )

    remote.main(["stop", "hopg", "mose2"])

    assert calls == [(["hopg", "mose2"], False)]


def test_stop_cli_accepts_all(monkeypatch):
    calls = []
    monkeypatch.setattr(
        remote, "stop_jobs", lambda materials, all_jobs: calls.append((materials, all_jobs))
    )

    remote.main(["stop", "--all"])

    assert calls == [([], True)]


# ---- clear <material> (checkpoint lifecycle, component 3) ----------------------
def _no_live_jobs(monkeypatch):
    monkeypatch.setattr(remote, "_live_jobs", lambda: [])


def test_clear_refuses_when_a_live_job_produces_the_stem(monkeypatch):
    # a live job producing hopg -> clearing hopg must refuse before any ssh
    monkeypatch.setattr(remote, "_live_jobs", lambda: [("job1", False, ["hopg"])])
    monkeypatch.setattr(
        remote, "_ssh_capture", lambda *a: pytest.fail("must not ssh when refusing")
    )
    with pytest.raises(SystemExit, match="refusing to clear"):
        remote.clear_remote("hopg", yes=True)


def test_clear_refuses_for_quick_stem_collision(monkeypatch):
    # a live --quick job producing hopg_quick still blocks a clear of hopg
    monkeypatch.setattr(remote, "_live_jobs", lambda: [("job1", True, ["hopg"])])
    with pytest.raises(SystemExit, match="refusing to clear"):
        remote.clear_remote("hopg")


def test_clear_dry_preview_lists_but_does_not_delete(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(remote, "_ssh_capture", lambda *a: "hopg.pkl\nhopg_quick.pkl\n")
    runs = []
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: runs.append(cmd))
    remote.clear_remote("hopg", yes=False)
    out = capsys.readouterr().out
    assert "would delete" in out and "hopg.pkl" in out and "hopg_quick.pkl" in out
    assert runs == []  # nothing deleted in a dry preview


def test_clear_yes_deletes_existing_files(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(remote, "_ssh_capture", lambda *a: "hopg.pkl hopg_quick.pkl")
    runs = []
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: runs.append(cmd))
    remote.clear_remote("hopg", yes=True)
    assert len(runs) == 1
    deletion = " ".join(runs[0])
    assert "rm -f" in deletion and "hopg.pkl" in deletion and "hopg_quick.pkl" in deletion
    assert "cleared on the box" in capsys.readouterr().out


def test_clear_reports_nothing_when_no_files(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(remote, "_ssh_capture", lambda *a: "\n")
    runs = []
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: runs.append(cmd))
    remote.clear_remote("hopg", yes=True)
    assert "nothing to clear" in capsys.readouterr().out
    assert runs == []  # nothing to delete


def test_clear_rejects_bad_material(monkeypatch):
    monkeypatch.setattr(
        remote, "_live_jobs", lambda: pytest.fail("must validate before touching jobs")
    )
    with pytest.raises(SystemExit):
        remote.clear_remote("rm -rf /")


def _bash_or_skip(tmp_path):
    """A bash that can reach tmp_path, or skip. Git Bash handles C:/-style posix
    paths; a WSL bash can't, and would hit the snippet's `|| exit 0` and silently
    return an empty listing -- so probe the cd before trusting the test."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    probe = subprocess.run([bash, "-c", f"cd '{tmp_path.as_posix()}'"], capture_output=True)
    if probe.returncode != 0:
        pytest.skip("bash cannot reach the pytest tmp dir")
    return bash


def test_clear_listing_snippet_exits_zero_when_quick_pkl_missing(monkeypatch, tmp_path, capsys):
    """Regression (checkpoint-lifecycle verification, 2026-07-07): the box-side
    listing loop's last command is `[ -f "$f" ] && echo "$f"`, whose failure
    status leaks out as the loop's -- and hence ssh's -- exit status whenever the
    last stem (<material>_quick.pkl) is missing, making _ssh_capture abort the
    clear before the dry preview prints. Execute the real snippet under bash
    against a checkpoints/ dir holding only hopg.pkl."""
    bash = _bash_or_skip(tmp_path)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "hopg.pkl").write_bytes(b"x")  # no hopg_quick.pkl

    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(remote, "REMOTE_DIR", tmp_path.as_posix())

    def local_bash_capture(remote_cmd):
        # same semantics as _ssh_capture, but run the snippet locally
        r = subprocess.run([bash, "-c", remote_cmd], capture_output=True, encoding="utf-8")
        if r.returncode != 0:
            raise SystemExit(f"ssh command failed (exit {r.returncode})")
        return r.stdout

    monkeypatch.setattr(remote, "_ssh_capture", local_bash_capture)
    monkeypatch.setattr(
        remote, "_run", lambda cmd, **kw: pytest.fail("dry preview must not delete")
    )
    remote.clear_remote("hopg", yes=False)  # must not raise SystemExit
    out = capsys.readouterr().out
    assert "hopg.pkl" in out and "hopg_quick.pkl" not in out


# ---- scan --quick --grid must be rejected at parse time -------------------------
def test_scan_rejects_quick_plus_grid_before_any_work(monkeypatch):
    """--quick checkpoints aren't grid-filterable (cxr slim rejects _quick stems),
    so scan --quick --grid must fail up front -- not run the whole sweep and then
    traceback on the trailing pull."""
    monkeypatch.setattr(
        remote, "_live_jobs", lambda: pytest.fail("must reject before the busy check")
    )
    monkeypatch.setattr(remote, "sync_code", lambda: pytest.fail("must reject before syncing"))
    monkeypatch.setattr(
        remote, "remote_scan", lambda *a, **kw: pytest.fail("must reject before scanning")
    )
    with pytest.raises(SystemExit, match="grid"):
        remote.main(["scan", "hopg", "--quick", "--grid"])


# ---- pull defaults to --grid; -f/--full opts into the plain whole-file pull -----
def test_pull_defaults_to_grid(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "pull", lambda *a, **kw: calls.append(kw))
    remote.main(["pull", "hopg"])
    assert calls[0]["grid"] is True


def test_pull_full_flag_disables_grid(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "pull", lambda *a, **kw: calls.append(kw))
    remote.main(["pull", "hopg", "--full"])
    assert calls[0]["grid"] is False


def test_pull_short_full_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "pull", lambda *a, **kw: calls.append(kw))
    remote.main(["pull", "hopg", "-f"])
    assert calls[0]["grid"] is False


# ---- cxr remote check (Zhai GPU reproduction) ------------------------------
def test_zhai_queue_script_has_ne_flags_and_meta():
    s = remote._zhai_queue_script("20260101-000000", ne=11, ne_brem=3, ne_supp=5, refresh=True)
    assert "reproduce_zhai.py" in s
    assert "--ne 11" in s and "--ne-brem 3" in s and "--ne-supp 5" in s and "--refresh" in s
    assert "materials: zhai" in s and "quick: False" in s
    assert "20260101-000000" in s


def test_zhai_queue_script_no_refresh_flag_when_unset():
    s = remote._zhai_queue_script("j", ne=1, ne_brem=1, ne_supp=1, refresh=False)
    assert "--refresh" not in s


def test_zhai_start_refuses_when_a_zhai_job_is_already_live(monkeypatch):
    monkeypatch.setattr(remote, "_live_jobs", lambda: [("job1", False, ["zhai"])])
    with pytest.raises(SystemExit, match="refusing to start"):
        remote.start_zhai_queue()


def test_zhai_start_dry_run_prints_without_ssh_or_sync(monkeypatch, capsys):
    monkeypatch.setattr(remote, "_live_jobs", lambda: pytest.fail("dry-run must not check busy"))
    monkeypatch.setattr(remote, "sync_code", lambda: pytest.fail("dry-run must not sync"))
    monkeypatch.setattr(
        remote.subprocess, "run", lambda *a, **kw: pytest.fail("dry-run must not ssh")
    )

    jobid = remote.start_zhai_queue(dry_run=True)

    out = capsys.readouterr().out
    assert jobid in out and "reproduce_zhai.py" in out


def test_remote_check_refuses_when_a_zhai_job_is_already_live(monkeypatch):
    monkeypatch.setattr(remote, "_live_jobs", lambda: [("job1", False, ["zhai"])])
    monkeypatch.setattr(remote, "sync_code", lambda: pytest.fail("must refuse before syncing"))
    with pytest.raises(SystemExit, match="refusing to start"):
        remote.remote_check()


def test_remote_check_syncs_runs_and_pulls(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "_live_jobs", lambda: [])
    monkeypatch.setattr(remote, "sync_code", lambda: calls.append("sync"))
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: calls.append(("run", cmd)))
    monkeypatch.setattr(remote, "pull_zhai_cache", lambda: calls.append("pull"))

    remote.remote_check(ne=11, ne_brem=3, ne_supp=5, refresh=True)

    assert calls[0] == "sync"
    assert calls[1][0] == "run"
    ssh_cmd = calls[1][1]
    assert ssh_cmd[:3] == ["ssh", "-n", remote.HOST]
    assert "reproduce_zhai.py" in ssh_cmd[3]
    assert "--ne 11" in ssh_cmd[3] and "--refresh" in ssh_cmd[3]
    assert calls[2] == "pull"


def test_remote_check_no_sync_skips_sync(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "_live_jobs", lambda: [])
    monkeypatch.setattr(remote, "sync_code", lambda: calls.append("sync"))
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: calls.append("run"))
    monkeypatch.setattr(remote, "pull_zhai_cache", lambda: calls.append("pull"))

    remote.remote_check(no_sync=True)

    assert calls == ["run", "pull"]


def test_pull_zhai_cache_fetches_every_listed_file(monkeypatch, tmp_path):
    monkeypatch.setattr(remote, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(
        remote,
        "_ssh_capture",
        lambda *a: (
            "/r/checkpoints/zhai_reproduction/zhai-a.pkl\n"
            "/r/checkpoints/zhai_reproduction/zhai-b.pkl\n"
        ),
    )
    runs = []
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: runs.append(cmd))

    remote.pull_zhai_cache()

    assert len(runs) == 2
    assert runs[0][0] == "scp" and runs[0][1].endswith("zhai-a.pkl")
    assert (tmp_path / "checkpoints" / "zhai_reproduction").is_dir()


def test_pull_zhai_cache_reports_when_empty(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(remote, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(remote, "_ssh_capture", lambda *a: "")
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: pytest.fail("nothing to pull"))

    remote.pull_zhai_cache()

    assert "no zhai cache files" in capsys.readouterr().out


def test_check_cli_pull_flag_skips_run(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "pull_zhai_cache", lambda: calls.append("pull"))
    monkeypatch.setattr(
        remote, "remote_check", lambda **kw: pytest.fail("--pull must not run the reproduction")
    )

    remote.main(["check", "--pull"])

    assert calls == ["pull"]


def test_check_cli_detached_starts_queue(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "start_zhai_queue", lambda **kw: calls.append(kw) or "jid")
    monkeypatch.setattr(remote, "attach", lambda jobid: pytest.fail("no --follow: must not attach"))

    remote.main(["check", "--detached", "--ne", "11"])

    assert calls[0]["ne"] == 11


def test_check_cli_detached_follow_attaches(monkeypatch):
    monkeypatch.setattr(remote, "start_zhai_queue", lambda **kw: "jid")
    attached = []
    monkeypatch.setattr(remote, "attach", attached.append)

    remote.main(["check", "--detached", "--follow"])

    assert attached == ["jid"]


def test_check_cli_foreground_calls_remote_check(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "remote_check", lambda **kw: calls.append(kw))

    remote.main(["check", "--ne", "11", "--refresh"])

    assert calls == [{"ne": 11, "ne_brem": 200, "ne_supp": 200, "refresh": True, "no_sync": False}]


def test_sync_paths_ship_checks_and_zhai_shim():
    assert "checks" in remote.SYNC_PATHS
    assert "reproduce_zhai.py" in remote.SYNC_PATHS
