import argparse
import subprocess
import sys

import pytest

from cxr_mc import line_grid


def _parse(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    line_grid.add_subparser(sub)
    return ap.parse_args(argv)


def test_status_delegates_to_remote_job_status_with_detail(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        line_grid.remote,
        "job_status",
        lambda jobid=None, detail=0: seen.update(jobid=jobid, detail=detail),
    )
    args = _parse(["line-grid", "status", "job7", "-vv"])
    args.func(args)
    assert seen == {"jobid": "job7", "detail": 2}


@pytest.mark.parametrize("status", [1, 130])
def test_follow_logs_propagates_remote_exit_status(monkeypatch, status):
    monkeypatch.setattr(
        line_grid.remote,
        "tail_logs",
        lambda _jobid, _follow: status,
    )
    args = _parse(["line-grid", "logs", "--follow"])

    assert args.func(args) == status


def test_submit_forwards_geometry_and_set_default(monkeypatch):
    seen = {}
    monkeypatch.setattr(line_grid.job, "start", lambda **kwargs: seen.update(kwargs))
    args = _parse(
        [
            "line-grid",
            "submit",
            "--materials",
            "diamond,wse2",
            "--energies",
            "100,200",
            "--tilts",
            "0,1.5",
            "--azimuths",
            "45,90",
            "--thickness",
            "1000,2000",
            "--set-default",
        ]
    )

    assert args.func(args) == 0
    assert seen["materials"] == "diamond,wse2"
    assert seen["energies"] == "100,200"
    assert seen["tilts"] == "0,1.5"
    assert seen["azimuths"] == "45,90"
    assert seen["thickness"] == "1000,2000"
    assert seen["set_default"] is True


def test_derive_forwards_brem_step(monkeypatch):
    seen = {}
    from cxr_mc.line_grid import derive

    monkeypatch.setattr(derive, "main", lambda argv: seen.update(argv=argv) or 0)
    args = _parse(["line-grid", "derive", "--brem-step", "12.5"])

    assert args.func(args) == 0
    assert seen["argv"] == ["--brem-step", "12.5"]


def test_apply_dispatches_with_pull_and_force(monkeypatch):
    seen = {}
    monkeypatch.setattr(line_grid, "_pull_combined", lambda: "combined.json")
    monkeypatch.setattr(
        line_grid.apply, "apply_file", lambda path, **kw: seen.update(path=path, **kw)
    )
    args = _parse(["line-grid", "apply", "--pull", "--force"])
    args.func(args)
    assert seen["path"] == "combined.json" and seen["force"] is True


def test_pull_combined_quotes_remote_scp_path(monkeypatch):
    calls = []
    monkeypatch.setattr(line_grid.remote.config, "HOST", "qlmc")
    monkeypatch.setattr(line_grid.remote.config, "REMOTE_DIR", "/srv/cxr data")
    monkeypatch.setattr(line_grid.remote, "_run", lambda command: calls.append(command))

    local = line_grid._pull_combined("bounds-result.json")

    assert local == "bounds-result.json"
    assert calls == [["scp", "qlmc:'/srv/cxr data/bounds-result.json'", "bounds-result.json"]]


def test_stop_falls_back_to_latest_jobid(monkeypatch):
    seen = {}
    monkeypatch.setattr(line_grid.remote, "_latest_jobid", lambda: "job9")
    monkeypatch.setattr(line_grid.remote, "_stop_jobid", lambda jobid: seen.update(jobid=jobid))
    args = _parse(["line-grid", "stop"])
    args.func(args)
    assert seen == {"jobid": "job9"}


def test_regen_golden_delegates_check(monkeypatch):
    from cxr_mc.line_grid import golden

    seen = {}

    def _fake_regen(check=False):
        seen["check"] = check
        return 0

    monkeypatch.setattr(golden, "regen", _fake_regen)
    args = _parse(["line-grid", "regen-golden", "--check"])
    assert args.func(args) == 0
    assert seen["check"] is True


@pytest.mark.parametrize(
    "argv",
    [
        ["line-grid", "set", "hopg", "--energy", "0", "--stop", "3000"],
        ["line-grid", "set", "hopg", "--energy", "30", "--stop", "-1"],
        ["line-grid", "set", "hopg", "--energy", "30", "--stop", "3000", "--num", "0"],
        ["line-grid", "set-brem", "hopg", "--stop", "1000", "--step", "0"],
        ["line-grid", "defaults", "--set", "--tilts", "90"],
        ["line-grid", "defaults", "--set", "--azimuths", "361"],
        ["line-grid", "defaults", "--set", "--thickness", "0"],
        ["line-grid", "defaults", "--set", "--brem-step", "nan"],
    ],
)
def test_numeric_domains_fail_at_cli_boundary(argv):
    with pytest.raises(SystemExit) as exc:
        _parse(argv)

    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("argv", "exit_code", "stderr_text"),
    [
        (
            ["line-grid", "defaults", "--tilts", "5"],
            2,
            "--tilts require --set",
        ),
        (
            ["line-grid", "show", "not-a-material"],
            1,
            "unknown material: not-a-material",
        ),
    ],
)
def test_cli_failure_exit_and_stream_contract(argv, exit_code, stderr_text):
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from cxr_mc.cli import main; raise SystemExit(main())",
            *argv,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == exit_code
    assert completed.stdout == ""
    assert stderr_text in completed.stderr
    assert "Traceback" not in completed.stderr
