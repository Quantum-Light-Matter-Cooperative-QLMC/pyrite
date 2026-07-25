import subprocess
import sys

import pytest

from cxr_mc import line_grid
from tests.cli_helpers import assert_clean_result, invoke

CLICK_COMMANDS = (
    "derive",
    "submit",
    "status",
    "attach",
    "logs",
    "stop",
    "apply",
    "set",
    "set-brem",
    "defaults",
    "show",
    "regen-golden",
)


def test_click_group_exposes_full_line_grid_tree():
    assert tuple(line_grid.command.commands) == CLICK_COMMANDS


@pytest.mark.parametrize("path", [(), *((name,) for name in CLICK_COMMANDS)])
def test_click_help_paths_are_clean(path):
    result = invoke(line_grid.command, [*path, "--help"])

    assert_clean_result(result)
    assert "Usage:" in result.stdout


@pytest.mark.parametrize(
    "module",
    ("cxr_mc.line_grid.derive", "cxr_mc.line_grid.job"),
)
def test_standalone_module_entry_points_remain_available(module):
    completed = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "usage:" in completed.stdout
    assert "Traceback" not in completed.stderr


def test_click_status_delegates_with_detail(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        line_grid.remote,
        "job_status",
        lambda jobid=None, detail=0: seen.update(jobid=jobid, detail=detail),
    )

    result = invoke(line_grid.command, ["status", "job7", "-vv"])

    assert_clean_result(result)
    assert seen == {"jobid": "job7", "detail": 2}


@pytest.mark.parametrize("status", [1, 75, 130])
def test_click_follow_logs_propagates_remote_exit_status(monkeypatch, status):
    monkeypatch.setattr(line_grid.remote, "tail_logs", lambda _jobid, _follow: status)

    result = invoke(line_grid.command, ["logs", "--follow"])

    assert_clean_result(result, exit_code=status)


def test_click_submit_forwards_geometry_and_set_default(monkeypatch):
    seen = {}
    monkeypatch.setattr(line_grid.job, "start", lambda **kwargs: seen.update(kwargs))

    result = invoke(
        line_grid.command,
        [
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
        ],
    )

    assert_clean_result(result)
    assert seen["materials"] == "diamond,wse2"
    assert seen["energies"] == "100,200"
    assert seen["tilts"] == "0,1.5"
    assert seen["azimuths"] == "45,90"
    assert seen["thickness"] == "1000,2000"
    assert seen["set_default"] is True


def test_click_submit_routes_legacy_message_to_stderr(monkeypatch):
    monkeypatch.setattr(
        line_grid.job,
        "start",
        lambda **_kwargs: (_ for _ in ()).throw(SystemExit("invalid material text")),
    )

    result = invoke(line_grid.command, ["submit", "--materials", "bad"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Error: invalid material text\n"
    assert "Traceback" not in result.output


def test_click_derive_forwards_brem_step(monkeypatch):
    seen = {}
    from cxr_mc.line_grid import derive

    monkeypatch.setattr(derive, "main", lambda argv: seen.update(argv=argv) or 0)

    result = invoke(line_grid.command, ["derive", "--brem-step", "12.5"])

    assert_clean_result(result)
    assert seen["argv"] == ["--brem-step", "12.5"]


@pytest.mark.parametrize("status", [1, 75, 130])
def test_click_derive_preserves_nonzero_status(monkeypatch, status):
    from cxr_mc.line_grid import derive

    monkeypatch.setattr(derive, "main", lambda _argv: status)

    result = invoke(line_grid.command, ["derive"])

    assert_clean_result(result, exit_code=status)


def test_click_apply_dispatches_with_pull_and_force(monkeypatch):
    seen = {}
    monkeypatch.setattr(line_grid, "_pull_combined", lambda: "combined.json")
    monkeypatch.setattr(
        line_grid.apply, "apply_file", lambda path, **kw: seen.update(path=path, **kw)
    )

    result = invoke(line_grid.command, ["apply", "--pull", "--force"])

    assert_clean_result(result)
    assert seen["path"] == "combined.json"
    assert seen["force"] is True


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (KeyError("section [materials.bad] not found"), "section [materials.bad] not found"),
        (ValueError("invalid derived bounds"), "invalid derived bounds"),
        (OSError("cannot read bounds.json"), "cannot read bounds.json"),
    ],
)
def test_click_apply_expected_failures_use_stderr(monkeypatch, error, message):
    monkeypatch.setattr(
        line_grid.apply, "apply_file", lambda *_args, **_kwargs: (_ for _ in ()).throw(error)
    )

    result = invoke(line_grid.command, ["apply", "bounds.json"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == f"Error: {message}\n"
    assert "Traceback" not in result.output


@pytest.mark.parametrize("command_name", ["set", "set-brem"])
def test_click_set_expected_domain_failure_uses_stderr(monkeypatch, command_name):
    target = "set_line_grid" if command_name == "set" else "set_brem_grid"
    monkeypatch.setattr(
        line_grid.apply,
        target,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("unknown material")),
    )
    argv = [command_name, "bad", "--stop", "100"]
    if command_name == "set":
        argv.extend(("--energy", "50"))

    result = invoke(line_grid.command, argv)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Error: unknown material\n"
    assert "Traceback" not in result.output


def test_pull_combined_quotes_remote_scp_path(monkeypatch):
    calls = []
    monkeypatch.setattr(line_grid.remote.config, "HOST", "qlmc")
    monkeypatch.setattr(line_grid.remote.config, "REMOTE_DIR", "/srv/cxr data")
    monkeypatch.setattr(line_grid.remote, "_run", lambda command: calls.append(command))

    local = line_grid._pull_combined("bounds-result.json")

    assert local == "bounds-result.json"
    assert calls == [["scp", "qlmc:'/srv/cxr data/bounds-result.json'", "bounds-result.json"]]


def test_click_stop_falls_back_to_latest_jobid(monkeypatch):
    seen = {}
    monkeypatch.setattr(line_grid.remote, "_latest_jobid", lambda: "job9")
    monkeypatch.setattr(line_grid.remote, "_stop_jobid", lambda jobid: seen.update(jobid=jobid))

    result = invoke(line_grid.command, ["stop"])

    assert_clean_result(result)
    assert seen == {"jobid": "job9"}


def test_click_regen_golden_delegates_check(monkeypatch):
    from cxr_mc.line_grid import golden

    seen = {}
    monkeypatch.setattr(
        golden,
        "regen",
        lambda check=False: seen.update(check=check) or 0,
    )

    result = invoke(line_grid.command, ["regen-golden", "--check"])

    assert_clean_result(result)
    assert seen["check"] is True


@pytest.mark.parametrize("status", [1, 75, 130])
def test_click_regen_golden_preserves_nonzero_status(monkeypatch, status):
    from cxr_mc.line_grid import golden

    monkeypatch.setattr(golden, "regen", lambda check=False: status)

    result = invoke(line_grid.command, ["regen-golden"])

    assert_clean_result(result, exit_code=status)


@pytest.mark.parametrize(
    "argv",
    [
        ["set", "hopg", "--energy", "0", "--stop", "3000"],
        ["set", "hopg", "--energy", "30", "--stop", "-1"],
        ["set", "hopg", "--energy", "30", "--stop", "3000", "--num", "0"],
        ["set-brem", "hopg", "--stop", "1000", "--step", "0"],
        ["defaults", "--set", "--tilts", "90"],
        ["defaults", "--set", "--azimuths", "361"],
        ["defaults", "--set", "--thickness", "0"],
        ["defaults", "--set", "--brem-step", "nan"],
        ["derive", "--energies", "0"],
        ["derive", "--tilts", "90"],
        ["submit", "--azimuths", "361"],
        ["submit", "--thickness", "0"],
        ["submit", "--slice-minutes", "0"],
    ],
)
def test_click_numeric_domains_fail_at_boundary(argv):
    result = invoke(line_grid.command, argv)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Invalid value" in result.stderr
    assert "Traceback" not in result.output


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


@pytest.mark.parametrize(
    ("argv", "exit_code", "stderr_text"),
    [
        (
            ["defaults", "--tilts", "5"],
            2,
            "--tilts require --set",
        ),
        (
            ["show", "not-a-material"],
            1,
            "unknown material: not-a-material",
        ),
        (
            ["apply"],
            2,
            "no JSON: pass a path or --pull",
        ),
    ],
)
def test_click_failure_exit_and_stream_contract(argv, exit_code, stderr_text):
    result = invoke(line_grid.command, argv)

    assert result.exit_code == exit_code
    assert result.stdout == ""
    assert stderr_text in result.stderr
    assert "Traceback" not in result.output
