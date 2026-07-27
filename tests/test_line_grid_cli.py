import subprocess
import sys

import pytest

from cxr_mc import line_grid
from tests.cli_helpers import assert_clean_result, invoke

CLICK_COMMANDS = (
    "derive",
    "submit",
    "job",
    "status",
    "attach",
    "logs",
    "stop",
    "apply",
    "line",
    "brem",
    "defaults",
    "show",
    "regen-golden",
)


def test_click_group_exposes_full_line_grid_tree():
    assert tuple(line_grid.command.commands) == (
        "line",
        "brem",
        "derive",
        "submit",
        "job",
        "status",
        "attach",
        "logs",
        "stop",
        "apply",
        "defaults",
        "show",
        "regen-golden",
    )


@pytest.mark.parametrize(
    "path",
    [
        (),
        *((name,) for name in CLICK_COMMANDS),
        *((("job", name)) for name in ("status", "attach", "logs", "stop")),
        *(((band, name)) for band in ("line", "brem") for name in ("set", "show")),
        ("line", "delete"),
    ],
)
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

    result = invoke(line_grid.command, ["job", "status", "job7", "-vv"])

    assert_clean_result(result)
    assert seen == {"jobid": "job7", "detail": 2}


def test_click_status_json_reuses_remote_machine_contract(monkeypatch):
    seen = {}

    def status(args):
        seen.update(vars(args))
        line_grid.emit_json_result(
            line_grid.cli_json.JsonResult("cxr.remote.status", {"job": {"job_id": args.jobid}})
        )

    monkeypatch.setattr(line_grid.remote, "_cli_status", status)

    result = invoke(line_grid.command, ["job", "status", "job7", "--json"])

    assert_clean_result(result)
    assert '"schema":"cxr.remote.status"' in result.stdout
    assert seen == {"jobid": "job7", "verbose": 0, "json_output": True}


@pytest.mark.parametrize("status", [1, 75, 130])
def test_click_follow_logs_propagates_remote_exit_status(monkeypatch, status):
    monkeypatch.setattr(line_grid.remote, "tail_logs", lambda _jobid, _follow: status)

    result = invoke(line_grid.command, ["job", "logs", "--follow"])

    assert_clean_result(result, exit_code=status)


def test_click_submit_with_invalid_azimuths_fails(monkeypatch):
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

    assert result.exit_code != 0
    print(result.output)
    assert "Invalid value for '--azimuths'" in result.output
    assert seen == {}

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
            "95,180,260",
            "--thickness",
            "1000,2000",
            "--set-default",
        ],
    )

    assert_clean_result(result)
    assert seen["materials"] == "diamond,wse2"
    assert seen["energies"] == "100,200"
    assert seen["tilts"] == "0,1.5"
    assert seen["azimuths"] == "95,180,260"
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


@pytest.mark.parametrize("command_name", ["line", "brem"])
def test_click_set_expected_domain_failure_uses_stderr(monkeypatch, command_name):
    target = "set_line_grid" if command_name == "line" else "set_brem_grid"
    monkeypatch.setattr(
        line_grid.apply,
        target,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("unknown material")),
    )
    argv = [command_name, "set", "bad", "--stop", "100"]
    if command_name == "line":
        argv.extend(("--energy", "50"))

    result = invoke(line_grid.command, argv)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Error: unknown material\n"
    assert "Traceback" not in result.output


def test_click_line_delete_confirmed_deletes_and_warns_stale_golden(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        line_grid.apply,
        "delete_line_grid",
        lambda material, energies, **kw: (
            seen.update(material=material, energies=list(energies)) or [30.0, 100.0]
        ),
    )

    result = invoke(
        line_grid.command,
        ["line", "delete", "hopg", "--energy", "30", "--energy", "100"],
        input="y\n",
    )

    assert result.exit_code == 0
    assert seen == {"material": "hopg", "energies": [30.0, 100.0]}
    assert "deleted hopg: 30, 100 keV" in result.stdout
    assert "cannot be undone" in result.stderr
    assert "golden is now stale" in result.stderr


def test_click_line_delete_declined_confirmation_aborts(monkeypatch):
    monkeypatch.setattr(
        line_grid.apply,
        "delete_line_grid",
        lambda *_a, **_kw: pytest.fail("delete_line_grid must not run when declined"),
    )

    result = invoke(line_grid.command, ["line", "delete", "hopg", "--energy", "30"], input="n\n")

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Aborted" in result.stderr


def test_click_line_delete_yes_skips_prompt(monkeypatch):
    monkeypatch.setattr(
        line_grid.apply, "delete_line_grid", lambda material, energies, **kw: [30.0]
    )

    result = invoke(line_grid.command, ["line", "delete", "hopg", "--energy", "30", "--yes"])

    assert result.exit_code == 0
    assert "deleted hopg: 30 keV" in result.stdout


def test_click_line_delete_dry_run_skips_prompt_and_confirms_via_kwarg(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        line_grid.apply,
        "delete_line_grid",
        lambda material, energies, **kw: seen.update(kw) or [30.0],
    )

    result = invoke(line_grid.command, ["line", "delete", "hopg", "--energy", "30", "--dry-run"])

    assert result.exit_code == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert seen == {"dry_run": True}


def test_click_line_delete_dry_run_and_json_conflict():
    result = invoke(
        line_grid.command, ["line", "delete", "hopg", "--energy", "30", "--dry-run", "--json"]
    )

    assert result.exit_code == 2
    assert "--dry-run and --json cannot be combined" in result.stderr


def test_click_line_delete_json_requires_yes():
    result = invoke(line_grid.command, ["line", "delete", "hopg", "--energy", "30", "--json"])

    assert result.exit_code == 2
    assert "--json requires --yes" in result.stderr


def test_click_line_delete_json_emits_one_envelope(monkeypatch):
    monkeypatch.setattr(
        line_grid.apply, "delete_line_grid", lambda material, energies, **kw: [30.0]
    )

    result = invoke(
        line_grid.command, ["line", "delete", "hopg", "--energy", "30", "--yes", "--json"]
    )

    assert_clean_result(result)
    assert result.stdout.count("\n") == 1
    import json

    document = json.loads(result.stdout)
    assert document["schema"] == "cxr.energy-grid.line-delete"
    assert document["payload"] == {"material": "hopg", "deleted_energies_keV": [30.0]}


def test_click_line_delete_expected_failure_uses_stderr(monkeypatch):
    monkeypatch.setattr(
        line_grid.apply,
        "delete_line_grid",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            ValueError("no energy_grids entry for material: bad")
        ),
    )

    result = invoke(line_grid.command, ["line", "delete", "bad", "--energy", "30", "--yes"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Error: no energy_grids entry for material: bad\n"
    assert "Traceback" not in result.output


def test_pull_combined_quotes_remote_scp_path(monkeypatch):
    calls = []
    monkeypatch.setattr(line_grid.remote.config, "HOST", "qlmc")
    monkeypatch.setattr(line_grid.remote.config, "REMOTE_DIR", "/srv/cxr data")
    monkeypatch.setattr(line_grid.remote, "_run", lambda command: calls.append(command))

    local = line_grid._pull_combined("bounds-result.json")

    assert local == "bounds-result.json"
    assert calls == [["scp", "qlmc:'/srv/cxr data/bounds-result.json'", "bounds-result.json"]]


def test_click_stop_requires_explicit_target():
    result = invoke(line_grid.command, ["job", "stop"])

    assert result.exit_code == 2
    assert "needs JOBID, or use --latest" in result.stderr


def test_click_stop_previews_latest_and_yes_cancels(monkeypatch):
    seen = {}
    monkeypatch.setattr(line_grid.remote, "_latest_jobid", lambda: "job9")
    monkeypatch.setattr(line_grid.remote, "_stop_jobid", lambda jobid: seen.update(jobid=jobid))

    preview = invoke(line_grid.command, ["job", "stop", "--latest"])
    confirmed = invoke(line_grid.command, ["job", "stop", "--latest", "--yes"])

    assert_clean_result(
        preview,
        stdout="would cancel remote job: job9\nre-run with --yes to cancel\n",
    )
    assert_clean_result(confirmed)
    assert seen == {"jobid": "job9"}


def test_hidden_line_grid_job_aliases_remain_callable():
    root_help = invoke(line_grid.command, ["--help"])
    command_lines = {
        line.split()[0]
        for line in root_help.stdout.splitlines()
        if line.startswith("  ") and line.strip() and not line.lstrip().startswith("-")
    }
    assert "job" in command_lines
    assert command_lines.isdisjoint({"status", "attach", "logs", "stop"})

    for alias in ("status", "attach", "logs", "stop"):
        result = invoke(line_grid.command, [alias, "--help"])
        assert_clean_result(result)


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
        ["line", "set", "hopg", "--energy", "0", "--stop", "3000"],
        ["line", "set", "hopg", "--energy", "30", "--stop", "-1"],
        ["line", "set", "hopg", "--energy", "30", "--stop", "3000", "--num", "0"],
        ["line", "delete", "hopg", "--energy", "0"],
        ["brem", "set", "hopg", "--stop", "1000", "--step", "0"],
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
            ["energy-grid", "defaults", "--tilts", "5"],
            2,
            "--tilts require --set",
        ),
        (
            ["energy-grid", "show", "not-a-material"],
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
