import subprocess
import sys

import pytest

from cxr_mc import energy_grid
from cxr_mc.cli import _core as _cli_core
from cxr_mc.cli._deprecations import message
from cxr_mc.energy_grid import _command
from tests.helpers.cli import assert_clean_result, invoke

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
    assert tuple(energy_grid.command.commands) == (
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
    result = invoke(energy_grid.command, [*path, "--help"])

    assert_clean_result(result)
    assert "Usage:" in result.stdout


@pytest.mark.parametrize(
    "module",
    ("cxr_mc.energy_grid.derive", "cxr_mc.energy_grid.job"),
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
        _command.remote,
        "job_status",
        lambda jobid=None, detail=0: seen.update(jobid=jobid, detail=detail),
    )

    result = invoke(energy_grid.command, ["job", "status", "job7", "-vv"])

    assert_clean_result(result, stderr=message("energy-grid job status") + "\n")
    assert seen == {"jobid": "job7", "detail": 2}


def test_click_status_json_reuses_remote_machine_contract(monkeypatch):
    seen = {}

    def status(args):
        seen.update(vars(args))
        _command.emit_json_result(
            _command.cli_json.JsonResult("cxr.remote.status", {"job": {"job_id": args.jobid}})
        )

    monkeypatch.setattr(_command.remote, "_cli_status", status)

    result = invoke(energy_grid.command, ["job", "status", "job7", "--json"])

    assert_clean_result(result, stderr=message("energy-grid job status") + "\n")
    assert '"schema":"cxr.remote.status"' in result.stdout
    assert seen == {"jobid": "job7", "verbose": 0, "json_output": True}


@pytest.mark.parametrize("status", [1, 75, 130])
def test_click_follow_logs_propagates_remote_exit_status(monkeypatch, status):
    monkeypatch.setattr(_command.remote, "tail_logs", lambda _jobid, _follow: status)

    result = invoke(energy_grid.command, ["job", "logs", "--follow"])

    assert_clean_result(
        result,
        exit_code=status,
        stderr=message("energy-grid job logs") + "\n",
    )


def test_click_submit_with_invalid_azimuths_fails(monkeypatch):
    seen = {}
    monkeypatch.setattr(energy_grid.job, "start", lambda **kwargs: seen.update(kwargs))

    result = invoke(
        energy_grid.command,
        [
            "submit",
            "--material",
            "diamond,wse2",
            "--energy",
            "100,200",
            "--polar",
            "0,1.5",
            "--azimuth",
            "45,90",
            "--thickness",
            "1000,2000",
            "--save-default",
        ],
    )

    assert result.exit_code != 0
    print(result.output)
    assert "Invalid value for '--azimuth'" in result.output
    assert seen == {}


def test_click_submit_forwards_geometry_and_set_default(monkeypatch):
    seen = {}
    monkeypatch.setattr(energy_grid.job, "start", lambda **kwargs: seen.update(kwargs))

    result = invoke(
        energy_grid.command,
        [
            "submit",
            "--material",
            "diamond,wse2",
            "--energy",
            "100,200",
            "--polar",
            "0,1.5",
            "--azimuth",
            "95,180,260",
            "--thickness",
            "1000,2000",
            "--save-default",
        ],
    )

    assert_clean_result(result)
    assert seen["materials"] == "diamond,wse2"
    assert seen["energies"] == "100,200"
    assert seen["tilts"] == "0,1.5"
    assert seen["azimuths"] == "95,180,260"
    assert seen["thickness"] == "1000,2000"
    assert seen["set_default"] is True


def test_click_submit_uses_persistent_materials_and_energies(monkeypatch):
    seen = {}
    monkeypatch.setattr(energy_grid.job, "start", lambda **kwargs: seen.update(kwargs))
    monkeypatch.setattr(
        energy_grid.defaults,
        "load_defaults",
        lambda: {
            **energy_grid.defaults.FALLBACK,
            "materials": ["wse2", "mose2"],
            "energies": [40.0, 60.0],
        },
    )

    result = invoke(energy_grid.command, ["submit", "--dry-run"])

    assert_clean_result(result)
    assert seen["materials"] == "wse2,mose2"
    assert seen["energies"] == "40,60"


def test_click_submit_routes_legacy_message_to_stderr(monkeypatch):
    monkeypatch.setattr(
        energy_grid.job,
        "start",
        lambda **_kwargs: (_ for _ in ()).throw(SystemExit("invalid material text")),
    )

    result = invoke(energy_grid.command, ["submit", "--material", "bad"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Error: invalid material text\n"
    assert "Traceback" not in result.output


def test_click_derive_forwards_brem_step(monkeypatch):
    seen = {}
    from cxr_mc.energy_grid import derive

    monkeypatch.setattr(derive, "main", lambda argv: seen.update(argv=argv) or 0)

    result = invoke(energy_grid.command, ["derive", "--brem-step", "12.5"])

    assert_clean_result(result)
    assert seen["argv"] == ["--brem-step", "12.5"]


@pytest.mark.parametrize("status", [1, 75, 130])
def test_click_derive_preserves_nonzero_status(monkeypatch, status):
    from cxr_mc.energy_grid import derive

    monkeypatch.setattr(derive, "main", lambda _argv: status)

    result = invoke(energy_grid.command, ["derive"])

    assert_clean_result(result, exit_code=status)


def test_click_apply_dispatches_with_pull_and_force(monkeypatch):
    seen = {}
    monkeypatch.setattr(_command, "_pull_combined", lambda: "combined.json")
    monkeypatch.setattr(
        energy_grid.apply, "apply_file", lambda path, **kw: seen.update(path=path, **kw)
    )

    result = invoke(energy_grid.command, ["apply", "--pull", "--force"])

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
        energy_grid.apply, "apply_file", lambda *_args, **_kwargs: (_ for _ in ()).throw(error)
    )

    result = invoke(energy_grid.command, ["apply", "bounds.json"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == f"Error: {message}\n"
    assert "Traceback" not in result.output


@pytest.mark.parametrize("command_name", ["line", "brem"])
def test_click_set_expected_domain_failure_uses_stderr(monkeypatch, command_name):
    target = "set_line_grid" if command_name == "line" else "set_brem_grid"
    monkeypatch.setattr(
        energy_grid.apply,
        target,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("unknown material")),
    )
    argv = [command_name, "set", "bad", "--stop", "100"]
    if command_name == "line":
        argv.extend(("--energy", "50"))

    result = invoke(energy_grid.command, argv)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Error: unknown material\n"
    assert "Traceback" not in result.output


def test_click_line_delete_confirmed_deletes_and_warns_stale_golden(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        energy_grid.apply,
        "delete_line_grid",
        lambda material, energies, **kw: (
            seen.update(material=material, energies=list(energies)) or [30.0, 100.0]
        ),
    )

    result = invoke(
        energy_grid.command,
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
        energy_grid.apply,
        "delete_line_grid",
        lambda *_a, **_kw: pytest.fail("delete_line_grid must not run when declined"),
    )

    result = invoke(energy_grid.command, ["line", "delete", "hopg", "--energy", "30"], input="n\n")

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Aborted" in result.stderr


def test_click_line_delete_yes_skips_prompt(monkeypatch):
    monkeypatch.setattr(
        energy_grid.apply, "delete_line_grid", lambda material, energies, **kw: [30.0]
    )

    result = invoke(energy_grid.command, ["line", "delete", "hopg", "--energy", "30", "--yes"])

    assert result.exit_code == 0
    assert "deleted hopg: 30 keV" in result.stdout


def test_click_line_delete_dry_run_skips_prompt_and_confirms_via_kwarg(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        energy_grid.apply,
        "delete_line_grid",
        lambda material, energies, **kw: seen.update(kw) or [30.0],
    )

    result = invoke(energy_grid.command, ["line", "delete", "hopg", "--energy", "30", "--dry-run"])

    assert result.exit_code == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert seen == {"dry_run": True}


def test_click_line_delete_dry_run_and_json_conflict():
    result = invoke(
        energy_grid.command, ["line", "delete", "hopg", "--energy", "30", "--dry-run", "--json"]
    )

    assert result.exit_code == 2
    assert "--dry-run and --json cannot be combined" in result.stderr


def test_click_line_delete_json_requires_yes():
    result = invoke(energy_grid.command, ["line", "delete", "hopg", "--energy", "30", "--json"])

    assert result.exit_code == 2
    assert "--json requires --yes" in result.stderr


def test_click_line_delete_json_emits_one_envelope(monkeypatch):
    monkeypatch.setattr(
        energy_grid.apply, "delete_line_grid", lambda material, energies, **kw: [30.0]
    )

    result = invoke(
        energy_grid.command, ["line", "delete", "hopg", "--energy", "30", "--yes", "--json"]
    )

    assert_clean_result(result)
    assert result.stdout.count("\n") == 1
    import json

    document = json.loads(result.stdout)
    assert document["schema"] == "cxr.energy-grid.line-delete"
    assert document["payload"] == {"material": "hopg", "deleted_energies_keV": [30.0]}


def test_click_line_delete_expected_failure_uses_stderr(monkeypatch):
    monkeypatch.setattr(
        energy_grid.apply,
        "delete_line_grid",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            ValueError("no energy_grids entry for material: bad")
        ),
    )

    result = invoke(energy_grid.command, ["line", "delete", "bad", "--energy", "30", "--yes"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Error: no energy_grids entry for material: bad\n"
    assert "Traceback" not in result.output


def test_pull_combined_quotes_remote_scp_path(monkeypatch):
    calls = []
    monkeypatch.setattr(_command.remote.config, "HOST", "qlmc")
    monkeypatch.setattr(_command.remote.config, "REMOTE_DIR", "/srv/cxr data")
    monkeypatch.setattr(_command.remote, "_run", lambda command: calls.append(command))

    local = _command._pull_combined("bounds-result.json")

    assert local == "bounds-result.json"
    assert calls == [["scp", "qlmc:'/srv/cxr data/bounds-result.json'", "bounds-result.json"]]


def test_click_stop_requires_explicit_target():
    result = invoke(energy_grid.command, ["job", "stop"])

    assert result.exit_code == 2
    assert "needs JOBID, or use --latest" in result.stderr


def test_click_stop_previews_latest_and_yes_cancels(monkeypatch):
    seen = {}
    monkeypatch.setattr(_command.remote, "_latest_jobid", lambda: "job9")
    monkeypatch.setattr(_command.remote, "_stop_jobid", lambda jobid: seen.update(jobid=jobid))

    preview = invoke(energy_grid.command, ["job", "stop", "--latest"])
    confirmed = invoke(energy_grid.command, ["job", "stop", "--latest", "-y"])

    assert_clean_result(
        preview,
        stdout=("would cancel remote job: job9\npreview only; re-run with -y/--yes to execute\n"),
        stderr=message("energy-grid job stop") + "\n",
    )
    assert_clean_result(confirmed, stderr=message("energy-grid job stop") + "\n")
    assert seen == {"jobid": "job9"}


def test_click_stop_prompts_on_tty(monkeypatch):
    seen = {}
    monkeypatch.setattr(_cli_core, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(_command.remote, "_latest_jobid", lambda: "job9")
    monkeypatch.setattr(_command.remote, "_stop_jobid", lambda jobid: seen.update(jobid=jobid))

    declined = invoke(energy_grid.command, ["job", "stop", "--latest"], input="n\n")
    accepted = invoke(energy_grid.command, ["job", "stop", "--latest"], input="y\n")

    assert_clean_result(
        declined,
        stdout="would cancel remote job: job9\n",
        stderr=message("energy-grid job stop") + "\nCancel this remote job? [y/N]: n\n",
    )
    assert_clean_result(
        accepted,
        stdout="would cancel remote job: job9\n",
        stderr=message("energy-grid job stop") + "\nCancel this remote job? [y/N]: y\n",
    )
    assert seen == {"jobid": "job9"}


def test_hidden_line_grid_job_aliases_remain_callable():
    root_help = invoke(energy_grid.command, ["--help"])
    command_lines = {
        line.split()[0]
        for line in root_help.stdout.splitlines()
        if line.startswith("  ") and line.strip() and not line.lstrip().startswith("-")
    }
    assert "job" in command_lines
    assert command_lines.isdisjoint({"status", "attach", "logs", "stop"})

    for alias in ("status", "attach", "logs", "stop"):
        result = invoke(energy_grid.command, [alias, "--help"])
        assert_clean_result(result)


def test_click_regen_golden_delegates_check(monkeypatch):
    from cxr_mc.energy_grid import golden

    seen = {}
    monkeypatch.setattr(
        golden,
        "regen",
        lambda check=False: seen.update(check=check) or 0,
    )

    result = invoke(energy_grid.command, ["regen-golden", "--check"])

    assert_clean_result(result)
    assert seen["check"] is True


@pytest.mark.parametrize("status", [1, 75, 130])
def test_click_regen_golden_preserves_nonzero_status(monkeypatch, status):
    from cxr_mc.energy_grid import golden

    monkeypatch.setattr(golden, "regen", lambda check=False: status)

    result = invoke(energy_grid.command, ["regen-golden"])

    assert_clean_result(result, exit_code=status)


def test_click_defaults_clear_fields(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        energy_grid.defaults,
        "reset_defaults",
        lambda *keys: seen.update(keys=keys) or energy_grid.defaults.FALLBACK,
    )

    result = invoke(
        energy_grid.command,
        ["defaults", "--clear", "tilts", "--clear", "brem-step"],
    )

    assert_clean_result(result)
    assert seen["keys"] == ("tilts", "brem_step_ev")


def test_click_defaults_reset_all(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        energy_grid.defaults,
        "reset_defaults",
        lambda *keys: seen.update(keys=keys) or energy_grid.defaults.FALLBACK,
    )

    result = invoke(energy_grid.command, ["defaults", "--reset"])

    assert_clean_result(result)
    assert seen["keys"] == ()


def test_click_defaults_explains_empty_angles(monkeypatch):
    monkeypatch.setattr(
        energy_grid.defaults,
        "load_defaults",
        lambda: {**energy_grid.defaults.FALLBACK, "tilts": [], "azimuths": []},
    )

    result = invoke(energy_grid.command, ["defaults"])

    assert_clean_result(result)
    assert "inherit each material's catalog-profile polar tilts" in result.stdout
    assert "inherit each material's catalog-profile azimuths" in result.stdout


@pytest.mark.parametrize(
    "argv",
    [
        ["defaults", "--set"],
        ["defaults", "--set", "--tilts", "5", "--clear", "azimuths"],
        ["defaults", "--reset", "--clear", "tilts"],
        ["defaults", "--json", "--reset"],
    ],
)
def test_click_defaults_rejects_ambiguous_mutations(argv):
    result = invoke(energy_grid.command, argv)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Error:" in result.stderr


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
    result = invoke(energy_grid.command, argv)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Invalid value" in result.stderr
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("argv", "exit_code", "stderr_text"),
    [
        (
            ["energy-grid", "defaults", "--polar", "5"],
            2,
            "--polar require --save-default",
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
            ["defaults", "--polar", "5"],
            2,
            "--polar require --save-default",
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
    result = invoke(energy_grid.command, argv)

    assert result.exit_code == exit_code
    assert result.stdout == ""
    assert stderr_text in result.stderr
    assert "Traceback" not in result.output
