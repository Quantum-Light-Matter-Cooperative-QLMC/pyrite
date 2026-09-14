import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyrite.cli.commands import energy_grid
from pyrite.cli.commands.energy_grid_surface import material_command, profile_command
from pyrite.console import output as _cli_core
from pyrite.devtools.cli_commands import energy_grid_command as dev_energy_grid_command
from pyrite.remote import config as remote_config
from tests.helpers.cli import assert_clean_result, invoke

CLICK_COMMANDS = (
    "derive",
    "add",
    "rm",
    "verify",
    "gc",
    "line",
    "brem",
    "defaults",
    "show",
)


def _invoke_grid(argv, **kwargs):
    """Invoke the canonical owner, retaining old paths only for alias tests."""
    head = argv[0] if argv else None
    if head in {"derive", "show"}:
        return invoke(material_command, argv, **kwargs)
    if head == "defaults":
        return invoke(profile_command, argv, **kwargs)
    if head in {"add", "rm", "verify", "gc"}:
        return invoke(dev_energy_grid_command, argv, **kwargs)
    if head in {"line", "brem"}:
        if len(argv) == 1 or argv[1].startswith("-") or argv[1] == "show":
            return invoke(material_command, argv, **kwargs)
        if argv[1] == "show":
            return invoke(material_command, argv, **kwargs)
        if argv[1] == "set":
            return invoke(dev_energy_grid_command, argv, **kwargs)
    raise AssertionError(f"no canonical owner for {argv!r}")


@pytest.mark.parametrize(
    "path",
    [
        *((name,) for name in CLICK_COMMANDS),
        *(((band, name)) for band in ("line", "brem") for name in ("set", "show")),
    ],
)
def test_click_help_paths_are_clean(path):
    result = _invoke_grid([*path, "--help"])

    assert_clean_result(result)
    assert "Usage:" in result.stdout


@pytest.mark.parametrize(
    "module",
    ("pyrite.energy_grid.derive", "pyrite.energy_grid.job"),
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


def test_click_derive_forwards_brem_step(monkeypatch):
    seen = {}
    from pyrite.energy_grid import derive

    monkeypatch.setattr(derive, "main", lambda argv: seen.update(argv=argv) or 0)
    monkeypatch.setattr(
        energy_grid.apply, "add_file", lambda path, **kwargs: seen.update(path=path, **kwargs)
    )

    result = _invoke_grid(["derive", "--brem-step", "12.5", "--profile", "survey"])

    assert_clean_result(result, stdout="installed derived grids for profile survey\n")
    assert seen["argv"][:2] == ["--brem-step", "12.5"]
    assert seen["argv"][2] == "--json-out"
    json_out = Path(seen["argv"][3])
    assert json_out.name == energy_grid.job.DEFAULT_JSON_OUT
    assert json_out.parent != Path(), "derive must not write into the working directory"
    assert seen["path"] == str(json_out)
    assert seen["profile"] == "survey"


def test_click_derive_remote_waits_pulls_and_restores_target(monkeypatch):
    seen = {}
    monkeypatch.setattr(remote_config, "HOST", "configured-box")
    monkeypatch.setattr(
        energy_grid.job,
        "start",
        lambda **kwargs: seen.update(kwargs, host=remote_config.remote_host()) or "job7",
    )
    monkeypatch.setattr(
        energy_grid.remote, "attach", lambda jobid: seen.update(attached=jobid) or True
    )
    monkeypatch.setattr(energy_grid.remote, "_job_succeeded", lambda jobid: jobid == "job7")
    monkeypatch.setattr(energy_grid, "_pull_combined", lambda *, dest_dir: "bounds.json")
    monkeypatch.setattr(
        energy_grid.apply, "add_file", lambda path, **kwargs: seen.update(path=path, **kwargs)
    )

    result = _invoke_grid(
        [
            "derive",
            "--remote=box-a",
            "--wait",
            "--brem-step",
            "12.5",
            "--profile",
            "hopg_hbn",
        ],
    )

    assert_clean_result(
        result,
        stdout=("pulled bounds.json\ninstalled derived grids for profile hopg_hbn\n"),
    )
    assert seen["host"] == "box-a"
    assert seen["brem_step"] == 12.5
    assert seen["attached"] == "job7"
    assert seen["path"] == "bounds.json"
    assert seen["profile"] == "hopg_hbn"
    assert remote_config.remote_host() == "configured-box"


def test_click_derive_remote_detach_skips_attach(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        energy_grid.defaults,
        "load_defaults",
        lambda: {
            **energy_grid.defaults.FALLBACK,
            "tilts": [1.5],
            "azimuths": [95.0],
            "thickness_ang": [2000.0],
            "brem_step_ev": 17.5,
        },
    )
    monkeypatch.setattr(energy_grid.job, "start", lambda **kwargs: seen.update(kwargs) or "job7")
    monkeypatch.setattr(
        energy_grid.remote,
        "attach",
        lambda _jobid: (_ for _ in ()).throw(AssertionError("must not attach")),
    )

    result = _invoke_grid(["derive", "--remote", "--detach"])

    assert_clean_result(result)
    assert seen["dry_run"] is False
    assert seen["tilts"] == "1.5"
    assert seen["azimuths"] == "95"
    assert seen["thickness"] == "2000"
    assert seen["brem_step"] == 17.5


def test_click_derive_remote_save_default_persists_locally(monkeypatch):
    seen = {}
    persisted = {
        **energy_grid.defaults.FALLBACK,
        "materials": ["mose2"],
        "energies": [60.0],
        "brem_step_ev": 12.5,
    }
    monkeypatch.setattr(energy_grid.defaults, "load_defaults", lambda: persisted)
    monkeypatch.setattr(
        energy_grid.defaults,
        "update_defaults",
        lambda **kwargs: seen.update(saved=kwargs) or persisted,
    )
    monkeypatch.setattr(
        energy_grid.job, "start", lambda **kwargs: seen.update(job=kwargs) or "job7"
    )

    result = _invoke_grid(
        ["derive", "--remote", "--detach", "--material", "mose2", "--save-default"],
    )

    assert_clean_result(result)
    assert seen["saved"]["materials"] == ["mose2"]
    assert seen["job"]["set_default"] is True


def test_click_derive_remote_failed_job_exits_nonzero(monkeypatch):
    monkeypatch.setattr(energy_grid.job, "start", lambda **_kwargs: "job7")
    monkeypatch.setattr(energy_grid.remote, "attach", lambda _jobid: True)
    monkeypatch.setattr(energy_grid.remote, "_job_succeeded", lambda _jobid: False)

    result = _invoke_grid(["derive", "--remote"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Error: energy-grid derivation failed; skipping automatic pull\n"


def test_click_derive_rejects_incompatible_locality_controls():
    conflict = _invoke_grid(["derive", "--remote", "--wait", "--detach"])
    local_only = _invoke_grid(["derive", "--dry-run"])

    assert conflict.exit_code == 2
    assert "--wait and --detach are mutually exclusive" in conflict.stderr
    assert local_only.exit_code == 2
    assert "remote-only option(s) require -R/--remote: --dry-run" in local_only.stderr


@pytest.mark.parametrize("status", [1, 75, 130])
def test_click_derive_preserves_nonzero_status(monkeypatch, status):
    from pyrite.energy_grid import derive

    monkeypatch.setattr(derive, "main", lambda _argv: status)

    result = _invoke_grid(["derive"])

    assert_clean_result(result, exit_code=status)


def test_click_add_pull_stages_into_a_removed_temp_dir(monkeypatch, tmp_path):
    """--pull must not leave the dated JSON in the working directory.

    ``apply.add_file`` ingests the payload into the content-addressed artifact
    store, so the pulled file is pure transport: it is staged in a temporary
    directory that is gone once the command returns.
    """
    seen = {}

    def fake_pull(json_name=None, *, dest_dir):
        local = Path(dest_dir) / "combined.json"
        local.write_text("{}")
        seen["dest_dir"] = dest_dir
        return str(local)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(energy_grid, "_pull_combined", fake_pull)
    monkeypatch.setattr(
        energy_grid.apply, "add_file", lambda path, **kw: seen.update(existed=Path(path).is_file())
    )

    result = _invoke_grid(["add", "--pull"])

    assert_clean_result(result)
    assert seen["existed"] is True
    assert not Path(seen["dest_dir"]).exists()
    assert list(tmp_path.iterdir()) == []


def test_click_add_dispatches_with_pull_force_and_resolved_profile(monkeypatch):
    seen = {}
    monkeypatch.setattr(energy_grid, "_pull_combined", lambda *, dest_dir: "combined.json")
    monkeypatch.setattr(
        energy_grid.apply, "add_file", lambda path, **kw: seen.update(path=path, **kw)
    )

    result = _invoke_grid(["add", "--pull", "--force", "--profile", "survey"])

    assert_clean_result(result)
    assert seen["path"] == "combined.json"
    assert seen["force"] is True
    assert seen["profile"] == "survey"


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (KeyError("section [materials.bad] not found"), "section [materials.bad] not found"),
        (ValueError("invalid derived bounds"), "invalid derived bounds"),
        (OSError("cannot read bounds.json"), "cannot read bounds.json"),
    ],
)
def test_click_add_expected_failures_use_stderr(monkeypatch, error, message):
    monkeypatch.setattr(
        energy_grid.apply, "add_file", lambda *_args, **_kwargs: (_ for _ in ()).throw(error)
    )

    result = _invoke_grid(["add", "bounds.json"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == f"Error: {message}\n"
    assert "Traceback" not in result.output


@pytest.mark.parametrize("command_name", ["line", "brem"])
def test_click_set_expected_domain_failure_uses_stderr(monkeypatch, command_name):
    target = "set_line_artifact" if command_name == "line" else "set_brem_artifact"
    monkeypatch.setattr(
        energy_grid.apply,
        target,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("unknown material")),
    )
    argv = [command_name, "set", "bad", "--stop", "100"]
    if command_name == "line":
        argv.extend(("--energy", "50"))

    result = _invoke_grid(argv)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Error: unknown material\n"
    assert "Traceback" not in result.output


def test_click_rm_confirmed_repoints_exact_preview(monkeypatch):
    calls = []
    monkeypatch.setattr(_cli_core, "_stdin_is_tty", lambda: True)

    def remove_line_rows(material, energies, **kwargs):
        calls.append((material, list(energies), kwargs))
        return [30.0, 100.0], "d" * 64

    monkeypatch.setattr(
        energy_grid.apply,
        "remove_line_rows",
        remove_line_rows,
    )

    result = _invoke_grid(
        ["rm", "hopg", "--energy", "30", "--energy", "100"],
        input="y\n",
    )

    assert result.exit_code == 0
    assert calls[0] == (
        "hopg",
        [30.0, 100.0],
        {"profile": "standard", "dry_run": True},
    )
    assert calls[1][0:2] == ("hopg", [30.0, 100.0])
    assert isinstance(calls[1][2]["expected_original"], str)
    assert len(calls) == 2
    assert "repointed standard/hopg" in result.stdout
    assert "old artifact remains recoverable until gc" in result.stderr


def test_click_rm_declined_confirmation_aborts(monkeypatch):
    monkeypatch.setattr(_cli_core, "_stdin_is_tty", lambda: True)

    def preview_only(_material, _energies, **kwargs):
        if not kwargs.get("dry_run"):
            pytest.fail("remove_line_rows must not execute when declined")
        return [30.0], "d" * 64

    monkeypatch.setattr(
        energy_grid.apply,
        "remove_line_rows",
        preview_only,
    )

    result = _invoke_grid(["rm", "hopg", "--energy", "30"], input="n\n")

    assert result.exit_code == 0
    assert result.stdout == ""
    assert "old artifact remains recoverable until gc" in result.stderr


def test_click_rm_non_tty_previews_without_mutating(monkeypatch):
    seen = []
    monkeypatch.setattr(_cli_core, "_stdin_is_tty", lambda: False)

    def preview_only(material, energies, **kwargs):
        seen.append((material, list(energies), kwargs))
        return [30.0], "d" * 64

    monkeypatch.setattr(energy_grid.apply, "remove_line_rows", preview_only)

    result = _invoke_grid(["rm", "hopg", "--energy", "30"])

    assert_clean_result(
        result,
        stdout="preview only; re-run with -y/--yes to execute\n",
    )
    assert seen == [("hopg", [30.0], {"profile": "standard", "dry_run": True})]


def test_click_rm_yes_skips_prompt(monkeypatch):
    monkeypatch.setattr(
        energy_grid.apply,
        "remove_line_rows",
        lambda material, energies, **kw: ([30.0], "d" * 64),
    )

    result = _invoke_grid(["rm", "hopg", "--energy", "30", "--yes"])

    assert result.exit_code == 0
    assert "repointed standard/hopg" in result.stdout


def test_click_verify_reports_integrity_and_failures(monkeypatch):
    monkeypatch.setattr(
        energy_grid.artifact_gc,
        "verify_artifacts",
        lambda *args, **kwargs: SimpleNamespace(ok=True, inventory=("a",), roots=("a",), issues=()),
    )
    clean = _invoke_grid(["verify"])
    assert_clean_result(clean, stdout="verified 1 stored artifact(s); 1 reachable ref(s)\n")

    issue = SimpleNamespace(digest="d" * 64, source="catalog", message="is missing")
    monkeypatch.setattr(
        energy_grid.artifact_gc,
        "verify_artifacts",
        lambda *args, **kwargs: SimpleNamespace(
            ok=False, inventory=(), roots=("d" * 64,), issues=(issue,)
        ),
    )
    failed = _invoke_grid(["verify"])
    assert failed.exit_code == 1
    assert "artifact verification failed" in failed.stderr
    assert "is missing" in failed.stderr


def test_click_gc_previews_non_tty_and_yes_executes_exact_plan(monkeypatch):
    candidate = SimpleNamespace(path=Path("store/aa/artifact.json"), digest="a" * 64)
    plan = SimpleNamespace(candidates=(candidate,), retained=("b" * 64,))
    monkeypatch.setattr(energy_grid.artifact_gc, "plan_gc", lambda *args, **kwargs: plan)
    executed = []
    monkeypatch.setattr(
        energy_grid.artifact_gc,
        "execute_gc",
        lambda selected: executed.append(selected) or (candidate.path,),
    )
    monkeypatch.setattr(_cli_core, "_stdin_is_tty", lambda: False)

    preview = _invoke_grid(["gc"])
    confirmed = _invoke_grid(["gc", "--prune-all", "-y"])

    assert "would delete unreachable energy-grid artifacts" in preview.stdout
    assert "preview only; re-run with -y/--yes to execute" in preview.stdout
    assert executed == [plan]
    assert "deleted 1 unreachable artifact(s)" in confirmed.stdout


def test_click_rm_dry_run_skips_prompt_and_confirms_via_kwarg(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        energy_grid.apply,
        "remove_line_rows",
        lambda material, energies, **kw: seen.update(kw) or ([30.0], "d" * 64),
    )

    result = _invoke_grid(["rm", "hopg", "--energy", "30", "--dry-run"])

    assert result.exit_code == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert seen == {"profile": "standard", "dry_run": True}


def test_click_rm_dry_run_and_json_conflict():
    result = _invoke_grid(
        ["rm", "hopg", "--energy", "30", "--dry-run", "-o", "json"],
    )

    assert result.exit_code == 2
    assert "--dry-run and --output json cannot be combined" in result.stderr


def test_click_rm_json_requires_yes():
    result = _invoke_grid(["rm", "hopg", "--energy", "30", "-o", "json"])

    assert result.exit_code == 2
    assert "--output json requires --yes" in result.stderr


def test_click_rm_json_emits_one_envelope(monkeypatch):
    monkeypatch.setattr(
        energy_grid.apply,
        "remove_line_rows",
        lambda material, energies, **kw: ([30.0], "d" * 64),
    )

    result = _invoke_grid(
        ["rm", "hopg", "--energy", "30", "--yes", "-o", "json"],
    )

    assert_clean_result(result)
    assert result.stdout.count("\n") == 1
    import json

    document = json.loads(result.stdout)
    assert document["schema"] == "cxr.energy-grid.rm"
    assert document["payload"] == {
        "material": "hopg",
        "profile": "standard",
        "removed_energies_keV": [30.0],
        "artifact_sha256": "d" * 64,
    }


def test_click_rm_expected_failure_uses_stderr(monkeypatch):
    monkeypatch.setattr(
        energy_grid.apply,
        "remove_line_rows",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            ValueError("no energy_grids entry for material: bad")
        ),
    )

    result = _invoke_grid(["rm", "bad", "--energy", "30", "--yes"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Error: no energy_grids entry for material: bad\n"
    assert "Traceback" not in result.output


def test_pull_combined_quotes_remote_scp_path(monkeypatch):
    calls = []
    monkeypatch.setattr(energy_grid.remote.config, "HOST", "qlmc")
    monkeypatch.setattr(energy_grid.remote.config, "REMOTE_DIR", "/srv/pyrite data")
    monkeypatch.setattr(energy_grid.remote, "_run", lambda command: calls.append(command))

    local = energy_grid._pull_combined("bounds-result.json", dest_dir="/tmp/pull-dest")

    assert local == "/tmp/pull-dest/bounds-result.json"
    assert calls == [
        ["scp", "qlmc:'/srv/pyrite data/bounds-result.json'", "/tmp/pull-dest/bounds-result.json"]
    ]


def test_click_defaults_clear_fields(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        energy_grid.defaults,
        "reset_defaults",
        lambda *keys: seen.update(keys=keys) or energy_grid.defaults.FALLBACK,
    )

    result = _invoke_grid(
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

    result = _invoke_grid(["defaults", "--reset"])

    assert_clean_result(result)
    assert seen["keys"] == ()


def test_click_defaults_explains_empty_angles(monkeypatch):
    monkeypatch.setattr(
        energy_grid.defaults,
        "load_defaults",
        lambda: {**energy_grid.defaults.FALLBACK, "tilts": [], "azimuths": []},
    )

    result = _invoke_grid(["defaults"])

    assert_clean_result(result)
    assert "inherit each material's catalog-profile polar tilts" in result.stdout
    assert "inherit each material's catalog-profile azimuths" in result.stdout


@pytest.mark.parametrize(
    "argv",
    [
        ["defaults", "--set"],
        ["defaults", "--set", "--tilts", "5", "--clear", "azimuths"],
        ["defaults", "--reset", "--clear", "tilts"],
        ["defaults", "-o", "json", "--reset"],
    ],
)
def test_click_defaults_rejects_ambiguous_mutations(argv):
    result = _invoke_grid(argv)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Error:" in result.stderr


@pytest.mark.parametrize(
    "argv",
    [
        ["line", "set", "hopg", "--energy", "0", "--stop", "3000"],
        ["line", "set", "hopg", "--energy", "30", "--stop", "-1"],
        ["line", "set", "hopg", "--energy", "30", "--stop", "3000", "--num", "0"],
        ["brem", "set", "hopg", "--stop", "1000", "--step", "0"],
        ["defaults", "--save-default", "--polar", "90"],
        ["defaults", "--save-default", "--azimuth", "361"],
        ["defaults", "--save-default", "--thickness", "0"],
        ["defaults", "--save-default", "--brem-step", "nan"],
        ["derive", "--energy", "0"],
        ["derive", "--polar", "90"],
    ],
)
def test_click_numeric_domains_fail_at_boundary(argv):
    result = _invoke_grid(argv)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Invalid value" in result.stderr
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("argv", "exit_code", "stderr_text"),
    [
        (
            ["profile", "energy-grid", "defaults", "--polar", "5"],
            2,
            "--polar require --save-default",
        ),
        (
            ["material", "energy-grid", "show", "not-a-material"],
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
            "from pyrite.cli import main; raise SystemExit(main())",
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
            ["add"],
            2,
            "no JSON: pass a path or --pull",
        ),
    ],
)
def test_click_failure_exit_and_stream_contract(argv, exit_code, stderr_text):
    result = _invoke_grid(argv)

    assert result.exit_code == exit_code
    assert result.stdout == ""
    assert stderr_text in result.stderr
    assert "Traceback" not in result.output
